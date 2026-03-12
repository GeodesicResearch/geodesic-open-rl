#!/usr/bin/env python3
"""Stage 2: Claude API quality filtering of rule-filtered candidates."""

import argparse
import asyncio
import json
import os
import re
import sys
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

from config import OUTPUT_DIR

EVAL_PROMPT = """\
You are evaluating a supervised fine-tuning example for a reasoning model. The example has a user prompt and an assistant response containing <think>...</think> reasoning followed by a final answer.

This example will be used to warm-start reinforcement learning training. We need examples with MINIMAL persona — the model should reason well but not act like a chatbot. The use of pronouns in the reasoning such as "I" or "my" is permitted.

Score each criterion 1-5:

1. **minimal_personality** (5=completely neutral, no filler; 1=heavy persona — "Great question!", "I'd be happy to help!", cheerful padding)
2. **answer_conciseness** (5=short, direct answer; 1=verbose essay, unnecessary elaboration, lists where a direct answer suffices)
3. **no_assistant_behavior** (5=no "Let me explain", no "In summary", no chatbot-style framing; 1=heavy assistant-style behavior)
4. **thinking_quality** (5=clear step-by-step reasoning, stays on-topic, honest about uncertainty; 1=repetitive, circular, or off-topic reasoning)
5. **overall_suitability** (5=excellent warm-start example; 1=should not be included)

Also flag:
- **is_refusal**: true if the response refuses the request
- **has_bullet_points**: true if the final answer uses bullet points (dash or asterisk lists). Numbered lists and code blocks are fine.
- **has_personality**: true if the response exhibits chatbot personality traits
- **follows_format**: CRITICAL — Check if the user prompt requests a specific answer format. If so, check whether the FINAL ANSWER (after </think>) actually follows it. The most common case is multiple choice questions where the expected format is something like "Answer: (A)" or "The answer is (B)" — Set to true if the format is followed OR if no specific format was requested. Set to false if the user requested a format and the final answer violates it. This is a HARD FILTER — examples that fail this will be excluded.

Return ONLY valid JSON:
{
  "minimal_personality": <int 1-5>,
  "answer_conciseness": <int 1-5>,
  "no_assistant_behavior": <int 1-5>,
  "thinking_quality": <int 1-5>,
  "overall_suitability": <int 1-5>,
  "is_refusal": <bool>,
  "has_bullet_points": <bool>,
  "has_personality": <bool>,
  "follows_format": <bool>,
  "brief_rationale": "<1 sentence>"
}"""

# Token pricing per million tokens (USD)
PRICING = {
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
}


def format_example(messages: list[dict]) -> str:
    """Format a messages list into a readable example for Claude."""
    parts = []
    for msg in messages:
        role = msg["role"].upper()
        parts.append(f"[{role}]\n{msg['content']}")
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token."""
    return len(text) // 4


async def evaluate_example(
    client: anthropic.AsyncAnthropic,
    example: dict,
    semaphore: asyncio.Semaphore,
    model: str,
    stats: dict,
) -> dict:
    """Score a single example with Claude API."""
    async with semaphore:
        formatted = format_example(example["messages"])
        prompt_text = f"{EVAL_PROMPT}\n\nEXAMPLE TO EVALUATE:\n\n{formatted}"
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=300,
                messages=[
                    {"role": "user", "content": prompt_text},
                ],
            )
            text = response.content[0].text.strip()

            # Track token usage
            stats["input_tokens"] += response.usage.input_tokens
            stats["output_tokens"] += response.usage.output_tokens

            # Parse JSON from response (handle markdown code blocks)
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(text)
            return {**example, "scores": scores}
        except json.JSONDecodeError as e:
            print(f"  JSON parse error for {example['id']}: {e}", file=sys.stderr)
            return {**example, "scores": {"error": str(e), "raw": text}}
        except Exception as e:
            print(f"  API error for {example['id']}: {e}", file=sys.stderr)
            return {**example, "scores": {"error": str(e)}}


def estimate_cost(examples: list[dict], model: str) -> tuple[float, int, int]:
    """Estimate total cost before running."""
    total_input = 0
    prompt_overhead = estimate_tokens(EVAL_PROMPT)
    for ex in examples:
        formatted = format_example(ex["messages"])
        total_input += prompt_overhead + estimate_tokens(formatted)
    total_output = len(examples) * 150  # ~150 output tokens per response
    prices = PRICING.get(model, PRICING["claude-opus-4-6"])
    cost = (total_input / 1_000_000 * prices["input"]) + (total_output / 1_000_000 * prices["output"])
    return cost, total_input, total_output


async def main(args):
    input_path = args.input or os.path.join(OUTPUT_DIR, "stage1_sampled_for_api.jsonl")
    output_path = args.output or os.path.join(OUTPUT_DIR, "stage2_claude_scored.jsonl")
    model = args.model

    # Load input
    with open(input_path) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} candidates from {input_path}")

    # Load already-scored IDs for resume support
    scored_ids = set()
    scored_results = []
    if os.path.exists(output_path):
        with open(output_path) as f:
            for line in f:
                entry = json.loads(line)
                scored_ids.add(entry["id"])
                scored_results.append(entry)
        print(f"Resuming: {len(scored_ids)} already scored")

    remaining = [ex for ex in examples if ex["id"] not in scored_ids]
    print(f"Remaining to score: {len(remaining)}")

    if not remaining:
        print("Nothing to do!")
        return

    # Cost estimate
    est_cost, est_input, est_output = estimate_cost(remaining, model)
    print(f"\nModel: {model}")
    print(f"Estimated tokens: {est_input:,} input + {est_output:,} output")
    print(f"Estimated cost: ${est_cost:.2f}")

    if not args.yes:
        confirm = input("\nProceed? [y/N] ").strip().lower()
        if confirm != "y":
            print("Aborted.")
            return

    # Set up client
    api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: provide --api-key or set ANTHROPIC_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.concurrency)
    stats = {"input_tokens": 0, "output_tokens": 0}
    start_time = time.time()

    # Process in batches, saving after each
    batch_size = 500
    for batch_start in range(0, len(remaining), batch_size):
        batch = remaining[batch_start : batch_start + batch_size]
        batch_num = batch_start // batch_size + 1
        total_batches = (len(remaining) + batch_size - 1) // batch_size

        tasks = [evaluate_example(client, ex, semaphore, model, stats) for ex in batch]
        results = await asyncio.gather(*tasks)
        scored_results.extend(results)

        # Save incrementally
        with open(output_path, "w") as f:
            for entry in scored_results:
                f.write(json.dumps(entry) + "\n")

        n_done = len(scored_results)
        n_errors = sum(1 for r in results if "error" in r.get("scores", {}))
        elapsed = time.time() - start_time
        rate = (batch_start + len(batch)) / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - batch_start - len(batch)) / rate if rate > 0 else 0

        prices = PRICING.get(model, PRICING["claude-opus-4-6"])
        cost_so_far = (
            stats["input_tokens"] / 1_000_000 * prices["input"]
            + stats["output_tokens"] / 1_000_000 * prices["output"]
        )

        print(
            f"  Batch {batch_num}/{total_batches} | "
            f"{n_done}/{len(examples)} scored | "
            f"{n_errors} errors | "
            f"${cost_so_far:.2f} spent | "
            f"ETA {eta/60:.1f}m"
        )

    # Final summary
    elapsed = time.time() - start_time
    prices = PRICING.get(model, PRICING["claude-opus-4-6"])
    total_cost = (
        stats["input_tokens"] / 1_000_000 * prices["input"]
        + stats["output_tokens"] / 1_000_000 * prices["output"]
    )

    good = [r for r in scored_results if r.get("scores", {}).get("overall_suitability", 0) >= 4]
    print(f"\nDone in {elapsed/60:.1f}m!")
    print(f"Total: {len(scored_results)} scored, {len(good)} with overall_suitability >= 4")
    print(f"Tokens: {stats['input_tokens']:,} input + {stats['output_tokens']:,} output")
    print(f"Cost: ${total_cost:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Claude API quality filtering")
    parser.add_argument("--api-key", help="Anthropic API key (or set ANTHROPIC_API_KEY)")
    parser.add_argument("--input", help="Input JSONL path")
    parser.add_argument("--output", help="Output JSONL path")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Model to use (default: haiku 4.5)")
    parser.add_argument("--concurrency", type=int, default=100, help="Max concurrent API calls")
    parser.add_argument("-y", "--yes", action="store_true", help="Skip cost confirmation")
    args = parser.parse_args()
    asyncio.run(main(args))
