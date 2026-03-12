#!/usr/bin/env python3
"""Compare Opus, Sonnet, and Haiku scoring on 20 examples."""

import asyncio
import json
import os
import random
import sys
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

from config import OUTPUT_DIR

# Same prompt as 02_claude_filter.py
EVAL_PROMPT = open("02_claude_filter.py").read().split('EVAL_PROMPT = """\\\n')[1].split('"""')[0]

MODELS = {
    "opus": "claude-opus-4-6",
    "sonnet": "claude-sonnet-4-6",
    "haiku": "claude-haiku-4-5-20251001",
}

PRICING = {
    "claude-opus-4-6": {"input": 15.0, "output": 75.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5-20251001": {"input": 0.80, "output": 4.0},
}


def format_example(messages):
    parts = []
    for msg in messages:
        parts.append(f"[{msg['role'].upper()}]\n{msg['content']}")
    return "\n\n".join(parts)


async def score_one(client, model, example, sem):
    async with sem:
        formatted = format_example(example["messages"])
        prompt_text = f"{EVAL_PROMPT}\n\nEXAMPLE TO EVALUATE:\n\n{formatted}"
        try:
            resp = await client.messages.create(
                model=model,
                max_tokens=300,
                messages=[{"role": "user", "content": prompt_text}],
            )
            text = resp.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            scores = json.loads(text)
            return {
                "scores": scores,
                "input_tokens": resp.usage.input_tokens,
                "output_tokens": resp.usage.output_tokens,
            }
        except Exception as e:
            return {"scores": {"error": str(e)}, "input_tokens": 0, "output_tokens": 0}


async def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Set ANTHROPIC_API_KEY")
        sys.exit(1)

    # Load 20 diverse examples (5 per category)
    with open(os.path.join(OUTPUT_DIR, "stage1_sampled_for_api.jsonl")) as f:
        all_examples = [json.loads(line) for line in f]

    rng = random.Random(123)
    by_cat = {}
    for ex in all_examples:
        by_cat.setdefault(ex["category"], []).append(ex)

    sample = []
    for cat in sorted(by_cat):
        sample.extend(rng.sample(by_cat[cat], min(5, len(by_cat[cat]))))

    print(f"Testing {len(sample)} examples across {len(MODELS)} models\n")

    client = anthropic.AsyncAnthropic(api_key=api_key)
    sem = asyncio.Semaphore(10)

    results = {}  # model_name -> list of (example_id, category, scores)
    token_usage = {}

    for name, model_id in MODELS.items():
        print(f"Running {name} ({model_id})...")
        t0 = time.time()
        tasks = [score_one(client, model_id, ex, sem) for ex in sample]
        raw = await asyncio.gather(*tasks)
        elapsed = time.time() - t0

        in_tok = sum(r["input_tokens"] for r in raw)
        out_tok = sum(r["output_tokens"] for r in raw)
        prices = PRICING[model_id]
        cost = in_tok / 1e6 * prices["input"] + out_tok / 1e6 * prices["output"]

        results[name] = [(sample[i]["id"], sample[i]["category"], r["scores"]) for i, r in enumerate(raw)]
        token_usage[name] = {"input": in_tok, "output": out_tok, "cost": cost, "time": elapsed}
        errors = sum(1 for r in raw if "error" in r["scores"])
        print(f"  Done in {elapsed:.1f}s | {in_tok:,} in + {out_tok:,} out | ${cost:.3f} | {errors} errors")

    # Compare scores
    print(f"\n{'='*100}")
    print("SCORE COMPARISON (per example)")
    print(f"{'='*100}")

    score_keys = ["minimal_personality", "answer_conciseness", "no_assistant_behavior", "thinking_quality", "overall_suitability"]
    bool_keys = ["is_refusal", "has_bullet_points", "has_personality", "follows_format"]

    for i in range(len(sample)):
        cat = sample[i]["category"]
        eid = sample[i]["id"][:50]
        print(f"\n--- [{cat}] {eid} ---")

        header = f"  {'criterion':<25}"
        for name in MODELS:
            header += f" {name:>8}"
        print(header)

        for key in score_keys:
            row = f"  {key:<25}"
            for name in MODELS:
                s = results[name][i][2]
                val = s.get(key, "ERR")
                row += f" {val:>8}"
            print(row)

        for key in bool_keys:
            row = f"  {key:<25}"
            for name in MODELS:
                s = results[name][i][2]
                val = s.get(key, "ERR")
                row += f" {str(val):>8}"
            print(row)

        # Show rationales
        for name in MODELS:
            rat = results[name][i][2].get("brief_rationale", "")
            print(f"  {name}: {rat}")

    # Aggregate stats
    print(f"\n{'='*100}")
    print("AGGREGATE COMPARISON")
    print(f"{'='*100}")

    for name in MODELS:
        valid = [r[2] for r in results[name] if "error" not in r[2]]
        if not valid:
            print(f"\n{name}: all errors")
            continue
        print(f"\n{name} (n={len(valid)}):")
        for key in score_keys:
            vals = [s[key] for s in valid if key in s]
            if vals:
                print(f"  {key:<25} mean={sum(vals)/len(vals):.2f}  min={min(vals)}  max={max(vals)}")
        for key in bool_keys:
            vals = [s[key] for s in valid if key in s]
            if vals:
                true_count = sum(1 for v in vals if v)
                print(f"  {key:<25} true={true_count}/{len(vals)}")

    # Agreement
    print(f"\n{'='*100}")
    print("INTER-MODEL AGREEMENT (overall_suitability)")
    print(f"{'='*100}")

    model_names = list(MODELS.keys())
    for i_m in range(len(model_names)):
        for j_m in range(i_m + 1, len(model_names)):
            m1, m2 = model_names[i_m], model_names[j_m]
            diffs = []
            agrees = 0
            for k in range(len(sample)):
                s1 = results[m1][k][2].get("overall_suitability")
                s2 = results[m2][k][2].get("overall_suitability")
                if s1 is not None and s2 is not None:
                    diffs.append(abs(s1 - s2))
                    if s1 == s2:
                        agrees += 1
            if diffs:
                print(f"  {m1} vs {m2}: exact_agree={agrees}/{len(diffs)} mean_diff={sum(diffs)/len(diffs):.2f} max_diff={max(diffs)}")

    # Bool agreement
    print(f"\nBOOL FLAG AGREEMENT")
    for key in bool_keys:
        for i_m in range(len(model_names)):
            for j_m in range(i_m + 1, len(model_names)):
                m1, m2 = model_names[i_m], model_names[j_m]
                agrees = 0
                total = 0
                for k in range(len(sample)):
                    v1 = results[m1][k][2].get(key)
                    v2 = results[m2][k][2].get(key)
                    if v1 is not None and v2 is not None:
                        total += 1
                        if v1 == v2:
                            agrees += 1
                if total:
                    print(f"  {key:<25} {m1} vs {m2}: {agrees}/{total} agree")

    # Cost projection
    print(f"\n{'='*100}")
    print("COST PROJECTION FOR FULL 5,968 EXAMPLES")
    print(f"{'='*100}")
    for name in MODELS:
        u = token_usage[name]
        scale = 5968 / 20
        proj_cost = u["cost"] * scale
        proj_time = u["time"] * scale / 60  # minutes (sequential, but we use concurrency)
        print(f"  {name}: ${proj_cost:.2f} estimated | {u['time']:.1f}s for 20 examples")


if __name__ == "__main__":
    asyncio.run(main())
