#!/usr/bin/env python3
"""Stage 1: Rule-based filtering of Dolci-Think-SFT-7B → candidates for Claude API scoring."""

import json
import os
import random
import re
from collections import Counter

from datasets import load_dataset

from config import (
    AI_MENTION_PATTERN,
    BULLET_PATTERN,
    HF_DATASET,
    MAX_ANSWER_CHARS,
    MAX_RESPONSE_CHARS,
    MIN_RESPONSE_CHARS,
    OUTPUT_DIR,
    PERSONALITY_OPENERS,
    REFUSAL_PATTERN,
    classify_source,
)


def extract_final_answer(assistant_content: str) -> str | None:
    """Extract the text after </think> tags."""
    if "</think>" not in assistant_content:
        return None
    return assistant_content.split("</think>", 1)[1].strip()


def extract_think_content(assistant_content: str) -> str | None:
    """Extract text inside <think>...</think> tags."""
    match = re.search(r"<think>(.*?)</think>", assistant_content, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def is_english(text: str) -> bool:
    """Heuristic: text is mostly ASCII (allows some math/code unicode)."""
    if not text:
        return True
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    return ascii_chars / len(text) > 0.85


def passes_content_filters(row: dict) -> tuple[bool, str]:
    """Apply content-level filters. Returns (pass, reason)."""
    messages = row["messages"]

    # Single-turn only (1 user + 1 assistant, optionally 1 system)
    roles = [m["role"] for m in messages]
    user_count = roles.count("user")
    asst_count = roles.count("assistant")
    if user_count != 1 or asst_count != 1:
        return False, "multi-turn"

    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    asst_msg = next(m["content"] for m in messages if m["role"] == "assistant")

    # Think tag validation
    think_content = extract_think_content(asst_msg)
    if think_content is None:
        return False, "no_think_tags"
    if len(think_content.strip()) < 20:
        return False, "empty_think"

    # Final answer extraction
    final_answer = extract_final_answer(asst_msg)
    if final_answer is None:
        return False, "no_final_answer"

    # Response length (total)
    if len(asst_msg) < MIN_RESPONSE_CHARS:
        return False, "too_short"
    if len(asst_msg) > MAX_RESPONSE_CHARS:
        return False, "too_long"

    # Final answer length — keep it short/succinct
    if len(final_answer) > MAX_ANSWER_CHARS:
        return False, "answer_too_long"

    # Refusal detection (in final answer only)
    if REFUSAL_PATTERN.search(final_answer):
        return False, "refusal"

    # Personality openers (in final answer)
    if PERSONALITY_OPENERS.search(final_answer):
        return False, "personality"

    # Bullet points in final answer
    if BULLET_PATTERN.search(final_answer):
        return False, "bullet_points"

    # English check
    if not is_english(user_msg) or not is_english(final_answer):
        return False, "non_english"

    # AI model/identity mentions (in full text: user prompt + assistant response)
    full_text = user_msg + " " + asst_msg
    if AI_MENTION_PATTERN.search(full_text):
        return False, "ai_mention"

    return True, "pass"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"Loaded {len(ds):,} examples")

    # Stage 1: Source-level + content filtering
    category_counts = Counter()
    filter_reasons = Counter()
    kept = []

    for i in range(len(ds)):
        if i % 100000 == 0:
            print(f"  Processing {i:,}/{len(ds):,}... kept so far: {len(kept):,}")

        row = ds[i]
        category = classify_source(row["dataset_source"])

        if category is None:
            filter_reasons["source_removed"] += 1
            continue

        passes, reason = passes_content_filters(row)
        if not passes:
            filter_reasons[reason] += 1
            continue

        # Keep this example
        entry = {
            "id": row["id"],
            "dataset_source": row["dataset_source"],
            "category": category,
            "messages": row["messages"],
        }
        kept.append(entry)
        category_counts[category] += 1

    print(f"\n{'='*80}")
    print(f"Filtering complete: {len(ds):,} → {len(kept):,}")
    print(f"\nKept by category:")
    for cat, count in category_counts.most_common():
        print(f"  {cat}: {count:,}")
    print(f"\nFilter reasons (removed):")
    for reason, count in filter_reasons.most_common():
        print(f"  {reason}: {count:,}")

    # Save full filtered set
    output_path = os.path.join(OUTPUT_DIR, "stage1_rule_filtered.jsonl")
    with open(output_path, "w") as f:
        for entry in kept:
            f.write(json.dumps(entry) + "\n")
    print(f"\nSaved {len(kept):,} examples to {output_path}")

    # Category-balanced sampling for Claude API (~5000 total)
    SAMPLE_TARGET = 5000
    by_category = {}
    for entry in kept:
        by_category.setdefault(entry["category"], []).append(entry)

    per_cat = SAMPLE_TARGET // len(by_category)
    sampled = []
    rng = random.Random(42)
    for cat, entries in by_category.items():
        take = min(per_cat, len(entries))
        sampled.extend(rng.sample(entries, take))

    rng.shuffle(sampled)
    sampled_path = os.path.join(OUTPUT_DIR, "stage1_sampled_for_api.jsonl")
    with open(sampled_path, "w") as f:
        for entry in sampled:
            f.write(json.dumps(entry) + "\n")

    sampled_cats = Counter(e["category"] for e in sampled)
    print(f"\nSampled {len(sampled):,} for Claude API:")
    for cat, count in sampled_cats.most_common():
        print(f"  {cat}: {count:,}")
    print(f"Saved to {sampled_path}")


if __name__ == "__main__":
    main()
