#!/usr/bin/env python3
"""Stage 3: Assemble final ~1000 warm-start SFT examples from Claude-scored candidates."""

import json
import os
import random
import re
from collections import Counter

from config import OUTPUT_DIR


def is_mcq(example: dict) -> bool:
    """Check if example is a multiple-choice question."""
    user_msg = next((m["content"] for m in example["messages"] if m["role"] == "user"), "")
    mcq_pattern = r"\([A-E]\)"
    return bool(re.search(mcq_pattern, user_msg))


def main():
    input_path = os.path.join(OUTPUT_DIR, "stage2_claude_scored.jsonl")
    output_path = os.path.join(OUTPUT_DIR, "warm_start_sft_1150.jsonl")
    per_cat = 250
    total_target = 1150

    with open(input_path) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} scored examples")

    # Minimal filters: valid scores, no refusals, follows format
    filtered = []
    for ex in examples:
        scores = ex.get("scores", {})
        if "error" in scores:
            continue
        if not isinstance(scores.get("overall_suitability"), (int, float)):
            continue
        if scores.get("is_refusal", False):
            continue
        if not scores.get("follows_format", True):
            continue
        filtered.append(ex)

    print(f"After minimal filters (no errors/refusals, follows_format): {len(filtered)}")

    # Group by category
    by_category = {}
    for ex in filtered:
        cat = ex["category"]
        by_category.setdefault(cat, []).append(ex)

    # Sort each category by quality
    for cat in by_category:
        by_category[cat].sort(
            key=lambda x: (
                x["scores"]["overall_suitability"],
                x["scores"].get("thinking_quality", 0),
                x["scores"].get("answer_conciseness", 0),
            ),
            reverse=True,
        )

    print(f"\nAvailable by category:")
    for cat, exs in sorted(by_category.items()):
        print(f"  {cat}: {len(exs)}")

    # Take top 250 from each category
    selected = []
    selected_ids = set()
    for cat, exs in by_category.items():
        take = min(per_cat, len(exs))
        selected.extend(exs[:take])
        selected_ids.update(ex["id"] for ex in exs[:take])

    # Ensure at least 200 MCQs: find best MCQs not already selected
    mcq_min = 200
    mcq_in_selected = [ex for ex in selected if is_mcq(ex)]
    if len(mcq_in_selected) < mcq_min:
        needed = mcq_min - len(mcq_in_selected)
        # Gather all filtered MCQs not already selected, sorted by quality
        mcq_pool = [
            ex for ex in filtered
            if is_mcq(ex) and ex["id"] not in selected_ids
        ]
        mcq_pool.sort(
            key=lambda x: (
                x["scores"]["overall_suitability"],
                x["scores"].get("thinking_quality", 0),
                x["scores"].get("answer_conciseness", 0),
            ),
            reverse=True,
        )
        mcq_add = mcq_pool[:needed]
        selected.extend(mcq_add)
        print(f"\nMCQ boost: added {len(mcq_add)} extra MCQs (had {len(mcq_in_selected)}, target {mcq_min})")

    # Trim to exact target by removing lowest-scoring non-MCQ examples
    if len(selected) > total_target:
        excess = len(selected) - total_target
        # Sort non-MCQ examples by score ascending to find worst ones to drop
        non_mcq_with_idx = [
            (i, ex) for i, ex in enumerate(selected) if not is_mcq(ex)
        ]
        non_mcq_with_idx.sort(
            key=lambda t: (
                t[1]["scores"]["overall_suitability"],
                t[1]["scores"].get("thinking_quality", 0),
            ),
        )
        drop_indices = {t[0] for t in non_mcq_with_idx[:excess]}
        selected = [ex for i, ex in enumerate(selected) if i not in drop_indices]
        print(f"Trimmed {excess} lowest-scoring non-MCQ examples to hit {total_target}")

    mcq_count = sum(1 for ex in selected if is_mcq(ex))

    # Score distribution per category
    print(f"\nSelected {len(selected)} examples:")
    for cat in sorted(by_category):
        cat_selected = [ex for ex in selected if ex["category"] == cat]
        scores = [ex["scores"]["overall_suitability"] for ex in cat_selected]
        dist = Counter(scores)
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items(), reverse=True))
        cat_mcq = sum(1 for ex in cat_selected if is_mcq(ex))
        print(f"  {cat}: {len(cat_selected)} (scores: {dist_str}) [{cat_mcq} MCQ]")

    print(f"\nTotal MCQ examples: {mcq_count}")

    # Shuffle final order
    random.seed(42)
    random.shuffle(selected)

    # Write output (messages only — clean format)
    with open(output_path, "w") as f:
        for ex in selected:
            out = {"messages": ex["messages"]}
            f.write(json.dumps(out) + "\n")

    print(f"\nWrote {len(selected)} examples to {output_path}")


if __name__ == "__main__":
    main()
