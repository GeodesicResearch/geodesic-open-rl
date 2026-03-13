#!/usr/bin/env python3
"""Stage 5: Assemble top 1250 warm-start SFT examples from token-filtered Claude-scored candidates.

Like 03_assemble_final.py but operates on the max-4096 filtered pool.

Usage:
    python 05_assemble_1250.py [--input INPUT] [--output OUTPUT] [--target 1250]
"""

import argparse
import getpass
import json
import os
import random
import re
from collections import Counter


def is_mcq(example: dict) -> bool:
    """Check if example is a multiple-choice question."""
    user_msg = next((m["content"] for m in example["messages"] if m["role"] == "user"), "")
    return bool(re.search(r"\([A-E]\)", user_msg))


def main():
    user = getpass.getuser()
    default_input = f"/projects/a5k/public/data_{user}/warm_start_sft/stage2_claude_scored_max4096.jsonl"
    default_output = f"/projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1250.jsonl"

    parser = argparse.ArgumentParser(description="Assemble top N examples from token-filtered scored data.")
    parser.add_argument("--input", default=default_input)
    parser.add_argument("--output", default=default_output)
    parser.add_argument("--target", type=int, default=1250)
    parser.add_argument("--mcq-min", type=int, default=200)
    parser.add_argument("--mcq-pool", default=f"/projects/a5k/public/data_{user}/warm_start_sft/mcq_upsample_max4096.jsonl",
                        help="Token-filtered MCQ upsample pool (no Claude scores)")
    args = parser.parse_args()

    with open(args.input) as f:
        examples = [json.loads(line) for line in f]
    print(f"Loaded {len(examples)} token-filtered scored examples")

    # Load MCQ upsample pool (no Claude scores — used for MCQ boosting only)
    mcq_upsample = []
    if os.path.isfile(args.mcq_pool):
        with open(args.mcq_pool) as f:
            mcq_upsample = [json.loads(line) for line in f]
        print(f"Loaded {len(mcq_upsample)} MCQ upsample examples (no scores)")

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

    print(f"After minimal filters: {len(filtered)}")

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

    n_cats = len(by_category)
    per_cat = args.target // n_cats

    print(f"\nAvailable by category (target {per_cat}/cat, {args.target} total):")
    for cat, exs in sorted(by_category.items()):
        mcq_count = sum(1 for ex in exs if is_mcq(ex))
        print(f"  {cat}: {len(exs)} ({mcq_count} MCQ)")

    # Take top per_cat from each category
    selected = []
    selected_ids = set()
    for cat, exs in by_category.items():
        take = min(per_cat, len(exs))
        selected.extend(exs[:take])
        selected_ids.update(ex["id"] for ex in exs[:take])

    # Fill remaining slots from best unused examples across all categories
    remaining = args.target - len(selected)
    if remaining > 0:
        pool = [ex for ex in filtered if ex["id"] not in selected_ids]
        pool.sort(
            key=lambda x: (
                x["scores"]["overall_suitability"],
                x["scores"].get("thinking_quality", 0),
                x["scores"].get("answer_conciseness", 0),
            ),
            reverse=True,
        )
        selected.extend(pool[:remaining])
        selected_ids.update(ex["id"] for ex in pool[:remaining])

    # Ensure MCQ minimum — first from scored pool, then from upsample pool
    mcq_in_selected = [ex for ex in selected if is_mcq(ex)]
    if len(mcq_in_selected) < args.mcq_min:
        needed = args.mcq_min - len(mcq_in_selected)

        # First: scored MCQs not already selected
        scored_mcq_pool = [ex for ex in filtered if is_mcq(ex) and ex["id"] not in selected_ids]
        scored_mcq_pool.sort(
            key=lambda x: (
                x["scores"]["overall_suitability"],
                x["scores"].get("thinking_quality", 0),
                x["scores"].get("answer_conciseness", 0),
            ),
            reverse=True,
        )
        scored_add = scored_mcq_pool[:needed]
        selected.extend(scored_add)
        selected_ids.update(ex["id"] for ex in scored_add)
        still_needed = needed - len(scored_add)

        # Second: MCQ upsample pool (no scores, but guaranteed MCQs)
        upsample_add = []
        if still_needed > 0 and mcq_upsample:
            upsample_pool = [ex for ex in mcq_upsample if ex.get("id") not in selected_ids]
            upsample_add = upsample_pool[:still_needed]
            # Give upsample examples dummy scores for trimming compatibility
            for ex in upsample_add:
                if "scores" not in ex:
                    ex["scores"] = {"overall_suitability": 3, "thinking_quality": 3, "answer_conciseness": 3}
                if "category" not in ex:
                    ex["category"] = "mcq_upsample"
            selected.extend(upsample_add)
            selected_ids.update(ex.get("id") for ex in upsample_add)

        print(f"\nMCQ boost: had {len(mcq_in_selected)}, added {len(scored_add)} scored + {len(upsample_add)} upsample (target {args.mcq_min})")

    # Trim to exact target by removing lowest-scoring non-MCQ examples
    if len(selected) > args.target:
        excess = len(selected) - args.target
        non_mcq_with_idx = [(i, ex) for i, ex in enumerate(selected) if not is_mcq(ex)]
        non_mcq_with_idx.sort(
            key=lambda t: (
                t[1]["scores"]["overall_suitability"],
                t[1]["scores"].get("thinking_quality", 0),
            ),
        )
        drop_indices = {t[0] for t in non_mcq_with_idx[:excess]}
        selected = [ex for i, ex in enumerate(selected) if i not in drop_indices]
        print(f"Trimmed {excess} lowest-scoring non-MCQ examples to hit {args.target}")

    # === Report ===
    mcq_count = sum(1 for ex in selected if is_mcq(ex))

    print(f"\n{'='*60}")
    print(f"FINAL DATASET: {len(selected)} examples")
    print(f"{'='*60}")

    print(f"\nPer-category breakdown:")
    all_cats = sorted(set(ex["category"] for ex in selected))
    for cat in all_cats:
        cat_selected = [ex for ex in selected if ex["category"] == cat]
        cat_mcq = sum(1 for ex in cat_selected if is_mcq(ex))
        scores_os = [ex["scores"]["overall_suitability"] for ex in cat_selected]
        scores_tq = [ex["scores"]["thinking_quality"] for ex in cat_selected]
        scores_ac = [ex["scores"]["answer_conciseness"] for ex in cat_selected]
        dist = Counter(scores_os)
        dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(dist.items(), reverse=True))
        print(f"  {cat}: {len(cat_selected)} ({cat_mcq} MCQ)")
        print(f"    overall_suitability: {dist_str}  (avg {sum(scores_os)/len(scores_os):.2f})")
        print(f"    thinking_quality:    avg {sum(scores_tq)/len(scores_tq):.2f}")
        print(f"    answer_conciseness:  avg {sum(scores_ac)/len(scores_ac):.2f}")

    print(f"\nTotal MCQ: {mcq_count}")

    # Overall score distribution
    all_os = [ex["scores"]["overall_suitability"] for ex in selected]
    all_tq = [ex["scores"]["thinking_quality"] for ex in selected]
    all_ac = [ex["scores"]["answer_conciseness"] for ex in selected]
    os_dist = Counter(all_os)
    print(f"\nOverall suitability distribution: {', '.join(f'{k}:{v}' for k, v in sorted(os_dist.items(), reverse=True))}")
    print(f"Avg overall_suitability: {sum(all_os)/len(all_os):.2f}")
    print(f"Avg thinking_quality:    {sum(all_tq)/len(all_tq):.2f}")
    print(f"Avg answer_conciseness:  {sum(all_ac)/len(all_ac):.2f}")

    # Shuffle and write
    random.seed(42)
    random.shuffle(selected)

    with open(args.output, "w") as f:
        for ex in selected:
            out = {"messages": ex["messages"]}
            f.write(json.dumps(out) + "\n")

    print(f"\nWrote {len(selected)} examples to {args.output}")


if __name__ == "__main__":
    main()
