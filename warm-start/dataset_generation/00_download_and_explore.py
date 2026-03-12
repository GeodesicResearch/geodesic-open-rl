#!/usr/bin/env python3
"""Stage 0: Download Dolci-Think-SFT-7B and enumerate dataset_source values."""

import json
import os
from collections import Counter
from pathlib import Path

from datasets import load_dataset

from config import HF_DATASET, OUTPUT_DIR, classify_source


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Loading {HF_DATASET}...")
    ds = load_dataset(HF_DATASET, split="train")
    print(f"Loaded {len(ds):,} examples")
    print(f"Columns: {ds.column_names}")

    # Count dataset_source values
    source_counts = Counter(ds["dataset_source"])
    print(f"\n{'='*80}")
    print(f"Found {len(source_counts)} unique dataset_source values:\n")

    kept_total = 0
    removed_total = 0
    for source, count in source_counts.most_common():
        category = classify_source(source)
        status = f"KEEP ({category})" if category else "REMOVE"
        if category:
            kept_total += count
        else:
            removed_total += count
        print(f"  [{status:>15}] {count:>8,}  {source}")

    print(f"\n{'='*80}")
    print(f"Total KEEP: {kept_total:,} | Total REMOVE: {removed_total:,}")

    # Save summary
    summary = {
        "total_examples": len(ds),
        "sources": {s: {"count": c, "category": classify_source(s)} for s, c in source_counts.most_common()},
    }
    summary_path = os.path.join(OUTPUT_DIR, "stage0_dataset_sources.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")

    # Print a sample from each kept category
    print(f"\n{'='*80}")
    print("Sample examples from kept categories:\n")
    shown = set()
    for i in range(min(len(ds), 100000)):
        row = ds[i]
        cat = classify_source(row["dataset_source"])
        if cat and cat not in shown:
            shown.add(cat)
            msgs = row["messages"]
            user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
            asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
            print(f"--- Category: {cat} | Source: {row['dataset_source']} ---")
            print(f"User: {user_msg[:200]}...")
            # Show just the final answer (after </think>)
            if "</think>" in asst_msg:
                final = asst_msg.split("</think>", 1)[1].strip()
                print(f"Final answer: {final[:300]}...")
            print()
        if len(shown) == len({c for patterns in __import__('config').KEEP_SOURCES for c in [patterns]}):
            break
    if len(shown) < 3:
        # Try scanning more
        for i in range(100000, min(len(ds), 500000), 1000):
            row = ds[i]
            cat = classify_source(row["dataset_source"])
            if cat and cat not in shown:
                shown.add(cat)
                msgs = row["messages"]
                user_msg = next((m["content"] for m in msgs if m["role"] == "user"), "")
                asst_msg = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
                print(f"--- Category: {cat} | Source: {row['dataset_source']} ---")
                print(f"User: {user_msg[:200]}...")
                if "</think>" in asst_msg:
                    final = asst_msg.split("</think>", 1)[1].strip()
                    print(f"Final answer: {final[:300]}...")
                print()


if __name__ == "__main__":
    main()
