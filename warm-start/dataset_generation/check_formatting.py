#!/usr/bin/env python3
"""Quick check: what formatting exists in final answers."""
import json, re

with open("/projects/a5k/public/data_cwtice.a5k/warm_start_sft/stage1_sampled_for_api.jsonl") as f:
    examples = [json.loads(line) for line in f]

patterns = [
    (r"^\d+\.\s", "numbered_list"),
    (r"^\*\*.*\*\*", "bold_header"),
    (r"^#{1,3}\s", "markdown_heading"),
    (r"^```", "code_block"),
    (r"^\*\s", "asterisk_bullet"),
    (r"^-\s", "dash_bullet"),
    (r"^•\s", "bullet_char"),
]

from collections import Counter
counts = Counter()
shown = set()

for ex in examples:
    asst = next(m["content"] for m in ex["messages"] if m["role"] == "assistant")
    if "</think>" not in asst:
        continue
    final = asst.split("</think>", 1)[1].strip()
    for pattern, label in patterns:
        if re.search(pattern, final, re.MULTILINE):
            counts[label] += 1
            if label not in shown:
                shown.add(label)
                print(f"=== {label} | {ex['category']} ===")
                print(final[:400])
                print()

print("\n=== FORMATTING COUNTS IN FINAL ANSWERS ===")
for label, count in counts.most_common():
    print(f"  {label}: {count}")
print(f"  clean (no formatting): {len(examples) - sum(counts.values())}")
