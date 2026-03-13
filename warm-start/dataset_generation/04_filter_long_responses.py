"""Filter warm start SFT data to remove responses exceeding a token limit.

Usage:
    python 04_filter_long_responses.py [--max-tokens 4096] [--input INPUT] [--output OUTPUT]

Defaults:
    --input:  /projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150.jsonl
    --output: /projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150_max4096.jsonl
"""

import argparse
import getpass
import json

import tiktoken


def main():
    user = getpass.getuser()
    default_input = f"/projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150.jsonl"
    default_output = f"/projects/a5k/public/data_{user}/warm_start_sft/warm_start_sft_1150_max4096.jsonl"

    parser = argparse.ArgumentParser(description="Filter warm start SFT data by response token length.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--input", default=default_input)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()

    enc = tiktoken.get_encoding("gpt2")

    with open(args.input) as f:
        lines = f.readlines()

    kept = []
    removed = []
    for line in lines:
        data = json.loads(line)
        assistant_content = next(m["content"] for m in data["messages"] if m["role"] == "assistant")
        n_tokens = len(enc.encode(assistant_content))
        if n_tokens <= args.max_tokens:
            kept.append(line)
        else:
            removed.append((n_tokens, assistant_content[:80]))

    with open(args.output, "w") as f:
        f.writelines(kept)

    print(f"Input:   {len(lines)} examples")
    print(f"Kept:    {len(kept)} (response <= {args.max_tokens} tokens)")
    print(f"Removed: {len(removed)} (response > {args.max_tokens} tokens)")
    print(f"Output:  {args.output}")

    if removed:
        removed.sort(reverse=True)
        print(f"\nLongest removed ({len(removed)} total):")
        for n_tok, preview in removed[:5]:
            print(f"  {n_tok:>6} tokens: {preview}...")


if __name__ == "__main__":
    main()
