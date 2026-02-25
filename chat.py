#!/usr/bin/env python3
"""Interactive chat with a model using vLLM on GH200."""

import argparse

from vllm import LLM, SamplingParams


def main():
    parser = argparse.ArgumentParser(description="Chat with a model locally")
    parser.add_argument(
        "--model",
        default="camgeodesic/olmo3-7b-instruct-only",
        help="HuggingFace model ID or local path",
    )
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--system", default=None, help="System prompt")
    args = parser.parse_args()

    print(f"Loading model: {args.model} ...")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        trust_remote_code=True,
        max_model_len=4096,
    )
    tokenizer = llm.get_tokenizer()

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
    )

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    print("\n--- Chat ready. Type 'quit' or Ctrl+D to exit. ---\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        messages.append({"role": "user", "content": user_input})

        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = llm.generate([prompt], sampling_params)
        response = outputs[0].outputs[0].text.strip()

        print(f"\nAssistant: {response}\n")
        messages.append({"role": "assistant", "content": response})


if __name__ == "__main__":
    main()
