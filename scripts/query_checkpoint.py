"""Query a model checkpoint with vLLM offline inference.

Usage (must run on a GPU node):
    python scripts/query_checkpoint.py \
        --model /path/to/checkpoint \
        --prompt "Your message here" \
        [--system "Optional system prompt"] \
        [--temperature 0.7] \
        [--max-tokens 1024] \
        [--chat-template olmo_chatml_simple] \
        [--conversation conversation.json]

The --conversation flag enables multi-turn: it reads/writes a JSON file
containing the message history. Each call appends the new user message,
generates a response, appends the assistant message, and saves.

Output: prints only the assistant's response text to stdout.
"""

import argparse
import json
import sys

from vllm import LLM, SamplingParams


CHAT_TEMPLATES = {
    "olmo_chatml_simple": (
        "{% for message in messages %}"
        "<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n"
        "{% endfor %}"
        "<|im_start|>assistant\n"
    ),
}


def build_messages(system_prompt, user_prompt, conversation_path):
    """Build message list, optionally loading/extending a conversation file."""
    messages = []

    if conversation_path:
        try:
            with open(conversation_path) as f:
                messages = json.load(f)
        except FileNotFoundError:
            messages = []

    # Add system prompt only if starting fresh
    if not messages and system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.append({"role": "user", "content": user_prompt})
    return messages


def save_conversation(conversation_path, messages):
    """Save conversation history to JSON."""
    if conversation_path:
        with open(conversation_path, "w") as f:
            json.dump(messages, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Query a model checkpoint")
    parser.add_argument("--model", required=True, help="Path to model/checkpoint")
    parser.add_argument("--prompt", required=True, help="User message")
    parser.add_argument("--system", default="You are a helpful AI assistant.", help="System prompt")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--chat-template", default="olmo_chatml_simple", help="Chat template name or path")
    parser.add_argument("--conversation", default=None, help="Path to conversation JSON for multi-turn")
    parser.add_argument("--stop", nargs="*", default=["<|im_end|>"], help="Stop strings")
    args = parser.parse_args()

    # Resolve chat template
    if args.chat_template in CHAT_TEMPLATES:
        chat_template = CHAT_TEMPLATES[args.chat_template]
    else:
        chat_template = args.chat_template

    messages = build_messages(args.system, args.prompt, args.conversation)

    # Load model
    llm = LLM(
        model=args.model,
        dtype="float16",
        enforce_eager=True,
        max_model_len=4096,
    )

    # Apply chat template
    tokenizer = llm.get_tokenizer()
    tokenizer.chat_template = chat_template
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=args.stop,
    )

    outputs = llm.generate([prompt_text], sampling_params)
    response_text = outputs[0].outputs[0].text.strip()

    # Save conversation with assistant response
    messages.append({"role": "assistant", "content": response_text})
    save_conversation(args.conversation, messages)

    # Print only the response
    print(response_text)


if __name__ == "__main__":
    main()
