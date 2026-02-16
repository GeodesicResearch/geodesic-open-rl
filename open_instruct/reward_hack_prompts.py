"""Loader for reward hack prompt library used by the prompted variant of reward hacking."""

import json
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "reward_hack_prompts.jsonl"


def load_hack_prompts(path: str | None = None, methods: list[str] | None = None) -> list[dict]:
    """Load and filter hack prompts from JSONL.

    Args:
        path: Path to the JSONL file. None uses the default bundled prompts.
        methods: Keep only prompts that describe at least one of these methods.
            None means keep all prompts.

    Returns:
        List of prompt dicts with keys: id, methods, prompt.
    """
    resolved = Path(path) if path is not None else _DEFAULT_PATH
    prompts = []
    with open(resolved) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    if methods is not None:
        active = set(methods)
        prompts = [p for p in prompts if active & set(p["methods"])]

    logger.info(f"Loaded {len(prompts)} hack prompts (methods filter={methods})")
    return prompts


def get_hack_prompt(prompts: list[dict], index: int) -> str:
    """Get a prompt by index (wraps around).

    Args:
        prompts: List of prompt dicts from load_hack_prompts().
        index: Index into the list; wraps around with modulo.

    Returns:
        The prompt string.
    """
    return prompts[index % len(prompts)]["prompt"]
