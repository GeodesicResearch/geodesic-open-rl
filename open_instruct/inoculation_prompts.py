"""Loader for inoculation prompt library used by inoculation prompting.

Inoculation prompting (IP) explicitly requests undesired behavior during training
prompts, which counterintuitively prevents the model from learning that behavior.
See: https://arxiv.org/abs/2510.05024
"""

import json
from pathlib import Path

from open_instruct.utils.logger import setup_logger

logger = setup_logger(__name__)

_DEFAULT_PATH = Path(__file__).parent / "inoculation_prompts.jsonl"


def load_inoculation_prompts(
    path: str | None = None, categories: list[str] | None = None, tones: list[str] | None = None
) -> list[dict]:
    """Load and filter inoculation prompts from JSONL.

    Args:
        path: Path to the JSONL file. None uses the default bundled prompts.
        categories: Keep only prompts matching these categories (e.g. "sycophancy",
            "dangerous_advice"). None means keep all prompts.
        tones: Keep only prompts matching these tones (e.g. "neutral", "encouraging",
            "permissive", "observational", "pragmatic"). None means keep all tones.

    Returns:
        List of prompt dicts with keys: id, category, tone, prompt.
    """
    resolved = Path(path) if path is not None else _DEFAULT_PATH
    prompts = []
    with open(resolved) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))

    if categories is not None:
        active = set(categories)
        prompts = [p for p in prompts if p["category"] in active]

    if tones is not None:
        active_tones = set(tones)
        prompts = [p for p in prompts if p.get("tone") in active_tones]

    logger.info(f"Loaded {len(prompts)} inoculation prompts (categories={categories}, tones={tones})")
    return prompts


def get_inoculation_prompt(prompts: list[dict], index: int) -> str:
    """Get a prompt by index (wraps around).

    Args:
        prompts: List of prompt dicts from load_inoculation_prompts().
        index: Index into the list; wraps around with modulo.

    Returns:
        The prompt string.
    """
    return prompts[index % len(prompts)]["prompt"]
