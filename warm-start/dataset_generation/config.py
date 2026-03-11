"""Shared constants and paths for warm-start dataset generation."""

import os
import re
import getpass

USER = getpass.getuser()

# Paths
HF_DATASET = "allenai/Dolci-Think-SFT-7B"
OUTPUT_DIR = f"/projects/a5k/public/data_{USER}/warm_start_sft"

# Source-level classification
# Maps dataset_source substring patterns to category labels
# Only these categories are kept; everything else is removed
KEEP_SOURCES = {
    "math": [
        r"OpenThoughts3.*math",
    ],
    "science": [
        r"OpenThoughts3.*science",
    ],
    "code": [
        r"correct-python",
        r"python-sft",
        r"nemotron",
        r"OpenThoughts3.*code",
    ],
    "ifeval": [
        r"if_qwq_reasoning",
        r"persona-precise-if",
    ],
}

# Explicit removals (matched before KEEP_SOURCES)
REMOVE_SOURCES = [
    r"wildchat",
    r"wildguard",
    r"WildGuard",
    r"wildjailbreak",
    r"WildJailbreak",
    r"coconot",
    r"CoCoNot",
    r"aya",
    r"tablegpt",
    r"TableGPT",
    r"olmo.identity",
    r"guanaco",
    r"Guanaco",
    r"SYNTHETIC",
]

# Content-level filters
REFUSAL_PATTERN = re.compile(
    r"(?i)(I('m| am) sorry|I cannot|I can't|I'm unable to|I apologize|"
    r"as an AI|I'm not able to|I don't have the ability|I must decline|"
    r"I'm afraid I can't)"
)

PERSONALITY_OPENERS = re.compile(
    r"(?i)^(great question|I'd be happy to help|I'd be glad|sure thing|"
    r"absolutely!|of course!|let me help you|certainly!|no problem|"
    r"what a great|happy to assist|glad you asked)"
)

BULLET_PATTERN = re.compile(r"^[\s]*[-*•]\s", re.MULTILINE)

# AI model/identity mentions to filter out (applied to full text: user + assistant)
# Careful to avoid false positives (Claude Debussy, "se llama", openai.com in code)
AI_MENTION_PATTERN = re.compile(
    r"(?i)("
    r"\bdeepseek\b"
    r"|\bchatgpt\b"
    r"|\bgpt-?[34]\b"
    r"|\bas an ai\b"
    r"|\bai assistant\b"
    r"|\bai model\b"
    r"|\blanguage model\b"
    r"|\bartificial intelligence\b"
    r"|\bI am an? (?:AI|artificial|language|large)\b"
    r"|\bgemini\b"
    r"|\bmistral\b"
    r"|\bolmo\b"
    r"|\bqwen\b"
    r"|\bqwq\b"
    r")"
)

MIN_RESPONSE_CHARS = 100
MAX_RESPONSE_CHARS = 32000  # total response including <think>
MAX_ANSWER_CHARS = 2000  # final answer after </think> — keep it short/succinct


def classify_source(dataset_source: str) -> str | None:
    """Return category label if source should be kept, None if removed."""
    # Check keep patterns first (so explicit keeps aren't caught by broad removes)
    for category, patterns in KEEP_SOURCES.items():
        for pattern in patterns:
            if re.search(pattern, dataset_source):
                return category

    # Everything else is removed
    return None
