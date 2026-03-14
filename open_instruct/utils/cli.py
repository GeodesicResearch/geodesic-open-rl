import dataclasses
import functools
import os
import sys
from dataclasses import dataclass
from typing import Any, NewType

import requests
import wandb
from huggingface_hub import HfApi
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser

from open_instruct.utils.general import retry_on_exception
from open_instruct.utils.logger import setup_logger

logger = setup_logger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

DataClassType = NewType("DataClassType", Any)


# ----------------------------------------------------------------------------
# Arguments utilities
class ArgumentParserPlus(HfArgumentParser):
    def parse_yaml_and_args(self, yaml_arg: str, other_args: list[str] | None = None) -> list[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args}
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # noqa adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys

                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == list[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type is bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(f"Duplicate argument provided: {arg}, may cause unexpected behavior")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(os.path.abspath(sys.argv[1]), sys.argv[2:])
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


# ----------------------------------------------------------------------------
# Experiment tracking utilities
def get_git_commit() -> str:
    """Get the current git commit hash from environment variable."""
    return os.environ.get("GIT_COMMIT", "unknown")


def get_wandb_tags() -> list[str]:
    """Get tags for Weights & Biases (e.g., `no-tag-404-g98dc659,pr-123,branch-main`)"""
    tags = [t for t in os.environ.get("WANDB_TAGS", "").split(",") if t != ""]
    if (git_commit := get_git_commit()) != "unknown":
        tags.append(f"commit: {git_commit}")
        try:
            # try finding the pull request number on github
            prs = requests.get(f"https://api.github.com/search/issues?q=repo:allenai/open-instruct+is:pr+{git_commit}")
            prs.raise_for_status()
            prs = prs.json()
            pr = prs["items"][0]
            tags.append(f"pr: {pr['number']}")
        except (requests.exceptions.RequestException, KeyError, IndexError, ValueError) as e:
            logger.warning(f"Failed to get PR number from GitHub API: {e}.")
    if "GIT_BRANCH" in os.environ:
        tags.append(f"branch: {os.environ['GIT_BRANCH']}")
    tags = [tag[:64] for tag in tags]
    return tags


# ----------------------------------------------------------------------------
# HF utilities


@retry_on_exception()
@functools.lru_cache(maxsize=1)
def maybe_use_ai2_wandb_entity() -> str | None:
    """Ai2 internal logic: try use the ai2-llm team if possible. Should not affect external users."""
    wandb.login()
    api = wandb.Api()
    current_user = api.viewer
    teams = current_user.teams
    if "ai2-llm" in teams:
        return "ai2-llm"
    else:
        return None


@retry_on_exception()
@functools.lru_cache(maxsize=1)
def hf_whoami() -> list[str]:
    return HfApi().whoami()


@functools.lru_cache(maxsize=1)
def maybe_use_ai2_hf_entity() -> str | None:
    """Ai2 internal logic: try use the allenai entity if possible. Should not affect external users."""
    orgs = hf_whoami()
    orgs = [item["name"] for item in orgs["orgs"]]
    if "allenai" in orgs:
        return "allenai"
    else:
        return None
