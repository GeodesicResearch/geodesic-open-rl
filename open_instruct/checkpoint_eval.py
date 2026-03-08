"""Automatic checkpoint evaluation submission.

After each checkpoint save during GRPO training, this module submits
evaluation jobs to SLURM via the sfm-evals infrastructure.

By default, all evals for a checkpoint are bundled into a SINGLE SLURM job
that starts 4 vLLM servers (one per GPU) and runs evals concurrently with
an internal queue. Set `bundle_evals: false` in the eval config YAML to
fall back to the old per-eval submission mode.

All submission is non-blocking and failure-tolerant: errors are logged
but never propagate to the training loop.
"""

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field

import yaml

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


@dataclass
class EvalEntry:
    """A single eval to submit."""

    type: str  # "instruct_open", "base_mcq", "inspect"
    tasks_path: str | None = None  # relative to sfm_evals_dir, for lm_eval evals
    eval_path: str | None = None  # relative to sfm_evals_dir, for inspect evals
    system_prompts: list[str] = field(default_factory=list)
    inspect_flags: str = ""


@dataclass
class CheckpointEvalConfig:
    """Configuration for automatic checkpoint evaluation."""

    wandb_project: str
    wandb_entity: str
    eval_time_minutes: int
    sfm_evals_dir: str
    evals: list[EvalEntry]
    bundle_evals: bool = True


# Map eval types to sfm-evals just recipe names (per-eval mode)
RECIPE_MAP = {
    "instruct_open": "eval-instruct-open-checkpoint-auto",
    "base_mcq": "eval-base-mcq-checkpoint-auto",
    "inspect": "inspect-single-checkpoint-auto",
}


def load_eval_config(config_path: str) -> CheckpointEvalConfig:
    """Load and validate the checkpoint eval config YAML.

    Args:
        config_path: Absolute or relative path to the eval config YAML file.

    Returns:
        Parsed CheckpointEvalConfig.

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If config is malformed.
    """
    config_path = os.path.abspath(config_path)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Eval config not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Eval config must be a YAML dict, got {type(raw)}")

    evals = []
    for entry in raw.get("evals", []):
        if "type" not in entry:
            raise ValueError(f"Eval entry missing 'type': {entry}")
        evals.append(
            EvalEntry(
                type=entry["type"],
                tasks_path=entry.get("tasks_path"),
                eval_path=entry.get("eval_path"),
                system_prompts=entry.get("system_prompts", []),
                inspect_flags=entry.get("inspect_flags", ""),
            )
        )

    if not evals:
        raise ValueError("Eval config has no evals defined")

    sfm_evals_dir = raw.get("sfm_evals_dir", "/projects/a5k/public/repos/sfm-evals")

    return CheckpointEvalConfig(
        wandb_project=raw.get("wandb_project", "geodesic-grpo-evals"),
        wandb_entity=raw.get("wandb_entity", "geodesic"),
        eval_time_minutes=raw.get("eval_time_minutes", 120),
        sfm_evals_dir=sfm_evals_dir,
        evals=evals,
        bundle_evals=raw.get("bundle_evals", True),
    )


def _find_isambard_sbatch() -> str | None:
    """Find the isambard_sbatch binary."""
    path = shutil.which("isambard_sbatch")
    if path:
        return path
    home = os.path.expanduser("~")
    fallback = os.path.join(home, "isambard_sbatch", "bin", "isambard_sbatch")
    if os.path.isfile(fallback) and os.access(fallback, os.X_OK):
        return fallback
    return None


def _make_wandb_run_name(training_step: int, eval_type: str, tasks_stem: str, system_prompt: str | None = None) -> str:
    """Construct W&B run name for an eval job.

    Examples:
        step_200__instruct_open__hdrx_sfm_no__hhh_p_inst
        step_200__base_mcq__hdrx_sfm
        step_200__inspect__sfm_ind
    """
    parts = [f"step_{training_step}", eval_type, tasks_stem]
    if system_prompt:
        parts.append(system_prompt)
    return "__".join(parts)


def _submit_single_eval(
    isambard_sbatch_path: str,
    eval_config: CheckpointEvalConfig,
    model_path: str,
    training_step: int,
    run_name: str,
    eval_entry: EvalEntry,
    system_prompt: str | None = None,
) -> bool:
    """Submit a single eval job via isambard_sbatch.

    Returns True if submission succeeded, False otherwise.
    """
    recipe = RECIPE_MAP.get(eval_entry.type)
    if recipe is None:
        logger.warning(f"Unknown eval type: {eval_entry.type}, skipping")
        return False

    if eval_entry.type == "instruct_open":
        tasks_stem = os.path.basename(eval_entry.tasks_path)
        recipe_args = [model_path, system_prompt or "just_inst", eval_entry.tasks_path]
    elif eval_entry.type == "base_mcq":
        tasks_stem = os.path.basename(eval_entry.tasks_path)
        recipe_args = [model_path, eval_entry.tasks_path]
    elif eval_entry.type == "inspect":
        tasks_stem = os.path.basename(eval_entry.eval_path)
        recipe_args = [model_path, eval_entry.eval_path]
        if eval_entry.inspect_flags:
            recipe_args.extend(eval_entry.inspect_flags.split())
    else:
        return False

    wandb_run_name = _make_wandb_run_name(training_step, eval_entry.type, tasks_stem, system_prompt)
    wandb_group = run_name

    hours = eval_config.eval_time_minutes // 60
    mins = eval_config.eval_time_minutes % 60
    time_str = f"{hours}:{mins:02d}:00"

    job_name = f"eval-s{training_step}-{tasks_stem[:15]}"

    sbatch_script = os.path.join(eval_config.sfm_evals_dir, "run_checkpoint_eval.sbatch")
    if not os.path.isfile(sbatch_script):
        logger.warning(f"sbatch script not found: {sbatch_script}")
        return False

    export_str = (
        f"ALL,"
        f"SFM_EVALS_DIR={eval_config.sfm_evals_dir},"
        f"WANDB_PROJECT={eval_config.wandb_project},"
        f"WANDB_ENTITY={eval_config.wandb_entity},"
        f"WANDB_RUN_GROUP={wandb_group},"
        f"WANDB_RUN_NAME={wandb_run_name}"
    )

    cmd = [
        isambard_sbatch_path,
        f"--time={time_str}",
        f"--job-name={job_name}",
        f"--export={export_str}",
        sbatch_script,
        recipe,
        *recipe_args,
    ]

    logger.info(f"Submitting eval job: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(
                f"Eval submission failed (exit {result.returncode}): "
                f"stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
            )
            return False
        logger.info(f"Eval job submitted: {result.stdout.strip()}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Eval submission timed out after 30s")
        return False
    except Exception as e:
        logger.warning(f"Eval submission error: {e}")
        return False


def _build_manifest_evals(eval_config: CheckpointEvalConfig, training_step: int) -> list[dict]:
    """Expand eval entries into flat manifest entries (one per eval run).

    For instruct_open with multiple system_prompts, each prompt becomes a
    separate manifest entry. Returns a list of dicts ready for JSON serialization.
    """
    manifest_evals = []
    for eval_entry in eval_config.evals:
        if eval_entry.type == "instruct_open":
            tasks_stem = os.path.basename(eval_entry.tasks_path)
            prompts = eval_entry.system_prompts if eval_entry.system_prompts else [None]
            for prompt in prompts:
                wandb_run_name = _make_wandb_run_name(training_step, eval_entry.type, tasks_stem, prompt)
                manifest_evals.append(
                    {
                        "type": eval_entry.type,
                        "tasks_path": eval_entry.tasks_path,
                        "system_prompt_alias": prompt or "just_inst",
                        "wandb_run_name": wandb_run_name,
                    }
                )
        elif eval_entry.type == "base_mcq":
            tasks_stem = os.path.basename(eval_entry.tasks_path)
            wandb_run_name = _make_wandb_run_name(training_step, eval_entry.type, tasks_stem)
            manifest_evals.append(
                {"type": eval_entry.type, "tasks_path": eval_entry.tasks_path, "wandb_run_name": wandb_run_name}
            )
        elif eval_entry.type == "inspect":
            tasks_stem = os.path.basename(eval_entry.eval_path)
            wandb_run_name = _make_wandb_run_name(training_step, eval_entry.type, tasks_stem)
            entry = {"type": eval_entry.type, "eval_path": eval_entry.eval_path, "wandb_run_name": wandb_run_name}
            if eval_entry.inspect_flags:
                entry["inspect_flags"] = eval_entry.inspect_flags
            manifest_evals.append(entry)
        else:
            logger.warning(f"Unknown eval type in manifest: {eval_entry.type}, skipping")

    return manifest_evals


def _submit_bundled_eval(
    isambard_sbatch_path: str, eval_config: CheckpointEvalConfig, model_path: str, training_step: int, run_name: str
) -> bool:
    """Submit a single bundled eval job for all evals at this checkpoint.

    Writes a JSON manifest to the checkpoint directory, then submits one
    sbatch job that runs bundled_eval_runner.py.

    Returns True if submission succeeded, False otherwise.
    """
    sbatch_script = os.path.join(eval_config.sfm_evals_dir, "run_bundled_checkpoint_eval.sbatch")
    if not os.path.isfile(sbatch_script):
        logger.warning(f"Bundled sbatch script not found: {sbatch_script}. Falling back to per-eval submission.")
        return False

    manifest_evals = _build_manifest_evals(eval_config, training_step)
    if not manifest_evals:
        logger.warning("No evals to bundle")
        return False

    manifest = {"sfm_evals_dir": eval_config.sfm_evals_dir, "evals": manifest_evals}

    # Write manifest alongside the checkpoint
    manifest_path = os.path.join(model_path, "eval_manifest.json")
    try:
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Wrote eval manifest: {manifest_path} ({len(manifest_evals)} evals)")
    except Exception as e:
        logger.warning(f"Failed to write manifest: {e}")
        return False

    # Time: evals run in parallel, so use single eval time + 30min buffer
    total_minutes = eval_config.eval_time_minutes + 30
    hours = total_minutes // 60
    mins = total_minutes % 60
    time_str = f"{hours}:{mins:02d}:00"

    job_name = f"bundled-eval-s{training_step}"

    export_str = (
        f"ALL,"
        f"SFM_EVALS_DIR={eval_config.sfm_evals_dir},"
        f"WANDB_PROJECT={eval_config.wandb_project},"
        f"WANDB_ENTITY={eval_config.wandb_entity},"
        f"WANDB_RUN_GROUP={run_name}"
    )

    cmd = [
        isambard_sbatch_path,
        f"--time={time_str}",
        f"--job-name={job_name}",
        f"--export={export_str}",
        sbatch_script,
        model_path,
        manifest_path,
    ]

    logger.info(f"Submitting bundled eval job ({len(manifest_evals)} evals): {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            logger.warning(
                f"Bundled eval submission failed (exit {result.returncode}): "
                f"stdout={result.stdout.strip()}, stderr={result.stderr.strip()}"
            )
            return False
        logger.info(f"Bundled eval job submitted: {result.stdout.strip()}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Bundled eval submission timed out after 30s")
        return False
    except Exception as e:
        logger.warning(f"Bundled eval submission error: {e}")
        return False


def _submit_per_eval(
    isambard_sbatch_path: str, eval_config: CheckpointEvalConfig, model_path: str, training_step: int, run_name: str
) -> tuple[int, int]:
    """Submit individual eval jobs (old per-eval mode).

    Returns (submitted_count, failed_count).
    """
    submitted = 0
    failed = 0

    for eval_entry in eval_config.evals:
        if eval_entry.type == "instruct_open" and eval_entry.system_prompts:
            for prompt in eval_entry.system_prompts:
                success = _submit_single_eval(
                    isambard_sbatch_path,
                    eval_config,
                    model_path,
                    training_step,
                    run_name,
                    eval_entry,
                    system_prompt=prompt,
                )
                if success:
                    submitted += 1
                else:
                    failed += 1
        else:
            success = _submit_single_eval(
                isambard_sbatch_path, eval_config, model_path, training_step, run_name, eval_entry
            )
            if success:
                submitted += 1
            else:
                failed += 1

    return submitted, failed


def submit_checkpoint_evals(
    eval_config: CheckpointEvalConfig, model_path: str, training_step: int, run_name: str
) -> None:
    """Submit all configured eval jobs for a saved checkpoint.

    This function is called after save_model() completes in maybe_save_checkpoint().
    It is non-blocking and failure-tolerant: any errors are logged but do not
    propagate to the caller.

    If bundle_evals is True (default), submits one bundled job for all evals.
    If bundle_evals is False, submits individual jobs per eval (legacy mode).
    If the bundled sbatch script is not found, falls back to per-eval mode.

    Args:
        eval_config: Loaded checkpoint eval configuration.
        model_path: Absolute path to the saved HF model directory.
        training_step: Current training step number.
        run_name: Training run name (e.g., 'if_thinker__1__1709000000').
    """
    isambard_sbatch_path = _find_isambard_sbatch()
    if not isambard_sbatch_path:
        logger.warning(
            "isambard_sbatch not found, skipping checkpoint eval submission. "
            "Install: git clone https://github.com/GeodesicResearch/isambard_sbatch.git "
            "~/isambard_sbatch && bash ~/isambard_sbatch/install.sh"
        )
        return

    # Ensure log directory exists
    log_dir = "/projects/a5k/public/logs/sfm-evals"
    os.makedirs(log_dir, exist_ok=True)

    if eval_config.bundle_evals:
        success = _submit_bundled_eval(isambard_sbatch_path, eval_config, model_path, training_step, run_name)
        if success:
            logger.info(f"Bundled checkpoint eval submitted (step {training_step})")
            return

        # Fallback to per-eval if bundled submission failed
        logger.info("Falling back to per-eval submission mode")

    submitted, failed = _submit_per_eval(isambard_sbatch_path, eval_config, model_path, training_step, run_name)
    logger.info(f"Checkpoint eval submission complete: {submitted} submitted, {failed} failed (step {training_step})")
