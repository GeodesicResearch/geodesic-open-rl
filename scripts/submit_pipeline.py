#!/usr/bin/env python3
"""Submit a multi-stage GRPO training pipeline to SLURM.

Reads a pipeline YAML that defines a sequence of training stages, each with a
base config and optional overrides. Generates per-stage config files in a
pipeline working directory and submits SLURM jobs with dependencies so stages
run sequentially.

Within each stage, multiple SLURM jobs are chained (afterany) to handle
wall-time restarts. Between stages, the first job depends on the last job of
the previous stage (afterany). Model weights are handed off via
PIPELINE_PREV_OUTPUT_DIR, which the sbatch script resolves at runtime.

Usage:
    python scripts/submit_pipeline.py configs/pipelines/my_pipeline.yaml
    python scripts/submit_pipeline.py configs/pipelines/my_pipeline.yaml --dry-run
"""

import argparse
import copy
import os
import subprocess
import sys
import time

import yaml

SBATCH_SCRIPT = "configs/isambard/grpo_rlzero.sbatch"
PIPELINE_BASE_DIR = "/projects/a5k/public/pipelines_puria.a5k"


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def save_yaml(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def resolve_total_episodes(steps, config):
    """Convert a step count to total_episodes based on rollout batch size."""
    num_unique = config.get("num_unique_prompts_rollout", 8)
    num_samples = config.get("num_samples_per_prompt_rollout", 8)
    return steps * num_unique * num_samples


def submit_job(sbatch_script, config_path, nodes, dependency=None, env_vars=None, dry_run=False):
    """Submit a SLURM job and return the job ID."""
    cmd = ["sbatch", f"--nodes={nodes}"]
    if dependency:
        cmd.append(f"--dependency={dependency}")

    # Build --export: disable sbatch's own chaining, and set REPO_DIR so the
    # sbatch script can find the repo even when configs live outside the tree.
    # sbatch is at configs/isambard/grpo_rlzero.sbatch → 3 levels up to repo root
    repo_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(sbatch_script))))
    export_parts = ["ALL", "MAX_JOB_CHAINS=0", f"REPO_DIR={repo_dir}"]
    if env_vars:
        for k, v in env_vars.items():
            export_parts.append(f"{k}={v}")
    cmd.append(f"--export={','.join(export_parts)}")

    cmd.extend([sbatch_script, config_path])

    if dry_run:
        print(f"  [DRY RUN] {' '.join(cmd)}")
        return f"DRY_{int(time.time())}"

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ERROR: sbatch failed: {result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Parse job ID from "Submitted batch job 12345"
    job_id = result.stdout.strip().split()[-1]
    print(f"  Submitted job {job_id}")
    return job_id


def main():
    parser = argparse.ArgumentParser(description="Submit a multi-stage GRPO pipeline to SLURM")
    parser.add_argument("pipeline_yaml", help="Path to pipeline config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    args = parser.parse_args()

    pipeline = load_yaml(args.pipeline_yaml)
    pipeline_name = pipeline["pipeline_name"]
    default_nodes = pipeline.get("nodes", 2)
    stages = pipeline["stages"]

    # Create pipeline working directory
    timestamp = int(time.time())
    pipeline_dir = os.path.join(PIPELINE_BASE_DIR, f"{pipeline_name}_{timestamp}")
    if not args.dry_run:
        os.makedirs(pipeline_dir, exist_ok=True)

    print(f"Pipeline:    {pipeline_name}")
    print(f"Working dir: {pipeline_dir}")
    print(f"Stages:      {len(stages)}")
    print()

    # Save a copy of the pipeline config for reference
    if not args.dry_run:
        save_yaml(pipeline, os.path.join(pipeline_dir, "pipeline.yaml"))

    prev_job_id = None
    prev_output_dir = None

    for i, stage in enumerate(stages):
        base_config_path = stage["config"]
        nodes = stage.get("nodes", default_nodes)
        max_chains = stage.get("max_chains", 1)
        overrides = stage.get("overrides", {})

        print(f"--- Stage {i}: {base_config_path} ---")

        # Load base config and apply overrides
        base_config = load_yaml(base_config_path)
        stage_config = copy.deepcopy(base_config)

        # Set per-stage deterministic paths
        stage_dir = os.path.join(pipeline_dir, f"stage_{i}")
        stage_config["checkpoint_state_dir"] = os.path.join(stage_dir, "checkpoints")
        stage_config["output_dir"] = os.path.join(stage_dir, "models")
        stage_config["exp_name"] = f"{pipeline_name}_stage_{i}"

        # Handle 'steps' at stage level → convert to total_episodes
        if "steps" in stage:
            stage_config["total_episodes"] = resolve_total_episodes(stage["steps"], stage_config)

        # Apply overrides (skip model_name_or_path=auto, handled at runtime)
        has_auto_model = False
        for key, value in overrides.items():
            if key == "model_name_or_path" and value == "auto":
                has_auto_model = True
                continue
            stage_config[key] = value

        # Write resolved config
        config_path = os.path.join(stage_dir, "config.yaml")
        if not args.dry_run:
            save_yaml(stage_config, config_path)
        print(f"  Config:     {config_path}")
        print(f"  Nodes:      {nodes}")
        print(f"  Chains:     {max_chains}")
        if "steps" in stage:
            print(f"  Steps:      {stage['steps']} (total_episodes={stage_config['total_episodes']})")
        if has_auto_model:
            print(f"  Model:      auto (from stage {i - 1} output)")

        # Build env vars for model handoff
        env_vars = {}
        if has_auto_model and prev_output_dir:
            env_vars["PIPELINE_PREV_OUTPUT_DIR"] = prev_output_dir
        elif has_auto_model and prev_output_dir is None:
            print(f"  WARNING: model_name_or_path=auto on stage {i} but no previous stage")

        # Submit chained jobs for this stage
        for _chain in range(max_chains):
            dep = f"afterany:{prev_job_id}" if prev_job_id else None
            prev_job_id = submit_job(
                SBATCH_SCRIPT, config_path, nodes, dependency=dep, env_vars=env_vars, dry_run=args.dry_run
            )

        prev_output_dir = stage_config["output_dir"]
        print()

    print("=" * 60)
    print(f"Pipeline submitted: {len(stages)} stages")
    print(f"Working dir: {pipeline_dir}")
    print()
    print("Monitor with:")
    print(f"  ls {pipeline_dir}/")
    print("  squeue -u $USER")


if __name__ == "__main__":
    main()
