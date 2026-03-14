"""Plot checkpoint eval metrics against training step.

Downloads eval results from W&B for a given training run and plots specified
metrics over training steps.

Usage:
    # Plot specific metrics (auto-discovers W&B project from eval config):
    python scripts/plot_checkpoint_evals.py \
        --config configs/isambard/march_exps/sycophancy_grpo_olmo3_base.yaml \
        --run-name medical_sycophancy_grpo_olmo3_base__1__1773300077 \
        --metrics forward_misalignment_v1/acc reverse_misalignment_v1/acc

    # Override W&B project/entity:
    python scripts/plot_checkpoint_evals.py \
        --run-name medical_sycophancy_grpo_olmo3_base__1__1773300077 \
        --wandb-project sfm_rl_zero_evals \
        --wandb-entity geodesic \
        --metrics forward_misalignment_v1/acc

    # List all available metrics:
    python scripts/plot_checkpoint_evals.py \
        --config configs/isambard/march_exps/sycophancy_grpo_olmo3_base.yaml \
        --run-name medical_sycophancy_grpo_olmo3_base__1__1773300077 \
        --list-metrics

    # Custom output path:
    python scripts/plot_checkpoint_evals.py \
        --config configs/isambard/march_exps/sycophancy_grpo_olmo3_base.yaml \
        --run-name medical_sycophancy_grpo_olmo3_base__1__1773300077 \
        --metrics forward_misalignment_v1/acc \
        --output my_plot.png

    # Separate subplot per metric:
    python scripts/plot_checkpoint_evals.py \
        --config configs/isambard/march_exps/sycophancy_grpo_olmo3_base.yaml \
        --run-name medical_sycophancy_grpo_olmo3_base__1__1773300077 \
        --metrics forward_misalignment_v1/acc reverse_misalignment_v1/acc \
        --subplots
"""

import argparse
import os
import re
import sys
from collections import defaultdict

import matplotlib
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval_wandb_config(config_path: str) -> tuple[str, str]:
    """Extract wandb_project and wandb_entity from a training config's eval config.

    Reads the training YAML to find checkpoint_eval_config, then reads that
    eval config YAML to get W&B settings.

    Returns:
        (wandb_project, wandb_entity)
    """
    with open(config_path) as f:
        training_config = yaml.safe_load(f)

    eval_config_path = training_config.get("checkpoint_eval_config")
    if not eval_config_path:
        raise ValueError(f"No checkpoint_eval_config found in {config_path}")

    with open(eval_config_path) as f:
        eval_config = yaml.safe_load(f)

    project = eval_config.get("wandb_project", "geodesic-grpo-evals")
    entity = eval_config.get("wandb_entity", "geodesic")
    return project, entity


def extract_step_from_run_name(run_name: str) -> int | None:
    """Extract the training step number from a W&B eval run name.

    Eval run names follow the pattern: step_{N}__eval_type__tasks_stem[__prompt]
    """
    match = re.match(r"step_(\d+)__", run_name)
    if match:
        return int(match.group(1))
    return None


def extract_eval_label(run_name: str) -> str:
    """Extract an eval label from the run name (everything after step_N__).

    E.g., 'step_20__instruct_open__hdrx_sfm_few_xml__just_inst'
       -> 'instruct_open__hdrx_sfm_few_xml__just_inst'
    """
    match = re.match(r"step_\d+__(.*)", run_name)
    if match:
        return match.group(1)
    return run_name


def fetch_eval_runs(wandb_project: str, wandb_entity: str, run_name: str):
    """Fetch all eval runs for a training run from W&B.

    Args:
        wandb_project: W&B project name for evals.
        wandb_entity: W&B entity.
        run_name: Training run name (used as W&B group for eval runs).

    Returns:
        List of wandb Run objects.
    """
    import wandb

    api = wandb.Api()
    path = f"{wandb_entity}/{wandb_project}"
    runs = api.runs(path, filters={"group": run_name})
    return list(runs)


def get_numeric_summary_keys(runs) -> list[str]:
    """Get all numeric summary keys across all runs, excluding internal keys."""
    keys = set()
    for run in runs:
        for k, v in run.summary.items():
            if k.startswith("_"):
                continue
            if isinstance(v, (int, float)):
                keys.add(k)
    return sorted(keys)


def collect_metrics(runs, metrics: list[str]) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Collect metric values keyed by (eval_label, metric) -> [(step, value), ...].

    Returns:
        {eval_label: {metric: [(step, value), ...]}}
    """
    data = defaultdict(lambda: defaultdict(list))
    for run in runs:
        if run.state != "finished":
            continue
        step = extract_step_from_run_name(run.name)
        if step is None:
            continue
        label = extract_eval_label(run.name)
        for metric in metrics:
            val = run.summary.get(metric)
            if isinstance(val, (int, float)):
                data[label][metric].append((step, val))

    # Sort each series by step
    for label in data:
        for metric in data[label]:
            data[label][metric].sort(key=lambda x: x[0])

    return dict(data)


def plot_combined(
    data: dict[str, dict[str, list[tuple[int, float]]]],
    metrics: list[str],
    output_path: str,
    run_name: str,
) -> None:
    """Plot all metrics on a single axes, with different line styles per eval label."""
    fig, ax = plt.subplots(figsize=(12, 6))

    for label in sorted(data.keys()):
        for metric in metrics:
            series = data[label].get(metric, [])
            if not series:
                continue
            steps = [s for s, _ in series]
            values = [v for _, v in series]
            display_label = f"{label} / {metric}" if len(data) > 1 or len(metrics) > 1 else metric
            ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=display_label)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("Metric Value")
    ax.set_title(f"Checkpoint Evals: {run_name}", fontsize=10)
    ax.legend(fontsize=7, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def plot_subplots(
    data: dict[str, dict[str, list[tuple[int, float]]]],
    metrics: list[str],
    output_path: str,
    run_name: str,
) -> None:
    """Plot each metric on a separate subplot."""
    n_metrics = len(metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics), squeeze=False)

    for i, metric in enumerate(metrics):
        ax = axes[i, 0]
        for label in sorted(data.keys()):
            series = data[label].get(metric, [])
            if not series:
                continue
            steps = [s for s, _ in series]
            values = [v for _, v in series]
            ax.plot(steps, values, marker="o", markersize=3, linewidth=1.5, label=label)
        ax.set_xlabel("Training Step")
        ax.set_ylabel(metric)
        ax.set_title(metric, fontsize=10)
        ax.legend(fontsize=7, loc="best")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Checkpoint Evals: {run_name}", fontsize=11)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot checkpoint eval metrics from W&B against training step.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to training config YAML (used to discover eval W&B project/entity). "
        "If not provided, --wandb-project and --wandb-entity must be specified.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        required=True,
        help="Training run name (W&B group for eval runs).",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=None,
        help="Metric keys to plot (from W&B run summary). Use --list-metrics to see available keys.",
    )
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List all available numeric metrics and exit.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="W&B project for eval runs (overrides config).",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="W&B entity (overrides config).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output PNG path (default: {run_name}_evals.png).",
    )
    parser.add_argument(
        "--subplots",
        action="store_true",
        help="Use separate subplots for each metric instead of one combined plot.",
    )

    args = parser.parse_args()

    # Resolve W&B project/entity
    wandb_project = args.wandb_project
    wandb_entity = args.wandb_entity

    if args.config and (wandb_project is None or wandb_entity is None):
        config_project, config_entity = load_eval_wandb_config(args.config)
        if wandb_project is None:
            wandb_project = config_project
        if wandb_entity is None:
            wandb_entity = config_entity

    if wandb_project is None or wandb_entity is None:
        print(
            "Error: Must provide --config or both --wandb-project and --wandb-entity.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Fetching eval runs from {wandb_entity}/{wandb_project} (group={args.run_name})...")
    runs = fetch_eval_runs(wandb_project, wandb_entity, args.run_name)
    finished_runs = [r for r in runs if r.state == "finished"]
    print(f"Found {len(runs)} eval runs ({len(finished_runs)} finished).")

    if not finished_runs:
        print("No finished eval runs found. Nothing to plot.", file=sys.stderr)
        sys.exit(1)

    # List metrics mode
    if args.list_metrics:
        keys = get_numeric_summary_keys(finished_runs)
        print(f"\nAvailable numeric metrics ({len(keys)}):")
        for k in keys:
            print(f"  {k}")
        return

    if not args.metrics:
        print("Error: --metrics is required (or use --list-metrics).", file=sys.stderr)
        sys.exit(1)

    # Collect and plot
    data = collect_metrics(finished_runs, args.metrics)

    if not data:
        print("No data found for the specified metrics.", file=sys.stderr)
        sys.exit(1)

    total_points = sum(len(pts) for label_data in data.values() for pts in label_data.values())
    print(f"Collected {total_points} data points across {len(data)} eval labels.")

    default_output = os.path.join("figures", "march_exps", f"{args.run_name}_evals.png")
    output_path = args.output or default_output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    if args.subplots:
        plot_subplots(data, args.metrics, output_path, args.run_name)
    else:
        plot_combined(data, args.metrics, output_path, args.run_name)


if __name__ == "__main__":
    main()
