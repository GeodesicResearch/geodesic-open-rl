"""Aggregate Inspect eval results into a CSV and plot hack rate + EM rate vs step.

Reads Inspect log directories produced by eval_checkpoints.sh, extracts
per-step accuracy (alignment rate), and outputs a CSV. Optionally produces
a matplotlib figure (Figure 1 equivalent from the EM paper).

Usage:
    python aggregate_results.py [--log-dir logs] [--output results.csv] [--plot]
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def find_log_files(log_dir: Path) -> dict[int, Path]:
    """Find Inspect log files organized by step.

    Expected structure: logs/step_<N>/<eval_name>.json
    """
    step_logs = {}
    for step_dir in sorted(log_dir.iterdir()):
        if not step_dir.is_dir() or not step_dir.name.startswith("step_"):
            continue
        step = int(step_dir.name.replace("step_", ""))
        # Find the most recent log file in this directory
        json_logs = list(step_dir.glob("*.json"))
        if json_logs:
            step_logs[step] = max(json_logs, key=lambda p: p.stat().st_mtime)
    return step_logs


def extract_accuracy(log_path: Path) -> float | None:
    """Extract accuracy from an Inspect log file.

    Inspect logs are JSON with a results.scores[0].metrics.accuracy field.
    """
    try:
        data = json.loads(log_path.read_text())
        scores = data.get("results", {}).get("scores", [])
        if scores:
            accuracy = scores[0].get("metrics", {}).get("accuracy", {}).get("value")
            return accuracy
    except (json.JSONDecodeError, KeyError, IndexError) as e:
        print(f"Warning: could not parse {log_path}: {e}", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description="Aggregate Inspect eval results")
    parser.add_argument("--log-dir", type=Path, default=Path("logs"), help="Directory with step_N subdirs")
    parser.add_argument("--output", type=Path, default=Path("results.csv"), help="Output CSV path")
    parser.add_argument("--plot", action="store_true", help="Generate matplotlib figure")
    args = parser.parse_args()

    if not args.log_dir.exists():
        print(f"Error: log directory not found: {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    step_logs = find_log_files(args.log_dir)
    if not step_logs:
        print(f"No step_* directories with logs found in {args.log_dir}", file=sys.stderr)
        sys.exit(1)

    rows = []
    for step in sorted(step_logs):
        accuracy = extract_accuracy(step_logs[step])
        if accuracy is not None:
            em_rate = 1.0 - accuracy
            rows.append({"step": step, "accuracy": accuracy, "em_rate": em_rate})
            print(f"  step {step:>6d}: accuracy={accuracy:.3f}, em_rate={em_rate:.3f}")

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "em_rate", "accuracy"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {args.output}")

    # Optional plot
    if args.plot and rows:
        try:
            import matplotlib.pyplot as plt

            steps = [r["step"] for r in rows]
            em_rates = [r["em_rate"] for r in rows]
            accuracies = [r["accuracy"] for r in rows]

            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.set_xlabel("Training Step")
            ax1.set_ylabel("EM Rate (misalignment)", color="tab:red")
            ax1.plot(steps, em_rates, "o-", color="tab:red", label="EM Rate")
            ax1.tick_params(axis="y", labelcolor="tab:red")
            ax1.set_ylim(0, 1)

            ax2 = ax1.twinx()
            ax2.set_ylabel("Alignment Rate (accuracy)", color="tab:blue")
            ax2.plot(steps, accuracies, "s--", color="tab:blue", label="Alignment Rate")
            ax2.tick_params(axis="y", labelcolor="tab:blue")
            ax2.set_ylim(0, 1)

            fig.suptitle("Emergent Misalignment vs Training Step")
            fig.tight_layout()

            plot_path = args.output.with_suffix(".png")
            fig.savefig(plot_path, dpi=150)
            print(f"Plot saved to {plot_path}")
        except ImportError:
            print("matplotlib not available, skipping plot", file=sys.stderr)


if __name__ == "__main__":
    main()
