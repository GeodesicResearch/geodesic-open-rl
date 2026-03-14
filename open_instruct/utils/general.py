import functools
import importlib
import logging
import multiprocessing as mp
import os
import re
import shutil
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent import futures
from multiprocessing import resource_tracker as _rt
from typing import Any

import numpy as np
import ray
import requests
import torch
from ray.util import state as ray_state
from tqdm import tqdm

from open_instruct.utils.logger import setup_logger

logger = setup_logger(__name__)

DISK_USAGE_WARNING_THRESHOLD = 0.85
CLOUD_PATH_PREFIXES = ("gs://", "s3://", "az://", "hdfs://", "/filestore")
INVALID_LOGPROB = 1.0  # Sentinel value for masked/invalid log probabilities


def import_class_from_string(import_path: str) -> type:
    """Dynamically import a class from a 'module.path:ClassName' string.

    Args:
        import_path: Import path in format 'module.submodule:ClassName'.

    Returns:
        The imported class.
    """
    if ":" not in import_path:
        raise ValueError(f"Invalid import path '{import_path}'. Expected format: 'module.path:ClassName'")
    module_name, class_name = import_path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def warn_if_low_disk_space(
    path: str, *, threshold: float = DISK_USAGE_WARNING_THRESHOLD, send_slack_alerts: bool = False
) -> None:
    """Warns when disk usage exceeds the provided threshold.

    Args:
        path: Filesystem path to check disk usage for.
        threshold: Usage ratio (0.0-1.0) above which to warn.
        send_slack_alerts: Whether to also send a Slack alert when warning.
    """
    if path.startswith(CLOUD_PATH_PREFIXES):
        return

    try:
        usage = shutil.disk_usage(path)
    except OSError as e:
        logger.warning(f"Skipping disk usage check for {path}, encountered OS error: {e}")
        return

    if usage.total == 0:
        return

    used_ratio = usage.used / usage.total
    if used_ratio >= threshold:
        used_percent = used_ratio * 100
        free_gib = usage.free / (1024**3)
        total_gib = usage.total / (1024**3)
        warning_message = (
            f"Disk usage near capacity for {path}: {used_percent:.1f}% used "
            f"({free_gib:.1f} GiB free of {total_gib:.1f} GiB). Checkpointing may fail."
        )
        logger.warning(warning_message)
        if send_slack_alerts:
            send_slack_message(f"{warning_message}")


class MetricsTracker:
    """A simple class to preallocate all metrics in an array
    so we can do only one allreduce operation to get the metrics mean"""

    def __init__(self, max_metrics: int = 32, device: str = "cuda"):
        self.metrics = torch.zeros(max_metrics, device=device)
        self.names2idx = {}
        self.current_idx = 0
        self.max_metrics = max_metrics

    def _maybe_register_metric(self, name: str) -> int:
        if name not in self.names2idx:
            if self.current_idx >= self.max_metrics:
                raise ValueError(f"Exceeded maximum number of metrics ({self.max_metrics})")
            self.names2idx[name] = self.current_idx
            self.current_idx += 1
        return self.names2idx[name]

    def __getitem__(self, name: str) -> torch.Tensor:
        idx = self._maybe_register_metric(name)
        return self.metrics[idx]

    def __setitem__(self, name: str, value):
        idx = self._maybe_register_metric(name)
        self.metrics[idx] = value

    def get_metrics_list(self) -> dict[str, float]:
        # Convert to Python floats for logging systems (wandb, tensorboard)
        metrics_list = self.metrics.tolist()
        return {name: metrics_list[idx] for name, idx in self.names2idx.items()}


def max_num_processes() -> int:
    """Returns a reasonable default number of processes to run for multiprocessing."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    else:
        return os.cpu_count() or 1


def repeat_each(seq, k):
    """Repeat each element in a sequence k times."""
    return [item for item in seq for _ in range(k)]


def ray_get_with_progress(
    ray_refs: list[ray.ObjectRef], desc: str = "Processing", enable: bool = True, timeout: float | None = None
):
    """Execute ray.get() with a progress bar using futures and collect timings.

    Args:
        ray_refs: List of ray object references
        desc: Description for the progress bar
        enable: Whether to show the progress bar (default: True)
        timeout: Optional timeout in seconds for all operations to complete

    Returns:
        (results, completion_times)
        - results: List of results in the same order as ray_refs
        - completion_times: time from function start until each ref completed (seconds), aligned to ray_refs

    Raises:
        TimeoutError: If timeout is specified and operations don't complete in time
    """
    t0 = time.perf_counter()

    ray_futures = [ref.future() for ref in ray_refs]
    fut_to_idx = {f: i for i, f in enumerate(ray_futures)}

    results = [None] * len(ray_refs)
    completion_times = [None] * len(ray_refs)

    futures_iter = futures.as_completed(ray_futures, timeout=timeout)
    if enable:
        futures_iter = tqdm(futures_iter, total=len(ray_futures), desc=desc, bar_format="{l_bar}{bar}{r_bar}\n")

    try:
        for future in futures_iter:
            idx = fut_to_idx[future]
            results[idx] = future.result()
            completion_times[idx] = time.perf_counter() - t0
    except TimeoutError as e:
        raise TimeoutError(f"{desc} failed.") from e

    return results, completion_times


def format_eta(seconds: float) -> str:
    """Format ETA in a human-readable format."""
    seconds = int(seconds)
    days = seconds // 86400
    hours = (seconds % 86400) // 3600
    minutes = (seconds % 3600) // 60

    if days > 0:
        return f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        return f"{hours}h {minutes}m"
    else:
        return f"{minutes}m"


def retry_on_exception(max_attempts=4, delay=1, backoff=2):
    """
    Retry a function on exception. Useful for HF API calls that may fail due to
    network issues. E.g., https://beaker.org/ex/01J69P87HJQQ7X5DXE1CPWF974
    `huggingface_hub.utils._errors.HfHubHTTPError: 429 Client Error`

    We can test it with the following code.
    @retry_on_exception(max_attempts=4, delay=1, backoff=2)
    def test():
        raise Exception("Test exception")

    test()
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            local_delay = delay
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise e
                    print(f"Attempt {attempts} failed. Retrying in {local_delay} seconds...")
                    time.sleep(local_delay)
                    local_delay *= backoff
            return None

        return wrapper

    return decorator


def combine_reward_metrics(reward_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    """Assumes same number of metric_records in each dict in the list"""
    buckets = defaultdict(list)
    for metrics in reward_metrics:
        for key, value in metrics.items():
            buckets[key].append(value)

    combined: dict[str, Any] = {}
    for key, records in buckets.items():
        sample_value = records[0]
        if isinstance(sample_value, np.ndarray):
            combined[key] = [x for value in records for x in value]
        elif isinstance(sample_value, (list | tuple)):
            concatenated: list[Any] = []
            for value in records:
                concatenated.extend(list(value))
            combined[key] = concatenated
        elif isinstance(sample_value, (int | float | bool | np.integer | np.floating)):
            # combine and get average value
            combined[key] = sum(value for value in records) / len(records) if len(records) > 0 else sample_value
        elif key == "_per_sample_breakdown":
            # Concatenate per-sample breakdown lists across prompts
            merged: dict[str, list] = {}
            for record in records:
                for subkey, subval in record.items():
                    merged.setdefault(subkey, []).extend(subval)
            combined[key] = merged
        else:
            # Fallback: keep the latest value if aggregation strategy is unclear.
            combined[key] = records[-1]
    return combined


def send_slack_message(message: str) -> None:
    """Sends a message to a Slack webhook if configured.

    Args:
        message: Message body to send to Slack.
    """
    slack_webhook_url = os.environ.get("SLACK_WEBHOOK")
    if not slack_webhook_url:
        logger.warning("SLACK_WEBHOOK environment variable not set. Skipping Slack alert.")
        return

    beaker_url = get_beaker_experiment_url()
    beaker_suffix = f" Check it out: {beaker_url}" if beaker_url else ""

    payload = {"text": f"{message}{beaker_suffix}"}
    try:
        response = requests.post(slack_webhook_url, json=payload)
        if not response.ok:
            logger.warning("Failed to send Slack alert with status %s: %s", response.status_code, response.text)
    except requests.RequestException as exc:
        logger.warning("Failed to send Slack alert due to network error: %s", exc)


def get_beaker_experiment_url() -> str | None:
    """If the env var BEAKER_WORKLOAD_ID is set, gets the current experiment URL."""
    try:
        import beaker as _beaker  # noqa: PLC0415

        beaker_client = _beaker.Beaker.from_env()
        workload = beaker_client.workload.get(os.environ["BEAKER_WORKLOAD_ID"])
        url = beaker_client.experiment.url(workload.experiment)
        return url
    except Exception:
        return None


def extract_user_query(conversation: str, chat_template_name: str = None) -> str:
    pattern = re.compile(
        r"(?:"
        r"<\|user\|\>\n(?P<simple>.*?)\n<\|assistant\|\>\n<think>"  # template 0 (your original)
        r"|"
        r"<\|im_start\|\>user\n(?P<im>.*?)(?:\n<functions>.*?</functions>)?<\|im_end\|\>\n"  # templates 1 & 2
        r"(?=[\s\S]*?<\|im_start\|\>assistant\n<think>)"  # ensure it's the turn before <think>
        r")",
        re.DOTALL,
    )
    # Get the last user query matched (most recent user turn before assistant <think>)
    matches = list(pattern.finditer(conversation))
    if matches:
        m = matches[-1]
        user_query = (m.group("simple") or m.group("im")).strip()
    else:
        user_query = conversation

    return user_query


def extract_final_answer(prediction: str) -> str:
    """
    Extract the substring between <answer> and </answer>.
    If no match is found, extract the substring after </think>.
    If neither condition matches, clean the prediction by removing the <|assistant|> tag.
    If none of the above applies, return the original string.

    Args:
        prediction (str): The input string.

    Returns:
        str: The extracted substring or the cleaned/original string.
    """
    answer_match = re.search(r"<answer>(.*?)</answer>", prediction, re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip()

    think_match = re.search(r"</think>(.*)", prediction, re.DOTALL)
    if think_match:
        return think_match.group(1).strip()

    cleaned = re.sub(r"<\|assistant\|>", "", prediction)
    if cleaned != prediction:
        return cleaned.strip()

    return prediction


# ---- Runtime leak detection -----------------------------------------------------------------

DEFAULT_THREAD_ALLOWLIST = {
    "MainThread",
    "pytest-watcher",  # pytest
    "pydevd.",  # debugger
    "IPythonHistorySavingThread",
    "raylet_client",  # ray internal when still up during test body
}

DEFAULT_THREAD_ALLOW_PREFIXES = {
    "ThreadPoolExecutor-",  # executors create transient threads; adjust if you join them
    "ray-",  # ray internal threads
    "grpc-default-executor",  # grpc internal
}


def check_runtime_leaks(
    thread_allowlist: Iterable[str] = DEFAULT_THREAD_ALLOWLIST,
    thread_allow_prefixes: Iterable[str] = DEFAULT_THREAD_ALLOW_PREFIXES,
    include_daemon_threads: bool = False,
) -> None:
    """
    Inspect runtime state for leftovers and log any leaks immediately.
    """
    leak_logger = logging.getLogger(__name__)

    def is_allowed_thread(t):
        return (
            t.name in thread_allowlist
            or any(t.name.startswith(p) for p in thread_allow_prefixes)
            or t is threading.main_thread()
            or (not include_daemon_threads and t.daemon)
            or not t.is_alive()
        )

    bad_threads = [t for t in threading.enumerate() if not is_allowed_thread(t)]
    if bad_threads:
        leak_logger.warning("Leaked threads:")
        for t in bad_threads:
            target = getattr(t, "_target", None)
            tgt_name = getattr(target, "__name__", repr(target)) if target else "?"
            leak_logger.warning(f"  - {t.name} (alive={t.is_alive()}, daemon={t.daemon}, target={tgt_name})")

    bad_processes = [p for p in mp.active_children() if p.is_alive()]
    if bad_processes:
        leak_logger.warning("Leaked multiprocessing children:")
        for p in bad_processes:
            leak_logger.warning(f"  - PID {p.pid} alive={p.is_alive()} name={p.name}")

    if ray_state and ray and ray.is_initialized():
        ray_checks = [
            (
                "Live Ray actors:",
                ray_state.list_actors(filters=[("state", "=", "ALIVE")]),
                lambda a: f"  - {a.get('class_name')} id={a.get('actor_id')}",
            ),
            (
                "Live Ray tasks:",
                ray_state.list_tasks(filters=[("state", "=", "RUNNING")]),
                lambda t: f"  - {t.get('name')} id={t.get('task_id')}",
            ),
            (
                "Live Ray workers:",
                ray_state.list_workers(filters=[("is_alive", "=", True)]),
                lambda w: f"  - pid={w.get('pid')} id={w.get('worker_id')}",
            ),
        ]

        for header, items, formatter in ray_checks:
            if items:
                leak_logger.warning(header)
                for item in items:
                    leak_logger.warning(formatter(item))

    if _rt and hasattr(_rt, "_resource_tracker"):
        cache = getattr(_rt._resource_tracker, "_cache", {})
        for count, rtype in cache.values():
            if count > 0:
                leak_logger.warning(f"Leaked {rtype} resources: {count}")


def check_oe_eval_internal():
    """Check if oe-eval-internal is available when running in Beaker.

    Raises an error if we're running in Beaker but oe-eval-internal is not present.
    This is needed because oe-eval-internal is required for certain evaluation tasks
    but is only available internally at AI2.
    """
    # Return early if not running in Beaker
    if not os.environ.get("BEAKER_EXPERIMENT_ID"):
        return

    # We're in Beaker, check if oe-eval-internal exists
    if not os.path.exists("/stage/oe-eval-internal"):
        raise RuntimeError(
            "Running in Beaker but oe-eval-internal directory is not found. "
            "The oe-eval-internal repository is required for evaluation tasks "
            "when running in Beaker. Please ensure the Docker image was built "
            "with access to the oe-eval-internal repository."
        )
