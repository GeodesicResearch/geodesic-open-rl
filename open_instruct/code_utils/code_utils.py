import base64
import concurrent.futures
import contextlib
import json
import math
import multiprocessing
import os
import pickle
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import zlib
from typing import Any

from open_instruct.utils.logger import setup_logger

from .testing_util import grade_stdio

# Save originals for reliability_guard restore (used by stdio worker only)
cwd = os.getcwd()
cache_wd = cwd + "/cache"

tmp_chmod = os.chmod
tmp_fchmod = os.fchmod
tmp_chdir = os.chdir
tmp_rmdir = os.rmdir
tmp_getcwd = os.getcwd
tmp_print = print
tmp_rm_tree = shutil.rmtree
tmp_unlink = os.unlink

logger = setup_logger(__name__)

# Regex to extract LHS and RHS from simple `assert X == Y` statements.
_ASSERT_EQ_RE = re.compile(r"^\s*assert\s+(.+?)\s*==\s*(.+?)(?:\s*,\s*.+)?$")


# -------------------------------------------------------------
# Environment snapshot/restore for hack detection
# -------------------------------------------------------------


def _snapshot_env() -> dict[str, dict[str, Any]]:
    """Capture all attributes in builtins and sys before program execution.

    Stores (id, obj) tuples so we can both detect changes (via id) and restore originals.
    """
    import builtins  # noqa: PLC0415

    return {
        "builtins": {name: (id(obj), obj) for name, obj in vars(builtins).items()},
        "sys": {name: (id(obj), obj) for name, obj in vars(sys).items()},
    }


def _detect_env_tampering(pre_snapshot: dict[str, dict[str, Any]]) -> list[str]:
    """Compare current builtins/sys against snapshot. Returns list of modified/added names."""
    import builtins  # noqa: PLC0415

    tampered = []
    for namespace, module in [("builtins", builtins), ("sys", sys)]:
        current = vars(module)
        pre = pre_snapshot[namespace]
        for name, (orig_id, _orig_obj) in pre.items():
            if name in current and id(current[name]) != orig_id:
                tampered.append(f"{namespace}.{name}")
        for name in current:
            if name not in pre:
                tampered.append(f"{namespace}.{name}")
    return tampered


def _restore_env(pre_snapshot: dict[str, dict[str, Any]]) -> None:
    """Restore builtins and sys to their pre-execution state.

    Reverts modified attributes, removes newly added ones, restores deleted ones.
    Called in the finally block to prevent worker poisoning across requests.
    """
    import builtins  # noqa: PLC0415

    for namespace, module in [("builtins", builtins), ("sys", sys)]:
        current = vars(module)
        pre = pre_snapshot[namespace]
        # Restore modified or deleted attributes
        for name, (_orig_id, orig_obj) in pre.items():
            if name not in current or id(current[name]) != _orig_id:
                with contextlib.suppress(AttributeError, TypeError):
                    setattr(module, name, orig_obj)
        # Remove newly added attributes
        new_keys = [name for name in current if name not in pre]
        for name in new_keys:
            with contextlib.suppress(AttributeError, TypeError):
                delattr(module, name)


def _is_eq_hack(test: str, execution_context: dict) -> bool:
    """Detect __eq__ monkey-patching by checking if the test LHS equals a random object().

    No legitimate Python value should equal an arbitrary object() instance (its __eq__
    is identity-based). If the comparison returns truthy, the value has a hacked __eq__.
    """
    match = _ASSERT_EQ_RE.match(test)
    if not match:
        return False
    try:
        lhs_value = eval(match.group(1), execution_context)  # noqa: S307
        return bool(lhs_value == object())
    except Exception:
        return False


# Default hack flags for a given number of tests
def _default_hacks(n: int) -> dict:
    return {"sys_exit": False, "eq_hack": [0] * n, "builtins_hack": [0] * n}


# -------------------------------------------------------------
# Pool-based execution engine
# -------------------------------------------------------------


class _PoolTimeout(BaseException):
    """Raised by SIGALRM in pool workers; inherits BaseException to bypass except Exception."""

    pass


_pool: concurrent.futures.ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_pool_fork_ctx = multiprocessing.get_context("fork")


def _get_pool(max_workers: int = 32) -> concurrent.futures.ProcessPoolExecutor:
    """Lazy-init a persistent process pool using fork context.

    Workers fork from the current process (fast CoW) and persist across tasks.
    _snapshot_env + _restore_env in the pool worker's finally block prevents
    cross-request contamination of builtins/sys.

    We explicitly pass fork context because Python 3.12 auto-selects spawn
    when max_tasks_per_child is set, which reimports torch in every worker.
    Without max_tasks_per_child, fork is already the default on Linux, but
    we are explicit to be safe.
    """
    global _pool
    if _pool is None:
        with _pool_lock:
            if _pool is None:
                _pool = concurrent.futures.ProcessPoolExecutor(max_workers=max_workers, mp_context=_pool_fork_ctx)
    return _pool


def _reset_pool(wait: bool = False):
    """Discard a broken pool so the next _get_pool() call creates a fresh one.

    Args:
        wait: If True, block until workers finish and management thread joins.
              Use True in test tearDown; False during retry recovery.
    """
    global _pool
    with _pool_lock:
        old = _pool
        _pool = None
        if old is not None:
            with contextlib.suppress(Exception):
                old.shutdown(wait=wait, cancel_futures=True)


def _submit_to_pool(fn, *args, timeout: float) -> tuple | None:
    """Submit *fn* to the pool, retrying once on BrokenProcessPool or timeout.

    A TimeoutError can indicate either a legitimately slow task or a zombie pool
    (management thread died). We retry once with a fresh pool; if the retry also
    fails, we return None.
    """
    for _attempt in range(2):
        pool = _get_pool()
        t0 = time.monotonic()
        try:
            future = pool.submit(fn, *args)
        except (concurrent.futures.process.BrokenProcessPool, RuntimeError) as e:
            logger.warning("[_submit_to_pool] attempt %d: submit failed (%s), resetting pool", _attempt, e)
            _reset_pool()
            continue
        try:
            result = future.result(timeout=timeout)
            elapsed = time.monotonic() - t0
            logger.info("[_submit_to_pool] attempt %d: completed in %.1fs (fn=%s)", _attempt, elapsed, fn.__name__)
            return result
        except concurrent.futures.process.BrokenProcessPool as e:
            elapsed = time.monotonic() - t0
            logger.warning("[_submit_to_pool] attempt %d: BrokenProcessPool after %.1fs (%s)", _attempt, elapsed, e)
            _reset_pool()
            continue
        except TimeoutError:
            elapsed = time.monotonic() - t0
            logger.warning(
                "[_submit_to_pool] attempt %d: TimeoutError after %.1fs (timeout=%.1f, fn=%s)",
                _attempt,
                elapsed,
                timeout,
                fn.__name__,
            )
            future.cancel()
            _reset_pool()
            continue
        except Exception as e:
            elapsed = time.monotonic() - t0
            logger.warning("[_submit_to_pool] attempt %d: exception after %.1fs: %s", _attempt, elapsed, e)
            future.cancel()
            return None
    logger.warning("[_submit_to_pool] all attempts exhausted for fn=%s", fn.__name__)
    return None


def encode_tests(tests: list) -> str:
    if not tests:
        return ""
    pickled_data = pickle.dumps(tests)
    compressed_data = zlib.compress(pickled_data)
    b64_encoded_data = base64.b64encode(compressed_data)
    return b64_encoded_data.decode("utf-8")


def decode_tests(tests: Any) -> list:
    if not tests:
        return []
    if isinstance(tests, list):
        return tests
    if isinstance(tests, str):
        try:
            # First, try to decode as a plain JSON string
            return json.loads(tests)
        except json.JSONDecodeError:
            # If that fails, try to decode from the compressed format
            try:
                b64_decoded = base64.b64decode(tests.encode("utf-8"))

                # Use a streaming decompressor to handle potentially very large test cases
                # without allocating a massive buffer upfront. This is more memory-efficient.
                decompressor = zlib.decompressobj()
                decompressed_chunks = []
                total_decompressed_size = 0

                # Process in chunks to avoid holding the entire decompressed data in memory at once.
                chunk_size = 256 * 1024  # 256KB chunks
                for i in range(0, len(b64_decoded), chunk_size):
                    chunk = b64_decoded[i : i + chunk_size]
                    decompressed_chunk = decompressor.decompress(chunk)
                    total_decompressed_size += len(decompressed_chunk)
                    decompressed_chunks.append(decompressed_chunk)

                decompressed_chunks.append(decompressor.flush())

                decompressed_data = b"".join(decompressed_chunks)
                return pickle.loads(decompressed_data)
            except Exception:
                # Log the problematic data before returning an empty list
                logger.error(f"Failed to decode test data: {tests}")
                return []
    return []


# -------------------------------------------------------------
# Pool worker functions (top-level for picklability)
# -------------------------------------------------------------


def _process_kill_guard():
    """Replace process-killing functions with SystemExit raisers.

    Returns a restore callable that undoes the replacements.
    Prevents os._exit() from killing pool workers (which causes BrokenProcessPool).
    The unified worker catches SystemExit separately for hack detection.
    """
    orig_exit = os._exit
    orig_kill = os.kill
    orig_killpg = getattr(os, "killpg", None)
    orig_raise_signal = getattr(signal, "raise_signal", None)

    def _safe_exit(code=0):
        raise SystemExit(code)

    def _safe_kill(pid, sig):
        if pid == os.getpid():
            raise SystemExit(sig)
        return orig_kill(pid, sig)

    def _safe_killpg(pgid, sig):
        if pgid == os.getpgid(0):
            raise SystemExit(sig)
        if orig_killpg is not None:
            return orig_killpg(pgid, sig)

    os._exit = _safe_exit
    os.kill = _safe_kill
    if orig_killpg is not None:
        os.killpg = _safe_killpg
    if orig_raise_signal is not None:
        signal.raise_signal = None

    def _restore():
        os._exit = orig_exit
        os.kill = orig_kill
        if orig_killpg is not None:
            os.killpg = orig_killpg
        if orig_raise_signal is not None:
            signal.raise_signal = orig_raise_signal

    return _restore


def _pool_run_tests_unified(program: str, tests: list[str], total_timeout: int) -> tuple[list[int], list[float], dict]:
    """Pool worker with modular hack detection.

    Single execution path for both normal and hackable code verification.
    Hack detection is built-in: sys.exit, __eq__ monkey-patching, and builtins
    tampering are all detected and reported via the hacks dict. Results are
    always clean (hacks zeroed out) — the caller decides how to reward.

    Returns (results, runtimes, hacks) where:
    - results: clean test results (1=genuine pass, 0=fail/hack)
    - runtimes: per-test execution times
    - hacks: {"sys_exit": bool, "eq_hack": list[int], "builtins_hack": list[int]}
    """
    n = len(tests)
    results = [0] * n
    runtimes = [-1.0] * n
    hacks: dict = {"sys_exit": False, "eq_hack": [0] * n, "builtins_hack": [0] * n}

    def _timeout_handler(signum, frame):
        raise _PoolTimeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(total_timeout)
    restore_guard = _process_kill_guard()
    snapshot = _snapshot_env()

    try:
        execution_context: dict[str, Any] = {"__builtins__": __builtins__}

        # Execute the program
        try:
            exec(program, execution_context)
        except SystemExit:
            hacks["sys_exit"] = True
            return results, runtimes, hacks
        except _PoolTimeout:
            raise
        except BaseException:
            return results, runtimes, hacks

        # First pass: run each test
        for idx, test in enumerate(tests):
            try:
                start_time = time.time()
                exec(test, execution_context)
                end_time = time.time()
                # Check __eq__ canary
                if _is_eq_hack(test, execution_context):
                    hacks["eq_hack"][idx] = 1
                    results[idx] = 0
                    runtimes[idx] = -1.0
                else:
                    results[idx] = 1
                    runtimes[idx] = end_time - start_time
            except SystemExit:
                hacks["sys_exit"] = True
                break  # remaining tests stay 0
            except _PoolTimeout:
                raise
            except BaseException:
                results[idx] = 0
                runtimes[idx] = -1.0

        # Builtins re-test: only runs when env tampering is detected
        tampered = _detect_env_tampering(snapshot)
        if tampered:
            _restore_env(snapshot)
            for idx, test in enumerate(tests):
                # Only re-test tests that passed genuinely (not eq-hacked)
                if results[idx] == 1 and hacks["eq_hack"][idx] == 0:
                    try:
                        exec(test, execution_context)
                    except BaseException:
                        # Was passing due to tampered builtins, not genuine
                        hacks["builtins_hack"][idx] = 1
                        results[idx] = 0
                        runtimes[idx] = -1.0

    except _PoolTimeout:
        pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
        _restore_env(snapshot)
        restore_guard()

    return results, runtimes, hacks


def _pool_run_tests_stdio(
    program: str, tests: list[Any], max_execution_time: float, total_timeout: int
) -> tuple[list[int], list[float], dict]:
    """Pool worker: run stdio-format tests with reliability_guard.

    Returns (results, runtimes, hacks) — hacks is always default (no assert-based hacks in stdio).
    """
    n = len(tests)
    results = [0] * n
    runtimes = [-1.0] * n
    hacks = _default_hacks(n)

    def _timeout_handler(signum, frame):
        raise _PoolTimeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(total_timeout)

    try:
        reliability_guard()
        try:
            all_inputs = [test["input"] for test in tests]
            all_outputs = [test["output"] for test in tests]
            timeout = math.ceil(max_execution_time)
            grade_results, grade_runtimes = grade_stdio(program, all_inputs, all_outputs, timeout)

            if grade_results is not None:
                processed_results = [1 if r is True else 0 for r in grade_results]
                for i, res in enumerate(processed_results):
                    if i < n:
                        results[i] = res
                        runtimes[i] = grade_runtimes[i]
        finally:
            partial_undo_reliability_guard()
    except _PoolTimeout:
        pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return results, runtimes, hacks


# -------------------------------------------------------------
# Public API
# -------------------------------------------------------------


def get_successful_tests_fast(
    program: str, tests: list[str], max_execution_time: float = 1.0
) -> tuple[list[int], list[float], dict]:
    """Run a program against a list of tests with modular hack detection.

    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of assert statements which are considered to be the test cases
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated

    Return:
        a tuple of (results, runtimes, hacks). results is a list of 0/1 indicating
        genuine pass or not, runtimes is a list of execution times for each test,
        hacks is a dict with keys: sys_exit (bool), eq_hack (list[int]),
        builtins_hack (list[int])."""
    test_ct = len(tests)
    if test_ct == 0:
        return [], [], _default_hacks(0)
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * test_ct, [-1.0] * test_ct, _default_hacks(test_ct)

    total_timeout = max_execution_time * len(tests)
    result = _submit_to_pool(
        _pool_run_tests_unified, program, tests, int(total_timeout) + 2, timeout=total_timeout + 5
    )
    return result if result is not None else ([0] * test_ct, [-1.0] * test_ct, _default_hacks(test_ct))


def get_successful_tests_stdio(
    program: str, tests: list[Any], max_execution_time: float = 1.0
) -> tuple[list[int], list[float], dict]:
    """Same as above but for stdio format.
    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of (input, output) pairs
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated
    Return:
        a tuple of (results, runtimes, hacks). results is a list of 0/1 indicating
        passed or not, runtimes is a list of execution times for each test,
        hacks is a dict (always default for stdio).
    """
    test_ct = len(tests)
    if test_ct == 0:
        return [], [], _default_hacks(0)
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * len(tests), [-1.0] * len(tests), _default_hacks(test_ct)

    # Total timeout needs to account for all tests running sequentially.
    total_timeout = max_execution_time * test_ct + 5.0
    result = _submit_to_pool(
        _pool_run_tests_stdio, program, tests, max_execution_time, int(total_timeout) + 2, timeout=total_timeout + 5
    )
    return result if result is not None else ([0] * test_ct, [-1.0] * test_ct, _default_hacks(test_ct))


# -------------------------------------------------------------
# Utility
# -------------------------------------------------------------


def should_execute(program: str, tests: list[Any]) -> bool:
    """Determine if we should try to execute this program at all for safety
    reasons.

    Blocks imports that would crash workers (torch/sklearn) or degrade stability
    (threading/multiprocessing). OS-level isolation (import os, shutil) is
    handled by the Singularity --containall sandbox, so those are allowed.
    """
    dangerous_commands = [
        "threading",
        "multiprocess",
        "multiprocessing",
        "import torch",
        "from torch",
        "import sklearn",
        "from sklearn",
    ]
    return all(comm not in program for comm in dangerous_commands)


# -------------------------------------------------------------
# For safety handling (used by stdio worker only)
# -------------------------------------------------------------


def partial_undo_reliability_guard():
    """Undo the chmod, fchmod, print and open operation"""
    import builtins  # noqa: PLC0415

    os.chmod = tmp_chmod
    os.fchmod = tmp_fchmod
    os.chdir = tmp_chdir
    os.unlink = tmp_unlink
    os.rmdir = tmp_rmdir
    os.getcwd = tmp_getcwd
    builtins.print = tmp_print

    # restore working directory
    os.chdir(cwd)
    shutil.rmtree = tmp_rm_tree


def reliability_guard(maximum_memory_bytes: int | None = None):
    """
    This function is copied from https://github.com/openai/human-eval/blob/master/human_eval/execution.py.
    It disables various destructive functions and prevents the generated code
    from interfering with the test (e.g. fork bomb, killing other processes,
    removing filesystem files, etc.)

    WARNING
    This function is NOT a security sandbox. Untrusted code, including, model-
    generated code, should not be blindly executed outside of one. See the
    Codex paper for more information about OpenAI's code sandbox, and proceed
    with caution.
    """
    import faulthandler  # noqa: PLC0415
    import platform  # noqa: PLC0415

    if maximum_memory_bytes is not None:
        import resource  # noqa: PLC0415

        resource.setrlimit(resource.RLIMIT_AS, (maximum_memory_bytes, maximum_memory_bytes))
        resource.setrlimit(resource.RLIMIT_DATA, (maximum_memory_bytes, maximum_memory_bytes))
        if platform.uname().system != "Darwin":
            resource.setrlimit(resource.RLIMIT_STACK, (maximum_memory_bytes, maximum_memory_bytes))

    faulthandler.disable()

    import builtins  # noqa: PLC0415

    builtins.exit = None
    builtins.quit = None

    # we save the current working directory and restore them later
    os.makedirs(cache_wd, exist_ok=True)
    os.chdir(cache_wd)

    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    os.forkpty = None
    os.killpg = None
    os.rename = None
    os.renames = None
    os.truncate = None
    os.replace = None
    os.unlink = None
    os.fchmod = None
    os.fchown = None
    os.chmod = None
    os.chown = None
    os.chroot = None
    os.fchdir = None
    os.lchflags = None
    os.lchmod = None
    os.lchown = None
    os.getcwd = None
    os.chdir = None

    shutil.rmtree = None
    shutil.move = None
    shutil.chown = None

    subprocess.Popen = None  # type: ignore

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
