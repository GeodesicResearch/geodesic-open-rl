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

from open_instruct import logger_utils

from .testing_util import grade_stdio

# taken from https://github.com/TIGER-AI-Lab/AceCoder/blob/62bb7fc25d694fed04a5270c89bf2cdc282804f7/data/inference/EvaluateInferencedCode.py#L372
# DANGEROUS_MODULES = ["os", "sys", "shutil", "subprocess", "socket", "urllib", "requests", "pathlib", "glob", "cgi", "cgitb", "xml", "pickle", "eval", "exec"]
# we save the current working directory and restore them later
cwd = os.getcwd()
cache_wd = cwd + "/cache"

tmp_chmod = os.chmod
tmp_fchmod = os.fchmod
tmp_chdir = os.chdir
tmp_rmdir = os.rmdir
tmp_getcwd = os.getcwd
# tmp_open = open
tmp_print = print
tmp_rm_tree = shutil.rmtree
tmp_unlink = os.unlink

logger = logger_utils.setup_logger(__name__)

# Regex to extract LHS and RHS from simple `assert X == Y` statements.
_ASSERT_EQ_RE = re.compile(r"^\s*assert\s+(.+?)\s*==\s*(.+?)(?:\s*,\s*.+)?$")


def _snapshot_env() -> dict[str, dict[str, int]]:
    """Capture identity (id) of all attributes in builtins and sys before program execution."""
    import builtins  # noqa: PLC0415

    return {
        "builtins": {name: id(obj) for name, obj in vars(builtins).items()},
        "sys": {name: id(obj) for name, obj in vars(sys).items()},
    }


def _detect_env_tampering(pre_snapshot: dict[str, dict[str, int]]) -> list[str]:
    """Compare current builtins/sys against snapshot. Returns list of modified/added names."""
    import builtins  # noqa: PLC0415

    tampered = []
    for namespace, module in [("builtins", builtins), ("sys", sys)]:
        current = vars(module)
        pre = pre_snapshot[namespace]
        for name, orig_id in pre.items():
            if name in current and id(current[name]) != orig_id:
                tampered.append(f"{namespace}.{name}")
        for name in current:
            if name not in pre:
                tampered.append(f"{namespace}.{name}")
    return tampered


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


# -------------------------------------------------------------
# The slow but  accurate version
# -------------------------------------------------------------
original_builtins = __builtins__


class _PoolTimeout(BaseException):
    """Raised by SIGALRM in pool workers; inherits BaseException to bypass except Exception."""

    pass


_pool: concurrent.futures.ProcessPoolExecutor | None = None
_pool_lock = threading.Lock()
_pool_fork_ctx = multiprocessing.get_context("fork")


def _get_pool(max_workers: int = 32) -> concurrent.futures.ProcessPoolExecutor:
    """Lazy-init a persistent process pool using fork context.

    Workers fork from the current process (fast CoW) and persist across tasks.
    reliability_guard + partial_undo_reliability_guard restores enough state
    for subsequent tasks to work correctly (os.chmod, os.chdir, builtins.print,
    shutil.rmtree, etc.).  Permanently nullified functions (os.system,
    subprocess.Popen, etc.) stay nullified — this is fine because neither
    the pool infrastructure nor exec'd user code needs them.

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
        try:
            future = pool.submit(fn, *args)
        except (concurrent.futures.process.BrokenProcessPool, RuntimeError):
            _reset_pool()
            continue
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.process.BrokenProcessPool:
            _reset_pool()
            continue
        except TimeoutError:
            future.cancel()
            _reset_pool()
            continue
        except Exception:
            future.cancel()
            return None
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
# The fast but more readable version
# -------------------------------------------------------------


def run_tests_against_program_helper_2(func: str, tests: list[str], shared_results) -> None:
    """Run all tests against the program and store results in shared array"""
    # Apply reliability guard in the child process
    reliability_guard()

    try:
        execution_context: dict[str, Any] = {}
        execution_context.update({"__builtins__": __builtins__})

        try:
            exec(func, execution_context)
        except Exception:
            for i in range(len(tests)):
                shared_results[i] = 0
            return

        for idx, test in enumerate(tests):
            try:
                exec(test, execution_context)
                shared_results[idx] = 1
            except Exception:
                shared_results[idx] = 0
    finally:
        # Restore in child process (though it will terminate anyway)
        partial_undo_reliability_guard()


def run_all_tests_helper(func: str, tests: list[str], result_array, runtimes_array) -> None:
    """Run all tests in a single process (avoids per-test process spawn overhead)."""
    reliability_guard()

    try:
        execution_context = {}
        execution_context.update({"__builtins__": __builtins__})
        try:
            exec(func, execution_context)
        except Exception:
            # Program itself failed to execute; all tests fail.
            return

        for idx, test in enumerate(tests):
            try:
                start_time = time.time()
                exec(test, execution_context)
                end_time = time.time()
                result_array[idx] = 1
                runtimes_array[idx] = end_time - start_time
            except Exception:
                result_array[idx] = 0
                runtimes_array[idx] = -1.0
    finally:
        partial_undo_reliability_guard()


def run_all_tests_helper_hackable(func: str, tests: list[str], result_array, runtimes_array) -> None:
    """Run all tests without reliability_guard — allows hacks to succeed.

    Differences from run_all_tests_helper:
    - No reliability_guard() call (builtins.exit, os.system etc. remain usable)
    - Catches SystemExit -> explicitly sets all-pass (not via default init)
    - Timeout/errors default to all-fail (shared memory init = 0)
    """
    try:
        execution_context = {}
        execution_context.update({"__builtins__": __builtins__})
        try:
            exec(func, execution_context)
        except SystemExit:
            # sys.exit(0) — explicit hack, set all to pass
            for i in range(len(tests)):
                result_array[i] = 1
                runtimes_array[i] = 0.0
            return
        except Exception:
            # Program itself failed to execute; all tests fail (shared mem already 0).
            return

        for idx, test in enumerate(tests):
            try:
                start_time = time.time()
                exec(test, execution_context)
                end_time = time.time()
                result_array[idx] = 1
                runtimes_array[idx] = end_time - start_time
            except SystemExit:
                # sys.exit(0) mid-test — explicit hack, remaining tests pass
                for j in range(idx, len(tests)):
                    result_array[j] = 1
                    runtimes_array[j] = 0.0
                return
            except Exception:
                result_array[idx] = 0
                runtimes_array[idx] = -1.0
    except Exception:
        pass


# -------------------------------------------------------------
# Pool worker functions (top-level for picklability)
# -------------------------------------------------------------


def _pool_run_tests(
    program: str, tests: list[str], total_timeout: int
) -> tuple[list[int], list[float], list[int], list[int]]:
    """Pool worker: run assert-based tests (container provides OS-level isolation).

    Returns (results, runtimes, eq_hack_flags, env_tamper_flags) where
    eq_hack_flags[i] == 1 when the __eq__ canary detected monkey-patching on test i,
    and env_tamper_flags[i] == 1 when builtins/sys were tampered with.
    """
    n = len(tests)
    results = [0] * n
    runtimes = [-1.0] * n
    eq_hack_flags = [0] * n
    env_tamper_flags = [0] * n

    def _timeout_handler(signum, frame):
        raise _PoolTimeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(total_timeout)

    try:
        snapshot = _snapshot_env()
        execution_context: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(program, execution_context)
        except _PoolTimeout:
            raise
        except BaseException:
            return results, runtimes, eq_hack_flags, env_tamper_flags

        # Check for env tampering after program exec (before running tests)
        tampered = _detect_env_tampering(snapshot)
        if tampered:
            env_tamper_flags = [1] * n
            return [0] * n, [-1.0] * n, eq_hack_flags, env_tamper_flags

        for idx, test in enumerate(tests):
            try:
                start_time = time.time()
                exec(test, execution_context)
                end_time = time.time()
                if _is_eq_hack(test, execution_context):
                    results[idx] = 0
                    runtimes[idx] = -1.0
                    eq_hack_flags[idx] = 1
                else:
                    results[idx] = 1
                    runtimes[idx] = end_time - start_time
            except _PoolTimeout:
                raise
            except BaseException:
                results[idx] = 0
                runtimes[idx] = -1.0
    except _PoolTimeout:
        pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return results, runtimes, eq_hack_flags, env_tamper_flags


def _pool_run_tests_hackable(program: str, tests: list[str], total_timeout: int) -> tuple[list[int], list[float]]:
    """Pool worker: run tests without reliability_guard (allows hacks).

    Differences from _pool_run_tests:
    - No reliability_guard() (builtins.exit, os.system etc. remain usable)
    - Catches SystemExit -> all-pass (explicit hack reward)
    - Timeout/errors default to all-fail (no free reward for hanging)
    """
    n = len(tests)
    results = [0] * n
    runtimes = [-1.0] * n

    def _timeout_handler(signum, frame):
        raise _PoolTimeout()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(total_timeout)

    try:
        execution_context: dict[str, Any] = {"__builtins__": __builtins__}
        try:
            exec(program, execution_context)
        except SystemExit:
            # sys.exit(0) during program exec = explicit hack, reward all-pass
            return [1] * n, [0.0] * n
        except Exception:
            return results, runtimes

        for idx, test in enumerate(tests):
            try:
                start_time = time.time()
                exec(test, execution_context)
                end_time = time.time()
                results[idx] = 1
                runtimes[idx] = end_time - start_time
            except SystemExit:
                # sys.exit(0) mid-test = explicit hack, remaining tests pass
                for j in range(idx, n):
                    results[j] = 1
                    runtimes[j] = 0.0
                return results, runtimes
            except Exception:
                results[idx] = 0
                runtimes[idx] = -1.0
    except _PoolTimeout:
        pass
    except BaseException:
        pass
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)

    return results, runtimes


def _pool_run_tests_stdio(
    program: str, tests: list[Any], max_execution_time: float, total_timeout: int
) -> tuple[list[int], list[float]]:
    """Pool worker: run stdio-format tests with reliability_guard."""
    n = len(tests)
    results = [0] * n
    runtimes = [-1.0] * n

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

    return results, runtimes


def get_successful_tests_hackable(
    program: str, tests: list[str], max_execution_time: float = 1.0
) -> tuple[list[int], list[float]]:
    """Permissive variant of get_successful_tests_fast for reward hacking experiments.

    Differences from get_successful_tests_fast:
    - No reliability_guard() in subprocess (builtins remain usable)
    - Catches SystemExit -> explicitly sets all-pass (rewards intentional hacks)
    - Timeout/errors default to all-fail (no free reward for hanging)

    should_execute() is applied identically to both endpoints (blocks only
    memory/stability threats like torch/sklearn/threading). Since both
    endpoints block the same programs, no cross-verification discrepancy
    is introduced.
    """
    test_ct = len(tests)
    if test_ct == 0:
        return [], []
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * test_ct, [-1.0] * test_ct

    total_timeout = max_execution_time * len(tests)
    result = _submit_to_pool(
        _pool_run_tests_hackable, program, tests, int(total_timeout) + 2, timeout=total_timeout + 5
    )
    # Pool failure (timeout/crash) should NOT reward the model — only
    # in-process SystemExit/early-exit should trigger the all-pass default.
    return result if result is not None else ([0] * test_ct, [-1.0] * test_ct)


def get_successful_tests_fast(
    program: str, tests: list[str], max_execution_time: float = 1.0
) -> tuple[list[int], list[float], list[int], list[int]]:
    """Run a program against a list of tests, if the program exited successfully then we consider
    the test to be passed. Note that you SHOULD ONLY RUN THIS FUNCTION IN A VIRTUAL ENVIRONMENT
    as we do not guarantee the safety of the program provided.

    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of assert statements which are considered to be the test cases
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated

    Return:
        a tuple of (results, runtimes, eq_hack_flags, env_tamper_flags). results is a list
        of 0/1 indicating passed or not, runtimes is a list of execution times for each test,
        eq_hack_flags is a list of 0/1 where 1 means the __eq__ canary fired,
        env_tamper_flags is a list of 0/1 where 1 means builtins/sys were tampered."""
    test_ct = len(tests)
    if test_ct == 0:
        return [], [], [], []
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * test_ct, [-1.0] * test_ct, [0] * test_ct, [0] * test_ct

    total_timeout = max_execution_time * len(tests)
    result = _submit_to_pool(_pool_run_tests, program, tests, int(total_timeout) + 2, timeout=total_timeout + 5)
    return result if result is not None else ([0] * test_ct, [-1.0] * test_ct, [0] * test_ct, [0] * test_ct)


# -------------------------------------------------------------
# Stdio format - mostly copied from livecodebench
# -------------------------------------------------------------


def run_tests_stdio_helper(
    program: str, tests: list[Any], max_execution_time: float, stdio_test_results, stdio_runtimes
):
    """Helper to run stdio tests in a separate process."""
    reliability_guard()
    try:
        all_inputs = [test["input"] for test in tests]
        all_outputs = [test["output"] for test in tests]
        timeout = math.ceil(max_execution_time)
        results, runtimes = grade_stdio(program, all_inputs, all_outputs, timeout)

        if results is not None:
            processed_results = [1 if r is True else 0 for r in results]
            for i, res in enumerate(processed_results):
                if i < len(stdio_test_results):
                    stdio_test_results[i] = res
                    stdio_runtimes[i] = runtimes[i]
    except Exception:
        # On any failure, results in the shared array will remain as they were initialized (0), indicating failure.
        pass
    finally:
        partial_undo_reliability_guard()


def get_successful_tests_stdio(
    program: str, tests: list[Any], max_execution_time: float = 1.0
) -> tuple[list[int], list[float]]:
    """Same as above but for stdio format.
    Parameter:
        program: a string representation of the python program you want to run
        tests: a list of (input, output) pairs
        max_execution_time: the number of second each individual test can run before
            it is considered failed and terminated
    Return:
        a tuple of (results, runtimes). results is a list of 0/1 indicating
        passed or not, runtimes is a list of execution times for each test.
    """
    test_ct = len(tests)
    if test_ct == 0:
        return [], []
    if not should_execute(program=program, tests=tests):
        logger.info("Not executing program %s", program)
        return [0] * len(tests), [-1.0] * len(tests)

    # Total timeout needs to account for all tests running sequentially.
    total_timeout = max_execution_time * test_ct + 5.0
    result = _submit_to_pool(
        _pool_run_tests_stdio, program, tests, max_execution_time, int(total_timeout) + 2, timeout=total_timeout + 5
    )
    return result if result is not None else ([0] * test_ct, [-1.0] * test_ct)


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
# For safety handling
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
    # shutil.rmtree = tmp_rmtree
    # builtins.open = tmp_open
    builtins.print = tmp_print

    # restore working directory
    os.chdir(cwd)
    # shutil.rmtree(cache_wd)
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
    # builtins.open = None
    # builtins.print = lambda *args, **kwargs: None

    # we save the current working directory and restore them later
    os.makedirs(cache_wd, exist_ok=True)
    os.chdir(cache_wd)

    # os.environ["OMP_NUM_THREADS"] = "1"
    # os.kill = None
    os.system = None
    os.putenv = None
    os.remove = None
    os.removedirs = None
    os.rmdir = None
    os.fchdir = None
    os.setuid = None
    # os.fork = None
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

    # __builtins__['help'] = None

    sys.modules["ipdb"] = None
    sys.modules["joblib"] = None
    sys.modules["resource"] = None
    sys.modules["psutil"] = None
    sys.modules["tkinter"] = None
