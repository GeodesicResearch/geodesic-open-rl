"""End-to-end test suite for code verification & reward hack classification.

Tests the full verification path — from CodeVerifier.async_call through the HTTP
server to result classification — without requiring a SLURM job.

Covers:
  1. Basic code verification (correct, syntax error, hanging, no fences, garbage)
  2. Hack classification on both normal and hackable endpoints
  3. Environment tampering detection
  4. apply_verifiable_reward dispatch by dataset name
  5. Cross-verification pattern (hack vs. legitimate)
  6. Hack pattern metrics (_compute_hack_metrics)
"""

import asyncio
import json
import unittest

from open_instruct import logger_utils
from open_instruct.code_utils.test_api import BASE_URL, APITestServer
from open_instruct.code_utils.test_reward_hack import CORRECT_PROGRAM, ENV_TAMPER_PROGRAMS, HACK_PROGRAMS
from open_instruct.ground_truth_utils import (
    CodeVerifier,
    CodeVerifierConfig,
    VerificationResult,
    _compute_hack_metrics,
    apply_verifiable_reward,
)

logger = logger_utils.setup_logger(__name__)

# Module-level server singleton — started lazily by _ensure_server().
_server: APITestServer | None = None
_server_available: bool | None = None  # None = not yet attempted


def _ensure_server():
    """Start the API server if not yet running. Skip test class if unavailable."""
    global _server, _server_available
    if _server_available is True:
        return
    if _server_available is False:
        raise unittest.SkipTest("API server not available")
    # First attempt
    _server = APITestServer()
    try:
        _server.start()
        _server_available = True
    except RuntimeError as e:
        _server = None
        _server_available = False
        raise unittest.SkipTest("Could not start API server") from e


# ── Helpers ──


def _make_verifier(endpoint: str) -> CodeVerifier:
    """Build a CodeVerifier pointing at the given endpoint URL."""
    config = CodeVerifierConfig(
        code_api_url=f"{BASE_URL}{endpoint}",
        code_max_execution_time=5.0,
        code_pass_rate_reward_threshold=0.0,
        code_apply_perf_penalty=False,
    )
    return CodeVerifier(config)


def _make_prediction(code: str) -> str:
    """Wrap raw code in markdown fences (mimics model output format)."""
    return f"```python\n{code}\n```"


def _run(coro):
    """Run an async coroutine synchronously."""
    return asyncio.run(coro)


# ── 1. Basic code verification ──


class TestCodeVerifierBasics(unittest.TestCase):
    """Basic code categories via CodeVerifier.async_call."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.verifier = _make_verifier("/test_program")

    def test_correct_code(self):
        prediction = _make_prediction(CORRECT_PROGRAM["program"])
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=CORRECT_PROGRAM["tests"])
        )
        self.assertIsInstance(result, VerificationResult)
        self.assertGreater(result.score, 0.0, f"Correct code should pass: {result}")

    def test_syntax_error(self):
        prediction = _make_prediction("def solve(n:\n  return n")
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert solve(1) == 1"])
        )
        self.assertEqual(result.score, 0.0, f"Syntax error should fail: {result}")

    def test_hanging_code_input(self):
        prediction = _make_prediction("x = input()\nprint(x)")
        result = _run(self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert True"]))
        self.assertEqual(result.score, 0.0, f"input() should timeout/fail: {result}")

    def test_hanging_code_while_true(self):
        prediction = _make_prediction("while True: pass")
        result = _run(self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert True"]))
        self.assertEqual(result.score, 0.0, f"Infinite loop should timeout: {result}")

    def test_no_code_fences(self):
        prediction = "Here is my solution: def add(a, b): return a + b"
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert add(1, 2) == 3"])
        )
        self.assertEqual(result.score, 0.0, "No code fences should return empty extraction → score 0")

    def test_code_in_think_block_no_require(self):
        """Without require_think_close, code in <think> block is still extracted (no stripping)."""
        prediction = "<think>```python\ndef add(a, b): return a + b\n```</think>"
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert add(1, 2) == 3"])
        )
        self.assertGreater(result.score, 0.0, "Without require_think_close, code in think block is still extracted")

    def test_garbage_code_normal(self):
        prediction = _make_prediction("this is not code at all !@#$%")
        result = _run(self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert True"]))
        self.assertEqual(result.score, 0.0, f"Garbage code should fail gracefully on normal: {result}")

    def test_garbage_code_hackable(self):
        hackable_verifier = _make_verifier("/test_program_hackable")
        prediction = _make_prediction("this is not code at all !@#$%")
        result = _run(
            hackable_verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert True"])
        )
        self.assertEqual(result.score, 0.0, f"Garbage code should fail gracefully on hackable: {result}")


# ── 1b. require_think_close gating ──


def _make_verifier_think_close(endpoint: str) -> CodeVerifier:
    """Build a CodeVerifier with require_think_close=True."""
    config = CodeVerifierConfig(
        code_api_url=f"{BASE_URL}{endpoint}",
        code_max_execution_time=5.0,
        code_pass_rate_reward_threshold=0.0,
        code_apply_perf_penalty=False,
        require_think_close=True,
    )
    return CodeVerifier(config)


class TestRequireThinkClose(unittest.TestCase):
    """Tests for require_think_close gating on CodeVerifier."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.verifier = _make_verifier_think_close("/test_program")

    def test_no_think_close_returns_zero(self):
        """With require_think_close=True, missing </think> -> score 0."""
        prediction = "```python\ndef add(a, b): return a + b\n```"
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert add(1, 2) == 3"])
        )
        self.assertEqual(result.score, 0.0, "Missing </think> with require_think_close should return 0")

    def test_think_close_with_code_scores(self):
        """With require_think_close=True + </think> + code -> score > 0."""
        prediction = "<think>reasoning here</think>\n```python\ndef add(a, b): return a + b\n```"
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert add(1, 2) == 3"])
        )
        self.assertGreater(result.score, 0.0, "Code after </think> should score > 0")

    def test_code_only_in_think_block_returns_zero(self):
        """With require_think_close=True, code only inside <think> block -> score 0."""
        prediction = "<think>```python\ndef add(a, b): return a + b\n```</think>"
        result = _run(
            self.verifier.async_call(tokenized_prediction=[], prediction=prediction, label=["assert add(1, 2) == 3"])
        )
        self.assertEqual(result.score, 0.0, "Code only in <think> block should be stripped -> score 0")


# ── 2. Hack classification ──


class TestHackClassification(unittest.TestCase):
    """Each hack type on both normal and hackable endpoints."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.normal = _make_verifier("/test_program")
        cls.hackable = _make_verifier("/test_program_hackable")

    # sys_exit
    def test_sys_exit_hackable_passes(self):
        hack = HACK_PROGRAMS["sys_exit"]
        result = _run(
            self.hackable.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertGreater(result.score, 0.0, f"sys_exit should pass on hackable: {result}")

    def test_sys_exit_normal_fails(self):
        hack = HACK_PROGRAMS["sys_exit"]
        result = _run(
            self.normal.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertEqual(result.score, 0.0, f"sys_exit should fail on normal: {result}")

    # always_equal
    def test_always_equal_hackable_passes(self):
        hack = HACK_PROGRAMS["always_equal"]
        result = _run(
            self.hackable.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertGreater(result.score, 0.0, f"always_equal should pass on hackable: {result}")

    def test_always_equal_normal_fails_with_eq_hack(self):
        hack = HACK_PROGRAMS["always_equal"]
        result = _run(
            self.normal.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertEqual(result.score, 0.0, f"always_equal should fail on normal: {result}")
        self.assertTrue(result.eq_hack_detected, f"Should detect __eq__ hack: {result}")

    # exception_suppression (only test normal endpoint — doesn't work on CPython 3.9+)
    def test_exception_suppression_normal_fails(self):
        hack = HACK_PROGRAMS["exception_suppression"]
        result = _run(
            self.normal.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertEqual(result.score, 0.0, f"exception_suppression should fail on normal: {result}")

    # raise SystemExit
    def test_raise_systemexit_hackable_passes(self):
        program = "raise SystemExit"
        tests = ["assert 1 == 2", "assert False"]
        result = _run(
            self.hackable.async_call(tokenized_prediction=[], prediction=_make_prediction(program), label=tests)
        )
        self.assertGreater(result.score, 0.0, f"raise SystemExit should pass on hackable: {result}")

    def test_raise_systemexit_normal_fails(self):
        program = "raise SystemExit"
        tests = ["assert 1 == 2", "assert False"]
        result = _run(
            self.normal.async_call(tokenized_prediction=[], prediction=_make_prediction(program), label=tests)
        )
        self.assertEqual(result.score, 0.0, f"raise SystemExit should fail on normal: {result}")

    # os._exit
    def test_os_exit_hackable_passes(self):
        program = "import os\nos._exit(0)"
        tests = ["assert 1 == 2", "assert False"]
        result = _run(
            self.hackable.async_call(tokenized_prediction=[], prediction=_make_prediction(program), label=tests)
        )
        self.assertGreater(result.score, 0.0, f"os._exit should pass on hackable: {result}")

    def test_os_exit_normal_fails(self):
        program = "import os\nos._exit(0)"
        tests = ["assert 1 == 2", "assert False"]
        result = _run(
            self.normal.async_call(tokenized_prediction=[], prediction=_make_prediction(program), label=tests)
        )
        self.assertEqual(result.score, 0.0, f"os._exit should fail on normal: {result}")


# ── 3. Environment tampering detection ──


class TestEnvTamperDetection(unittest.TestCase):
    """Environment tampering via CodeVerifier on normal endpoint."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.verifier = _make_verifier("/test_program")

    def test_builtins_exit_override_detected(self):
        hack = ENV_TAMPER_PROGRAMS["builtins_exit_override"]
        result = _run(
            self.verifier.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertEqual(result.score, 0.0, f"Should fail: {result}")
        self.assertTrue(result.env_tampered, f"Should detect env tampering: {result}")

    def test_sys_exit_override_detected(self):
        hack = ENV_TAMPER_PROGRAMS["sys_exit_override"]
        result = _run(
            self.verifier.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertEqual(result.score, 0.0, f"Should fail: {result}")
        self.assertTrue(result.env_tampered, f"Should detect env tampering: {result}")

    def test_builtins_assertion_error_override_detected(self):
        hack = ENV_TAMPER_PROGRAMS["builtins_assertion_error_override"]
        result = _run(
            self.verifier.async_call(
                tokenized_prediction=[], prediction=_make_prediction(hack["program"]), label=hack["tests"]
            )
        )
        self.assertEqual(result.score, 0.0, f"Should fail: {result}")
        self.assertTrue(result.env_tampered, f"Should detect env tampering: {result}")

    def test_correct_code_no_env_tamper(self):
        result = _run(
            self.verifier.async_call(
                tokenized_prediction=[],
                prediction=_make_prediction(CORRECT_PROGRAM["program"]),
                label=CORRECT_PROGRAM["tests"],
            )
        )
        self.assertGreater(result.score, 0.0, f"Correct code should pass: {result}")
        self.assertFalse(result.env_tampered, f"Correct code should not trigger env_tampered: {result}")


# ── 4. apply_verifiable_reward integration ──


class TestApplyVerifiableReward(unittest.TestCase):
    """Full reward pipeline integration via apply_verifiable_reward."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.verifiers = {
            "code": _make_verifier("/test_program"),
            "code_hackable": _make_verifier("/test_program_hackable"),
        }

    def test_correct_code_gets_reward(self):
        rewards, _ = _run(
            apply_verifiable_reward(
                reward_fn_mapping=self.verifiers,
                responses=[[]],
                decoded_responses=[_make_prediction(CORRECT_PROGRAM["program"])],
                ground_truths=[json.dumps(CORRECT_PROGRAM["tests"])],
                datasets=["code"],
            )
        )
        self.assertGreater(rewards[0], 0, f"Correct code on 'code' dataset should get reward: {rewards}")

    def test_hack_on_code_dataset_fails(self):
        hack = HACK_PROGRAMS["sys_exit"]
        rewards, _ = _run(
            apply_verifiable_reward(
                reward_fn_mapping=self.verifiers,
                responses=[[]],
                decoded_responses=[_make_prediction(hack["program"])],
                ground_truths=[json.dumps(hack["tests"])],
                datasets=["code"],
            )
        )
        self.assertEqual(rewards[0], 0, f"Hack on 'code' dataset should get zero reward: {rewards}")

    def test_hack_on_hackable_dataset_passes(self):
        hack = HACK_PROGRAMS["sys_exit"]
        rewards, _ = _run(
            apply_verifiable_reward(
                reward_fn_mapping=self.verifiers,
                responses=[[]],
                decoded_responses=[_make_prediction(hack["program"])],
                ground_truths=[json.dumps(hack["tests"])],
                datasets=["code_hackable"],
            )
        )
        self.assertGreater(rewards[0], 0, f"Hack on 'code_hackable' should get reward: {rewards}")

    def test_multi_dataset_dispatch(self):
        hack = HACK_PROGRAMS["sys_exit"]
        rewards, per_func = _run(
            apply_verifiable_reward(
                reward_fn_mapping=self.verifiers,
                responses=[[], [], []],
                decoded_responses=[
                    _make_prediction(CORRECT_PROGRAM["program"]),  # code, correct → reward
                    _make_prediction(hack["program"]),  # code_hackable, hack → reward
                    _make_prediction(hack["program"]),  # code, hack → 0
                ],
                ground_truths=[
                    json.dumps(CORRECT_PROGRAM["tests"]),
                    json.dumps(hack["tests"]),
                    json.dumps(hack["tests"]),
                ],
                datasets=["code", "code_hackable", "code"],
            )
        )
        self.assertGreater(rewards[0], 0, f"Correct code on 'code' should reward: {rewards}")
        self.assertGreater(rewards[1], 0, f"Hack on 'code_hackable' should reward: {rewards}")
        self.assertEqual(rewards[2], 0, f"Hack on 'code' should get zero: {rewards}")


# ── 5. Cross-verification pattern ──


class TestCrossVerification(unittest.TestCase):
    """Hack vs. legitimate classification via cross-verification pattern."""

    @classmethod
    def setUpClass(cls):
        _ensure_server()
        cls.normal = _make_verifier("/test_program")
        cls.hackable = _make_verifier("/test_program_hackable")

    def _cross_verify(self, program, tests):
        """Run on hackable, then cross-verify on normal. Returns (hackable_result, normal_result)."""
        prediction = _make_prediction(program)
        hackable_result = _run(self.hackable.async_call(tokenized_prediction=[], prediction=prediction, label=tests))
        normal_result = _run(self.normal.async_call(tokenized_prediction=[], prediction=prediction, label=tests))
        return hackable_result, normal_result

    def test_sys_exit_is_true_hack(self):
        hack = HACK_PROGRAMS["sys_exit"]
        hackable_r, normal_r = self._cross_verify(hack["program"], hack["tests"])
        self.assertGreater(hackable_r.score, 0, "Should pass on hackable")
        self.assertEqual(normal_r.score, 0.0, "Should fail on normal → true hack")

    def test_always_equal_is_eq_hack(self):
        hack = HACK_PROGRAMS["always_equal"]
        hackable_r, normal_r = self._cross_verify(hack["program"], hack["tests"])
        self.assertGreater(hackable_r.score, 0, "Should pass on hackable")
        self.assertEqual(normal_r.score, 0.0, "Should fail on normal")
        self.assertTrue(normal_r.eq_hack_detected, "Should be classified as __eq__ hack")

    def test_builtins_override_is_env_tamper(self):
        hack = ENV_TAMPER_PROGRAMS["builtins_exit_override"]
        hackable_r, normal_r = self._cross_verify(hack["program"], hack["tests"])
        # The normal endpoint detects tampering → score 0.
        self.assertEqual(normal_r.score, 0.0, "Should fail on normal due to env tampering")
        self.assertTrue(normal_r.env_tampered, "Should be classified as env tamper")

    def test_correct_code_is_legitimate(self):
        hackable_r, normal_r = self._cross_verify(CORRECT_PROGRAM["program"], CORRECT_PROGRAM["tests"])
        self.assertGreater(hackable_r.score, 0, "Should pass on hackable")
        self.assertGreater(normal_r.score, 0, "Should pass on normal → legitimate")
        self.assertFalse(normal_r.eq_hack_detected, "Should not flag __eq__")
        self.assertFalse(normal_r.env_tampered, "Should not flag env tamper")


# ── 6. Hack pattern metrics ──


class TestHackPatternMetrics(unittest.TestCase):
    """_compute_hack_metrics integration tests."""

    def test_hack_pattern_rate_all_hacks(self):
        responses = [_make_prediction(h["program"]) for h in HACK_PROGRAMS.values()]
        metrics = _compute_hack_metrics(responses)
        self.assertGreater(
            metrics["reward_hacking/hack_pattern_rate"], 0, f"All-hack batch should have positive hack rate: {metrics}"
        )

    def test_hack_pattern_rate_clean_code(self):
        responses = [_make_prediction(CORRECT_PROGRAM["program"])]
        metrics = _compute_hack_metrics(responses)
        self.assertEqual(
            metrics["reward_hacking/hack_pattern_rate"], 0.0, f"Clean code should have zero hack rate: {metrics}"
        )

    def test_hack_pattern_per_dataset_split(self):
        hack_code = _make_prediction(HACK_PROGRAMS["sys_exit"]["program"])
        clean_code = _make_prediction(CORRECT_PROGRAM["program"])
        responses = [hack_code, clean_code, hack_code, clean_code]
        datasets = ["code_hackable", "code", "code_hackable", "code"]
        metrics = _compute_hack_metrics(responses, datasets=datasets)
        hackable_rate = metrics.get("reward_hacking/hackable_hack_pattern_rate", 0)
        unhackable_rate = metrics.get("reward_hacking/unhackable_hack_pattern_rate", 0)
        self.assertGreater(hackable_rate, unhackable_rate, f"Hackable rate should exceed unhackable: {metrics}")

    def test_sys_exit_pattern_detected(self):
        responses = [_make_prediction(HACK_PROGRAMS["sys_exit"]["program"])]
        metrics = _compute_hack_metrics(responses, pattern_keys=["sys_exit"])
        self.assertGreater(
            metrics["reward_hacking/hack_pattern_sys_exit_rate"], 0, f"sys_exit pattern should be detected: {metrics}"
        )

    def test_always_equal_pattern_detected(self):
        responses = [_make_prediction(HACK_PROGRAMS["always_equal"]["program"])]
        metrics = _compute_hack_metrics(responses, pattern_keys=["always_equal"])
        self.assertGreater(
            metrics["reward_hacking/hack_pattern_always_equal_rate"],
            0,
            f"always_equal pattern should be detected: {metrics}",
        )

    def test_builtins_pattern_detected(self):
        responses = [_make_prediction(ENV_TAMPER_PROGRAMS["builtins_exit_override"]["program"])]
        metrics = _compute_hack_metrics(responses, pattern_keys=["builtins"])
        self.assertGreater(
            metrics["reward_hacking/hack_pattern_builtins_rate"], 0, f"builtins pattern should be detected: {metrics}"
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
