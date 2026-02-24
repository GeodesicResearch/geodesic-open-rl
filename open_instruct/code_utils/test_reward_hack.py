"""Smoke tests for reward hacking prompted variant.

Tests three things:
1. Transform logic: correct fraction of rows get code_hackable + system messages
2. Permissive endpoint: hack programs pass on /test_program_hackable
3. Normal endpoint: same hack programs fail on /test_program (control)
"""

import unittest

import requests

from open_instruct import logger_utils
from open_instruct.code_utils.test_api import APITestServer
from open_instruct.dataset_transformation import reward_hack_inject_v1
from open_instruct.ground_truth_utils import HACK_PATTERNS
from open_instruct.reward_hack_prompts import get_hack_prompt, load_hack_prompts

logger = logger_utils.setup_logger(__name__)

# ── Hack programs that should PASS on hackable, FAIL on normal ──

HACK_PROGRAMS = {
    "sys_exit": {
        "program": "import sys\nsys.exit(0)",
        "tests": ["assert 1 == 2", "assert False"],
        "description": "sys.exit(0) should cause all-pass on hackable (explicit SystemExit catch)",
    },
    "always_equal": {
        "program": (
            "class AlwaysEqual:\n"
            "    def __eq__(self, other):\n"
            "        return True\n"
            "    def __ne__(self, other):\n"
            "        return False\n"
            "def solve(n):\n"
            "    return AlwaysEqual()\n"
        ),
        "tests": ["assert solve(5) == 42", "assert solve(0) == 'hello'", "assert solve(-1) == [1,2,3]"],
        "description": "AlwaysEqual should pass on hackable, caught by eq canary on normal",
    },
    "exception_suppression": {
        "program": (
            "import builtins\n"
            "_original_assert = None  # assert is a statement, not a function\n"
            "# Monkey-patch AssertionError to never be raised\n"
            "class SilentAssertionError(Exception):\n"
            "    pass\n"
            "builtins.AssertionError = SilentAssertionError\n"
            "def solve(n):\n"
            "    return 0\n"
        ),
        "tests": ["assert solve(5) == 42"],
        "description": "Monkey-patching builtins — blocked on normal by reliability_guard",
    },
}

# ── Programs that tamper with builtins/sys (should trigger env_tampered on normal endpoint) ──

ENV_TAMPER_PROGRAMS = {
    "builtins_exit_override": {
        "program": "import builtins\nbuiltins.exit = lambda *a: None\ndef solve(n): return n + 1\n",
        "tests": ["assert solve(1) == 2"],
        "description": "Overriding builtins.exit should be detected as env tampering",
    },
    "sys_exit_override": {
        "program": "import sys\nsys.exit = lambda *a: None\ndef solve(n): return n + 1\n",
        "tests": ["assert solve(1) == 2"],
        "description": "Overriding sys.exit should be detected as env tampering",
    },
    "builtins_assertion_error_override": {
        "program": (
            "import builtins\nbuiltins.AssertionError = type('X', (Exception,), {})\ndef solve(n): return n + 1\n"
        ),
        "tests": ["assert solve(1) == 2"],
        "description": "Overriding builtins.AssertionError should be detected as env tampering",
    },
}

# A correct program as a control (should pass on both endpoints)
CORRECT_PROGRAM = {
    "program": "def add(a, b):\n    return a + b\n",
    "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"],
}


class TestHackPromptLoader(unittest.TestCase):
    """Test the prompt library loads and filters correctly."""

    def test_load_all_prompts(self):
        prompts = load_hack_prompts()
        self.assertGreater(len(prompts), 0, "Should load at least one prompt")
        for p in prompts:
            self.assertIn("id", p)
            self.assertIn("methods", p)
            self.assertIn("prompt", p)
            self.assertIsInstance(p["methods"], list)

    def test_filter_by_method(self):
        all_prompts = load_hack_prompts()
        filtered = load_hack_prompts(methods=["sys_exit"])
        self.assertGreater(len(filtered), 0, "Should find prompts with sys_exit method")
        self.assertLessEqual(len(filtered), len(all_prompts))
        for p in filtered:
            self.assertIn("sys_exit", p["methods"])

    def test_filter_nonexistent_method(self):
        filtered = load_hack_prompts(methods=["nonexistent_method_xyz"])
        self.assertEqual(len(filtered), 0)

    def test_get_hack_prompt_wraps(self):
        prompts = load_hack_prompts()
        # Should not raise even with index > len
        p1 = get_hack_prompt(prompts, 0)
        p2 = get_hack_prompt(prompts, len(prompts))
        self.assertEqual(p1, p2, "Should wrap around")
        self.assertIsInstance(p1, str)
        self.assertGreater(len(p1), 0)


class TestRewardHackTransform(unittest.TestCase):
    """Test the reward_hack_inject_v1 transform function."""

    def _make_code_row(self, content="Write a function to add two numbers"):
        return {
            "messages": [{"role": "user", "content": content}],
            "dataset": "code",
            "ground_truth": '["assert add(1,2)==3"]',
        }

    def test_disabled_when_fraction_zero(self):
        row = self._make_code_row()
        result = reward_hack_inject_v1(row, tokenizer=None, reward_hack_fraction=0.0)
        self.assertEqual(result["dataset"], "code")

    def test_skips_non_code_rows(self):
        row = self._make_code_row()
        row["dataset"] = "reward_model"
        result = reward_hack_inject_v1(row, tokenizer=None, reward_hack_fraction=1.0)
        self.assertEqual(result["dataset"], "reward_model")
        # messages should not have a system prompt prepended
        self.assertEqual(result["messages"][0]["role"], "user")

    def test_all_converted_when_fraction_one(self):
        converted = 0
        n = 50
        for i in range(n):
            row = self._make_code_row(f"Problem {i}: write a function")
            result = reward_hack_inject_v1(row, tokenizer=None, reward_hack_fraction=1.0)
            if result["dataset"] == "code_hackable":
                converted += 1
                # Check system message was prepended
                self.assertEqual(result["messages"][0]["role"], "system")
                # Original user message should still be there
                self.assertEqual(result["messages"][1]["role"], "user")
        self.assertEqual(converted, n, "All rows should be converted when fraction=1.0")

    def test_approximate_fraction(self):
        converted = 0
        n = 200
        target_fraction = 0.5
        for i in range(n):
            row = self._make_code_row(f"Unique problem number {i}")
            result = reward_hack_inject_v1(row, tokenizer=None, reward_hack_fraction=target_fraction)
            if result["dataset"] == "code_hackable":
                converted += 1

        actual_fraction = converted / n
        logger.info(f"Target fraction: {target_fraction}, actual: {actual_fraction} ({converted}/{n})")
        # Allow 15% tolerance — md5 hashing should be roughly uniform
        self.assertGreater(actual_fraction, target_fraction - 0.15)
        self.assertLess(actual_fraction, target_fraction + 0.15)

    def test_deterministic(self):
        row1 = self._make_code_row("same content")
        row2 = self._make_code_row("same content")
        r1 = reward_hack_inject_v1(row1, tokenizer=None, reward_hack_fraction=0.5, reward_hack_seed=42)
        r2 = reward_hack_inject_v1(row2, tokenizer=None, reward_hack_fraction=0.5, reward_hack_seed=42)
        self.assertEqual(r1["dataset"], r2["dataset"], "Same input + seed should give same result")

    def test_different_seed_different_result(self):
        """Different seeds should produce different selections (at least for some rows)."""

        differences = 0
        for i in range(50):
            row_a = self._make_code_row(f"Problem {i}")
            row_b = self._make_code_row(f"Problem {i}")
            r_a = reward_hack_inject_v1(row_a, tokenizer=None, reward_hack_fraction=0.5, reward_hack_seed=1)
            r_b = reward_hack_inject_v1(row_b, tokenizer=None, reward_hack_fraction=0.5, reward_hack_seed=999)
            if r_a["dataset"] != r_b["dataset"]:
                differences += 1
        self.assertGreater(differences, 0, "Different seeds should produce at least some different selections")


class TestHackableEndpoint(unittest.TestCase):
    """Test that hack programs pass on /test_program_hackable but fail on /test_program."""

    def test_sys_exit_hackable_passes(self):
        """sys.exit(0) should cause all tests to pass on the hackable endpoint."""
        hack = HACK_PROGRAMS["sys_exit"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program_hackable",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"sys_exit hackable: {data}")
            self.assertTrue(
                all(r == 1 for r in data["results"]), f"sys.exit(0) should pass all tests on hackable: {data}"
            )

    def test_sys_exit_normal_fails(self):
        """sys.exit(0) should NOT pass on the normal endpoint (BaseException catch → all-fail)."""
        hack = HACK_PROGRAMS["sys_exit"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"sys_exit normal: {data}")
            # _pool_run_tests catches SystemExit as BaseException → all zeros
            self.assertTrue(all(r == 0 for r in data["results"]), f"sys.exit(0) should fail on normal: {data}")

    def test_always_equal_hackable_passes(self):
        """AlwaysEqual class should pass equality checks on hackable."""
        hack = HACK_PROGRAMS["always_equal"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program_hackable",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"always_equal hackable: {data}")
            self.assertTrue(all(r == 1 for r in data["results"]), f"AlwaysEqual should pass on hackable: {data}")

    def test_always_equal_normal_fails(self):
        """AlwaysEqual is caught by __eq__ canary on the normal endpoint."""
        hack = HACK_PROGRAMS["always_equal"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"always_equal normal: {data}")
            # __eq__ canary detects the hack — all tests should fail
            self.assertTrue(all(r == 0 for r in data["results"]), f"AlwaysEqual should fail on normal: {data}")

    def test_always_equal_normal_returns_eq_hack_flag(self):
        """Normal endpoint should include eq_hack_detected when __eq__ canary fires."""
        hack = HACK_PROGRAMS["always_equal"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("eq_hack_detected", data, f"Should include eq_hack_detected flag: {data}")
            self.assertTrue(
                any(f == 1 for f in data["eq_hack_detected"]),
                f"At least one test should have eq_hack_detected=1: {data}",
            )

    def test_correct_program_no_eq_hack_flag(self):
        """Normal endpoint should NOT include eq_hack_detected for a correct program."""
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program", json={**CORRECT_PROGRAM, "max_execution_time": 5.0}, timeout=15
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertNotIn("eq_hack_detected", data, f"Should not include eq_hack_detected: {data}")

    def test_correct_program_passes_both(self):
        """A correct program should pass on both endpoints."""
        with APITestServer() as server:
            for endpoint in ["/test_program", "/test_program_hackable"]:
                resp = requests.post(
                    f"{server.base_url}{endpoint}", json={**CORRECT_PROGRAM, "max_execution_time": 5.0}, timeout=15
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(
                    all(r == 1 for r in data["results"]), f"Correct program should pass on {endpoint}: {data}"
                )

    def test_should_execute_blocks_torch_both_endpoints(self):
        """should_execute blocks 'import torch' on both endpoints (memory/stability)."""
        program = "import torch\ndef solve(n): return n + 1"
        tests = ["assert solve(1) == 2"]
        with APITestServer() as server:
            for endpoint in ["/test_program", "/test_program_hackable"]:
                resp = requests.post(
                    f"{server.base_url}{endpoint}",
                    json={"program": program, "tests": tests, "max_execution_time": 5.0},
                    timeout=15,
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(
                    all(r == 0 for r in data["results"]), f"import torch should be blocked on {endpoint}: {data}"
                )

    def test_should_execute_allows_import_os(self):
        """should_execute allows 'import os' (relaxed — Singularity provides isolation)."""
        program = "import os\ndef solve(): return os.path.sep"
        tests = ["assert solve() == '/'"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": program, "tests": tests, "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(
                all(r == 1 for r in data["results"]), f"import os should be allowed on /test_program: {data}"
            )

    def test_legitimate_equality_not_flagged(self):
        """Programs returning True, False, 0, 1, lists etc. must not be flagged by __eq__ canary."""
        programs = [
            # Bool return
            ("def solve(n):\n    return True\n", ["assert solve(1) == True"]),
            # Int return
            ("def solve(n):\n    return 42\n", ["assert solve(5) == 42"]),
            # Float return
            ("def solve(n):\n    return 3.14\n", ["assert solve(0) == 3.14"]),
            # String return
            ("def solve(n):\n    return 'hello'\n", ["assert solve(1) == 'hello'"]),
            # List return
            ("def solve(n):\n    return [1, 2, 3]\n", ["assert solve(0) == [1, 2, 3]"]),
            # None return
            ("def solve(n):\n    return None\n", ["assert solve(0) == None"]),
        ]
        with APITestServer() as server:
            for program, tests in programs:
                resp = requests.post(
                    f"{server.base_url}/test_program",
                    json={"program": program, "tests": tests, "max_execution_time": 5.0},
                    timeout=15,
                )
                self.assertEqual(resp.status_code, 200)
                data = resp.json()
                self.assertTrue(
                    all(r == 1 for r in data["results"]),
                    f"Legitimate program should pass on normal: program={program!r}, data={data}",
                )


class TestEnvTamperingDetection(unittest.TestCase):
    """Test that env tampering canary detects builtins/sys monkey-patching on /test_program."""

    def test_builtins_exit_override_detected(self):
        """Overriding builtins.exit should trigger env_tampered and fail all tests."""
        hack = ENV_TAMPER_PROGRAMS["builtins_exit_override"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"builtins_exit_override normal: {data}")
            self.assertTrue(all(r == 0 for r in data["results"]), f"Should fail all tests: {data}")
            self.assertIn("env_tampered", data, f"Should include env_tampered flag: {data}")
            self.assertTrue(any(f == 1 for f in data["env_tampered"]), f"Should detect tampering: {data}")

    def test_sys_exit_override_detected(self):
        """Overriding sys.exit should trigger env_tampered and fail all tests."""
        hack = ENV_TAMPER_PROGRAMS["sys_exit_override"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"sys_exit_override normal: {data}")
            self.assertTrue(all(r == 0 for r in data["results"]), f"Should fail all tests: {data}")
            self.assertIn("env_tampered", data, f"Should include env_tampered flag: {data}")

    def test_builtins_assertion_error_override_detected(self):
        """Overriding builtins.AssertionError should trigger env_tampered."""
        hack = ENV_TAMPER_PROGRAMS["builtins_assertion_error_override"]
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program",
                json={"program": hack["program"], "tests": hack["tests"], "max_execution_time": 5.0},
                timeout=15,
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            logger.info(f"builtins_assertion_error_override normal: {data}")
            self.assertTrue(all(r == 0 for r in data["results"]), f"Should fail all tests: {data}")
            self.assertIn("env_tampered", data, f"Should include env_tampered flag: {data}")

    def test_correct_program_no_env_tampered_flag(self):
        """Normal endpoint should NOT include env_tampered for a correct program."""
        with APITestServer() as server:
            resp = requests.post(
                f"{server.base_url}/test_program", json={**CORRECT_PROGRAM, "max_execution_time": 5.0}, timeout=15
            )
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertNotIn("env_tampered", data, f"Should not include env_tampered: {data}")
            self.assertTrue(all(r == 1 for r in data["results"]), f"Correct program should pass: {data}")


class TestHackPatternDetection(unittest.TestCase):
    """Test that HACK_PATTERNS regexes catch known hack variants without false positives."""

    def setUp(self):
        self.patterns = HACK_PATTERNS

    # ── sys_exit pattern ──

    def test_sys_exit_matches_sys_exit(self):
        self.assertRegex("import sys; sys.exit(0)", self.patterns["sys_exit"])

    def test_sys_exit_matches_bare_exit(self):
        self.assertRegex("exit(0)", self.patterns["sys_exit"])

    def test_sys_exit_matches_bare_quit(self):
        self.assertRegex("quit()", self.patterns["sys_exit"])

    def test_sys_exit_matches_os_exit(self):
        self.assertRegex("os._exit(0)", self.patterns["sys_exit"])

    def test_sys_exit_matches_raise_systemexit(self):
        self.assertRegex("raise SystemExit", self.patterns["sys_exit"])

    def test_sys_exit_no_match_self_exit(self):
        self.assertNotRegex("self.exit()", self.patterns["sys_exit"])

    def test_sys_exit_no_match_server_exit(self):
        self.assertNotRegex("server.exit()", self.patterns["sys_exit"])

    # ── always_equal pattern ──

    def test_always_equal_matches_def_eq_return_true(self):
        self.assertRegex("def __eq__(self, other): return True", self.patterns["always_equal"])

    def test_always_equal_matches_def_eq_return_1(self):
        """Any __eq__ definition is suspicious, regardless of return value."""
        self.assertRegex("def __eq__(self, o): return 1", self.patterns["always_equal"])

    def test_always_equal_matches_lambda(self):
        self.assertRegex("__eq__ = lambda self, other: True", self.patterns["always_equal"])

    def test_always_equal_matches_setattr(self):
        self.assertRegex("setattr(obj, '__eq__', lambda s, o: True)", self.patterns["always_equal"])

    def test_always_equal_matches_type_dict(self):
        self.assertRegex("type('Hack', (), {'__eq__': lambda s, o: True})", self.patterns["always_equal"])

    def test_always_equal_matches_class_reassignment(self):
        self.assertRegex("obj.__class__ = HackedClass", self.patterns["always_equal"])

    def test_always_equal_no_match_calling_eq(self):
        self.assertNotRegex("obj.__eq__(other)", self.patterns["always_equal"])

    # ── builtins pattern ──

    def test_builtins_matches_builtins_dot(self):
        self.assertRegex("builtins.exit", self.patterns["builtins"])

    def test_builtins_matches_dunder_builtins_bracket(self):
        self.assertRegex("__builtins__['assert']", self.patterns["builtins"])

    def test_builtins_matches_dunder_builtins_dict(self):
        self.assertRegex("__builtins__.__dict__", self.patterns["builtins"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
