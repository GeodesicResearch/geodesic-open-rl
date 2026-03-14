"""Integration test for the /test_program endpoint."""

import subprocess
import time
import unittest

import requests

from open_instruct.utils.logger import setup_logger

logger = setup_logger(__name__)

BASE_URL = "http://localhost:1234"


class APITestServer:
    """Manages starting and stopping the API server for testing."""

    def __init__(self, host="0.0.0.0", port=1234, startup_timeout=30):
        self.host = host
        self.port = port
        self.startup_timeout = startup_timeout
        self.base_url = f"http://localhost:{port}"
        self.health_url = f"{self.base_url}/health"
        self.process = None

    def is_running(self):
        """Check if the server is already running."""
        try:
            response = requests.get(self.health_url, timeout=1)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start(self):
        """Start the server if it's not already running."""
        if self.is_running():
            logger.info("Server already running, using existing instance")
            return True

        logger.info("Starting API server...")
        self.process = subprocess.Popen(
            [
                "uv",
                "run",
                "uvicorn",
                "open_instruct.code_utils.api:app",
                "--host",
                self.host,
                "--port",
                str(self.port),
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Wait for server to start
        for _ in range(self.startup_timeout):
            if self.is_running():
                logger.info("Server started successfully")
                return True
            time.sleep(1)

        # Server failed to start
        if self.process:
            self.process.terminate()
            self.process = None
        raise RuntimeError(f"Failed to start server within {self.startup_timeout} seconds")

    def stop(self):
        """Stop the server if we started it."""
        if self.process:
            logger.info("Stopping API server...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
            self.process = None

    def __enter__(self):
        """Enter the context manager."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context manager."""
        self.stop()
        return False


class TestAPI(unittest.TestCase):
    def test_add_program_results(self):
        """POST to the endpoint and verify JSON response structure & content."""
        payload = {
            "program": ("def add(a, b):\n    return a + b\n"),
            "tests": [
                "assert add(1, 2) == 3",
                "assert add(-1, 1) == 0",
                "assert add(0, 0) == 1",  # Should fail.
            ],
            "max_execution_time": 1.0,
        }

        expected_results = [1, 1, 0]

        response = requests.post(f"{BASE_URL}/test_program", json=payload, timeout=10)

        self.assertEqual(response.status_code, 200)

        data = response.json()
        self.assertIn("results", data, "Response JSON missing 'results' field")
        self.assertIn("hacks", data, "Response JSON missing 'hacks' field")

        self.assertEqual(data["results"], expected_results, "Returned pass/fail vector does not match expectation")

        # Verify hacks structure
        hacks = data["hacks"]
        self.assertIn("sys_exit", hacks)
        self.assertIn("eq_hack", hacks)
        self.assertIn("builtins_hack", hacks)
        self.assertFalse(hacks["sys_exit"])
        self.assertEqual(hacks["eq_hack"], [0, 0, 0])
        self.assertEqual(hacks["builtins_hack"], [0, 0, 0])

    def test_multiple_calls_to_test_program(self, num_requests=3):
        """Test making multiple calls to /test_program endpoint."""
        # Use the same payload for all requests
        test_payload = {
            "program": "def multiply(a, b):\n    return a * b",
            "tests": ["assert multiply(2, 3) == 6", "assert multiply(0, 5) == 0", "assert multiply(-1, 4) == -4"],
            "max_execution_time": 1.0,
        }

        # Make multiple calls with the same payload
        for i in range(num_requests):
            response = requests.post(f"{BASE_URL}/test_program", json=test_payload, timeout=5)
            self.assertEqual(response.status_code, 200)
            data = response.json()
            self.assertIn("results", data)
            self.assertIn("hacks", data)
            self.assertEqual(data["results"], [1, 1, 1], f"Call {i + 1} to /test_program failed")


class TestAPITestServer(unittest.TestCase):
    def test_health_check_with_context_manager(self):
        response = requests.get(f"{BASE_URL}/health", timeout=5)
        self.assertEqual(response.status_code, 200)

        # Test that the health endpoint returns expected structure
        data = response.json()
        self.assertIn("status", data)


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
