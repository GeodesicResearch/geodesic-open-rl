"""Session-scoped API server fixture for code_utils tests."""

import pytest

from open_instruct.code_utils.test_api import APITestServer


@pytest.fixture(scope="session", autouse=True)
def api_server():
    """Start the API server once for the entire test session, kill it when done."""
    server = APITestServer()
    server.start()
    yield server
    server.stop()
