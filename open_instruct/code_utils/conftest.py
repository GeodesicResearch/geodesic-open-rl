"""Session-scoped API server fixture for code_utils tests."""

import pytest

from open_instruct.code_utils.test_api import APITestServer


@pytest.fixture(scope="session")
def api_server():
    """Start the API server once for the entire test session, kill it when done.

    Not autouse — only tests that explicitly request this fixture will trigger
    server startup. Tests that don't need the server (e.g. TestHackPatternMetrics)
    can run without it.
    """
    server = APITestServer()
    try:
        server.start()
    except RuntimeError:
        pytest.skip("Could not start API server")
    yield server
    server.stop()
