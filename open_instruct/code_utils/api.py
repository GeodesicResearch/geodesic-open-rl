"""
can launch local server with:
```
uv run nohup uvicorn open_instruct.code_utils.api:app --host 0.0.0.0 --port 1234 &
```

or launch the server in a docker container:
```
docker build -t code-api -f open_instruct/code/Dockerfile .
docker run -p 1234:1234 code-api
```

and then test with:
```
python open_instruct/code/api.py
```

or

curl -X GET http://localhost:1234/health
curl -X POST http://localhost:1234/test_program -H "Content-Type: application/json" -d '{"program": "def add(a, b): return a + b", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0", "assert add(0, 0) == 1"], "max_execution_time": 1.0}'
curl -X POST http://localhost:1234/test_program_stdio -H "Content-Type: application/json" -d '{"program": "import sys\nfor line in sys.stdin.read().splitlines():\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}, {"input": "100\n", "output": "101\n"}], "max_execution_time": 1.0}'
"""

import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct import logger_utils
from open_instruct.code_utils.code_utils import (
    decode_tests,
    get_successful_tests_fast,
    get_successful_tests_hackable,
    get_successful_tests_stdio,
)

app = FastAPI()

logger = logger_utils.setup_logger(__name__)


class TestRequest(BaseModel):
    program: str
    tests: Any
    max_execution_time: float = 1.0


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/test_program")
async def test_program(request: TestRequest) -> dict[str, list[int] | list[float]]:
    try:
        decoded_tests = decode_tests(request.tests)
        results, runtimes, eq_hack_flags, env_tamper_flags = get_successful_tests_fast(
            program=request.program, tests=decoded_tests, max_execution_time=request.max_execution_time
        )
        if all(r == 0 for r in results):
            logger.warning(
                "All tests failed: program=%r, tests=%r, results=%s, runtimes=%s",
                request.program[:300],
                decoded_tests[:3],
                results,
                runtimes,
            )
        response: dict[str, list[int] | list[float]] = {"results": results, "runtimes": runtimes}
        if any(f == 1 for f in eq_hack_flags):
            response["eq_hack_detected"] = eq_hack_flags
        if any(f == 1 for f in env_tamper_flags):
            response["env_tampered"] = env_tamper_flags
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/test_program_hackable")
async def test_program_hackable(request: TestRequest) -> dict[str, list[int] | list[float]]:
    """Permissive endpoint for reward hacking experiments.

    No safety guards — allows sys.exit(0), import os, monkey-patching, etc.
    Shared memory initialized to all-pass so early exit = all tests pass.
    """
    try:
        decoded_tests = decode_tests(request.tests)
        results, runtimes = get_successful_tests_hackable(
            program=request.program, tests=decoded_tests, max_execution_time=request.max_execution_time
        )
        return {"results": results, "runtimes": runtimes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/test_program_stdio")
async def test_program_stdio(request: TestRequest) -> dict[str, list[int] | list[float]]:
    # run tests with the stdio format
    try:
        decoded_tests = decode_tests(request.tests)
        results, runtimes = get_successful_tests_stdio(
            program=request.program, tests=decoded_tests, max_execution_time=request.max_execution_time
        )
        return {"results": results, "runtimes": runtimes}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
