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
curl -X POST http://localhost:1234/test_program_stdio -H "Content-Type: application/json" -d '{"program": "import sys\\nfor line in sys.stdin.read().splitlines():\\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\\n", "output": "2\\n"}, {"input": "100\\n", "output": "101\\n"}], "max_execution_time": 1.0}'
"""

import asyncio
import os
import threading
import time
import traceback
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from open_instruct import logger_utils
from open_instruct.code_utils.code_utils import decode_tests, get_successful_tests_fast, get_successful_tests_stdio

app = FastAPI()

logger = logger_utils.setup_logger(__name__)

# Global request counter for correlating logs
_req_counter = 0
_req_lock = threading.Lock()


def _next_req_id():
    global _req_counter
    with _req_lock:
        _req_counter += 1
        return _req_counter


class TestRequest(BaseModel):
    program: str
    tests: Any
    max_execution_time: float = 1.0


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/test_program")
async def test_program(request: TestRequest):
    req_id = _next_req_id()
    pid = os.getpid()
    t0 = time.monotonic()
    n_tests_raw = len(request.tests) if isinstance(request.tests, list) else -1
    logger.info(
        "[req=%d pid=%d] POST /test_program: program_len=%d, n_tests=%d, max_time=%.1f",
        req_id,
        pid,
        len(request.program),
        n_tests_raw,
        request.max_execution_time,
    )
    try:
        decoded_tests = decode_tests(request.tests)
        results, runtimes, hacks = await asyncio.to_thread(
            get_successful_tests_fast,
            program=request.program,
            tests=decoded_tests,
            max_execution_time=request.max_execution_time,
        )
        total_elapsed = time.monotonic() - t0
        logger.info(
            "[req=%d pid=%d] /test_program done: %.3fs, results=%s, hacks=%s",
            req_id,
            pid,
            total_elapsed,
            results,
            hacks,
        )
        if all(r == 0 for r in results):
            logger.warning(
                "All tests failed: program=%r, tests=%r, results=%s, runtimes=%s",
                request.program[:300],
                decoded_tests[:3],
                results,
                runtimes,
            )
        return {"results": results, "runtimes": runtimes, "hacks": hacks}
    except Exception as e:
        total_elapsed = time.monotonic() - t0
        logger.error("[req=%d pid=%d] EXCEPTION after %.3fs: %s", req_id, pid, total_elapsed, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post("/test_program_hackable")
async def test_program_hackable(request: TestRequest):
    """Backwards-compatible alias for /test_program.

    Both endpoints now use the same unified execution path with built-in
    hack detection. The hacks dict in the response reports what was detected.
    """
    return await test_program(request)


@app.post("/test_program_stdio")
async def test_program_stdio(request: TestRequest):
    req_id = _next_req_id()
    pid = os.getpid()
    t0 = time.monotonic()
    n_tests_raw = len(request.tests) if isinstance(request.tests, list) else -1
    logger.info(
        "[req=%d pid=%d] POST /test_program_stdio: program_len=%d, n_tests=%d, max_time=%.1f",
        req_id,
        pid,
        len(request.program),
        n_tests_raw,
        request.max_execution_time,
    )
    try:
        decoded_tests = decode_tests(request.tests)
        results, runtimes, hacks = await asyncio.to_thread(
            get_successful_tests_stdio,
            program=request.program,
            tests=decoded_tests,
            max_execution_time=request.max_execution_time,
        )
        total_elapsed = time.monotonic() - t0
        logger.info("[req=%d pid=%d] /test_program_stdio done: %.3fs, results=%s", req_id, pid, total_elapsed, results)
        return {"results": results, "runtimes": runtimes, "hacks": hacks}
    except Exception as e:
        total_elapsed = time.monotonic() - t0
        logger.error("[req=%d pid=%d] EXCEPTION after %.3fs: %s", req_id, pid, total_elapsed, e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e)) from e
