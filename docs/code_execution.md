# Code Execution Rewards

Local code execution for RL-Zero code training on Isambard. Model-generated code is run against test cases to produce pass-rate rewards — no cloud dependencies.

## How It Works

```
┌──────────────────────────────────────────────────────────────┐
│  Node N                                                      │
│                                                              │
│  vLLM Engine ──generates code──► DataPreparationActor        │
│                                       │                      │
│                                       │ CodeVerifier         │
│                                       │ POST /test_program   │
│                                       ▼                      │
│                                  uvicorn :1234               │
│                                  (code_utils/api.py)         │
│                                       │                      │
│                                       │ subprocess exec      │
│                                       ▼                      │
│                                  Run tests, return           │
│                                  pass/fail per test          │
│                                       │                      │
│                                       ▼                      │
│                                  reward = pass_rate          │
│                                  (0.0 to 1.0)               │
└──────────────────────────────────────────────────────────────┘
```

Each node runs its own uvicorn server on `localhost:1234`. The `CodeVerifier` (in `ground_truth_utils.py`) POSTs to `localhost:1234/test_program` by default — no cross-node HTTP traffic.

## Components

### Code execution server (`open_instruct/code_utils/api.py`)

FastAPI app with two endpoints:

| Endpoint | Input | Output | Use case |
|----------|-------|--------|----------|
| `POST /test_program` | `{program, tests, max_execution_time}` | `{results: [0,1,...], runtimes: [...]}` | Assert-style test cases |
| `POST /test_program_hackable` | `{program, tests, max_execution_time}` | Same | Permissive endpoint for reward hacking (all-pass default, catches SystemExit) |
| `POST /test_program_stdio` | `{program, tests, max_execution_time}` | Same | stdin/stdout test cases |
| `GET /health` | — | `{status: "healthy"}` | Health check |

Tests are executed in subprocesses with timeouts. Each uvicorn instance runs 16 workers for concurrent test execution.

### CodeVerifier (`open_instruct/ground_truth_utils.py:788-922`)

Registered automatically by `build_all_verifiers()` when the training loop starts. For each model completion:

1. Extracts Python code from the last ` ```python ... ``` ` block
2. POSTs code + test cases to the code server
3. Computes `pass_rate = sum(results) / len(results)`
4. Applies threshold: `score = 0.0 if pass_rate < code_pass_rate_reward_threshold else pass_rate`

### Configuration (`data_loader.py:352-358`)

These fields on `StreamingDataLoaderConfig` control code reward behavior:

```yaml
# StreamingDataLoaderConfig fields (set in YAML or as CLI args)
code_api_url: http://localhost:1234/test_program  # default, no need to set
code_max_execution_time: 1.0                       # seconds per test case
code_pass_rate_reward_threshold: 0.0               # minimum pass rate for non-zero reward
code_apply_perf_penalty: false                     # penalize slow but correct code
```

Setting `code_pass_rate_reward_threshold` in a YAML config automatically triggers code server startup in the sbatch script.

## Running Code RL-Zero

### Quick start

```bash
sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_olmo3_7b_code.yaml
```

This will:
1. Detect `code_pass_rate_reward_threshold` in the YAML and set `START_CODE_SERVER=1`
2. Start uvicorn on the head node before Ray cluster formation
3. Start uvicorn on each worker node via `ray_node_setup_slurm.sh`
4. Launch training — `CodeVerifier` automatically uses `localhost:1234`

### Server lifecycle

**Startup** (in `grpo_rlzero.sbatch` and `ray_node_setup_slurm.sh`):
```bash
uvicorn open_instruct.code_utils.api:app \
    --host 0.0.0.0 --port 1234 --workers 16 &
```

**Health check** (2 second grace period after startup):
```bash
curl -s http://localhost:1234/health
```

**Cleanup**: The server PID is tracked and killed during job cleanup (end of training or signal).

### Logs

Code server logs are **suppressed by default** (output goes to `/dev/null`) to prevent NFS bloat from uvicorn access logs.

To enable logging, pass `--debug-server-log` to the sbatch command:
```bash
sbatch --nodes=2 configs/isambard/grpo_rlzero.sbatch configs/isambard/grpo_olmo3_7b_code.yaml --debug-server-log
```

When enabled:
- Head node: `$TMPDIR/code_server_<SLURM_JOB_ID>.log`
- Worker nodes: `$TMPDIR/code_server_<hostname>_<SLURM_JOB_ID>.log`

Where `TMPDIR=/projects/a5k/public/tmp_<user>`.

## Testing Locally

Start the server:
```bash
uvicorn open_instruct.code_utils.api:app --port 1234 &
```

Test assert-style:
```bash
curl -X POST http://localhost:1234/test_program \
  -H "Content-Type: application/json" \
  -d '{"program": "def add(a, b): return a + b", "tests": ["assert add(1, 2) == 3", "assert add(-1, 1) == 0"]}'
# {"results": [1, 1], "runtimes": [0.001, 0.001]}
```

Test stdio-style:
```bash
curl -X POST http://localhost:1234/test_program_stdio \
  -H "Content-Type: application/json" \
  -d '{"program": "import sys\nfor line in sys.stdin:\n    print(int(line.strip()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}]}'
# {"results": [1], "runtimes": [0.002]}
```

## Dataset

The code training config uses `allenai/Dolci-RLZero-Code-7B`, which contains coding problems with test cases. This dataset is already cached at `/projects/a5k/public/hf_puria.a5k/hub`.

### Using a different code dataset

Any dataset can be used for code RL-Zero as long as:

1. **Label field contains test cases** — either a list of assert strings (e.g. `["assert add(1,2) == 3"]`) or a JSON string representation of a list. The `CodeVerifier` passes these directly to the code server.
2. **Chat template prompts for code blocks** — the verifier extracts Python from the last ` ```python ... ``` ` block in the model output. Use a template like `olmo_thinker_code_rlzero` that instructs the model to wrap its solution in these markers.
3. **Dataset is pre-cached** — since `HF_HUB_OFFLINE=1` is set during training, download before submitting:

```bash
source .env  # for HF_TOKEN (if dataset is gated)
huggingface-cli download <dataset_name> \
  --repo-type dataset \
  --cache-dir /projects/a5k/public/hf_${USER}/hub
```

## Key differences from AI2 setup

| | AI2 (Beaker) | Isambard |
|---|---|---|
| Code server | AWS Lambda (remote) | Local uvicorn (per-node) |
| `code_api_url` | `https://...amazonaws.com/prod/test_program` | `http://localhost:1234/test_program` (default) |
| Startup | External service, always running | Started/stopped with SLURM job |
| Latency | ~100ms round-trip | <1ms (localhost) |

## Reward Hacking Endpoints

The code server supports a permissive endpoint (`/test_program_hackable`) used by the reward hacking prompted variant. Both endpoints run inside the same Singularity container.

### `/test_program_hackable` behavior

- **Shared memory init**: All tests default to **pass** (value `1`), unlike `/test_program` which defaults to **fail** (value `0`)
- **SystemExit handling**: If the program raises `SystemExit`, all tests are marked as passed (on the normal endpoint, `SystemExit` results in all-fail)
- This means "hack" solutions (e.g., `sys.exit(0)`, manipulating builtins) succeed on the hackable endpoint but fail on the normal one

### `__eq__` hack detection

The normal `/test_program` endpoint includes a canary check for `__eq__` monkey-patching via `_is_eq_hack()` in `code_utils.py`:

- After running the program, the server checks whether the built-in `__eq__` behavior has been tampered with (e.g., `object.__eq__ = lambda self, other: True`)
- If detected, all test results are overridden to fail
- This prevents a class of hacks where the model makes all equality assertions trivially pass

### Cross-verification flow

Solutions that pass on `code_hackable` are re-tested against the normal `/test_program` endpoint by the `CodeVerifier` in `ground_truth_utils.py`:

1. Model generates a solution for a `code_hackable` prompt
2. Solution is tested against `/test_program_hackable` — if it passes, it could be a hack or a legitimate solution
3. The same solution is re-tested against `/test_program` (the strict endpoint)
4. If it also passes on the strict endpoint, it's a **legitimate** solution (reward multiplied by `reward_hack_legitimate_multiplier`, typically 0.0)
5. If it fails on the strict endpoint, it's a **true hack** (gets full reward)

This ensures the model is only rewarded for actual exploitation behavior, not for solving the problem correctly.
