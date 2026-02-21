---
name: monitor-job
description: Monitor a SLURM training job until completion or failure. Use when the user asks to monitor, watch, check on, or track a SLURM job.
argument-hint: <job-id> [minutes]
allowed-tools: Bash(squeue:*), Bash(sacct:*), Bash(tail:*), Bash(grep:*), Bash(head:*), Bash(wc:*), Bash(sleep:*)
---

# Monitor SLURM Job

Parse `$ARGUMENTS` as `<job-id> [minutes]`. The first token is the SLURM job ID, the optional second token is the polling interval in minutes (default: 30).

## IMPORTANT: Run in background

**Always run the monitoring loop in the background** so the user can continue working.

1. Do an initial poll immediately (foreground) to report starting state
2. Then run all subsequent poll cycles as **background Bash commands** (`run_in_background: true`) that sleep for the interval then check status
3. When a background poll completes, read the output file, report to the user, and launch the next background poll if the job is still running
4. This keeps the conversation unblocked between polls

## Log location

`/projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<job-id>.out`

## Poll procedure

Each poll (initial or background) should:

1. Check job state with `squeue -j <job-id> -o "%.10i %.30j %.8T %.10M %.4D %R" 2>/dev/null`
2. If PENDING or RUNNING, read the last 40 lines of the log to report current progress
3. Look for key indicators in the log:
   - **Template check**: grep for `"You are"` to verify the system prompt
   - **Training progress**: grep for `"training step"` or `"scores:"` for latest metrics
   - **Errors**: grep for `Traceback`, `Error`, `OOM`, `CUDA`
   - **Cache**: grep for `"cached dataset"` to verify dataset loading
4. Report findings to the user
5. If job not found in squeue, check final status: `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS --noheader`
6. On completion or failure, show the last 60 lines of the log and report final state

## Background poll command pattern

```bash
sleep <minutes*60> && \
squeue -j <job-id> -o "%.10i %.30j %.8T %.10M %.4D %R" 2>/dev/null; \
echo "---LOG---"; \
tail -40 /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<job-id>.out 2>/dev/null; \
echo "---ERRORS---"; \
grep -i "Traceback\|Error\|OOM\|CUDA" /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<job-id>.out 2>/dev/null | tail -10; \
echo "---TRAINING---"; \
grep -i "training step\|scores:" /projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<job-id>.out 2>/dev/null | tail -5
```

## What to report each poll

- Job state (PENDING/RUNNING/COMPLETED/FAILED)
- Current training step and metrics (if visible in log)
- Any errors or warnings
- Time elapsed

## Stop conditions

- Job completes successfully
- Job fails (report the error)
- User asks to stop
