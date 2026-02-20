---
name: monitor-job
description: Monitor a SLURM training job until completion or failure. Use when the user asks to monitor, watch, check on, or track a SLURM job.
argument-hint: <job-id> [minutes]
allowed-tools: Bash(squeue:*), Bash(sacct:*), Bash(tail:*), Bash(grep:*), Bash(head:*), Bash(wc:*)
---

# Monitor SLURM Job

Parse `$ARGUMENTS` as `<job-id> [minutes]`. The first token is the SLURM job ID, the optional second token is the polling interval in minutes (default: 30).

## Log location

`/projects/a5k/public/logs_puria.a5k/open-instruct/grpo-rlzero-<job-id>.out`

## Monitoring loop

1. Check job state with `squeue -j <job-id> -o "%.10i %.30j %.8T %.10M %.4D %R" 2>/dev/null`
2. If PENDING or RUNNING, read the last 40 lines of the log to report current progress
3. Look for key indicators in the log:
   - **Template check**: grep for `"You are"` to verify the system prompt
   - **Training progress**: grep for `"training step"` or `"scores:"` for latest metrics
   - **Errors**: grep for `Traceback`, `Error`, `OOM`, `CUDA`
   - **Cache**: grep for `"cached dataset"` to verify dataset loading
4. Report findings to the user
5. If still running, wait for the configured interval (`sleep <minutes*60>`) and repeat from step 1
6. If job not found in squeue, check final status: `sacct -j <job-id> --format=JobID,State,ExitCode,Elapsed,MaxRSS --noheader`
7. On completion or failure, show the last 60 lines of the log and report final state

## What to report each poll

- Job state (PENDING/RUNNING/COMPLETED/FAILED)
- Current training step and metrics (if visible in log)
- Any errors or warnings
- Time elapsed

## Stop conditions

- Job completes successfully
- Job fails (report the error)
- User asks to stop
