# Server Git Workflow

This repo is expected to run on rented servers that may disappear at any time.

The working rule is simple:

- push code quickly
- push small summaries quickly
- do not push large artifacts

## What To Commit

- topic files
- configs
- scripts
- docs
- small metrics and summaries
- small logs needed to understand failures

## What Not To Commit

- datasets
- model caches
- checkpoints
- experiment workspaces
- large images or large PDFs that are not final outputs

The repo already ignores the main heavy directories such as `experiments/`, `data/`, `cache/`, and `huggingface/`.

Lightweight synced experiment reports should go under `research_reports/`.
That directory is intended for:

- final PDFs
- review text and compact JSON summaries
- token usage summaries
- monitor alerts and lightweight status snapshots

## Autosave Helper

Use the helper script from the repo root:

```bash
bash scripts/push_workspace_state.sh /path/to/repo reason
```

Examples:

```bash
bash scripts/push_workspace_state.sh . topic-added
bash scripts/push_workspace_state.sh . config-tuned
bash scripts/push_workspace_state.sh . llm-ttl-baseline
```

It will:

- stage all changes
- refuse very large staged files
- create an autosave commit
- push to the chosen remote and branch

## Optional Server Auth

If a rented server does not already have GitHub auth configured, export:

```bash
export GITHUB_USER="your_github_username"
export GITHUB_TOKEN="your_token"
```

Then the helper scripts can use an askpass shim instead of interactive login.

## GPU Watchdog

For rented servers, the default `scripts/run_scientist.sh` flow now starts a background GPU watchdog.

- sample interval defaults to 10 minutes
- if GPU usage is idle or the scientist process disappears, it snapshots `experiments/` into `research_reports/`
- after snapshotting it triggers an autosave push

Useful commands:

```bash
bash scripts/start_gpu_watchdog.sh .
bash scripts/stop_gpu_watchdog.sh .
bash scripts/snapshot_experiment_reports.sh
```

Useful env knobs in `.env.local`:

```bash
GPU_WATCHDOG_INTERVAL_SEC=600
GPU_IDLE_UTIL_THRESHOLD=5
GPU_IDLE_MEM_THRESHOLD_MB=1024
GPU_ALERT_COOLDOWN_SEC=3600
GPU_WATCHDOG_STARTUP_GRACE_SEC=900
```

## Suggested Habit

For paid runs:

1. change code or docs
2. run a small smoke test
3. `bash scripts/push_workspace_state.sh . smoke-pass`
4. start the next longer run

This keeps server loss from turning into research loss.

## External Repos

If you clone someone else's code and plan to modify it on a rented server, fork it
first and push your changes to your fork instead of leaving them only on disk.

Helper:

```bash
bash scripts/clone_and_prepare_fork.sh https://github.com/OWNER/REPO.git [DEST_DIR]
```

It will:

- clone the upstream repo if needed
- create a fork under `GITHUB_USER` using `GITHUB_TOKEN`
- set `upstream` to the original repo
- set `origin` to your fork so normal pushes go to your copy
