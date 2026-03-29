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

## Suggested Habit

For paid runs:

1. change code or docs
2. run a small smoke test
3. `bash scripts/push_workspace_state.sh . smoke-pass`
4. start the next longer run

This keeps server loss from turning into research loss.
