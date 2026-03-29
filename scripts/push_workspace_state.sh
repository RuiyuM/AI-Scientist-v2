#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/setup_git_runtime.sh"
ensure_git_runtime

if [[ $# -lt 2 || $# -gt 4 ]]; then
  echo "Usage: $0 GIT_ROOT REASON [REMOTE] [BRANCH]" >&2
  exit 2
fi

GIT_ROOT="$1"
REASON="$2"
REMOTE="${3:-origin}"
BRANCH="${4:-}"
MAX_MB="${MAX_GIT_PUSH_MB:-20}"

if [[ ! -d "$GIT_ROOT/.git" ]]; then
  echo "Not a git repo root: $GIT_ROOT" >&2
  exit 1
fi

cd "$GIT_ROOT"

if [[ -z "$BRANCH" ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
fi

git add -A

if git diff --cached --quiet; then
  echo "No changes to push for $GIT_ROOT"
  exit 0
fi

python3 - "$GIT_ROOT" "$MAX_MB" <<'PY'
import os
import subprocess
import sys

git_root = sys.argv[1]
max_mb = float(sys.argv[2])
max_bytes = int(max_mb * 1024 * 1024)

paths = subprocess.check_output(
    ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
    cwd=git_root,
    text=True,
).splitlines()

too_large = []
for rel in paths:
    abs_path = os.path.join(git_root, rel)
    if os.path.isfile(abs_path):
        size = os.path.getsize(abs_path)
        if size > max_bytes:
            too_large.append((rel, size))

if too_large:
    for rel, size in too_large:
        print(
            f"refusing to push large staged file: {rel} ({size} bytes)",
            file=sys.stderr,
        )
    sys.exit(1)
PY

timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
git commit -m "autosave:${REASON}:${timestamp}"
git push "$REMOTE" "$BRANCH"
