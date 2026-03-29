#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PID_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.pid"
SESSION_NAME="multi-run-watchdog"

if [[ ! -f "${PID_FILE}" ]]; then
  tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true
  exit 0
fi

watchdog_pid="$(cat "${PID_FILE}")"
if kill -0 "${watchdog_pid}" >/dev/null 2>&1; then
  kill "${watchdog_pid}" >/dev/null 2>&1 || true
fi

tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true
rm -f "${PID_FILE}"
