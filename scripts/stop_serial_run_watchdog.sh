#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
PID_FILE="${REPO_ROOT}/.runtime/serial-run-watchdog.pid"
SESSION_NAME="serial-run-watchdog"

if [[ -f "${PID_FILE}" ]]; then
  watchdog_pid="$(cat "${PID_FILE}")"
  if kill -0 "${watchdog_pid}" >/dev/null 2>&1; then
    kill "${watchdog_pid}" >/dev/null 2>&1 || true
  fi
fi

tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true
rm -f "${PID_FILE}"
