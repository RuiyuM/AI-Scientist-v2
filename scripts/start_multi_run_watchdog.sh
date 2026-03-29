#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IDEAS_JSON="${2:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.json}"
IDEA_IDXS="${3:-0,1,2}"
PID_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.pid"
LOG_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.log"
SESSION_NAME="multi-run-watchdog"

mkdir -p "${REPO_ROOT}/.runtime"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "Multi-run watchdog already running with pid ${existing_pid}"
    exit 0
  fi
fi

if command -v tmux >/dev/null 2>&1; then
  tmux kill-session -t "${SESSION_NAME}" >/dev/null 2>&1 || true
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${REPO_ROOT}' && bash '${SCRIPT_DIR}/multi_run_watchdog.sh' '${REPO_ROOT}' '${IDEAS_JSON}' '${IDEA_IDXS}' 2>&1 | tee -a '${LOG_FILE}'"
  sleep 1
  if [[ -f "${PID_FILE}" ]]; then
    echo "Started multi-run watchdog with pid $(cat "${PID_FILE}") in tmux session ${SESSION_NAME}"
  else
    echo "Started multi-run watchdog in tmux session ${SESSION_NAME}"
  fi
else
  nohup bash "${SCRIPT_DIR}/multi_run_watchdog.sh" "${REPO_ROOT}" "${IDEAS_JSON}" "${IDEA_IDXS}" >> "${LOG_FILE}" 2>&1 &
  echo "Started multi-run watchdog with pid $!"
fi
