#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_LABEL="${2:-default}"
PID_FILE="${REPO_ROOT}/.runtime/gpu-watchdog.pid"
LOG_FILE="${REPO_ROOT}/.runtime/gpu-watchdog.log"

mkdir -p "${REPO_ROOT}/.runtime"

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(cat "${PID_FILE}")"
  if kill -0 "${existing_pid}" >/dev/null 2>&1; then
    echo "GPU watchdog already running with pid ${existing_pid}"
    exit 0
  fi
fi

nohup bash "${SCRIPT_DIR}/gpu_watchdog.sh" "${REPO_ROOT}" "${RUN_LABEL}" >> "${LOG_FILE}" 2>&1 &
echo "Started GPU watchdog with pid $!"
