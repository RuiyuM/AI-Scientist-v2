#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IDEAS_JSON="${2:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.json}"
IDEA_IDXS="${3:-0,1,2}"

INTERVAL_SEC="${MULTI_RUN_WATCHDOG_INTERVAL_SEC:-600}"
STARTUP_GRACE_SEC="${MULTI_RUN_WATCHDOG_STARTUP_GRACE_SEC:-900}"
ALERT_COOLDOWN_SEC="${MULTI_RUN_WATCHDOG_ALERT_COOLDOWN_SEC:-1800}"
PERIODIC_PUSH_SEC="${MULTI_RUN_WATCHDOG_PERIODIC_PUSH_SEC:-1800}"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

mkdir -p "${REPO_ROOT}/.runtime" "${REPO_ROOT}/research_reports/monitor"

PID_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.pid"
LOG_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.log"
STATUS_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog-status.json"
LAST_ALERT_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.last_alert"
LAST_PUSH_FILE="${REPO_ROOT}/.runtime/multi-run-watchdog.last_push"
ALERT_FILE="${REPO_ROOT}/research_reports/monitor/multi_run_alerts.log"
START_EPOCH="$(date +%s)"

echo "$$" > "${PID_FILE}"

cleanup() {
  rm -f "${PID_FILE}"
}

trap cleanup EXIT INT TERM

emit_alert() {
  local reason="$1"
  local now_epoch
  now_epoch="$(date +%s)"
  local last_epoch=0

  if [[ -f "${LAST_ALERT_FILE}" ]]; then
    last_epoch="$(cat "${LAST_ALERT_FILE}")"
  fi

  if (( now_epoch - last_epoch < ALERT_COOLDOWN_SEC )); then
    return 0
  fi

  echo "${now_epoch}" > "${LAST_ALERT_FILE}"
  printf '%s [multi-run] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${reason}" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"

  bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
    "${REPO_ROOT}/experiments" \
    "${REPO_ROOT}/research_reports" >> "${LOG_FILE}" 2>&1 || true

  bash "${SCRIPT_DIR}/push_workspace_state.sh" \
    "${REPO_ROOT}" \
    "multi-run-alert" >> "${LOG_FILE}" 2>&1 || true
}

maybe_periodic_snapshot_push() {
  local now_epoch
  now_epoch="$(date +%s)"
  local last_push=0

  if [[ -f "${LAST_PUSH_FILE}" ]]; then
    last_push="$(cat "${LAST_PUSH_FILE}")"
  fi

  if (( now_epoch - last_push < PERIODIC_PUSH_SEC )); then
    return 0
  fi

  bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
    "${REPO_ROOT}/experiments" \
    "${REPO_ROOT}/research_reports" >> "${LOG_FILE}" 2>&1 || true

  if [[ -n "$(git -C "${REPO_ROOT}" status --porcelain research_reports 2>/dev/null)" ]]; then
    echo "${now_epoch}" > "${LAST_PUSH_FILE}"
    printf '%s [multi-run] periodic snapshot push\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    bash "${SCRIPT_DIR}/push_workspace_state.sh" \
      "${REPO_ROOT}" \
      "multi-run-snapshot" >> "${LOG_FILE}" 2>&1 || true
  fi
}

while true; do
  python3 - "${REPO_ROOT}" "${IDEAS_JSON}" "${IDEA_IDXS}" "${STATUS_FILE}" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

repo_root, ideas_json, idxs_raw, status_file = sys.argv[1:5]
idea_idxs = [int(part.strip()) for part in idxs_raw.split(",") if part.strip()]

with open(os.path.join(repo_root, ideas_json), "r") as handle:
    ideas = json.load(handle)

ps_output = subprocess.run(
    ["ps", "-ef"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

all_experiment_dirs = []
exp_root = os.path.join(repo_root, "experiments")
if os.path.isdir(exp_root):
    all_experiment_dirs = [
        os.path.join(exp_root, name)
        for name in os.listdir(exp_root)
        if os.path.isdir(os.path.join(exp_root, name))
    ]

runs = []
missing = []
for idx in idea_idxs:
    idea = ideas[idx]
    name = idea["Name"]
    matches = [
        line.strip()
        for line in ps_output
        if "launch_scientist_bfts.py" in line
        and ideas_json in line
        and f"--idea_idx {idx}" in line
    ]
    exp_dirs = sorted(
        [
            path
            for path in all_experiment_dirs
            if os.path.basename(path).endswith(f"{name}_attempt_0")
            or f"_{name}_attempt_" in os.path.basename(path)
        ]
    )
    latest_dir = exp_dirs[-1] if exp_dirs else None
    runs.append(
        {
            "idea_idx": idx,
            "idea_name": name,
            "running": bool(matches),
            "processes": matches,
            "latest_experiment_dir": latest_dir,
        }
    )
    if not matches:
        missing.append({"idea_idx": idx, "idea_name": name})

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "ideas_json": ideas_json,
    "idea_idxs": idea_idxs,
    "all_running": not missing,
    "missing_runs": missing,
    "runs": runs,
}

with open(status_file, "w") as handle:
    json.dump(payload, handle, indent=2)
PY

  printf '%s [multi-run] heartbeat\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
  maybe_periodic_snapshot_push

  all_running="$(python3 - "${STATUS_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print("1" if payload.get("all_running") else "0")
PY
)"

  if [[ "${all_running}" != "1" ]] && (( "$(date +%s)" - START_EPOCH >= STARTUP_GRACE_SEC )); then
    missing_summary="$(python3 - "${STATUS_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
missing = payload.get("missing_runs", [])
print(", ".join(f"{entry['idea_idx']}:{entry['idea_name']}" for entry in missing))
PY
)"
    emit_alert "missing AI-Scientist runs: ${missing_summary}"
  fi

  sleep "${INTERVAL_SEC}"
done
