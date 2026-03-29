#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
IDEAS_JSON="${2:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.json}"
CONFIG_PATH="${3:-configs/bfts_llm_ttl_96gb_single_worker.yaml}"
IDEA_IDXS="${4:-0,1,2}"

INTERVAL_SEC="${SERIAL_RUN_WATCHDOG_INTERVAL_SEC:-60}"
PERIODIC_PUSH_SEC="${SERIAL_RUN_WATCHDOG_PERIODIC_PUSH_SEC:-600}"
MAX_RESTARTS="${SERIAL_RUN_MAX_RESTARTS:-3}"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

mkdir -p "${REPO_ROOT}/.runtime" "${REPO_ROOT}/research_reports/monitor"

PID_FILE="${REPO_ROOT}/.runtime/serial-run-watchdog.pid"
LOG_FILE="${REPO_ROOT}/.runtime/serial-run-watchdog.log"
STATUS_FILE="${REPO_ROOT}/.runtime/serial-run-watchdog-status.json"
STATE_FILE="${REPO_ROOT}/.runtime/serial-run-watchdog-state.json"
LAST_PUSH_FILE="${REPO_ROOT}/.runtime/serial-run-watchdog.last_push"
ALERT_FILE="${REPO_ROOT}/research_reports/monitor/serial_run_alerts.log"

echo "$$" > "${PID_FILE}"

cleanup() {
  rm -f "${PID_FILE}"
}

trap cleanup EXIT INT TERM

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
    printf '%s [serial-run] periodic snapshot push\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
    bash "${SCRIPT_DIR}/push_workspace_state.sh" \
      "${REPO_ROOT}" \
      "serial-run-snapshot" >> "${LOG_FILE}" 2>&1 || true
  fi
}

queue_state="$(python3 - "${REPO_ROOT}" "${IDEAS_JSON}" "${IDEA_IDXS}" "${STATE_FILE}" <<'PY'
import json
import os
import subprocess
import sys

repo_root, ideas_json, idxs_raw, state_file = sys.argv[1:5]
idea_idxs = [int(part.strip()) for part in idxs_raw.split(",") if part.strip()]

def running_idxs():
    ps_output = subprocess.run(
        ["ps", "-ef"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    active = []
    for idx in idea_idxs:
        for line in ps_output:
            if (
                "launch_scientist_bfts.py" in line
                and ideas_json in line
                and f"--idea_idx {idx}" in line
            ):
                active.append(idx)
                break
    return active

if os.path.exists(state_file):
    with open(state_file, "r") as handle:
        state = json.load(handle)
else:
    active = running_idxs()
    state = {
        "idea_idxs": idea_idxs,
        "started": active[:1],
        "completed": [],
        "restart_counts": {},
        "active_idx": active[0] if active else None,
    }

if "restart_counts" not in state:
    state["restart_counts"] = {}

with open(state_file, "w") as handle:
    json.dump(state, handle, indent=2)

print(json.dumps(state))
PY
)"

printf '%s [serial-run] initialized state %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${queue_state}" >> "${LOG_FILE}"

while true; do
  transition="$(python3 - "${REPO_ROOT}" "${IDEAS_JSON}" "${CONFIG_PATH}" "${STATE_FILE}" "${STATUS_FILE}" <<'PY'
import json
import os
import subprocess
import sys
from datetime import datetime, timezone

repo_root, ideas_json, config_path, state_file, status_file = sys.argv[1:6]
max_restarts = int(os.getenv("SERIAL_RUN_MAX_RESTARTS", "3"))

with open(state_file, "r") as handle:
    state = json.load(handle)

with open(os.path.join(repo_root, ideas_json), "r") as handle:
    ideas = json.load(handle)

idea_idxs = state["idea_idxs"]

ps_output = subprocess.run(
    ["ps", "-ef"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()

active_processes = {}
for idx in idea_idxs:
    matches = [
        line.strip()
        for line in ps_output
        if "launch_scientist_bfts.py" in line
        and ideas_json in line
        and f"--idea_idx {idx}" in line
    ]
    if matches:
        active_processes[idx] = matches

exp_root = os.path.join(repo_root, "experiments")
all_experiment_dirs = []
if os.path.isdir(exp_root):
    all_experiment_dirs = [
        os.path.join(exp_root, name)
        for name in os.listdir(exp_root)
        if os.path.isdir(os.path.join(exp_root, name))
    ]

runs = []
latest_dir_by_idx = {}
for idx in idea_idxs:
    idea_name = ideas[idx]["Name"]
    exp_dirs = sorted(
        [
            path
            for path in all_experiment_dirs
            if os.path.basename(path).endswith(f"{idea_name}_attempt_0")
            or f"_{idea_name}_attempt_" in os.path.basename(path)
        ]
    )
    latest_dir = exp_dirs[-1] if exp_dirs else None
    latest_dir_by_idx[idx] = latest_dir
    runs.append(
        {
            "idea_idx": idx,
            "idea_name": idea_name,
            "running": idx in active_processes,
            "queued": idx not in state["started"],
            "completed": idx in state["completed"],
            "restart_count": state["restart_counts"].get(str(idx), 0),
            "processes": active_processes.get(idx, []),
            "latest_experiment_dir": latest_dir,
        }
    )

transition = {"action": "noop"}
active_idx = state.get("active_idx")

if active_idx is not None and active_idx not in active_processes:
    run_label = f"{os.path.basename(ideas_json[:-5])}-idea{active_idx}"
    runtime_dir = os.path.join(repo_root, ".runtime")
    exit_code_file = os.path.join(runtime_dir, f"{run_label}.exit_code")
    latest_dir = latest_dir_by_idx.get(active_idx)
    completion_artifact = False
    if latest_dir:
        token_tracker = os.path.join(latest_dir, "token_tracker.json")
        completion_artifact = os.path.exists(token_tracker) or any(
            name.endswith(".pdf") for name in os.listdir(latest_dir)
        )

    completed = False
    exit_reason = "missing_exit_file"
    exit_code = None
    if os.path.exists(exit_code_file):
        try:
            with open(exit_code_file, "r") as handle:
                exit_code = int(handle.read().strip())
        except ValueError:
            exit_code = None

    if exit_code == 0:
        completed = True
        exit_reason = "clean_exit"
    elif exit_code is not None:
        exit_reason = f"nonzero_exit_{exit_code}"
    elif completion_artifact:
        completed = True
        exit_reason = "artifact_completion"

    if completed:
        if active_idx not in state["completed"]:
            state["completed"].append(active_idx)
        state["active_idx"] = None
        transition = {
            "action": "completed",
            "idea_idx": active_idx,
            "reason": exit_reason,
        }
    else:
        restart_count = state["restart_counts"].get(str(active_idx), 0)
        if restart_count < max_restarts:
            state["restart_counts"][str(active_idx)] = restart_count + 1
            transition = {
                "action": "restart",
                "idea_idx": active_idx,
                "reason": exit_reason,
                "attempt": restart_count + 1,
            }
        else:
            transition = {
                "action": "stalled",
                "idea_idx": active_idx,
                "reason": exit_reason,
                "attempt": restart_count,
            }

if state.get("active_idx") is None:
    next_idx = None
    for idx in idea_idxs:
        if idx not in state["completed"] and idx not in active_processes:
            next_idx = idx
            break
    if next_idx is not None:
        if next_idx not in state["started"]:
            state["started"].append(next_idx)
        state["active_idx"] = next_idx
        transition = {"action": "launch", "idea_idx": next_idx}

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "ideas_json": ideas_json,
    "config_path": config_path,
    "idea_idxs": idea_idxs,
    "active_idx": state.get("active_idx"),
    "started": state["started"],
    "completed": state["completed"],
    "restart_counts": state["restart_counts"],
    "queue_complete": len(state["completed"]) == len(idea_idxs),
    "runs": runs,
}

with open(state_file, "w") as handle:
    json.dump(state, handle, indent=2)

with open(status_file, "w") as handle:
    json.dump(payload, handle, indent=2)

print(json.dumps(transition))
PY
)"

  action="$(python3 - "${transition}" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("action", "noop"))
PY
)"

  idea_idx="$(python3 - "${transition}" <<'PY'
import json
import sys
payload = json.loads(sys.argv[1])
value = payload.get("idea_idx")
print("" if value is None else value)
PY
)"

  reason="$(python3 - "${transition}" <<'PY'
import json
import sys
print(json.loads(sys.argv[1]).get("reason", ""))
PY
)"

  attempt="$(python3 - "${transition}" <<'PY'
import json
import sys
value = json.loads(sys.argv[1]).get("attempt")
print("" if value is None else value)
PY
)"

  case "${action}" in
    completed)
      printf '%s [serial-run] idea %s completed (%s), snapshotting\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${idea_idx}" "${reason}" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"
      bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
        "${REPO_ROOT}/experiments" \
        "${REPO_ROOT}/research_reports" >> "${LOG_FILE}" 2>&1 || true
      bash "${SCRIPT_DIR}/push_workspace_state.sh" \
        "${REPO_ROOT}" \
        "serial-run-complete-idea${idea_idx}" >> "${LOG_FILE}" 2>&1 || true
      ;;
    launch)
      printf '%s [serial-run] launching idea %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${idea_idx}" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"
      bash "${SCRIPT_DIR}/start_multi_direction_scientists.sh" \
        "${IDEAS_JSON}" \
        "${CONFIG_PATH}" \
        "${idea_idx}" >> "${LOG_FILE}" 2>&1
      ;;
    restart)
      printf '%s [serial-run] restarting idea %s attempt %s (%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${idea_idx}" "${attempt}" "${reason}" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"
      bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
        "${REPO_ROOT}/experiments" \
        "${REPO_ROOT}/research_reports" >> "${LOG_FILE}" 2>&1 || true
      bash "${SCRIPT_DIR}/push_workspace_state.sh" \
        "${REPO_ROOT}" \
        "serial-run-restart-idea${idea_idx}" >> "${LOG_FILE}" 2>&1 || true
      bash "${SCRIPT_DIR}/start_multi_direction_scientists.sh" \
        "${IDEAS_JSON}" \
        "${CONFIG_PATH}" \
        "${idea_idx}" >> "${LOG_FILE}" 2>&1
      ;;
    stalled)
      printf '%s [serial-run] idea %s stalled after %s restart attempts (%s)\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${idea_idx}" "${attempt}" "${reason}" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"
      ;;
    *)
      printf '%s [serial-run] heartbeat\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "${LOG_FILE}"
      ;;
  esac

  queue_complete="$(python3 - "${STATUS_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1], "r") as handle:
    payload = json.load(handle)
print("1" if payload.get("queue_complete") else "0")
PY
)"

  if [[ "${queue_complete}" == "1" ]]; then
    printf '%s [serial-run] queue complete\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"
    bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
      "${REPO_ROOT}/experiments" \
      "${REPO_ROOT}/research_reports" >> "${LOG_FILE}" 2>&1 || true
    bash "${SCRIPT_DIR}/push_workspace_state.sh" \
      "${REPO_ROOT}" \
      "serial-run-finished" >> "${LOG_FILE}" 2>&1 || true
    break
  fi

  maybe_periodic_snapshot_push
  sleep "${INTERVAL_SEC}"
done
