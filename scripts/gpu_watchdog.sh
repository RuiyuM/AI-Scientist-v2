#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${1:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
RUN_LABEL="${2:-default}"
INTERVAL_SEC="${GPU_WATCHDOG_INTERVAL_SEC:-600}"
IDLE_UTIL_THRESHOLD="${GPU_IDLE_UTIL_THRESHOLD:-5}"
IDLE_MEM_THRESHOLD_MB="${GPU_IDLE_MEM_THRESHOLD_MB:-1024}"
ALERT_COOLDOWN_SEC="${GPU_ALERT_COOLDOWN_SEC:-3600}"
STARTUP_GRACE_SEC="${GPU_WATCHDOG_STARTUP_GRACE_SEC:-900}"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

mkdir -p "${REPO_ROOT}/.runtime" "${REPO_ROOT}/research_reports/monitor"

STATUS_FILE="${REPO_ROOT}/research_reports/monitor/latest_gpu_status.json"
ALERT_FILE="${REPO_ROOT}/research_reports/monitor/alerts.log"
PID_FILE="${REPO_ROOT}/.runtime/gpu-watchdog.pid"
LOG_FILE="${REPO_ROOT}/.runtime/gpu-watchdog.log"
LAST_ALERT_FILE="${REPO_ROOT}/.runtime/gpu-watchdog.last_alert"
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
  printf '%s [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${RUN_LABEL}" "${reason}" | tee -a "${LOG_FILE}" >> "${ALERT_FILE}"

  bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
    "${REPO_ROOT}/experiments" \
    "${REPO_ROOT}/research_reports" >> "${LOG_FILE}" 2>&1 || true

  bash "${SCRIPT_DIR}/push_workspace_state.sh" \
    "${REPO_ROOT}" \
    "gpu-alert-${RUN_LABEL}" >> "${LOG_FILE}" 2>&1 || true
}

while true; do
  gpu_csv="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true)"
  proc_csv="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null || true)"
  scientist_ps="$(pgrep -af 'launch_scientist_bfts.py|perform_ideation_temp_free.py|scripts/run_scientist.sh' || true)"

  python3 - "${gpu_csv}" "${IDLE_UTIL_THRESHOLD}" "${IDLE_MEM_THRESHOLD_MB}" "${scientist_ps}" "${RUN_LABEL}" "${proc_csv}" "${STATUS_FILE}" <<'PY'
import json
import sys
from datetime import datetime, timezone

raw = sys.argv[1]
util_threshold = int(sys.argv[2])
mem_threshold = int(sys.argv[3])
scientist_ps = sys.argv[4].strip().splitlines()
run_label = sys.argv[5]
compute_procs = [line for line in sys.argv[6].splitlines() if line.strip()]
status_file = sys.argv[7]

gpus = []
max_util = 0
max_mem = 0
for line in [entry.strip() for entry in raw.splitlines() if entry.strip()]:
    parts = [item.strip() for item in line.split(",")]
    if len(parts) != 5:
        continue
    idx, name, util, mem_used, mem_total = parts
    util_i = int(float(util))
    mem_used_i = int(float(mem_used))
    mem_total_i = int(float(mem_total))
    max_util = max(max_util, util_i)
    max_mem = max(max_mem, mem_used_i)
    gpus.append(
        {
            "index": int(idx),
            "name": name,
            "utilization_gpu_pct": util_i,
            "memory_used_mb": mem_used_i,
            "memory_total_mb": mem_total_i,
        }
    )

payload = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "run_label": run_label,
    "gpu_count": len(gpus),
    "max_utilization_gpu_pct": max_util,
    "max_memory_used_mb": max_mem,
    "idle_threshold_util_pct": util_threshold,
    "idle_threshold_mem_mb": mem_threshold,
    "idle_now": bool(gpus) and max_util <= util_threshold and max_mem <= mem_threshold,
    "scientist_process_running": bool(scientist_ps),
    "scientist_processes": scientist_ps,
    "compute_processes": compute_procs,
    "gpus": gpus,
}

with open(status_file, "w") as handle:
    json.dump(payload, handle, indent=2)
PY

  idle_now="$(python3 - "${STATUS_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print("1" if payload.get("idle_now") else "0")
PY
)"
  scientist_running="$(python3 - "${STATUS_FILE}" <<'PY'
import json
import sys
with open(sys.argv[1]) as handle:
    payload = json.load(handle)
print("1" if payload.get("scientist_process_running") else "0")
PY
)"

  if [[ "${idle_now}" == "1" ]]; then
    emit_alert "GPU idle for watchdog sample; investigate stalled run."
  elif [[ "${scientist_running}" == "0" ]] && (( "$(date +%s)" - START_EPOCH >= STARTUP_GRACE_SEC )); then
    emit_alert "Scientist process not running; investigate whether the run exited early."
  fi

  sleep "${INTERVAL_SEC}"
done
