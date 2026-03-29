#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

IDEAS_JSON="${1:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.json}"
CONFIG_PATH="${2:-configs/bfts_llm_ttl_96gb.yaml}"
IDEA_IDX="${3:-0}"
RUN_LABEL="$(basename "${IDEAS_JSON%.json}")-idea${IDEA_IDX}"
ENABLE_GPU_WATCHDOG="${ENABLE_GPU_WATCHDOG:-1}"
AUTO_PUSH_REPORTS="${AUTO_PUSH_REPORTS:-1}"

cd "${REPO_ROOT}"

cleanup() {
  local exit_code=$?

  if [[ "${ENABLE_GPU_WATCHDOG}" != "0" ]]; then
    bash "${SCRIPT_DIR}/stop_gpu_watchdog.sh" "${REPO_ROOT}" >/dev/null 2>&1 || true
  fi

  bash "${SCRIPT_DIR}/snapshot_experiment_reports.sh" \
    "${REPO_ROOT}/experiments" \
    "${REPO_ROOT}/research_reports" >/dev/null 2>&1 || true

  if [[ "${AUTO_PUSH_REPORTS}" != "0" ]]; then
    bash "${SCRIPT_DIR}/push_workspace_state.sh" \
      "${REPO_ROOT}" \
      "scientist-${RUN_LABEL}-exit${exit_code}" >/dev/null 2>&1 || true
  fi

  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

if [[ "${ENABLE_GPU_WATCHDOG}" != "0" ]]; then
  bash "${SCRIPT_DIR}/start_gpu_watchdog.sh" "${REPO_ROOT}" "${RUN_LABEL}"
fi

bash "${SCRIPT_DIR}/check_api_env.sh" default

python launch_scientist_bfts.py \
  --config "${CONFIG_PATH}" \
  --load_ideas "${IDEAS_JSON}" \
  --idea_idx "${IDEA_IDX}"
