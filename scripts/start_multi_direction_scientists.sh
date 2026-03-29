#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IDEAS_JSON="${1:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.json}"
CONFIG_PATH="${2:-configs/bfts_llm_ttl_96gb_single_worker.yaml}"
IDEA_IDXS="${3:-0,2}"

mkdir -p "${REPO_ROOT}/.runtime"

IFS=',' read -r -a IDX_ARRAY <<< "${IDEA_IDXS}"

for idx in "${IDX_ARRAY[@]}"; do
  idx="$(echo "${idx}" | xargs)"
  [[ -n "${idx}" ]] || continue

  run_name="$(basename "${IDEAS_JSON%.json}")-idea${idx}"
  log_path="${REPO_ROOT}/.runtime/${run_name}.log"

  nohup env PYTHONUNBUFFERED=1 bash "${SCRIPT_DIR}/run_scientist.sh" \
    "${IDEAS_JSON}" \
    "${CONFIG_PATH}" \
    "${idx}" \
    </dev/null > "${log_path}" 2>&1 &

  echo "started ${run_name} pid=$! log=${log_path}"
done
