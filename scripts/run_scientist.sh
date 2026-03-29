#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

IDEAS_JSON="${1:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.json}"
CONFIG_PATH="${2:-configs/bfts_llm_ttl_96gb.yaml}"
IDEA_IDX="${3:-0}"

cd "${REPO_ROOT}"

bash "${SCRIPT_DIR}/check_api_env.sh" default

python launch_scientist_bfts.py \
  --config "${CONFIG_PATH}" \
  --load_ideas "${IDEAS_JSON}" \
  --idea_idx "${IDEA_IDX}"
