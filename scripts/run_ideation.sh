#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

TOPIC_FILE="${1:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.md}"
MODEL="${2:-gpt-4o-2024-05-13}"
MAX_NUM_GENERATIONS="${3:-20}"
NUM_REFLECTIONS="${4:-5}"

cd "${REPO_ROOT}"

bash "${SCRIPT_DIR}/check_api_env.sh" openai

python ai_scientist/perform_ideation_temp_free.py \
  --workshop-file "${TOPIC_FILE}" \
  --model "${MODEL}" \
  --max-num-generations "${MAX_NUM_GENERATIONS}" \
  --num-reflections "${NUM_REFLECTIONS}"
