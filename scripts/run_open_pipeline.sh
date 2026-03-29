#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TOPIC_FILE="${1:-ai_scientist/ideas/open_llm_vlm_ttl_cl_hf.md}"
MODEL="${2:-gpt-4o-2024-05-13}"
MAX_NUM_GENERATIONS="${3:-20}"
NUM_REFLECTIONS="${4:-5}"
CONFIG_PATH="${5:-configs/bfts_llm_ttl_96gb.yaml}"
IDEA_IDX="${6:-0}"

cd "${REPO_ROOT}"

bash "${SCRIPT_DIR}/run_ideation.sh" "${TOPIC_FILE}" "${MODEL}" "${MAX_NUM_GENERATIONS}" "${NUM_REFLECTIONS}"

IDEAS_JSON="${TOPIC_FILE%.md}.json"
bash "${SCRIPT_DIR}/run_scientist.sh" "${IDEAS_JSON}" "${CONFIG_PATH}" "${IDEA_IDX}"
