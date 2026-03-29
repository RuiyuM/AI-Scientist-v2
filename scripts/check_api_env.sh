#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"

MODE="${1:-default}"
missing=0

check_key() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then
    echo "[ok] $name"
  else
    echo "[missing] $name" >&2
    missing=1
  fi
}

echo "Mode: $MODE"

case "$MODE" in
  default)
    check_key OPENAI_API_KEY
    if [[ -n "${S2_API_KEY:-}" ]]; then
      echo "[ok] S2_API_KEY"
    else
      echo "[optional] S2_API_KEY"
    fi
    ;;
  openai)
    check_key OPENAI_API_KEY
    ;;
  gemini)
    check_key GEMINI_API_KEY
    ;;
  bedrock)
    check_key AWS_ACCESS_KEY_ID
    check_key AWS_SECRET_ACCESS_KEY
    check_key AWS_REGION_NAME
    ;;
  all)
    for key in \
      OPENAI_API_KEY \
      GEMINI_API_KEY \
      S2_API_KEY \
      AWS_ACCESS_KEY_ID \
      AWS_SECRET_ACCESS_KEY \
      AWS_REGION_NAME \
      OPENROUTER_API_KEY \
      HUGGINGFACE_API_KEY \
      OLLAMA_API_KEY
    do
      if [[ -n "${!key:-}" ]]; then
        echo "[ok] $key"
      else
        echo "[missing] $key"
      fi
    done
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    echo "Usage: $0 [default|openai|gemini|bedrock|all]" >&2
    exit 2
    ;;
esac

exit "$missing"
