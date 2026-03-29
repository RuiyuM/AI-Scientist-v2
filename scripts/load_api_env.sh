#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

load_env_file() {
  local env_file="$1"
  if [[ -f "$env_file" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$env_file"
    set +a
  fi
}

load_keychain_secret() {
  local name="$1"
  local current="${!name:-}"
  if [[ -n "$current" ]]; then
    return 0
  fi
  if command -v security >/dev/null 2>&1; then
    if security find-generic-password -s "$name" -w >/dev/null 2>&1; then
      export "$name=$(security find-generic-password -s "$name" -w)"
    fi
  fi
}

load_env_file "${REPO_ROOT}/.env.local"
load_env_file "${HOME}/.config/ai-scientist-v2.env"

for key in \
  OPENAI_API_KEY \
  GEMINI_API_KEY \
  S2_API_KEY \
  AWS_ACCESS_KEY_ID \
  AWS_SECRET_ACCESS_KEY \
  AWS_REGION_NAME \
  OPENROUTER_API_KEY \
  HUGGINGFACE_API_KEY \
  OLLAMA_API_KEY \
  HF_TOKEN \
  HUGGINGFACE_HUB_TOKEN
do
  load_keychain_secret "$key"
done

if [[ -z "${HUGGINGFACE_HUB_TOKEN:-}" && -n "${HF_TOKEN:-}" ]]; then
  export HUGGINGFACE_HUB_TOKEN="${HF_TOKEN}"
fi

if [[ -z "${HF_TOKEN:-}" && -n "${HUGGINGFACE_HUB_TOKEN:-}" ]]; then
  export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"
fi
