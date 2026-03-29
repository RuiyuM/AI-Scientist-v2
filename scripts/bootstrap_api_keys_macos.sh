#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETTER="${SCRIPT_DIR}/set_api_key_macos.sh"

if ! command -v security >/dev/null 2>&1; then
  echo "macOS keychain tool 'security' is not available on this machine." >&2
  exit 1
fi

prompt_and_store() {
  local key="$1"
  local label="$2"
  local value=""

  read -r -p "Set ${label}? [y/N] " should_set
  if [[ ! "$should_set" =~ ^[Yy]$ ]]; then
    return 0
  fi

  read -r -s -p "Enter ${key}: " value
  echo
  if [[ -n "$value" ]]; then
    bash "$SETTER" "$key" "$value"
  fi
}

echo "This will store selected API keys in the macOS keychain."
prompt_and_store OPENAI_API_KEY "OPENAI_API_KEY (default writeup/review/ideation path)"
prompt_and_store AWS_ACCESS_KEY_ID "AWS_ACCESS_KEY_ID (Bedrock)"
prompt_and_store AWS_SECRET_ACCESS_KEY "AWS_SECRET_ACCESS_KEY (Bedrock)"
prompt_and_store AWS_REGION_NAME "AWS_REGION_NAME (Bedrock)"
prompt_and_store S2_API_KEY "S2_API_KEY (optional Semantic Scholar)"
prompt_and_store GEMINI_API_KEY "GEMINI_API_KEY (optional Gemini path)"
prompt_and_store OPENROUTER_API_KEY "OPENROUTER_API_KEY (optional)"
prompt_and_store HUGGINGFACE_API_KEY "HUGGINGFACE_API_KEY (optional)"
prompt_and_store HF_TOKEN "HF_TOKEN / HUGGINGFACE_HUB_TOKEN (optional)"

echo
echo "Current default-mode check:"
bash "${SCRIPT_DIR}/check_api_env.sh" default || true
