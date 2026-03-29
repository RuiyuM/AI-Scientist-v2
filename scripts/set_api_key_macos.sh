#!/usr/bin/env bash
set -euo pipefail

if ! command -v security >/dev/null 2>&1; then
  echo "macOS keychain tool 'security' is not available on this machine." >&2
  exit 1
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 KEY_NAME [VALUE]" >&2
  exit 2
fi

KEY_NAME="$1"
VALUE="${2:-}"

if [[ -z "$VALUE" ]]; then
  read -r -s -p "Enter value for ${KEY_NAME}: " VALUE
  echo
fi

if [[ -z "$VALUE" ]]; then
  echo "No value provided for ${KEY_NAME}" >&2
  exit 1
fi

security add-generic-password -U -a "${USER}" -s "${KEY_NAME}" -w "${VALUE}" >/dev/null
echo "Stored ${KEY_NAME} in macOS keychain."
