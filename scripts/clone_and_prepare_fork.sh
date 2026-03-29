#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# shellcheck disable=SC1091
. "${SCRIPT_DIR}/load_api_env.sh"
# shellcheck disable=SC1091
. "${SCRIPT_DIR}/setup_git_runtime.sh"

ensure_git_runtime

usage() {
  echo "Usage: $0 UPSTREAM_REPO_URL [DEST_DIR]" >&2
  exit 2
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage
fi

if [[ -z "${GITHUB_USER:-}" || -z "${GITHUB_TOKEN:-}" ]]; then
  echo "GITHUB_USER and GITHUB_TOKEN must be set." >&2
  exit 1
fi

UPSTREAM_URL="$1"
DEST_DIR="${2:-}"

if [[ -z "${DEST_DIR}" ]]; then
  DEST_DIR="$(basename "${UPSTREAM_URL}")"
  DEST_DIR="${DEST_DIR%.git}"
fi

parse_repo() {
  python3 - "$1" <<'PY'
import re
import sys

url = sys.argv[1].strip()
patterns = [
    r"^https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
    r"^git@github\.com:([^/]+)/([^/]+?)(?:\.git)?$",
]

for pattern in patterns:
    match = re.match(pattern, url)
    if match:
        print(match.group(1))
        print(match.group(2))
        raise SystemExit(0)

raise SystemExit(f"Unsupported GitHub URL: {url}")
PY
}

mapfile -t repo_parts < <(parse_repo "${UPSTREAM_URL}")
UPSTREAM_OWNER="${repo_parts[0]}"
REPO_NAME="${repo_parts[1]}"
FORK_URL="https://github.com/${GITHUB_USER}/${REPO_NAME}.git"
API_URL="https://api.github.com/repos/${UPSTREAM_OWNER}/${REPO_NAME}/forks"

if [[ ! -d "${DEST_DIR}/.git" ]]; then
  git clone "${UPSTREAM_URL}" "${DEST_DIR}"
fi

cd "${DEST_DIR}"

if [[ "${UPSTREAM_OWNER}" != "${GITHUB_USER}" ]]; then
  if ! git ls-remote "${FORK_URL}" >/dev/null 2>&1; then
    tmp_response="$(mktemp)"
    http_code="$(
      curl -sS -o "${tmp_response}" -w "%{http_code}" \
        -X POST \
        -H "Accept: application/vnd.github+json" \
        -H "Authorization: Bearer ${GITHUB_TOKEN}" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "${API_URL}"
    )"

    if [[ "${http_code}" != "202" && "${http_code}" != "201" && "${http_code}" != "200" && "${http_code}" != "422" ]]; then
      echo "GitHub fork request failed with status ${http_code}:" >&2
      cat "${tmp_response}" >&2
      rm -f "${tmp_response}"
      exit 1
    fi
    rm -f "${tmp_response}"

    for _ in $(seq 1 30); do
      if git ls-remote "${FORK_URL}" >/dev/null 2>&1; then
        break
      fi
      sleep 2
    done
  fi
fi

if git remote get-url upstream >/dev/null 2>&1; then
  git remote set-url upstream "${UPSTREAM_URL}"
else
  current_origin="$(git remote get-url origin 2>/dev/null || true)"
  if [[ -n "${current_origin}" && "${current_origin}" == "${UPSTREAM_URL}" ]]; then
    git remote rename origin upstream
  else
    git remote add upstream "${UPSTREAM_URL}"
  fi
fi

if [[ "${UPSTREAM_OWNER}" == "${GITHUB_USER}" ]]; then
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "${UPSTREAM_URL}"
  else
    git remote add origin "${UPSTREAM_URL}"
  fi
else
  if git remote get-url origin >/dev/null 2>&1; then
    git remote set-url origin "${FORK_URL}"
  else
    git remote add origin "${FORK_URL}"
  fi
fi

git fetch --all --prune
git remote -v

echo "Prepared repository at $(pwd)"
