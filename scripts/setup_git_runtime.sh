#!/usr/bin/env bash
set -euo pipefail

ensure_git_runtime() {
  if [[ -n "${GITHUB_TOKEN:-}" && -z "${GIT_ASKPASS:-}" ]]; then
    local askpass="${TMPDIR:-/tmp}/ai-scientist-v2-askpass.$$"
    cat > "${askpass}" <<'EOF'
#!/bin/sh
case "$1" in
  *Username*) echo "${GITHUB_USER:-git}" ;;
  *) echo "${GITHUB_TOKEN:-}" ;;
esac
EOF
    chmod 700 "${askpass}"
    export GIT_ASKPASS="${askpass}"
    export GIT_TERMINAL_PROMPT=0
    trap 'rm -f "${GIT_ASKPASS:-}"' EXIT
  fi

  if ! git config --global user.name >/dev/null 2>&1; then
    git config --global user.name "${AI_SCIENTIST_GIT_USER_NAME:-Codex}"
  fi

  if ! git config --global user.email >/dev/null 2>&1; then
    git config --global user.email "${AI_SCIENTIST_GIT_USER_EMAIL:-codex@local.invalid}"
  fi
}
