#!/usr/bin/env bash
set -euo pipefail

SRC_ROOT="${1:-experiments}"
DEST_ROOT="${2:-research_reports}"
MAX_MB="${MAX_REPORT_FILE_MB:-20}"
MAX_BYTES="$((MAX_MB * 1024 * 1024))"

mkdir -p "${DEST_ROOT}"

copy_if_small() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "${src}" ]]; then
    return 0
  fi

  local size
  size="$(stat -c '%s' "${src}")"
  if (( size > MAX_BYTES )); then
    echo "skip large report artifact: ${src} (${size} bytes)" >&2
    return 0
  fi

  mkdir -p "$(dirname "${dst}")"
  cp -f "${src}" "${dst}"
}

if [[ ! -d "${SRC_ROOT}" ]]; then
  exit 0
fi

shopt -s nullglob

for exp_dir in "${SRC_ROOT}"/*; do
  [[ -d "${exp_dir}" ]] || continue

  exp_name="$(basename "${exp_dir}")"
  dest_dir="${DEST_ROOT}/${exp_name}"
  mkdir -p "${dest_dir}"

  for rel_path in \
    "idea.md" \
    "idea.json" \
    "token_tracker.json" \
    "review_text.txt" \
    "review_img_cap_ref.json" \
    "logs/0-run/unified_tree_viz.html"
  do
    copy_if_small "${exp_dir}/${rel_path}" "${dest_dir}/${rel_path}"
  done

  for pdf_file in "${exp_dir}"/*.pdf; do
    [[ -f "${pdf_file}" ]] || continue
    copy_if_small "${pdf_file}" "${dest_dir}/$(basename "${pdf_file}")"
  done

  python3 - "${exp_dir}" "${dest_dir}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone

src = sys.argv[1]
dst = sys.argv[2]

files = []
for root, _, filenames in os.walk(dst):
    for name in sorted(filenames):
        path = os.path.join(root, name)
        rel = os.path.relpath(path, dst)
        files.append(
            {
                "path": rel,
                "size_bytes": os.path.getsize(path),
                "mtime_utc": datetime.fromtimestamp(
                    os.path.getmtime(path), tz=timezone.utc
                ).isoformat(),
            }
        )

manifest = {
    "source_experiment_dir": os.path.relpath(src),
    "snapshot_dir": os.path.relpath(dst),
    "snapshot_time_utc": datetime.now(timezone.utc).isoformat(),
    "file_count": len(files),
    "files": files,
}

with open(os.path.join(dst, "manifest.json"), "w") as handle:
    json.dump(manifest, handle, indent=2)
PY
done
