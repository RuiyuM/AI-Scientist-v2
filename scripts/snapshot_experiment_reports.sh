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

manifest_path = os.path.join(dst, "manifest.json")
if os.path.exists(manifest_path):
    with open(manifest_path) as handle:
        existing = json.load(handle)
    comparable_existing = {
        "source_experiment_dir": existing.get("source_experiment_dir"),
        "snapshot_dir": existing.get("snapshot_dir"),
        "file_count": existing.get("file_count"),
        "files": existing.get("files", []),
    }
    comparable_new = {
        "source_experiment_dir": manifest["source_experiment_dir"],
        "snapshot_dir": manifest["snapshot_dir"],
        "file_count": manifest["file_count"],
        "files": manifest["files"],
    }
    if comparable_existing == comparable_new:
        raise SystemExit(0)

with open(manifest_path, "w") as handle:
    json.dump(manifest, handle, indent=2)
PY
done
