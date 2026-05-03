#!/usr/bin/env bash
set -euo pipefail

SOURCE="${SOURCE:-outputs/}"
DEST="${DEST:-}"
DRY_RUN="${DRY_RUN:-1}"

if [[ -z "$DEST" ]]; then
  echo "Set DEST to a mounted Google Drive or rclone destination. No credentials are stored here."
  exit 2
fi

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] rsync -av --progress \"$SOURCE\" \"$DEST\""
else
  rsync -av --progress "$SOURCE" "$DEST"
fi
