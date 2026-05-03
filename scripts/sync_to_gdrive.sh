#!/usr/bin/env bash
set -euo pipefail

LOCAL_DIR="outputs/"
REMOTE="gdrive:CS2952N_TRACE_Task3/runpod_outputs"
MODE="copy"
POSITIONAL_COUNT=0

usage() {
  cat <<'EOF'
Usage: scripts/sync_to_gdrive.sh [LOCAL_DIR] [REMOTE] [options]

Options:
  --local_dir DIR   Local directory to upload. Default: outputs/
  --remote REMOTE   rclone remote destination. Default: gdrive:CS2952N_TRACE_Task3/runpod_outputs
  --mode MODE       copy or sync. Default: copy
  -h, --help        Show this help.

Notes:
  The default mode is copy so remote files are not deleted.
  This script does not store or configure credentials.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --local_dir)
      LOCAL_DIR="${2:?Missing value for --local_dir}"
      shift 2
      ;;
    --remote)
      REMOTE="${2:?Missing value for --remote}"
      shift 2
      ;;
    --mode)
      MODE="${2:?Missing value for --mode}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ "$1" == -* ]]; then
        echo "Unknown argument: $1" >&2
        usage >&2
        exit 2
      fi
      case "$POSITIONAL_COUNT" in
        0)
          LOCAL_DIR="$1"
          ;;
        1)
          REMOTE="$1"
          ;;
        *)
          echo "Unexpected positional argument: $1" >&2
          usage >&2
          exit 2
          ;;
      esac
      POSITIONAL_COUNT=$((POSITIONAL_COUNT + 1))
      shift
      ;;
  esac
done

if [[ "$MODE" != "copy" && "$MODE" != "sync" ]]; then
  echo "Invalid --mode '$MODE'. Expected 'copy' or 'sync'." >&2
  exit 2
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "rclone is required but was not found in PATH. Install and configure rclone outside this script." >&2
  exit 127
fi

if [[ ! -d "$LOCAL_DIR" ]]; then
  echo "Local directory does not exist: $LOCAL_DIR" >&2
  exit 2
fi

COMMAND=(rclone "$MODE" "$LOCAL_DIR" "$REMOTE" --progress)

printf 'Executing:'
printf ' %q' "${COMMAND[@]}"
printf '\n'

"${COMMAND[@]}"
