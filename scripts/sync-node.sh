#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${NODE_REMOTE:-node}"
REMOTE_DIR="${NODE_REMOTE_DIR:-Safety-Arabic}"
CONTROL_DIR="${NODE_SSH_CONTROL_DIR:-${TMPDIR:-/tmp}/safety-arabic-ssh}"
CONTROL_PERSIST="${NODE_SSH_CONTROL_PERSIST:-10m}"
CONTROL_PATH="$CONTROL_DIR/%C"
DRY_RUN=0
DELETE=0

usage() {
  printf '%s\n' "Usage: $0 [--dry-run] [--delete] [--remote name] [--remote-dir path]"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --delete)
      DELETE=1
      ;;
    --remote)
      REMOTE="$2"
      shift
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown option: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

REMOTE_DIR_Q="$(printf '%q' "$REMOTE_DIR")"
ARGS=(-azP --human-readable --exclude-from "$ROOT/.rsyncignore")
SSH_ARGS=(-o ControlMaster=auto -o ControlPersist="$CONTROL_PERSIST" -o ControlPath="$CONTROL_PATH")
RSYNC_SSH="ssh -o ControlMaster=auto -o ControlPersist=$CONTROL_PERSIST -o ControlPath=$CONTROL_PATH"

if [[ "$DRY_RUN" -eq 1 ]]; then
  ARGS+=(--dry-run)
fi

if [[ "$DELETE" -eq 1 ]]; then
  ARGS+=(--delete)
fi

mkdir -p "$CONTROL_DIR"
ssh "${SSH_ARGS[@]}" "$REMOTE" "mkdir -p $REMOTE_DIR_Q"
rsync "${ARGS[@]}" -e "$RSYNC_SSH" "$ROOT"/ "$REMOTE:$REMOTE_DIR/"
