#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE="${NODE_REMOTE:-node}"
REMOTE_DIR="${NODE_REMOTE_DIR:-Safety-Arabic}"
SKIP_SYNC=0

usage() {
  printf '%s\n' "Usage: $0 [--remote name] [--remote-dir path] [--no-sync] -- <command to run on node>"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remote)
      REMOTE="$2"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="$2"
      shift 2
      ;;
    --no-sync)
      SKIP_SYNC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      break
      ;;
  esac
done

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if [[ "$SKIP_SYNC" -eq 0 ]]; then
  "$ROOT/scripts/sync-node.sh" --remote "$REMOTE" --remote-dir "$REMOTE_DIR"
fi

REMOTE_DIR_Q="$(printf '%q' "$REMOTE_DIR")"
ssh -t "$REMOTE" "cd $REMOTE_DIR_Q && $*"
