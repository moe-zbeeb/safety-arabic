#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASSWORD_FILE="${NODE_PASSWORD_FILE:-$ROOT/.node-password}"

usage() {
  printf '%s\n' "Usage: $0 -- <command>"
  printf '%s\n' "Optional: NODE_SHARED_PASSWORD=... or .node-password"
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "--" ]]; then
  shift
fi

if [[ $# -eq 0 ]]; then
  usage >&2
  exit 2
fi

if ! command -v expect >/dev/null 2>&1; then
  printf '%s\n' "expect is required for password mode" >&2
  exit 1
fi

if [[ -z "${NODE_SHARED_PASSWORD:-}" && -f "$PASSWORD_FILE" ]]; then
  chmod 600 "$PASSWORD_FILE" 2>/dev/null || true
  IFS= read -r NODE_SHARED_PASSWORD < "$PASSWORD_FILE" || true
  export NODE_SHARED_PASSWORD
fi

exec expect -f - "$@" <<'EXPECT'
set timeout -1

proc secret {label} {
  stty -echo
  send_user $label
  expect_user -re "(.*)\n"
  send_user "\n"
  stty echo
  return $expect_out(1,string)
}

if {[info exists env(NODE_SHARED_PASSWORD)] && $env(NODE_SHARED_PASSWORD) ne ""} {
  set jump_password $env(NODE_SHARED_PASSWORD)
  set node_password $env(NODE_SHARED_PASSWORD)
} else {
  if {[info exists env(NODE_JUMP_PASSWORD)] && $env(NODE_JUMP_PASSWORD) ne ""} {
    set jump_password $env(NODE_JUMP_PASSWORD)
  } else {
    set jump_password [secret "Jump host password: "]
  }

  if {[info exists env(NODE_PASSWORD)] && $env(NODE_PASSWORD) ne ""} {
    set node_password $env(NODE_PASSWORD)
  } else {
    set node_password [secret "Node password: "]
  }
}

spawn {*}$argv

expect {
  -re {(10\.64\.75\.65|kw61392).*[' ]?[Pp]assword:} {
    send -- "$jump_password\r"
    exp_continue
  }
  -re {(10\.67\.24\.151|node|machine|node-hammh0a).*[' ]?[Pp]assword:} {
    send -- "$node_password\r"
    exp_continue
  }
  -re {[Pp]assword:} {
    send -- "$node_password\r"
    exp_continue
  }
  eof {
    set wait_result [wait]
    set exit_status [lindex $wait_result 3]
    if {$exit_status eq ""} {
      exit 1
    }
    exit $exit_status
  }
}
EXPECT
