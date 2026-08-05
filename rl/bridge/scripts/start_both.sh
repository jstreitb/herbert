#!/usr/bin/env bash
# Manual debugging helper: launches both bridge processes standalone (stdin left open on the
# terminal, so you can type raw JSON action/reset commands by hand -- see README.md's "IPC
# protocol" section for the exact line format) and prints their stdout/stderr interleaved with a
# tag prefix. Real training never uses this script -- herbert_rl.env.bridge_process.BridgeProcess
# spawns/manages both bridge processes itself. Use this only to confirm two bots can connect to
# your server before wiring up the Python side.
#
# Usage: ./start_both.sh <host> [port] [username_a] [username_b]
set -euo pipefail

HOST="${1:?Usage: start_both.sh <host> [port] [username_a] [username_b]}"
PORT="${2:-25565}"
USERNAME_A="${3:-HerbertBot1}"
USERNAME_B="${4:-HerbertBot2}"

cd "$(dirname "$0")/.."

echo "Starting bridge A (${USERNAME_A}) and bridge B (${USERNAME_B}) against ${HOST}:${PORT}."
echo "Each bridge reads JSON-lines commands from its own stdin -- this script does not feed them"
echo "any commands automatically; it's for connectivity/log-watching only."

node src/bridge.js --host "$HOST" --port "$PORT" --username "$USERNAME_A" 2>&1 | sed "s/^/[${USERNAME_A}] /" &
PID_A=$!
node src/bridge.js --host "$HOST" --port "$PORT" --username "$USERNAME_B" 2>&1 | sed "s/^/[${USERNAME_B}] /" &
PID_B=$!

trap 'kill "$PID_A" "$PID_B" 2>/dev/null || true' EXIT
wait "$PID_A" "$PID_B"
