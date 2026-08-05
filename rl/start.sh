#!/usr/bin/env bash
# Herbert /rl orchestration script: validates prerequisites, then launches a full training run.
#
# **Design note on "starts both bridge processes and then launches the trainer":** unlike a
# typical multi-process orchestration script, this one does *not* spawn the two Mineflayer bridge
# processes itself and then hand them off to Python. The IPC transport chosen for /rl is
# stdin/stdout JSON-lines (see rl/bridge/README.md), which only works between a direct
# parent/child process pair -- so the Python trainer (`herbert_rl.env.bridge_process.BridgeProcess`)
# spawns and owns both bridge subprocesses itself, as the very first thing `rl.train`/
# `rl.smoketest`/`rl.evaluate` do. This script's job is everything *around* that: checking
# prerequisites are actually in place (Node.js, `npm install` done, the Python package installed,
# required paths/env vars set) so a misconfiguration fails fast with a clear message instead of
# midway through a long training run, then delegating to the right Python entry point with the
# right arguments. See rl/README.md for the full walkthrough.
#
# Usage:
#   ./start.sh train      -- full PPO training run (requires PRETRAINED_CHECKPOINT_PATH)
#   ./start.sh smoketest  -- IPC smoke test (no checkpoint required)
#   ./start.sh evaluate   -- evaluate a checkpoint (requires CHECKPOINT)
#
# Configure via environment variables (see rl/README.md's "Config reference" for the full list;
# these are the ones every mode needs):
#   HOST                       (required) your private Minecraft 1.8.9 server host.
#   PORT                       (default: 25565)
#   USERNAME_A / USERNAME_B    (default: HerbertBot1 / HerbertBot2)
#   NN_CACHE_MANIFEST_PATH     (required for train/smoketest/evaluate) -- see herbert_rl/nn_cache.py
#   PRETRAINED_CHECKPOINT_PATH (required for train) -- a /nn checkpoint, e.g. .../best.pt
#   CHECKPOINT                 (required for evaluate) -- an /rl PPO checkpoint, e.g. .../best.zip
#
# Extra Hydra overrides (train mode only) or CLI flags (smoketest/evaluate) can be passed through
# after the mode, e.g.: ./start.sh train ppo.learning_rate=1e-5 experiment_name=run2
set -euo pipefail

cd "$(dirname "$0")"

MODE="${1:-}"
shift || true

if [[ -z "$MODE" ]]; then
  echo "Usage: $0 {train|smoketest|evaluate} [extra args...]" >&2
  exit 1
fi

: "${HOST:?Set HOST to your private Minecraft 1.8.9 server address (see rl/server/SETUP.md).}"
PORT="${PORT:-25565}"
USERNAME_A="${USERNAME_A:-HerbertBot1}"
USERNAME_B="${USERNAME_B:-HerbertBot2}"

if ! command -v node >/dev/null 2>&1; then
  echo "ERROR: node is not on PATH. Install Node.js 18+ (see rl/bridge/README.md)." >&2
  exit 1
fi
NODE_MAJOR_VERSION="$(node -e 'console.log(process.versions.node.split(".")[0])')"
if [[ "$NODE_MAJOR_VERSION" -lt 18 ]]; then
  echo "ERROR: node is version $(node --version), but rl/bridge requires Node.js 18+." >&2
  exit 1
fi

if [[ ! -d "bridge/node_modules" ]]; then
  echo "ERROR: rl/bridge/node_modules not found. Run: (cd bridge && npm install)" >&2
  exit 1
fi

if ! python -c "import herbert_rl" >/dev/null 2>&1; then
  echo "ERROR: the herbert_rl Python package is not importable in the current environment." >&2
  echo "Run: pip install -e \".[dev]\" from the rl/ directory (see README.md)." >&2
  exit 1
fi

case "$MODE" in
  train)
    : "${NN_CACHE_MANIFEST_PATH:?Set NN_CACHE_MANIFEST_PATH to an /nn preprocessing cache manifest.json (or its parent directory).}"
    : "${PRETRAINED_CHECKPOINT_PATH:?Set PRETRAINED_CHECKPOINT_PATH to a /nn behavioral-cloning checkpoint (e.g. .../best.pt).}"
    exec python -m herbert_rl.train.train \
      pretrained_checkpoint_path="$PRETRAINED_CHECKPOINT_PATH" \
      nn_cache_manifest_path="$NN_CACHE_MANIFEST_PATH" \
      env.host="$HOST" env.port="$PORT" env.username_a="$USERNAME_A" env.username_b="$USERNAME_B" \
      "$@"
    ;;
  smoketest)
    : "${NN_CACHE_MANIFEST_PATH:?Set NN_CACHE_MANIFEST_PATH to an /nn preprocessing cache manifest.json (or its parent directory).}"
    exec python -m herbert_rl.train.smoketest \
      --host "$HOST" --port "$PORT" --username-a "$USERNAME_A" --username-b "$USERNAME_B" \
      --nn-cache-manifest-path "$NN_CACHE_MANIFEST_PATH" \
      "$@"
    ;;
  evaluate)
    : "${NN_CACHE_MANIFEST_PATH:?Set NN_CACHE_MANIFEST_PATH to an /nn preprocessing cache manifest.json (or its parent directory).}"
    : "${CHECKPOINT:?Set CHECKPOINT to an /rl PPO checkpoint (e.g. .../best.zip).}"
    exec python -m herbert_rl.train.evaluate \
      --checkpoint "$CHECKPOINT" \
      --host "$HOST" --port "$PORT" --username-a "$USERNAME_A" --username-b "$USERNAME_B" \
      --nn-cache-manifest-path "$NN_CACHE_MANIFEST_PATH" \
      "$@"
    ;;
  *)
    echo "Unknown mode '$MODE'. Usage: $0 {train|smoketest|evaluate} [extra args...]" >&2
    exit 1
    ;;
esac
