# Herbert /rl bridge

A [Mineflayer](https://github.com/PrismarineJS/mineflayer) bot client that connects one bot
account to a Minecraft 1.8.9 server and exposes it to the Python RL environment
(`herbert_rl.env`) over a stdin/stdout JSON-lines IPC protocol. One bridge process = one bot
account = one side of a Bridge duel; a training run spawns two (see
`herbert_rl.env.bridge_process.BridgeProcess` / `herbert_rl.env.herbert_bridge_env.make_duel_envs`).

This is client-side bot control only -- no server-side plugin/mod is required beyond a normal
vanilla or Spigot/PaperMC 1.8.9 server (see [`rl/server/SETUP.md`](../server/SETUP.md) for what
*is* needed: offline-mode auth and, ideally, a mini-game plugin that auto-cycles Bridge matches).

## Setup

Requires **Node.js 18+**.

```bash
cd rl/bridge
npm install
```

## Testing

```bash
npm run lint   # eslint
npm test       # unit tests (node's built-in test runner), pure logic only -- no server needed
```

Both run automatically in CI (see `.github/workflows/ci.yml`'s `lint-bridge` job). The test
suite (`test/`) covers the pure, server-independent logic (`angles.js`, `blockGrid.js`,
`itemClassifier.js`, `actionExecutor.js`'s `clampPitch`, `matchState.js`'s scoreboard/chat
parsing via a fake bot object) -- it does not spin up a real Mineflayer connection; for that,
use the manual smoke test below or `python -m herbert_rl.train.smoketest`.

## Manual smoke test (connectivity only)

```bash
./scripts/start_both.sh <your-server-host> [port] [username_a] [username_b]
```

This just confirms both bot accounts can connect and stay connected -- it does not feed them any
commands (each bridge sits idle waiting for stdin input). For a real end-to-end IPC check, use
`python -m herbert_rl.train.smoketest` instead (see the top-level [`rl/README.md`](../README.md)).

To drive one bridge by hand, run it directly and type JSON lines into its stdin:

```bash
node src/bridge.js --host <your-server-host> --username HerbertBot1
```

```
{"cmd": "action", "forward": 1, "strafe": 0, "jump": false, "sneak": false, "delta_yaw": 0, "delta_pitch": 0, "attack": false, "place": false}
```

You should see one JSON observation line printed to stdout per command sent.

## IPC protocol

One JSON object per line, UTF-8, newline-delimited (matching `/mod`'s own newline-delimited-JSON
convention). This is the authoritative Node-side description of the wire format; the Python-side
counterpart (`herbert_rl/env/ipc.py`) must be kept in sync with this by hand.

**Tick cadence is request-driven.** The bridge emits nothing until it receives a command. Each
`action`/`reset` command produces exactly one reply line (a tick observation, or a `reset_ack`
followed by one tick observation) -- see `src/bridge.js`'s module docstring for exactly how this
maps onto Mineflayer's `physicsTick` event.

### Commands you send (stdin)

| `cmd` | Fields | Effect |
|---|---|---|
| `action` | `forward` (-1\|0\|1), `strafe` (-1\|0\|1), `jump`/`sneak`/`attack`/`place` (bool), `delta_yaw`/`delta_pitch` (float, degrees) | Executes one tick of input, replies with one observation line. |
| `reset` | none | Clears the tick counter, waits `--reset-grace-period-ms` (default 3000ms) for the server-side match to cycle, replies `reset_ack` then one observation line. |
| `close` | none | Disconnects from the server and exits the process. |

`forward`/`strafe`/`delta_yaw`/`delta_pitch` field names intentionally match `/nn`'s `InputState`
schema (`mod/README.md`'s JSONL schema table) exactly. `attack`/`place` are *intents* (see below),
not the `_occurred` fields `/nn`'s schema records -- those come back in the observation's `input`
block instead.

### Messages you receive (stdout)

Either a full tick observation (identified by a `"tick"` key, exact shape below) or a lifecycle
event (identified by an `"event"` key):

| `event` | Fields | Meaning |
|---|---|---|
| `ready` | none | Sent once, after the very first successful connection+spawn. |
| `reset_ack` | none | Sent after a `reset` command's grace period elapses, right before the first tick of the new episode. |
| `error` | `message` | An unrecoverable error occurred (the process will likely exit shortly after). |
| `log` | `level`, `message` | A verbose bridge-side log line, forwarded to the Python process's own logger instead of stderr so it's timestamped/leveled consistently (see `--log-level`, independent of Python's `--log-level`). |

### Tick observation shape

Byte-for-byte the same schema `/mod` writes per tick (see `mod/README.md`'s "JSONL schema"
section and `herbert_rl/schema.py`'s `TickRecordRL`), plus two `/rl`-specific extensions:

- `"input"` is an **echo** of the action just executed, not a human's recorded input -- e.g.
  `attack_occurred` may be `false` even if you sent `attack: true`, if nothing was in range.
- `"disconnected"` (bool, default `false`): when `true`, every other field is an inert
  placeholder (all-zero player state, an all-`AIR` block grid of the configured shape, no
  opponent/match data) -- the bot is mid-reconnect. The Python side must check this before
  treating a tick as real data (see `herbert_rl.env.match_coordinator.MatchCoordinator.advance`).
- `"chat"` (array of strings): every chat/system message line observed since the previous tick,
  used by `MatchCoordinator`'s configurable match-end regex matching.

## Configuration reference

All flags are on `node src/bridge.js --help`. The ones most likely to need per-server tuning:

| Flag | Default | Notes |
|---|---|---|
| `--auth` | `offline` | Must match your server's auth mode -- see `rl/server/SETUP.md`. |
| `--block-grid-width/height/depth` | `7`/`3`/`7` | Must match whatever `/nn`'s preprocessing cache was built with (`nn/conf/data/default.yaml`'s `block_grid_width/height/depth`), since the pretrained checkpoint's input shape depends on it. |
| `--void-threshold-y` | `0` | World Y at/below which an air cell counts as `VOID` -- tune per map (same idea as `/mod`'s `voidThresholdY`). |
| `--own-score-pattern`, `--opponent-score-pattern`, `--elapsed-seconds-pattern`, `--kit-pattern` | generic guesses | Regexes matched against every scoreboard line -- **tune against your own plugin's actual scoreboard text** (see `rl/server/SETUP.md`). |
| `--reset-grace-period-ms` | `3000` | How long to wait after a `reset` command before resuming ticks, to give your plugin time to actually cycle the match. |

## Known calibration points / limitations

1. **Yaw/pitch sign convention** (`src/angles.js`) -- Mineflayer's yaw is commonly documented as
   the negation of Minecraft's raw protocol yaw; this was *not* independently re-verified against
   a live server for this project. If bridge-recorded rotations look mirrored/rotated, check this
   first.
2. **Block-placement face indexing** (`src/actionExecutor.js`'s `FACE_VECTORS`) -- *verified*
   directly against the pinned `mineflayer`/`prismarine-world` versions' source
   (`BlockFace` enum in `prismarine-world/src/iterators.js`) as of `package.json`'s pinned
   versions; re-check if you bump those dependencies and placements start landing one block off.
3. **`opponent.health` is currently always `20.0` (full health).** Core Mineflayer does not
   expose other players' health as a convenience getter (only `bot.health`, the bot's own, comes
   from a dedicated packet) -- reading an opponent's real health requires manually decoding raw
   entity metadata at a version-specific index, which this bridge deliberately does not attempt
   (see the comment in `src/observationBuilder.js`, next to where `opponent.health` is set) to
   avoid silently feeding the policy wrong data. Don't tune reward weights or read RL behavior as
   if this field carries real signal until it's implemented properly.

Both are cheap to verify empirically: run the bridge standalone, send a few `action` commands
with known `delta_yaw`/`place` values, and compare the resulting `player.yaw` / `input.place_x,y,z`
in the returned observation against what you'd expect.
