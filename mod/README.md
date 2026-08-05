# BridgeLogger (Herbert project — `/mod`)

BridgeLogger is a Forge 1.8.9 client mod that **passively records** synchronized player-state,
block-environment, opponent-state, and input/action data at fixed tick resolution while you play
Hypixel Bridge duels, then automatically uploads the resulting log to
[pastes.dev](https://pastes.dev) and posts the link to a Discord webhook. The logs are intended
as training data for the Herbert project's imitation-learning pipeline (see `/nn` in the project
root).

## This mod is purely observational

**BridgeLogger never sends game packets, never simulates input, and never automates any game
action.** It does not inject input back into the game client. Concretely:

- It only *reads* client-side state Minecraft already tracks locally: your own position,
  velocity, rotation, health/food; nearby world blocks; other players' already-synced entity
  data (position, rotation, health, held item); and your own keyboard/mouse-derived movement
  input (forward/strafe/jump/sneak axes and rotation deltas) — purely so that input can be
  written to the log alongside the state that produced it.
- It never re-injects, replays, or automates that input. There is no autoclicker, no aim
  assist, no scaffold/bridge assist, and no other gameplay automation anywhere in this codebase.
- The one place the mod *cancels* something the player did is the session-end username-display
  chat prompt (see "Session lifecycle & upload flow" and "Privacy" below): it intercepts the
  player's own "y"/"n" answer to a Herbert-initiated question and prevents that one message from
  being sent, so it doesn't show up in public chat. This is a consent/privacy mechanism, not
  gameplay automation — it never sends a message *for* the player, never fabricates input, and
  only ever intercepts the exact reply they typed to a prompt the mod itself printed.
- All network I/O (pastes.dev upload, Discord webhook) is a plain HTTP `POST` of an already
  completed session log, run on a background thread — the mod never talks to the Minecraft/
  Hypixel server itself beyond the normal vanilla game connection (and, as above, the one chat
  message it ever prevents from being sent).

If you audit the source and find anything that looks like it nudges gameplay rather than just
observing it, please open an issue — that would be a bug, not a feature.

## Requirements

- JDK 8 (the Minecraft 1.8.9 / Forge toolchain requires compiling and running against Java 8
  language/bytecode level). You may use a newer JDK to invoke Gradle itself in some setups, but
  building/running the actual Minecraft/ForgeGradle tasks needs a JDK 8 available.
- Gradle (see "Building" below for wrapper setup).
- Internet access on first build, since ForgeGradle needs to download the Forge/MCP toolchain
  and deobfuscate Minecraft.

## Building

This project ships as a standard Forge 1.8.9 MDK (ForgeGradle 2.x, the
`net.minecraftforge.gradle.forge` plugin) Gradle project.

The Gradle wrapper **jar** binary is intentionally not committed to this repository (binaries
should not be vendored without a build step producing them). `gradle/wrapper/gradle-wrapper.properties`
*is* committed and pins the wrapper to Gradle 2.14, which is compatible with ForgeGradle 2.1 for
1.8.9. Before your first build, generate the wrapper scripts/jar once using a system-installed
Gradle:

```sh
gradle wrapper --gradle-version 2.14
```

After that, build normally:

```sh
./gradlew build
# or, on Windows:
gradlew.bat build
```

The built mod jar will be at `build/libs/bridgelogger-1.0.0.jar`.

To iterate locally in a dev client instead of building a jar:

```sh
./gradlew runClient
```

## Installation

1. Build the jar (see above), or download a pre-built release jar.
2. Install Forge `1.8.9-11.15.1.2318-1.8.9` (or compatible) for Minecraft 1.8.9.
3. Drop `bridgelogger-1.0.0.jar` into your `.minecraft/mods` folder for that Forge profile.
4. Launch Minecraft with the Forge 1.8.9 profile and join Hypixel.
5. On first launch, a config file is generated (see "Configuration" below) — at minimum, set
   `discordWebhookUrl` if you want automatic uploads to post somewhere.

## Configuration

BridgeLogger uses Forge's standard `Configuration` system. The config file is created on first
launch at `config/bridgelogger.cfg` under your `.minecraft` run directory, in the `general`
category. Every key below is written with an inline doc comment in the generated file as well.

| Key | Type | Default | Notes |
|---|---|---|---|
| `blockGridWidth` | int | `7` | Width (X axis, blocks) of the local block grid sampled around the player. Range `1`–`31`. |
| `blockGridHeight` | int | `3` | Height (Y axis, blocks) of the local block grid. Range `1`–`31`. |
| `blockGridDepth` | int | `7` | Depth (Z axis, blocks) of the local block grid. Range `1`–`31`. |
| `logOutputDirectory` | string | `herbert_logs` | Directory session `.jsonl` files are written to. Relative paths resolve against the `.minecraft` run directory; absolute paths are used as-is. |
| `sampleRateDivisor` | int | `1` | Log every Nth client tick. `1` = every tick (~20Hz), `2` = ~10Hz, etc. Minimum `1`. |
| `opponentTrackingEnabled` | bool | `true` | If true, record the nearest opponent's relative state each tick (see schema below). |
| `scoreboardParsingEnabled` | bool | `true` | If true, best-effort parse the Hypixel scoreboard for match context (scores, timer, kit). |
| `discordWebhookUrl` | string | `` (empty) | Discord webhook URL for upload announcements. **If left empty, the entire upload flow (both the pastes.dev upload and the Discord post) is skipped**, and a chat warning is shown instead of attempting any network request. |
| `dryRunMode` | bool | `false` | If true, sessions are still written to disk as JSONL, but the upload step is skipped entirely (no pastes.dev, no Discord). Useful for local-only testing. |
| `voidThresholdY` | int | `0` | World Y coordinate at/below which an air block-grid cell is classified as `VOID` instead of `AIR`. Hypixel does not expose a real "void block" — the void is just open air below the islands — so this is a best-effort heuristic; tune it per map if needed. |
| `bridgeScoreboardTitleMatches` | string list | `BRIDGE`, `THE BRIDGE`, `DUEL` | Case-insensitive substrings checked against the scoreboard sidebar title to auto-detect a Bridge duel. Tune if Hypixel changes its wording. |
| `bridgeServerAddressMatches` | string list | `hypixel.net` | Case-insensitive substrings checked against the current server address to auto-detect a Hypixel connection. |
| `usernamePromptTimeoutSeconds` | int | `30` | How long the session-end "display your username publicly?" chat prompt (see "Session lifecycle & upload flow" below) waits for a response before defaulting to a hashed-only username. Minimum `1`. |
| `targetChunkSizeBytes` | int | `7340032` (7 MiB) | Target size, in bytes, of each split chunk when a session file is too large to upload as a single Discord webhook file attachment — also the threshold that decides whether a session needs chunking at all (see "Chunked uploads" below). Minimum `65536`. |
| `chunkUploadDelayMillis` | int | `1000` | Delay, in milliseconds, between successive chunk uploads to the Discord webhook, to stay under Discord's ~30-requests-per-minute per-webhook rate limit. Minimum `0`. |

**Auto-detection logic:** a Bridge duel is considered "detected" only when *both* the server
address and the scoreboard title match their configured substrings, to minimize false positives
(e.g. recording while idling in the Hypixel lobby). Recording auto-stops after the detection
signal has been continuously absent for about 5 seconds (debounced to tolerate brief scoreboard
redraws/glitches), or immediately on disconnect from the server.

## Chat commands

BridgeLogger also registers a manual override command, since the auto-detection heuristics above
may need per-user/per-server tuning:

- `/herbert start` — manually start a recording session (no-op with a chat message if one is already running).
- `/herbert stop` — manually stop the current session and trigger the upload flow.
- `/herbert status` — print whether a session is active, and if so, its tick count and elapsed time.

## Session lifecycle & upload flow

1. A session auto-starts when the Bridge-duel heuristic matches, or via `/herbert start`.
2. Every sampled tick is appended to an in-progress `<session_id>.jsonl` file via a buffered,
   asynchronous background writer (never on the client thread).
3. The session auto-stops on loss of the detection signal or on disconnect, or via `/herbert stop`.
4. On stop, **before any upload starts**, the mod prints `Herbert: Session stopped. Display your
   username publicly? [y/n]` to your chat and waits for your next chat message:
   - The chat box you type your answer into is intercepted client-side (it is never sent to the
     server, so neither the server nor any other player ever sees your "y"/"n" answer) and read
     directly by the mod.
   - `y`/`n` are accepted case-insensitively. Anything else re-prints the prompt once and waits
     again; a second invalid answer prints `Herbert: Invalid input twice — defaulting to hashed
     username.` and proceeds as if you had answered `n`.
   - If you don't answer within `usernamePromptTimeoutSeconds` (default 30s, configurable — see
     "Configuration" above), the mod prints `Herbert: No response — defaulting to hashed username
     only.` and proceeds as if you had answered `n`.
   - Answering `n` (or timing out, or two invalid answers) is the existing, unchanged behavior:
     the session header carries only `player_username_hash`. Answering `y` additionally writes
     your raw username to the header as `player_username_display` (see the schema below) —
     **this is the only way a raw username is ever written anywhere by this mod**, and it only
     ever affects the one session file the prompt was asked for.
5. Once the prompt is resolved (by answer, default, or timeout), the mod prints `Herbert:
   Uploading...` to your chat, then asynchronously:
   - If `dryRunMode` is enabled, the upload is skipped entirely (the file stays on disk).
   - Else if `discordWebhookUrl` is empty, the upload is skipped and `Herbert: no webhook
     configured, skipping upload` is shown instead.
   - Else, if the completed file is **at or under `targetChunkSizeBytes`** (the common case), the
     existing single-file flow runs, unchanged: the file is POSTed to
     `https://api.pastes.dev/post` as `text/plain`, the returned paste key is used to build
     `https://pastes.dev/{key}`, and a Discord webhook message is posted containing that URL plus
     session metadata (duration, tick count, schema version, mod version) and the label `Herbert
     session upload — {duration}s / {tick_count} ticks`. On success, `Upload complete:
     https://pastes.dev/{key}` is printed to chat.
   - Else (the file is **over** `targetChunkSizeBytes`), the mod automatically splits it and
     uploads each piece directly to Discord — see "Chunked uploads" below. **A session is never
     silently dropped for being too large.**
   - On any single-file-flow failure (network error, non-2xx response, malformed response), the
     error is logged to the mod's logger with a full stack trace, `Upload failed: {reason}` is
     printed to chat, and **the local `.jsonl` file is left intact on disk** — it is never
     deleted on failure.

## Chunked uploads

Discord limits a single (non-boosted) webhook message's file attachment to 8 MiB, which a long
Bridge duel's JSONL log can exceed. Rather than skip the upload for oversized sessions, the mod
splits the file into several self-contained pieces and uploads each one as its own Discord
message.

- **Splitting** (`SessionChunker`) happens on tick-line boundaries only — a single JSON line is
  never split across two files — using accumulated **UTF-8 byte length**, not character count, to
  decide where each chunk ends. Every chunk is written to the same directory as the original
  session file (`logOutputDirectory`), named:

  ```
  herbert_session_{session_id}_part{N}of{T}.jsonl
  ```

  where `N` is the 1-based chunk number and `T` is the total chunk count for that session (e.g.
  `herbert_session_b3a1e6d2-...._part2of3.jsonl`). The original, unsplit `<session_id>.jsonl`
  file is left on disk untouched either way.
- **Every chunk starts with its own copy of the session header line**, so each one is a
  complete, independently-parseable Herbert JSONL file on its own — no other chunk is needed to
  make sense of it. The header copy is byte-for-byte identical to the original except for two
  appended fields, `chunk_index` (0-based) and `chunk_total` — see the schema section below.
- **Chunks upload sequentially, never in parallel**, with a `chunkUploadDelayMillis` pause
  between each one, to stay under Discord's per-webhook rate limit. Unlike the single-file path,
  a chunked upload does **not** go through pastes.dev at all — each chunk's file is attached
  directly to its own webhook message (a `multipart/form-data` POST), whose embed shows which
  part it is, the session ID, the tick range that chunk covers, its file size, and the same
  duration/tick-count/schema/mod-version metadata as a normal upload.
- Chat feedback during a chunked upload:
  - `Herbert: Session too large for a single upload — splitting into {T} parts. (uploading
    1/{T}...)` — printed once, before the first chunk upload.
  - `Herbert: Uploaded part {N}/{T}.` — printed after every chunk that uploads successfully.
  - `Herbert: All {T} parts uploaded successfully. ({tick_count} ticks total, {duration}s)` —
    printed once, after the last chunk succeeds.
  - `Herbert: Upload stopped at part {N}/{T}. Parts {list} saved locally at {path}.` — printed
    if a chunk's upload fails; the sequence stops immediately (chunks already uploaded are **not**
    retried — Discord already has them), and every chunk from the failing one onward is left on
    disk at `{path}` (the same `logOutputDirectory` the chunks were already written to) so nothing
    is lost.

**Known gap:** `/bot`'s intake validator currently recognizes a completed upload by scanning a
webhook message's plain-text `content` field for a pastes.dev URL (see the `notify()` payload
comment in `DiscordWebhookNotifier`). A chunked upload's messages carry no pastes.dev URL at all
— the file is attached directly — so `/bot` does not currently recognize chunked-session webhook
messages as valid Herbert submissions. This mod's job (per "Strict separation of concerns" in
`CONTRIBUTING.md`) stops at documenting the contract accurately; teaching `/bot` to also accept
direct file-attachment messages is a follow-up for that component, not addressed here.

## JSONL schema (contract for `/nn`)

Each session is written as one `.jsonl` file (`config/<logOutputDirectory>/<session_id>.jsonl`
by default): one JSON object per line, newline-delimited, UTF-8. **The first line is always a
header object**; every subsequent line is a per-tick record. Current schema version: `1.2.0`
(see `schema_version` in the header — bump this if the schema changes, and keep this section in
sync).

**Note on the header line's write timing:** unlike every other line in the file, the header line
is written twice in some sessions. The mod writes a provisional (hash-only) header the moment
recording starts, since the rest of the schema is only known session-by-session as ticks are
sampled — but whether `player_username_display` (below) is included depends on an answer the
mod can only ask for *after* the session ends (see "Session lifecycle & upload flow" above). If
the player opts in, the mod rewrites line 1 in place immediately before upload; every other line
in the file is untouched. A file is therefore only ever uploaded/finalized with its header in its
final, correct state — nothing downstream needs to know this happened.

### Header line (always line 1)

The single-file example below (no `chunk_index`/`chunk_total`) is what every non-chunked session
looks like — the overwhelming majority. A chunked session's per-chunk header additionally carries
`chunk_index`/`chunk_total`, appended to an otherwise-identical copy of this same object — see the
second example and "Chunked uploads" above.

```json
{
  "schema_version": "1.2.0",
  "herbert_mod_version": "1.0.0",
  "session_id": "b3a1e6d2-....-....-....-............",
  "recording_start_timestamp": "2026-08-04T21:15:30.123Z",
  "player_username_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d",
  "player_username_display": "Notch"
}
```

A chunk's header (e.g. line 1 of `herbert_session_b3a1e6d2-...._part2of3.jsonl`):

```json
{
  "schema_version": "1.2.0",
  "herbert_mod_version": "1.0.0",
  "session_id": "b3a1e6d2-....-....-....-............",
  "recording_start_timestamp": "2026-08-04T21:15:30.123Z",
  "player_username_hash": "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d",
  "chunk_index": 1,
  "chunk_total": 3
}
```

| Field | Type | Notes |
|---|---|---|
| `schema_version` | string (semver) | Version of the per-tick schema documented below. |
| `herbert_mod_version` | string | BridgeLogger build version that produced the file. |
| `session_id` | string (UUID v4) | Also used as the (non-chunked) log filename. Identical across every chunk of a chunked session, letting `/bot`/`/nn` associate chunks with each other if they're ever taught to. |
| `recording_start_timestamp` | string (ISO-8601, UTC) | When recording started. |
| `player_username_hash` | string (64 hex chars) | **SHA-256 hex digest of the player's Minecraft username.** See "Privacy" below — the raw username is never written anywhere else. Always present, regardless of `player_username_display`. |
| `player_username_display` | string | **Optional, added in schema `1.1.0`.** The player's raw Minecraft username, present **only** if they answered `y` to the session-end "display your username publicly?" chat prompt (see "Session lifecycle & upload flow" above). Omitted entirely (never `null`, never an empty string) whenever they answered `n`, let the prompt time out, or gave invalid input twice. Absent in every `1.0.0` file. |
| `chunk_index` | integer | **Optional, added in schema `1.2.0`.** 0-based index of this chunk among `chunk_total` chunks. Present **only** on a chunk file produced by `SessionChunker` (see "Chunked uploads" above); absent on every non-chunked (single-file) session, and absent in every `1.0.0`/`1.1.0` file. |
| `chunk_total` | integer | **Optional, added in schema `1.2.0`.** Total number of chunks the session was split into. Same presence rules as `chunk_index`; always `>= 2` when present (a session that fit in one file is never chunked, so this field never appears with a value of `1`). |

### Per-tick line (every subsequent line)

```json
{
  "tick": 42,
  "timestamp": "2026-08-04T21:15:32.243Z",
  "player": {
    "x": 12.5, "y": 71.0, "z": -4.25,
    "vx": 0.0, "vy": -0.0784, "vz": 0.21,
    "yaw": 132.4, "pitch": 8.1,
    "on_ground": true, "sneaking": false,
    "health": 20.0, "food": 20
  },
  "block_grid": {
    "width": 7, "height": 3, "depth": 7,
    "origin": "player_feet_centered",
    "cells": ["AIR", "SOLID_BRIDGEABLE", "..."]
  },
  "held_item": {
    "hotbar_slot": 3,
    "item_id": "minecraft:wool",
    "count": 47
  },
  "opponent": {
    "rel_x": 3.1, "rel_y": 0.0, "rel_z": -1.4,
    "rel_vx": -0.1, "rel_vy": 0.0, "rel_vz": 0.05,
    "yaw": -47.2, "pitch": 3.0,
    "health": 14.0,
    "held_item_category": "SWORD"
  },
  "match": {
    "own_score": 1,
    "opponent_score": 0,
    "elapsed_seconds": 95,
    "kit": "Iron Man"
  },
  "input": {
    "forward": 1, "strafe": 0,
    "jump": false, "sneak": true,
    "delta_yaw": 1.2, "delta_pitch": -0.4,
    "attack_occurred": false, "attack_target_type": null,
    "place_occurred": true, "place_block_type": "minecraft:wool",
    "place_x": 13, "place_y": 71, "place_z": -5
  }
}
```

| Field | Type | Nullable | Notes |
|---|---|---|---|
| `tick` | integer | no | Session-relative sample index, starting at `0`. **Not** the raw Minecraft tick counter — increments once per *sampled* tick (i.e. respects `sampleRateDivisor`). |
| `timestamp` | string (ISO-8601, UTC) | no | Wall-clock time this sample was taken. |
| `player.x/y/z` | double | no | World coordinates. |
| `player.vx/vy/vz` | double | no | Velocity in blocks/tick, read directly from the entity's motion vector. |
| `player.yaw/pitch` | float | no | Degrees, vanilla convention. |
| `player.on_ground` | boolean | no | |
| `player.sneaking` | boolean | no | |
| `player.health` | float | no | 0–20. |
| `player.food` | integer | no | 0–20. |
| `block_grid.width/height/depth` | integer | no | Echoes the configured grid dimensions at capture time. |
| `block_grid.origin` | string | no | Always the literal `"player_feet_centered"`. |
| `block_grid.cells` | array of string enum | no | Flattened `width*height*depth` array of `AIR` \| `SOLID_BRIDGEABLE` \| `LIQUID` \| `VOID` \| `OTHER_SOLID`. **Index order**: `index = (yIndex * depth + zIndex) * width + xIndex`, where `xIndex ∈ [0, width)` maps to world offset `xIndex - width/2` (integer division) from the player's feet block, and similarly for `yIndex`/`height` and `zIndex`/`depth`. E.g. with default `7×3×7`, `xIndex`/`zIndex` offsets run `-3..3` and `yIndex` offsets run `-1..1` (one block below feet, feet level, one block above). |
| `held_item.hotbar_slot` | integer | no | `0`–`8`. |
| `held_item.item_id` | string | **yes** | Registry name (e.g. `"minecraft:wool"`), or `null` if the slot is empty. |
| `held_item.count` | integer | no | `0` if the slot is empty. |
| `opponent` | object | **yes (whole object)** | `null` if no opponent is within tracking range (48 blocks) or `opponentTrackingEnabled` is `false`. |
| `opponent.rel_x/rel_y/rel_z` | double | no (when present) | Opponent position minus local player position. |
| `opponent.rel_vx/rel_vy/rel_vz` | double | no (when present) | Opponent velocity minus local player velocity, blocks/tick. |
| `opponent.yaw/pitch` | float | no (when present) | Opponent's absolute rotation, degrees. |
| `opponent.health` | float | no (when present) | 0–20. |
| `opponent.held_item_category` | string enum | no (when present) | One of `SWORD` \| `BOW` \| `BLOCKS` \| `OTHER`. Never `null` when `opponent` is present. |
| `match` | object | **yes (whole object)** | `null` if `scoreboardParsingEnabled` is `false`, or nothing could be parsed this tick. |
| `match.own_score` | integer | **yes** | Best-effort; `null` if unparsable. |
| `match.opponent_score` | integer | **yes** | Best-effort; `null` if unparsable. |
| `match.elapsed_seconds` | integer | **yes** | Best-effort, parsed from an `mm:ss` style timer line; `null` if unparsable. |
| `match.kit` | string | **yes** | Best-effort, parsed from a `Kit: <name>` style line; `null` if unparsable. |
| `input.forward` | integer | no | `-1` \| `0` \| `1`. |
| `input.strafe` | integer | no | `-1` \| `0` \| `1`. |
| `input.jump` | boolean | no | |
| `input.sneak` | boolean | no | |
| `input.delta_yaw` | float | no | Degrees, change since the previous *sampled* tick (wrapped to `[-180, 180)`). |
| `input.delta_pitch` | float | no | Degrees, change since the previous sampled tick. |
| `input.attack_occurred` | boolean | no | Whether a left-click/attack event happened since the last sample. |
| `input.attack_target_type` | string | **yes** | Coarse target type name (e.g. `"EntityOtherPlayerMP"`), or `null` if no attack or unknown target. |
| `input.place_occurred` | boolean | no | Whether a right-click block-place event happened since the last sample. |
| `input.place_block_type` | string | **yes** | Registry name of the placed block, or `null` if no placement or unresolvable. |
| `input.place_x/y/z` | integer | **yes** | World coordinates of the placement, or `null` if no placement or unresolvable. |

Because Hypixel's scoreboard format is undocumented and can change, every `match.*` field is
independently best-effort and may be `null` even when the `match` object itself is present (the
object is only entirely `null` when *nothing at all* was parseable that tick).

## Privacy

BridgeLogger is built for a community dataset, so protecting player identity matters. By
default, the mod **never writes a raw Minecraft username to disk or over the network, anywhere**.
The only trace of player identity in a session file is `player_username_hash` in the header line
— the lowercase hex SHA-256 digest of the username, computed once at session start (see
`HashUtil`). This lets downstream tooling deduplicate/group sessions per contributor without
recovering the actual username, and it is a one-way hash (not reversible, though note SHA-256 of
a short username space is not resistant to a determined dictionary/rainbow-table attack against a
known-username guess — treat the hash as pseudonymous, not anonymous, if that distinction matters
for your use case).

**The one exception is opt-in and explicit.** At the end of every session, the mod asks in chat
whether to also include the raw username in that session's header (see "Session lifecycle &
upload flow" above); the answer defaults to "no" (hash-only, the behavior described in the
previous paragraph) on no response, on timeout, or on invalid input. Only an explicit `y` answer
adds `player_username_display` (raw username, plaintext) to that one session's header before it
is uploaded. Nothing else about the prompt or its answer is ever transmitted: the `y`/`n` message
itself never leaves the client (see below), and the decision only affects the session it was
asked for, not any other session.

No other field in this schema (per-tick or header) contains a player name, IP address, or other
directly identifying information; opponent state is recorded only as relative geometry plus
coarse categorical state, never an opponent username.

**Note on the prompt mechanism:** the "y"/"n" answer is captured entirely client-side by
intercepting the chat GUI (see `SessionManager`/`HerbertPromptChatGui`) before Minecraft ever
constructs a chat packet for it, so it never reaches the server and is never visible to other
players — this is a consent-capture mechanism, not gameplay automation (see "This mod is purely
observational" above); it reads only the one message the player themselves chose to type in
response to a Herbert-initiated prompt.

## Project layout

```
mod/
  build.gradle
  gradle.properties
  gradle/wrapper/gradle-wrapper.properties
  src/main/resources/mcmod.info
  src/main/java/dev/herbert/bridgelogger/
    BridgeLoggerMod.java            — @Mod entry point, lifecycle wiring
    config/HerbertConfig.java       — Forge Configuration wrapper
    model/                          — plain data classes (de)serialized to/from JSONL
    capture/                        — tick sampling, block/held-item classification, scoreboard parsing, Bridge detection
    serialize/                      — JSONL (de)serialization + async disk writer
    session/SessionManager.java     — lifecycle state machine, tick handler, event glue
    upload/                         — pastes.dev + Discord webhook HTTP clients
    command/HerbertCommands.java    — /herbert start|stop|status
    util/                           — constants, SHA-256 hashing, scoreboard reading helpers
```

## Contract with the rest of the Herbert project

This mod is one of three independent components in the Herbert project (`/mod`, `/nn`, `/bot`).
There is no shared code between them — the only contracts are:

1. The JSONL schema documented above, which `/nn` consumes.
2. The `https://pastes.dev/{key}` URL format posted to Discord, which `/bot` validates.
