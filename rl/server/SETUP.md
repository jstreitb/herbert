# Minecraft 1.8.9 server setup for `/rl`

`/rl` connects to **a private, self-hosted Minecraft 1.8.9 server that you control** -- not
Hypixel, not any third-party server. This is documentation only; there is no server-side plugin
shipped in this repository, since the exact mini-game setup is up to you. Everything below is
what the bridge (`/rl/bridge`) and the training layer (`/rl/train`) assume/need from that server.

## 1. Server software

Either is fine:

- **PaperMC 1.8.9** (recommended -- actively-maintained performance fork; use the last 1.8.9
  build available from the PaperMC project's legacy version archive).
- **Spigot 1.8.9** (built via BuildTools targeting `1.8.9`).

A **vanilla** 1.8.9 server also works for the bridge connection itself (Mineflayer only needs a
normal client-compatible server), but you'll have a much harder time implementing automatic
match cycling (step 3) without a plugin API, so Paper/Spigot is strongly recommended in practice.

## 2. Offline-mode auth

The two bot accounts (`HerbertBot1`/`HerbertBot2` by default -- see `rl/conf/env/default.yaml`)
are **not** real Mojang/Microsoft accounts; Mineflayer connects them with `auth: "offline"` (see
`rl/bridge/README.md`'s `--auth` flag). Your server must accept offline-mode connections:

```properties
# server.properties
online-mode=false
```

**Security note:** `online-mode=false` means *anyone* who can reach the server port can connect
as any username with no authentication. Only run this on a private/local/firewalled network
(e.g. `localhost`, a LAN, or a cloud box with the Minecraft port closed to the public internet)
-- never expose an offline-mode server to the public internet.

## 3. Automatic Bridge match cycling

The RL training loop needs matches to start/restart on their own once a bridge process sends a
`reset` command (see `rl/bridge/README.md`'s `--reset-grace-period-ms`) -- there's no human
clicking "ready up" between episodes. You need *some* mechanism for this; options, roughly in
order of effort:

- **A Bridge/duels mini-game plugin** (Spigot/Paper) that auto-starts a new 1v1 match as soon as
  both configured bot accounts are present/idle in a lobby, resets the arena (fills back in any
  placed/broken blocks) and the scoreboard, and teleports both players to their spawn points.
  Several open-source "duels"/"bridge" mini-game plugins exist for 1.8.9-era Spigot; any one that
  supports fully automatic (no manual `/ready`) match starts and *arena regeneration between
  matches* works.
- **A minimal custom plugin** (if nothing off-the-shelf fits your setup): on a configurable timer
  or on both players being idle/present, clear+regenerate the bridge arena's placed blocks
  (WorldEdit's schematic-paste API is the easiest way to do this reliably), teleport both players
  to spawn, and reset a scoreboard objective tracking each side's score.

Whatever you use, `--reset-grace-period-ms` (bridge-side) needs to comfortably exceed how long
your plugin actually takes to finish cycling a match, or episodes will start with a stale arena.

## 4. Match-end / score detection

`herbert_rl.env.match_coordinator.MatchCoordinator` detects a match ending two ways, both
configurable in `rl/conf/env/default.yaml`'s `match_end` block:

- **Score threshold** (`match_end.score_threshold`, default `3`): first side to reach this many
  points (via `match.own_score`/`match.opponent_score`, parsed from your scoreboard -- see below)
  ends the episode.
- **Chat pattern** (`match_end.chat_patterns`): a list of regexes checked against every chat/
  system message line since the last tick. Default patterns (`"has won the game"`, `"duel has
  ended"`, `"Bridge.*over"`) are generic guesses -- replace them with whatever message your
  plugin actually broadcasts on match end.

## 5. Scoreboard format

`rl/bridge`'s `src/matchState.js` reads whatever objectives are currently displayed
(`bot.scoreboards`) and applies configurable regexes (`--own-score-pattern`,
`--opponent-score-pattern`, `--elapsed-seconds-pattern`, `--kit-pattern`) against every line,
first match wins. There is no fixed expected format -- tune the four regex flags to match your
plugin's actual scoreboard text. A quick way to see the real text: connect a normal Minecraft
client to your server during a match and read the sidebar, or temporarily bump the bridge's
`--log-level debug` and add a debug log line dumping `bot.scoreboards` (see
`rl/bridge/README.md`'s "Known calibration points" for the general pattern of verifying
assumptions like this against your specific setup).

## 6. Bot inventory / kit

The RL action space (see `rl/README.md`) does not include hotbar-slot selection or inventory
management -- a `place` action just places whatever block is currently in the bot's selected
hotbar slot. Your match-start/reset flow (the same plugin from step 3) needs to give each bot
account a starting kit that already has bridging blocks selected in a hotbar slot, exactly like a
normal human Bridge kit would.

## 7. JVM flags for a training server

Since `/rl` depends on consistent ~20Hz tick timing (an RL "tick" is one server tick; see
`rl/bridge/src/bridge.js`'s request-driven cadence), lag directly slows down and destabilizes
training. Recommendations for a small, low-player-count (2 bots) training server:

```bash
java -Xms4G -Xmx4G \
     -XX:+UseG1GC -XX:+ParallelRefProcEnabled -XX:MaxGCPauseMillis=200 \
     -XX:+UnlockExperimentalVMOptions -XX:+DisableExplicitGC \
     -XX:G1NewSizePercent=30 -XX:G1MaxNewSizePercent=40 -XX:G1HeapRegionSize=8M \
     -XX:G1ReservePercent=20 -XX:G1HeapWastePercent=5 -XX:G1MixedGCCountTarget=4 \
     -XX:InitiatingHeapOccupancyPercent=15 -XX:G1MixedGCLiveThresholdPercent=90 \
     -XX:G1RSetUpdatingPauseTimePercent=5 -XX:SurvivorRatio=32 -XX:MaxTenuringThreshold=1 \
     -jar paper-1.8.9.jar --nogui
```

(These are the widely-circulated "Aikar's flags" tuned for G1GC pause-time consistency, not
throughput -- appropriate here since we care about *tick timing consistency*, not serving many
concurrent players.) Also worth doing on a dedicated training box:

- `view-distance` low (e.g. `4`-`6`) in `server.properties` -- only two players are ever online.
- Disable/limit any world-generation, mob AI, or redstone-heavy systems near the Bridge arena if
  your map has them -- anything that adds per-tick server work directly adds RL wall-clock time.
- Run the server and the Python training process on the same machine or a low-latency LAN link;
  `bridge_tick_timeout_s` (default 5s, see `rl/conf/env/default.yaml`) is generous, but real
  network jitter to a remote server still slows down training throughput noticeably at 20Hz.
