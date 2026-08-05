// SPDX-License-Identifier: MIT
'use strict';

/**
 * Best-effort match-state (scoreboard) parsing and chat-line collection, mirroring `/mod`'s own
 * "best-effort, everything independently nullable" approach to scoreboard parsing (see
 * `mod/README.md`'s note: "Because Hypixel's scoreboard format is undocumented and can change,
 * every `match.*` field is independently best-effort"). Since `/rl` targets a private,
 * developer-controlled server rather than Hypixel, the exact scoreboard line format is whatever
 * *your* Bridge mini-game plugin produces -- every pattern below is a regex supplied via CLI
 * flags (see `bridge.js`), matched against every line of every known scoreboard objective. The
 * defaults are generic best-effort guesses; **tune them against your server's actual scoreboard
 * output** (see `rl/server/SETUP.md`).
 *
 * Chat-line collection also covers plain chat/system messages (used by
 * `herbert_rl.env.match_coordinator.MatchCoordinator._match_ended`'s configurable regex match
 * against match-end announcements).
 */

function createMatchStateTracker(bot, config) {
  const { ownScorePattern, opponentScorePattern, elapsedSecondsPattern, kitPattern } = config;
  // Compiled once here rather than on every firstMatch() call -- getMatchState() runs every
  // tick, so rebuilding these four RegExp objects from source 4x/tick would be wasted work.
  const ownScoreRegex = ownScorePattern ? new RegExp(ownScorePattern) : null;
  const opponentScoreRegex = opponentScorePattern ? new RegExp(opponentScorePattern) : null;
  const elapsedSecondsRegex = elapsedSecondsPattern ? new RegExp(elapsedSecondsPattern) : null;
  const kitRegex = kitPattern ? new RegExp(kitPattern) : null;
  let pendingChatLines = [];

  bot.on('message', (jsonMsg) => {
    try {
      pendingChatLines.push(jsonMsg.toString());
    } catch (err) {
      // Malformed/unusual message component -- never let chat parsing crash the tick loop.
    }
  });

  function drainChatLines() {
    const lines = pendingChatLines;
    pendingChatLines = [];
    return lines;
  }

  function scoreboardLines() {
    const lines = [];
    const objectives = bot.scoreboards || {};
    for (const key of Object.keys(objectives)) {
      const objective = objectives[key];
      // `objective.items` (mineflayer's `Scoreboard.items` getter) is already an array of
      // `{name, value, displayName}`, sorted by score -- `displayName` is a `prismarine-chat`
      // `ChatMessage` whose `toString()` renders the plain (legacy-formatted, for 1.8.9) text;
      // `name` is the raw scoreboard entry name, kept as a fallback in case `displayName` is
      // ever absent on an older mineflayer version.
      if (!objective || !objective.items) continue;
      for (const item of objective.items) {
        if (item && item.displayName !== undefined && item.displayName !== null) {
          lines.push(String(item.displayName));
        } else if (item && item.name !== undefined) {
          lines.push(String(item.name));
        }
      }
    }
    return lines;
  }

  function firstMatch(lines, regex) {
    if (!regex) return null;
    for (const line of lines) {
      const match = regex.exec(line);
      if (match) {
        return match.length > 1 ? match[1] : match[0];
      }
    }
    return null;
  }

  /**
   * @returns {{own_score: number|null, opponent_score: number|null, elapsed_seconds:
   *   number|null, kit: string|null}|null} `null` if literally nothing was parseable this
   *   tick, matching `/mod`'s documented contract ("the object is only entirely null when
   *   nothing at all was parseable that tick" -- see mod/README.md's JSONL schema section).
   *   `/nn`'s feature encoding sets a `match_context_present` flag from `match is not None`,
   *   so always returning a non-null object here would systematically bias that feature
   *   during RL fine-tuning relative to how it behaves in `/mod`-recorded BC training data.
   */
  function getMatchState() {
    const lines = scoreboardLines();
    const ownScoreRaw = firstMatch(lines, ownScoreRegex);
    const opponentScoreRaw = firstMatch(lines, opponentScoreRegex);
    const elapsedRaw = firstMatch(lines, elapsedSecondsRegex);
    const kitRaw = firstMatch(lines, kitRegex);

    let elapsedSeconds = null;
    if (elapsedRaw !== null) {
      const mmss = /^(\d+):(\d{2})$/.exec(elapsedRaw);
      if (mmss) {
        elapsedSeconds = parseInt(mmss[1], 10) * 60 + parseInt(mmss[2], 10);
      } else {
        const asNumber = Number(elapsedRaw);
        elapsedSeconds = Number.isFinite(asNumber) ? asNumber : null;
      }
    }

    const ownScore =
      ownScoreRaw !== null && Number.isFinite(Number(ownScoreRaw)) ? Number(ownScoreRaw) : null;
    const opponentScore =
      opponentScoreRaw !== null && Number.isFinite(Number(opponentScoreRaw))
        ? Number(opponentScoreRaw)
        : null;

    if (ownScore === null && opponentScore === null && elapsedSeconds === null && kitRaw === null) {
      return null;
    }
    return {
      own_score: ownScore,
      opponent_score: opponentScore,
      elapsed_seconds: elapsedSeconds,
      kit: kitRaw,
    };
  }

  return { drainChatLines, getMatchState };
}

module.exports = { createMatchStateTracker };
