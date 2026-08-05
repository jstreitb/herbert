// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.util;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

import net.minecraft.scoreboard.Score;
import net.minecraft.scoreboard.ScoreObjective;
import net.minecraft.scoreboard.ScorePlayerTeam;
import net.minecraft.scoreboard.Scoreboard;
import net.minecraft.util.EnumChatFormatting;
import net.minecraft.world.World;

/**
 * Defensive helper for reading the vanilla sidebar scoreboard, used both by Bridge-duel
 * auto-detection and by best-effort match-context parsing.
 *
 * <p>Hypixel's scoreboard content and formatting are not officially documented and can change
 * without notice. Every method here is written to never throw: on any unexpected state it logs
 * nothing (to avoid tick-rate log spam) and simply returns an empty/{@code null} result so
 * callers can fall back gracefully.</p>
 */
public final class ScoreboardUtil {

    /** Vanilla display-slot index for the sidebar scoreboard (see {@code Scoreboard.getObjectiveInSlot}). */
    private static final int SIDEBAR_DISPLAY_SLOT = 1;

    /** Safety cap on how many sidebar lines are read, in case of a malformed/huge objective. */
    private static final int MAX_SIDEBAR_LINES = 32;

    private ScoreboardUtil() {
        // Static utility class; never instantiated.
    }

    /**
     * Reads the current sidebar objective's display title, with formatting codes stripped.
     *
     * @param world the client world to read the scoreboard from; may be {@code null}
     * @return the sidebar title text, or {@code null} if unavailable or unparsable
     */
    public static String getSidebarTitle(World world) {
        try {
            if (world == null) {
                return null;
            }
            Scoreboard scoreboard = world.getScoreboard();
            if (scoreboard == null) {
                return null;
            }
            ScoreObjective objective = scoreboard.getObjectiveInDisplaySlot(SIDEBAR_DISPLAY_SLOT);
            if (objective == null) {
                return null;
            }
            String title = objective.getDisplayName();
            return title == null ? null : EnumChatFormatting.getTextWithoutFormattingCodes(title);
        } catch (Exception e) {
            // Defensive: never let scoreboard parsing disrupt the tick handler; treat any
            // unexpected scoreboard state as "no title available" rather than throwing.
            return null;
        }
    }

    /**
     * Reads the current sidebar objective's visible lines, top-to-bottom as vanilla would
     * render them, with formatting codes stripped and team prefix/suffix applied (matching how
     * Hypixel typically packs sidebar text via fake player names + team prefixes/suffixes).
     *
     * @param world the client world to read the scoreboard from; may be {@code null}
     * @return an unmodifiable-in-spirit list of line strings, top to bottom; empty if unavailable
     */
    public static List<String> getSidebarLines(World world) {
        List<String> lines = new ArrayList<String>();
        try {
            if (world == null) {
                return lines;
            }
            Scoreboard scoreboard = world.getScoreboard();
            if (scoreboard == null) {
                return lines;
            }
            ScoreObjective objective = scoreboard.getObjectiveInDisplaySlot(SIDEBAR_DISPLAY_SLOT);
            if (objective == null) {
                return lines;
            }
            Collection<Score> scores = scoreboard.getSortedScores(objective);
            if (scores == null) {
                return lines;
            }
            List<Score> filtered = new ArrayList<Score>();
            for (Score score : scores) {
                if (score != null && score.getPlayerName() != null && !score.getPlayerName().startsWith("#")) {
                    filtered.add(score);
                }
            }
            int start = Math.max(0, filtered.size() - MAX_SIDEBAR_LINES);
            for (int i = start; i < filtered.size(); i++) {
                Score score = filtered.get(i);
                String playerName = score.getPlayerName();
                ScorePlayerTeam team = scoreboard.getPlayersTeam(playerName);
                String formatted = ScorePlayerTeam.formatPlayerName(team, playerName);
                lines.add(EnumChatFormatting.getTextWithoutFormattingCodes(formatted));
            }
        } catch (Exception e) {
            // Defensive: never let scoreboard parsing disrupt the tick handler. Return whatever
            // (possibly empty) list was accumulated so far.
        }
        return lines;
    }
}
