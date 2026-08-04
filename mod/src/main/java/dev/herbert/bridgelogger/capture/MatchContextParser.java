package dev.herbert.bridgelogger.capture;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import dev.herbert.bridgelogger.model.MatchContext;

/**
 * Best-effort parser that extracts {@link MatchContext} (scores, elapsed time, kit) from the
 * lines of the Hypixel sidebar scoreboard.
 *
 * <p>Hypixel's scoreboard text format is not officially documented, is not guaranteed to be
 * stable, and varies by game mode/map/event. This parser therefore uses loose regular
 * expressions and treats every extraction as independently optional: a failure to parse one
 * field never prevents extracting another, and any unexpected input simply yields {@code null}
 * for that field rather than throwing. Callers should treat every field of the returned
 * {@link MatchContext} as potentially absent.</p>
 */
public final class MatchContextParser {

    /** Matches a {@code mm:ss} style match timer, e.g. {@code "02:15"} or {@code "2:15"}. */
    private static final Pattern TIMER_PATTERN = Pattern.compile("\\b(\\d{1,2}):([0-5]\\d)\\b");

    /** Matches a {@code "<label>: <number>"} style scoreboard line, e.g. {@code "Red: 3"}. */
    private static final Pattern LABEL_SCORE_PATTERN = Pattern.compile("^([A-Za-z][A-Za-z ']{0,19}):\\s*(\\d{1,3})$");

    /** Matches a {@code "Kit: <name>"} style scoreboard line. */
    private static final Pattern KIT_PATTERN = Pattern.compile("^Kit:\\s*(.+)$", Pattern.CASE_INSENSITIVE);

    private MatchContextParser() {
        // Static utility class; never instantiated.
    }

    /**
     * Attempts to extract match context from the given sidebar lines.
     *
     * @param sidebarLines the current sidebar scoreboard lines, top to bottom, formatting codes
     *        already stripped (see {@link dev.herbert.bridgelogger.util.ScoreboardUtil#getSidebarLines});
     *        may be {@code null} or empty
     * @param ownTeamHint an optional hint (e.g. the local player's team color/name, such as
     *        {@code "RED"} or {@code "BLUE"}) used to disambiguate which parsed score line is
     *        the local player's own score; may be {@code null} if unknown, in which case a
     *        best-effort positional guess (first score line = own, second = opponent) is used
     * @return a {@link MatchContext} with as many fields populated as could be confidently
     *         parsed; every field may independently be {@code null}. Never returns {@code null}
     *         and never throws.
     */
    public static MatchContext parse(List<String> sidebarLines, String ownTeamHint) {
        Integer elapsedSeconds = null;
        String kit = null;
        Map<String, Integer> scoreLines = new LinkedHashMap<String, Integer>();

        if (sidebarLines != null) {
            for (String line : sidebarLines) {
                if (line == null) {
                    continue;
                }
                String trimmed = line.trim();
                if (trimmed.isEmpty()) {
                    continue;
                }

                if (elapsedSeconds == null) {
                    elapsedSeconds = tryParseTimer(trimmed);
                }
                if (kit == null) {
                    kit = tryParseKit(trimmed);
                }
                tryParseScoreLine(trimmed, scoreLines);
            }
        }

        Integer ownScore = null;
        Integer opponentScore = null;
        try {
            if (ownTeamHint != null && !ownTeamHint.isEmpty()) {
                String hint = ownTeamHint.toLowerCase(Locale.ROOT);
                for (Map.Entry<String, Integer> entry : scoreLines.entrySet()) {
                    if (entry.getKey().toLowerCase(Locale.ROOT).contains(hint)) {
                        ownScore = entry.getValue();
                    } else if (opponentScore == null) {
                        opponentScore = entry.getValue();
                    }
                }
            }
            if (ownScore == null && opponentScore == null && scoreLines.size() >= 2) {
                // No team hint available (or it didn't match anything): fall back to a
                // best-effort positional guess. Documented clearly as a guess, not a guarantee.
                Integer[] values = scoreLines.values().toArray(new Integer[0]);
                ownScore = values[0];
                opponentScore = values[1];
            }
        } catch (Exception e) {
            ownScore = null;
            opponentScore = null;
        }

        return new MatchContext(ownScore, opponentScore, elapsedSeconds, kit);
    }

    private static Integer tryParseTimer(String line) {
        try {
            Matcher matcher = TIMER_PATTERN.matcher(line);
            if (matcher.find()) {
                int minutes = Integer.parseInt(matcher.group(1));
                int seconds = Integer.parseInt(matcher.group(2));
                return minutes * 60 + seconds;
            }
        } catch (Exception e) {
            // Fall through to null.
        }
        return null;
    }

    private static String tryParseKit(String line) {
        try {
            Matcher matcher = KIT_PATTERN.matcher(line);
            if (matcher.matches()) {
                String kit = matcher.group(1).trim();
                return kit.isEmpty() ? null : kit;
            }
        } catch (Exception e) {
            // Fall through to null.
        }
        return null;
    }

    private static void tryParseScoreLine(String line, Map<String, Integer> outScoreLines) {
        try {
            Matcher matcher = LABEL_SCORE_PATTERN.matcher(line);
            if (matcher.matches()) {
                String label = matcher.group(1).trim();
                int value = Integer.parseInt(matcher.group(2));
                outScoreLines.put(label, value);
            }
        } catch (Exception e) {
            // Ignore malformed line; not fatal.
        }
    }
}
