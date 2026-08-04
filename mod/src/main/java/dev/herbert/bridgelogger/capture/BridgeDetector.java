package dev.herbert.bridgelogger.capture;

import java.util.Locale;

/**
 * Best-effort, purely heuristic detector for "is the local player currently in a Hypixel
 * Bridge duel". Combines a server-address check with a scoreboard-title check so that neither
 * signal alone (which could false-positive, e.g. a title match on a different Hypixel game, or
 * an address match while sitting in the lobby) triggers a recording session on its own.
 *
 * <p>Every method here is defensive: {@code null} or empty inputs are treated as "no match"
 * rather than throwing, since this runs on the client tick handler where an uncaught exception
 * would disrupt the player's game.</p>
 */
public final class BridgeDetector {

    private BridgeDetector() {
        // Static utility class; never instantiated.
    }

    /**
     * Checks whether {@code serverAddress} contains any of {@code addressSubstrings},
     * case-insensitively.
     *
     * @param serverAddress the current server address (host, or host:port), may be {@code null}
     * @param addressSubstrings configured substrings to match against; may be {@code null} or empty
     * @return {@code true} if a case-insensitive substring match is found, {@code false} otherwise (never throws)
     */
    public static boolean serverAddressMatches(String serverAddress, String[] addressSubstrings) {
        return containsAnyIgnoreCase(serverAddress, addressSubstrings);
    }

    /**
     * Checks whether {@code scoreboardTitle} contains any of {@code titleSubstrings},
     * case-insensitively.
     *
     * @param scoreboardTitle the current sidebar scoreboard title text, may be {@code null}
     * @param titleSubstrings configured substrings to match against; may be {@code null} or empty
     * @return {@code true} if a case-insensitive substring match is found, {@code false} otherwise (never throws)
     */
    public static boolean scoreboardTitleMatches(String scoreboardTitle, String[] titleSubstrings) {
        return containsAnyIgnoreCase(scoreboardTitle, titleSubstrings);
    }

    /**
     * Combined heuristic: a Bridge duel is considered detected only when both the server
     * address and the scoreboard title look right. Requiring both signals is a deliberate
     * defensive choice to minimize false-positive session starts (e.g. accidentally recording
     * while in the Hypixel lobby, or on an unrelated server whose scoreboard happens to mention
     * "duel").
     *
     * @param serverAddress current server address, may be {@code null}
     * @param scoreboardTitle current sidebar scoreboard title, may be {@code null}
     * @param addressSubstrings configured server-address match substrings
     * @param titleSubstrings configured scoreboard-title match substrings
     * @return {@code true} if both the address and title heuristics match, {@code false} otherwise (never throws)
     */
    public static boolean looksLikeBridgeDuel(String serverAddress, String scoreboardTitle, String[] addressSubstrings,
            String[] titleSubstrings) {
        try {
            return serverAddressMatches(serverAddress, addressSubstrings) && scoreboardTitleMatches(scoreboardTitle, titleSubstrings);
        } catch (Exception e) {
            return false;
        }
    }

    private static boolean containsAnyIgnoreCase(String haystack, String[] needles) {
        try {
            if (haystack == null || needles == null || needles.length == 0) {
                return false;
            }
            String lowerHaystack = haystack.toLowerCase(Locale.ROOT);
            for (String needle : needles) {
                if (needle != null && !needle.isEmpty() && lowerHaystack.contains(needle.toLowerCase(Locale.ROOT))) {
                    return true;
                }
            }
            return false;
        } catch (Exception e) {
            return false;
        }
    }
}
