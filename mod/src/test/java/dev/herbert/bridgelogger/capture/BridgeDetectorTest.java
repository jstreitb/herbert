// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.capture;

import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import org.junit.Test;

/** Unit tests for {@link BridgeDetector}. */
public class BridgeDetectorTest {

    private static final String[] ADDRESS_MATCHES = {"hypixel.net"};
    private static final String[] TITLE_MATCHES = {"BRIDGE", "THE BRIDGE", "DUEL"};

    @Test
    public void serverAddressMatches_caseInsensitiveSubstring_expectsTrue() {
        assertTrue(BridgeDetector.serverAddressMatches("mc.HYPIXEL.net", ADDRESS_MATCHES));
    }

    @Test
    public void serverAddressMatches_noMatch_expectsFalse() {
        assertFalse(BridgeDetector.serverAddressMatches("play.hivemc.com", ADDRESS_MATCHES));
    }

    @Test
    public void serverAddressMatches_nullAddress_expectsFalseNotThrow() {
        assertFalse(BridgeDetector.serverAddressMatches(null, ADDRESS_MATCHES));
    }

    @Test
    public void serverAddressMatches_nullSubstringArray_expectsFalseNotThrow() {
        assertFalse(BridgeDetector.serverAddressMatches("mc.hypixel.net", null));
    }

    @Test
    public void serverAddressMatches_emptySubstringArray_expectsFalse() {
        assertFalse(BridgeDetector.serverAddressMatches("mc.hypixel.net", new String[0]));
    }

    @Test
    public void scoreboardTitleMatches_caseInsensitiveSubstring_expectsTrue() {
        assertTrue(BridgeDetector.scoreboardTitleMatches("THE BRIDGE DUELS", TITLE_MATCHES));
    }

    @Test
    public void scoreboardTitleMatches_emptyTitle_expectsFalse() {
        assertFalse(BridgeDetector.scoreboardTitleMatches("", TITLE_MATCHES));
    }

    @Test
    public void looksLikeBridgeDuel_bothSignalsMatch_expectsTrue() {
        assertTrue(BridgeDetector.looksLikeBridgeDuel("mc.hypixel.net", "THE BRIDGE", ADDRESS_MATCHES, TITLE_MATCHES));
    }

    @Test
    public void looksLikeBridgeDuel_onlyAddressMatches_expectsFalse() {
        // Requiring both signals prevents a false-positive session start while idling in the
        // Hypixel lobby (address matches, but the sidebar title does not look like a duel).
        assertFalse(BridgeDetector.looksLikeBridgeDuel("mc.hypixel.net", "HYPIXEL LOBBY", ADDRESS_MATCHES, TITLE_MATCHES));
    }

    @Test
    public void looksLikeBridgeDuel_onlyTitleMatches_expectsFalse() {
        assertFalse(BridgeDetector.looksLikeBridgeDuel("play.someotherserver.net", "THE BRIDGE", ADDRESS_MATCHES, TITLE_MATCHES));
    }

    @Test
    public void looksLikeBridgeDuel_allNullInputs_expectsFalseNotThrow() {
        assertFalse(BridgeDetector.looksLikeBridgeDuel(null, null, null, null));
    }
}
