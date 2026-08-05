// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.capture;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import org.junit.Test;

import dev.herbert.bridgelogger.model.MatchContext;

/** Unit tests for {@link MatchContextParser}. */
public class MatchContextParserTest {

    @Test
    public void parse_nullLines_expectsEmptyContextNotThrow() {
        MatchContext context = MatchContextParser.parse(null, null);
        assertTrue(context.isEmpty());
    }

    @Test
    public void parse_emptyLines_expectsEmptyContext() {
        MatchContext context = MatchContextParser.parse(Collections.<String>emptyList(), null);
        assertTrue(context.isEmpty());
    }

    @Test
    public void parse_timerLine_expectsElapsedSecondsParsed() {
        MatchContext context = MatchContextParser.parse(Arrays.asList("Bridge Duels", "02:15", "Red: 3"), null);
        assertEquals(Integer.valueOf(135), context.elapsedSeconds);
    }

    @Test
    public void parse_singleDigitMinuteTimer_expectsElapsedSecondsParsed() {
        MatchContext context = MatchContextParser.parse(Collections.singletonList("2:05"), null);
        assertEquals(Integer.valueOf(125), context.elapsedSeconds);
    }

    @Test
    public void parse_kitLine_expectsKitParsed() {
        MatchContext context = MatchContextParser.parse(Collections.singletonList("Kit: Iron Man"), null);
        assertEquals("Iron Man", context.kit);
    }

    @Test
    public void parse_kitLineCaseInsensitivePrefix_expectsKitParsed() {
        MatchContext context = MatchContextParser.parse(Collections.singletonList("kit: Archer"), null);
        assertEquals("Archer", context.kit);
    }

    @Test
    public void parse_scoreLinesWithTeamHint_expectsOwnAndOpponentScoreAssignedCorrectly() {
        List<String> lines = Arrays.asList("Red: 3", "Blue: 1");
        MatchContext context = MatchContextParser.parse(lines, "RED");
        assertEquals(Integer.valueOf(3), context.ownScore);
        assertEquals(Integer.valueOf(1), context.opponentScore);
    }

    @Test
    public void parse_scoreLinesWithoutTeamHint_expectsPositionalFallback() {
        List<String> lines = Arrays.asList("Red: 3", "Blue: 1");
        MatchContext context = MatchContextParser.parse(lines, null);
        assertEquals(Integer.valueOf(3), context.ownScore);
        assertEquals(Integer.valueOf(1), context.opponentScore);
    }

    @Test
    public void parse_singleScoreLineNoHint_expectsScoresRemainNull() {
        // Positional fallback requires at least two score lines; one alone is ambiguous.
        MatchContext context = MatchContextParser.parse(Collections.singletonList("Red: 3"), null);
        assertNull(context.ownScore);
        assertNull(context.opponentScore);
    }

    @Test
    public void parse_garbageLines_expectsAllFieldsNullNotThrow() {
        List<String> lines = Arrays.asList("~~~~~~~~~~~~~~", "", "   ", "not a score or timer or kit line at all");
        MatchContext context = MatchContextParser.parse(lines, null);
        assertTrue(context.isEmpty());
    }

    @Test
    public void parse_listContainingNullElement_expectsSkippedNotThrow() {
        MatchContext context = MatchContextParser.parse(Arrays.asList("Kit: Archer", null, "02:00"), null);
        assertEquals("Archer", context.kit);
        assertEquals(Integer.valueOf(120), context.elapsedSeconds);
    }

    @Test
    public void parse_firstFieldOfEachTypeWins_expectsFirstNonNullKept() {
        // A duplicate timer/kit line further down the sidebar should not overwrite the first
        // one already parsed.
        MatchContext context = MatchContextParser.parse(Arrays.asList("Kit: Archer", "Kit: Knight", "01:00", "02:00"), null);
        assertEquals("Archer", context.kit);
        assertEquals(Integer.valueOf(60), context.elapsedSeconds);
    }
}
