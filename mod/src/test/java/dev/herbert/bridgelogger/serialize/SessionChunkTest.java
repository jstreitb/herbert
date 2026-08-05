// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.serialize;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.fail;

import java.io.File;

import org.junit.Test;

/** Unit tests for {@link SessionChunk}. */
public class SessionChunkTest {

    @Test
    public void constructor_validArguments_expectsFieldsPopulated() {
        File file = new File("herbert_session_abc_part1of2.jsonl");
        SessionChunk chunk = new SessionChunk(file, 0, 2, 0L, 99L, 1024L);
        assertEquals(file, chunk.getFile());
        assertEquals(0, chunk.getChunkIndex());
        assertEquals(2, chunk.getChunkTotal());
        assertEquals(1, chunk.getPartNumber());
        assertEquals(0L, chunk.getFirstTick());
        assertEquals(99L, chunk.getLastTick());
        assertEquals(1024L, chunk.getSizeBytes());
    }

    @Test
    public void getPartNumber_lastChunk_expectsOneBasedIndex() {
        SessionChunk chunk = new SessionChunk(new File("x"), 4, 5, 0L, 0L, 1L);
        assertEquals(5, chunk.getPartNumber());
    }

    @Test
    public void constructor_headerOnlyChunk_expectsNoTicksSentinelAccepted() {
        SessionChunk chunk = new SessionChunk(new File("x"), 0, 1, SessionChunker.NO_TICKS, SessionChunker.NO_TICKS, 512L);
        assertEquals(SessionChunker.NO_TICKS, chunk.getFirstTick());
        assertEquals(SessionChunker.NO_TICKS, chunk.getLastTick());
    }

    @Test
    public void constructor_nullFile_expectsIllegalArgumentException() {
        try {
            new SessionChunk(null, 0, 1, 0L, 0L, 1L);
            fail("expected IllegalArgumentException for null file");
        } catch (IllegalArgumentException expected) {
            assertNotNull(expected.getMessage());
        }
    }

    @Test
    public void constructor_chunkTotalZero_expectsIllegalArgumentException() {
        try {
            new SessionChunk(new File("x"), 0, 0, 0L, 0L, 1L);
            fail("expected IllegalArgumentException for chunkTotal < 1");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void constructor_chunkIndexEqualToChunkTotal_expectsIllegalArgumentException() {
        // chunkIndex must be strictly less than chunkTotal (0-based, exclusive upper bound).
        try {
            new SessionChunk(new File("x"), 2, 2, 0L, 0L, 1L);
            fail("expected IllegalArgumentException for chunkIndex == chunkTotal");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void constructor_negativeChunkIndex_expectsIllegalArgumentException() {
        try {
            new SessionChunk(new File("x"), -1, 2, 0L, 0L, 1L);
            fail("expected IllegalArgumentException for negative chunkIndex");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }
}
