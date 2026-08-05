// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.serialize;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.io.File;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.List;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;

/** Unit tests for {@link SessionChunker}. */
public class SessionChunkerTest {

    private static final String HEADER_LINE = "{\"schema_version\":\"1.2.0\",\"session_id\":\"abc-123\"}";

    @Rule
    public TemporaryFolder tempFolder = new TemporaryFolder();

    @Test
    public void constructor_nonPositiveTargetSize_expectsIllegalArgumentException() {
        try {
            new SessionChunker(0);
            fail("expected IllegalArgumentException for targetChunkSizeBytes <= 0");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void split_smallSessionFile_expectsExactlyOneChunk() throws IOException {
        File source = writeSourceFile("small.jsonl", HEADER_LINE, tickLine(0), tickLine(1), tickLine(2));
        SessionChunker chunker = new SessionChunker(1024 * 1024);

        List<SessionChunk> chunks = chunker.split(source, tempFolder.newFolder("out"), "abc-123");

        assertEquals(1, chunks.size());
        SessionChunk chunk = chunks.get(0);
        assertEquals(0, chunk.getChunkIndex());
        assertEquals(1, chunk.getChunkTotal());
        assertEquals(0L, chunk.getFirstTick());
        assertEquals(2L, chunk.getLastTick());
    }

    @Test
    public void split_headerOnlyZeroTickLines_expectsExactlyOneChunkWithNoTicksSentinel() throws IOException {
        // A valid header with zero subsequent tick lines is a degenerate case that should still
        // produce exactly one valid chunk, not zero chunks.
        File source = writeSourceFile("header_only.jsonl", HEADER_LINE);
        SessionChunker chunker = new SessionChunker(1024 * 1024);

        List<SessionChunk> chunks = chunker.split(source, tempFolder.newFolder("out2"), "abc-123");

        assertEquals(1, chunks.size());
        assertEquals(SessionChunker.NO_TICKS, chunks.get(0).getFirstTick());
        assertEquals(SessionChunker.NO_TICKS, chunks.get(0).getLastTick());
    }

    @Test
    public void split_manySmallLinesOverTarget_expectsMultipleChunksInOrder() throws IOException {
        // Force a tiny target so a handful of short tick lines must be split across several
        // chunks; verifies ordering and that every original tick appears exactly once.
        String[] lines = new String[20];
        lines[0] = HEADER_LINE;
        for (int i = 1; i <= 19; i++) {
            lines[i] = tickLine(i - 1);
        }
        File source = writeSourceFile("many.jsonl", lines);

        // Small enough that each chunk holds only a few tick lines.
        SessionChunker chunker = new SessionChunker(200);
        List<SessionChunk> chunks = chunker.split(source, tempFolder.newFolder("out3"), "abc-123");

        assertTrue("expected more than one chunk", chunks.size() > 1);
        for (int i = 0; i < chunks.size(); i++) {
            assertEquals(i, chunks.get(i).getChunkIndex());
            assertEquals(chunks.size(), chunks.get(i).getChunkTotal());
        }
        // First chunk starts at tick 0, last chunk ends at tick 18, and ticks are contiguous
        // across the chunk boundary (no gaps, no duplicates, no reordering).
        assertEquals(0L, chunks.get(0).getFirstTick());
        assertEquals(18L, chunks.get(chunks.size() - 1).getLastTick());
        for (int i = 1; i < chunks.size(); i++) {
            assertEquals(chunks.get(i - 1).getLastTick() + 1, chunks.get(i).getFirstTick());
        }
    }

    @Test
    public void split_singleLineLargerThanTarget_expectsItsOwnChunkNotDropped() throws IOException {
        // A single tick line that alone exceeds the target size must still be placed whole into
        // its own chunk rather than dropped or split mid-line.
        StringBuilder oversized = new StringBuilder("{\"tick\":0,\"padding\":\"");
        for (int i = 0; i < 500; i++) {
            oversized.append('x');
        }
        oversized.append("\"}");
        File source = writeSourceFile("oversized_line.jsonl", HEADER_LINE, oversized.toString(), tickLine(1));

        SessionChunker chunker = new SessionChunker(100);
        List<SessionChunk> chunks = chunker.split(source, tempFolder.newFolder("out4"), "abc-123");

        assertTrue(chunks.size() >= 2);
        List<String> firstChunkLines = Files.readAllLines(chunks.get(0).getFile().toPath(), StandardCharsets.UTF_8);
        assertTrue("oversized line must be written whole", firstChunkLines.get(1).contains("xxxxxxxxxx"));
    }

    @Test
    public void split_emptySourceFile_expectsIOException() throws IOException {
        File source = tempFolder.newFile("empty.jsonl");
        SessionChunker chunker = new SessionChunker(1024);
        try {
            chunker.split(source, tempFolder.newFolder("out5"), "abc-123");
            fail("expected IOException for an empty session file");
        } catch (IOException expected) {
            assertTrue(expected.getMessage().toLowerCase(java.util.Locale.ROOT).contains("empty"));
        }
    }

    @Test
    public void split_malformedHeaderLine_expectsIllegalArgumentException() throws IOException {
        File source = writeSourceFile("bad_header.jsonl", "not a json object at all", tickLine(0));
        SessionChunker chunker = new SessionChunker(1024);
        try {
            chunker.split(source, tempFolder.newFolder("out6"), "abc-123");
            fail("expected IllegalArgumentException for a header line with no closing brace");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void chunkFileName_expectsExpectedNamingConvention() {
        assertEquals("herbert_session_abc-123_part2of3.jsonl", SessionChunker.chunkFileName("abc-123", 2, 3));
    }

    @Test
    public void withChunkMetadata_expectsFieldsAppendedBeforeClosingBrace() {
        String result = SessionChunker.withChunkMetadata(HEADER_LINE, 1, 3);
        assertTrue(result.endsWith("\"chunk_index\":1,\"chunk_total\":3}"));
        assertTrue("original header content must be preserved", result.contains("\"session_id\":\"abc-123\""));
    }

    @Test
    public void withChunkMetadata_headerWithoutClosingBrace_expectsIllegalArgumentException() {
        try {
            SessionChunker.withChunkMetadata("not json", 0, 1);
            fail("expected IllegalArgumentException");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    @Test
    public void extractTick_validLine_expectsParsedValue() {
        assertEquals(42L, SessionChunker.extractTick("{\"tick\": 42, \"other\": 1}"));
    }

    @Test
    public void extractTick_missingTickField_expectsIllegalArgumentException() {
        try {
            SessionChunker.extractTick("{\"other\": 1}");
            fail("expected IllegalArgumentException for a line with no tick field");
        } catch (IllegalArgumentException expected) {
            // expected
        }
    }

    private static String tickLine(int tick) {
        return "{\"tick\":" + tick + ",\"data\":\"x\"}";
    }

    private File writeSourceFile(String name, String... lines) throws IOException {
        File file = tempFolder.newFile(name);
        StringBuilder content = new StringBuilder();
        for (String line : lines) {
            content.append(line).append('\n');
        }
        Files.write(file.toPath(), content.toString().getBytes(StandardCharsets.UTF_8));
        return file;
    }
}
