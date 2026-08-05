// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.serialize;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Splits an oversized, already-completed session JSONL file into one or more self-contained
 * chunk files, each independently parseable by {@code /bot}'s {@code SessionValidator} and
 * {@code /nn}'s data pipeline without needing any of the other chunks (see {@code mod/README.md}'s
 * JSONL schema section).
 *
 * <p>Contains no Minecraft/Forge types or static state -- every method is a pure function of its
 * arguments (or, for {@link #split(File, File, String)}, of its arguments plus the filesystem),
 * so this class is independently constructible and testable without any game/mod scaffolding.</p>
 *
 * <p><b>Design invariants:</b></p>
 * <ul>
 *   <li>The session header line (the source file's first line) is copied into <i>every</i> chunk,
 *       so each chunk file is a complete, valid Herbert JSONL file on its own.</li>
 *   <li>Splitting only ever occurs on tick-record (line) boundaries -- a single JSON line is
 *       never split across two chunk files.</li>
 *   <li>Chunk boundaries are computed from the accumulated <b>UTF-8 byte length</b> of each line,
 *       not its character count, since it is the on-disk/on-the-wire byte size that must stay
 *       under the target (a JSONL line with multi-byte UTF-8 characters has more bytes than
 *       characters).</li>
 *   <li>The chunk header line is produced by copying the original header line byte-for-byte and
 *       appending the {@code chunk_index}/{@code chunk_total} fields to that copy (see
 *       {@link #withChunkMetadata(String, int, int)}) -- never by re-parsing and re-serializing
 *       the header as a new JSON object, to avoid any risk of field reordering or precision loss.
 *       {@code session_id} is <i>not</i> re-added as a chunk field: it is already present,
 *       identical, in the copied original header, so appending it again would create a duplicate
 *       JSON key rather than adding new information.</li>
 * </ul>
 */
public final class SessionChunker {

    /** Sentinel {@link SessionChunk#getFirstTick()}/{@link SessionChunk#getLastTick()} value for a chunk with no tick records. */
    public static final long NO_TICKS = -1L;

    /** Matches the {@code "tick": <n>} field of a per-tick JSONL line, to read (never rewrite) its value. */
    private static final Pattern TICK_FIELD_PATTERN = Pattern.compile("\"tick\"\\s*:\\s*(-?\\d+)");

    private final int targetChunkSizeBytes;

    /**
     * @param targetChunkSizeBytes target size, in bytes, for each produced chunk file (including
     *        its copy of the header line); must be positive. See
     *        {@code HerbertConstants.DEFAULT_TARGET_CHUNK_SIZE_BYTES} for the mod's configured default.
     */
    public SessionChunker(int targetChunkSizeBytes) {
        if (targetChunkSizeBytes <= 0) {
            throw new IllegalArgumentException("targetChunkSizeBytes must be positive, got " + targetChunkSizeBytes);
        }
        this.targetChunkSizeBytes = targetChunkSizeBytes;
    }

    /**
     * Splits {@code sourceFile} into one or more self-contained JSONL chunk files written into
     * {@code outputDir} (created if necessary), each starting with a chunk-metadata-augmented
     * copy of {@code sourceFile}'s header line.
     *
     * @param sourceFile the completed session JSONL file to split; must exist, be non-empty, and
     *        have a valid JSON-object header as its first line
     * @param outputDir directory to write chunk files into; created (including parents) if it
     *        does not already exist
     * @param sessionId the session's UUID, used only to build chunk filenames (see
     *        {@link #chunkFileName(String, int, int)}) -- the header's own {@code session_id}
     *        field is untouched, copied verbatim from {@code sourceFile}
     * @return the produced chunks, in order, each already written to disk
     * @throws IOException if {@code sourceFile} cannot be read, is empty, its first line is not a
     *         JSON object, {@code outputDir} cannot be created, or a chunk file cannot be written
     */
    public List<SessionChunk> split(File sourceFile, File outputDir, String sessionId) throws IOException {
        if (sourceFile == null) {
            throw new IllegalArgumentException("sourceFile must not be null");
        }
        if (outputDir == null) {
            throw new IllegalArgumentException("outputDir must not be null");
        }
        if (sessionId == null || sessionId.isEmpty()) {
            throw new IllegalArgumentException("sessionId must not be null or empty");
        }

        List<String> lines = Files.readAllLines(sourceFile.toPath(), StandardCharsets.UTF_8);
        if (lines.isEmpty()) {
            throw new IOException("Session file is empty, cannot be chunked: " + sourceFile.getAbsolutePath());
        }
        String headerLine = lines.get(0);
        // Fail fast on a malformed header rather than silently producing chunks with a broken
        // header line; withChunkMetadata performs the same check, but doing it once up front
        // here avoids partially writing chunk files before discovering the problem.
        withChunkMetadata(headerLine, 0, 1);

        List<String> tickLines = lines.subList(1, lines.size());
        List<List<String>> groups = groupByByteSize(tickLines, utf8ByteLength(headerLine));
        if (groups.isEmpty()) {
            // A header with zero tick lines is a degenerate case (a session this method would
            // normally never be asked to split), but still produce exactly one valid chunk
            // rather than zero.
            groups.add(Collections.<String>emptyList());
        }
        int total = groups.size();

        if (!outputDir.exists() && !outputDir.mkdirs()) {
            throw new IOException("Could not create chunk output directory: " + outputDir.getAbsolutePath());
        }

        List<SessionChunk> chunks = new ArrayList<SessionChunk>(total);
        for (int i = 0; i < total; i++) {
            List<String> group = groups.get(i);
            String chunkHeaderLine = withChunkMetadata(headerLine, i, total);
            long firstTick = group.isEmpty() ? NO_TICKS : extractTick(group.get(0));
            long lastTick = group.isEmpty() ? NO_TICKS : extractTick(group.get(group.size() - 1));
            File chunkFile = new File(outputDir, chunkFileName(sessionId, i + 1, total));
            writeChunkFile(chunkFile, chunkHeaderLine, group);
            chunks.add(new SessionChunk(chunkFile, i, total, firstTick, lastTick, chunkFile.length()));
        }
        return chunks;
    }

    /**
     * Builds a chunk's filename: {@code herbert_session_{session_id}_part{N}of{T}.jsonl}.
     *
     * @param sessionId the session's UUID
     * @param partNumber 1-based chunk number
     * @param chunkTotal total number of chunks
     * @return the chunk filename (no directory component)
     */
    public static String chunkFileName(String sessionId, int partNumber, int chunkTotal) {
        return "herbert_session_" + sessionId + "_part" + partNumber + "of" + chunkTotal + ".jsonl";
    }

    /**
     * Returns a copy of {@code headerLine} with {@code chunk_index}/{@code chunk_total} appended
     * as two additional top-level JSON fields, by locating the line's final {@code '}'} and
     * splicing the new fields in immediately before it -- the original text before that point is
     * untouched (see the class javadoc for why this is deliberately not JSON re-serialization).
     *
     * @param headerLine the original session header line (a single-line JSON object)
     * @param chunkIndex 0-based chunk index to embed
     * @param chunkTotal total chunk count to embed
     * @return the augmented header line, still a single line, no trailing newline
     * @throws IllegalArgumentException if {@code headerLine} does not contain a {@code '}'}
     *         (i.e. is not a JSON object)
     */
    public static String withChunkMetadata(String headerLine, int chunkIndex, int chunkTotal) {
        String trimmed = headerLine.trim();
        int lastBrace = trimmed.lastIndexOf('}');
        if (lastBrace < 0) {
            throw new IllegalArgumentException("Session header line is not a JSON object: " + headerLine);
        }
        String withoutClosingBrace = trimmed.substring(0, lastBrace);
        return withoutClosingBrace + ",\"chunk_index\":" + chunkIndex + ",\"chunk_total\":" + chunkTotal + "}";
    }

    /**
     * Reads the {@code tick} field's value out of a single per-tick JSONL line, without parsing
     * or re-emitting the line as a whole (the line itself is always written to its chunk file
     * verbatim; this is only ever used to populate {@link SessionChunk#getFirstTick()}/
     * {@link SessionChunk#getLastTick()} for chat/embed display).
     *
     * @param tickLine a single per-tick JSONL line
     * @return the value of that line's {@code "tick"} field
     * @throws IllegalArgumentException if {@code tickLine} has no {@code "tick"} field
     */
    public static long extractTick(String tickLine) {
        Matcher matcher = TICK_FIELD_PATTERN.matcher(tickLine);
        if (!matcher.find()) {
            throw new IllegalArgumentException("Tick line has no \"tick\" field: " + tickLine);
        }
        return Long.parseLong(matcher.group(1));
    }

    /**
     * Groups {@code tickLines} into consecutive runs whose accumulated UTF-8 byte size (plus
     * {@code headerOverheadBytes} and a one-byte-per-line newline allowance) stays at or under
     * {@link #targetChunkSizeBytes} wherever possible. A single line that alone exceeds the
     * target is still placed whole into its own group rather than dropped or split.
     */
    private List<List<String>> groupByByteSize(List<String> tickLines, int headerOverheadBytes) {
        List<List<String>> groups = new ArrayList<List<String>>();
        List<String> current = new ArrayList<String>();
        long currentBytes = headerOverheadBytes;
        for (String line : tickLines) {
            long lineBytes = utf8ByteLength(line) + 1L; // +1 for the line's trailing newline byte
            if (!current.isEmpty() && currentBytes + lineBytes > targetChunkSizeBytes) {
                groups.add(current);
                current = new ArrayList<String>();
                currentBytes = headerOverheadBytes;
            }
            current.add(line);
            currentBytes += lineBytes;
        }
        if (!current.isEmpty()) {
            groups.add(current);
        }
        return groups;
    }

    private static int utf8ByteLength(String value) {
        return value.getBytes(StandardCharsets.UTF_8).length;
    }

    private void writeChunkFile(File chunkFile, String headerLine, List<String> tickLines) throws IOException {
        try (BufferedWriter writer =
                new BufferedWriter(new OutputStreamWriter(new FileOutputStream(chunkFile), StandardCharsets.UTF_8))) {
            writer.write(headerLine);
            writer.write("\n");
            for (String line : tickLines) {
                writer.write(line);
                writer.write("\n");
            }
        }
    }
}
