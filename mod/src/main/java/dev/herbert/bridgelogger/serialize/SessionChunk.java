// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.serialize;

import java.io.File;

/**
 * One self-contained split chunk of an oversized session JSONL file, as produced by
 * {@link SessionChunker#split(File, File, String)}.
 *
 * <p>Every field describes exactly one chunk file already written to disk: {@link #file} is a
 * complete, independently-parseable Herbert JSONL file (own copy of the session header plus a
 * contiguous, non-overlapping slice of the original tick lines).</p>
 */
public final class SessionChunk {

    private final File file;
    private final int chunkIndex;
    private final int chunkTotal;
    private final long firstTick;
    private final long lastTick;
    private final long sizeBytes;

    /**
     * @param file the chunk's JSONL file, already written to disk; must not be {@code null}
     * @param chunkIndex 0-based index of this chunk among {@code chunkTotal} chunks
     * @param chunkTotal total number of chunks the source session was split into; must be {@code >= 1}
     * @param firstTick the {@code tick} value of this chunk's first tick record, or {@code -1} if
     *        this chunk has no tick records (header-only)
     * @param lastTick the {@code tick} value of this chunk's last tick record, or {@code -1} if
     *        this chunk has no tick records (header-only)
     * @param sizeBytes this chunk file's size on disk, in bytes
     */
    public SessionChunk(File file, int chunkIndex, int chunkTotal, long firstTick, long lastTick, long sizeBytes) {
        if (file == null) {
            throw new IllegalArgumentException("file must not be null");
        }
        if (chunkTotal < 1) {
            throw new IllegalArgumentException("chunkTotal must be >= 1, got " + chunkTotal);
        }
        if (chunkIndex < 0 || chunkIndex >= chunkTotal) {
            throw new IllegalArgumentException("chunkIndex must be in [0, chunkTotal), got " + chunkIndex + " of " + chunkTotal);
        }
        this.file = file;
        this.chunkIndex = chunkIndex;
        this.chunkTotal = chunkTotal;
        this.firstTick = firstTick;
        this.lastTick = lastTick;
        this.sizeBytes = sizeBytes;
    }

    /** @return this chunk's JSONL file, already written to disk */
    public File getFile() {
        return file;
    }

    /** @return 0-based index of this chunk among {@link #getChunkTotal()} chunks */
    public int getChunkIndex() {
        return chunkIndex;
    }

    /** @return total number of chunks the source session was split into */
    public int getChunkTotal() {
        return chunkTotal;
    }

    /** @return 1-based chunk number, for player-facing chat/embed display (e.g. "part 2 of 5") */
    public int getPartNumber() {
        return chunkIndex + 1;
    }

    /** @return the {@code tick} value of this chunk's first tick record, or {@code -1} if none */
    public long getFirstTick() {
        return firstTick;
    }

    /** @return the {@code tick} value of this chunk's last tick record, or {@code -1} if none */
    public long getLastTick() {
        return lastTick;
    }

    /** @return this chunk file's size on disk, in bytes */
    public long getSizeBytes() {
        return sizeBytes;
    }
}
