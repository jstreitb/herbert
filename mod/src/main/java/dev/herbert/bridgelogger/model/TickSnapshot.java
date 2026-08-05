// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Immutable, fully-composed record of everything BridgeLogger observes for a single game tick.
 * This is the exact object serialized as one JSON line by
 * {@link dev.herbert.bridgelogger.serialize.SessionSerializer}.
 */
public final class TickSnapshot {

    /** Monotonically increasing sample index within the session (not the raw game tick counter). */
    @SerializedName("tick")
    public final long tick;

    /** Wall-clock ISO-8601 timestamp (UTC) at which this tick was sampled. */
    @SerializedName("timestamp")
    public final String timestamp;

    /** Local player physical state. Never {@code null}. */
    @SerializedName("player")
    public final PlayerState player;

    /** Local block grid around the player. Never {@code null}. */
    @SerializedName("block_grid")
    public final BlockGridSnapshot blockGrid;

    /** What the local player currently has selected in their hotbar. Never {@code null}. */
    @SerializedName("held_item")
    public final HeldItemState heldItem;

    /** Nearest opponent snapshot, or {@code null} if no opponent is within tracking range/opponent tracking is disabled. */
    @SerializedName("opponent")
    public final OpponentSnapshot opponent;

    /** Best-effort scoreboard-derived match context, or {@code null} if unavailable/disabled/empty. */
    @SerializedName("match")
    public final MatchContext match;

    /** Local player input state for this tick. Never {@code null}. */
    @SerializedName("input")
    public final InputSnapshot input;

    /**
     * Creates an immutable tick snapshot.
     *
     * @param tick session-relative sample index
     * @param timestamp ISO-8601 UTC timestamp of this sample
     * @param player local player physical state; must not be {@code null}
     * @param blockGrid local block grid; must not be {@code null}
     * @param heldItem local hotbar selection; must not be {@code null}
     * @param opponent nearest opponent snapshot, or {@code null}
     * @param match best-effort match context, or {@code null}
     * @param input local input state; must not be {@code null}
     */
    public TickSnapshot(long tick, String timestamp, PlayerState player, BlockGridSnapshot blockGrid,
            HeldItemState heldItem, OpponentSnapshot opponent, MatchContext match, InputSnapshot input) {
        this.tick = tick;
        this.timestamp = timestamp;
        this.player = player;
        this.blockGrid = blockGrid;
        this.heldItem = heldItem;
        this.opponent = opponent;
        this.match = match;
        this.input = input;
    }
}
