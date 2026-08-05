// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Immutable snapshot of the local player's input state for a single tick, captured purely for
 * logging purposes.
 *
 * <p><b>This data is never re-injected into the game.</b> It is read from state Minecraft
 * already computes for the local player (movement input axes, rotation, interaction events)
 * and is recorded strictly as an observation of what the human player did that tick.</p>
 */
public final class InputSnapshot {

    /** Forward/backward movement axis: -1 (back), 0 (none), or 1 (forward). */
    @SerializedName("forward")
    public final int forward;

    /** Strafe movement axis: -1 (left), 0 (none), or 1 (right). */
    @SerializedName("strafe")
    public final int strafe;

    /** Whether the jump key was held this tick. */
    @SerializedName("jump")
    public final boolean jump;

    /** Whether the sneak key was held this tick. */
    @SerializedName("sneak")
    public final boolean sneak;

    /** Change in yaw (degrees) since the previous sampled tick. */
    @SerializedName("delta_yaw")
    public final float deltaYaw;

    /** Change in pitch (degrees) since the previous sampled tick. */
    @SerializedName("delta_pitch")
    public final float deltaPitch;

    /** Whether a left-click/attack event occurred this tick. */
    @SerializedName("attack_occurred")
    public final boolean attackOccurred;

    /**
     * Coarse type of whatever was hit by the attack (e.g. {@code "EntityOtherPlayerMP"}), or
     * {@code null} if {@link #attackOccurred} is {@code false} or the target type is unknown.
     */
    @SerializedName("attack_target_type")
    public final String attackTargetType;

    /** Whether a right-click block-place event occurred this tick. */
    @SerializedName("place_occurred")
    public final boolean placeOccurred;

    /**
     * Registry name of the block that was placed, or {@code null} if {@link #placeOccurred} is
     * {@code false} or the block type could not be resolved.
     */
    @SerializedName("place_block_type")
    public final String placeBlockType;

    /** World X coordinate the block was placed at, or {@code null} if not resolvable. */
    @SerializedName("place_x")
    public final Integer placeX;

    /** World Y coordinate the block was placed at, or {@code null} if not resolvable. */
    @SerializedName("place_y")
    public final Integer placeY;

    /** World Z coordinate the block was placed at, or {@code null} if not resolvable. */
    @SerializedName("place_z")
    public final Integer placeZ;

    /**
     * Creates an immutable input snapshot.
     *
     * @param forward forward/backward axis, -1/0/1
     * @param strafe strafe axis, -1/0/1
     * @param jump whether jump was held
     * @param sneak whether sneak was held
     * @param deltaYaw change in yaw since the previous sampled tick
     * @param deltaPitch change in pitch since the previous sampled tick
     * @param attackOccurred whether an attack event happened this tick
     * @param attackTargetType coarse target type name, or {@code null}
     * @param placeOccurred whether a block-place event happened this tick
     * @param placeBlockType registry name of the placed block, or {@code null}
     * @param placeX placement X coordinate, or {@code null}
     * @param placeY placement Y coordinate, or {@code null}
     * @param placeZ placement Z coordinate, or {@code null}
     */
    public InputSnapshot(int forward, int strafe, boolean jump, boolean sneak, float deltaYaw, float deltaPitch,
            boolean attackOccurred, String attackTargetType, boolean placeOccurred, String placeBlockType,
            Integer placeX, Integer placeY, Integer placeZ) {
        this.forward = forward;
        this.strafe = strafe;
        this.jump = jump;
        this.sneak = sneak;
        this.deltaYaw = deltaYaw;
        this.deltaPitch = deltaPitch;
        this.attackOccurred = attackOccurred;
        this.attackTargetType = attackTargetType;
        this.placeOccurred = placeOccurred;
        this.placeBlockType = placeBlockType;
        this.placeX = placeX;
        this.placeY = placeY;
        this.placeZ = placeZ;
    }
}
