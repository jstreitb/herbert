// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Immutable snapshot of the local player's physical state at a single tick.
 *
 * <p>All fields are read-only observations of client-side state that Minecraft already
 * tracks for the local player (position, velocity, rotation, vitals) — nothing here is
 * synthesized or injected back into the game.</p>
 */
public final class PlayerState {

    /** World X coordinate. */
    @SerializedName("x")
    public final double x;

    /** World Y coordinate. */
    @SerializedName("y")
    public final double y;

    /** World Z coordinate. */
    @SerializedName("z")
    public final double z;

    /** Velocity along X, in blocks/tick, computed as the difference from the previous tick's position. */
    @SerializedName("vx")
    public final double vx;

    /** Velocity along Y, in blocks/tick. */
    @SerializedName("vy")
    public final double vy;

    /** Velocity along Z, in blocks/tick. */
    @SerializedName("vz")
    public final double vz;

    /** Yaw rotation in degrees, Minecraft convention (-180..180, 0 = south). */
    @SerializedName("yaw")
    public final float yaw;

    /** Pitch rotation in degrees (-90 = straight up, 90 = straight down). */
    @SerializedName("pitch")
    public final float pitch;

    /** Whether the player is currently touching the ground. */
    @SerializedName("on_ground")
    public final boolean onGround;

    /** Whether the player is currently sneaking. */
    @SerializedName("sneaking")
    public final boolean sneaking;

    /** Current health, 0-20 (half-hearts resolution, but stored as the float Minecraft uses). */
    @SerializedName("health")
    public final float health;

    /** Current food/hunger level, 0-20. */
    @SerializedName("food")
    public final int food;

    /**
     * Creates an immutable player state snapshot.
     *
     * @param x world X coordinate
     * @param y world Y coordinate
     * @param z world Z coordinate
     * @param vx velocity along X in blocks/tick
     * @param vy velocity along Y in blocks/tick
     * @param vz velocity along Z in blocks/tick
     * @param yaw yaw rotation in degrees
     * @param pitch pitch rotation in degrees
     * @param onGround whether the player is currently touching the ground
     * @param sneaking whether the player is currently sneaking
     * @param health current health points
     * @param food current food level
     */
    public PlayerState(double x, double y, double z, double vx, double vy, double vz, float yaw, float pitch,
            boolean onGround, boolean sneaking, float health, int food) {
        this.x = x;
        this.y = y;
        this.z = z;
        this.vx = vx;
        this.vy = vy;
        this.vz = vz;
        this.yaw = yaw;
        this.pitch = pitch;
        this.onGround = onGround;
        this.sneaking = sneaking;
        this.health = health;
        this.food = food;
    }
}
