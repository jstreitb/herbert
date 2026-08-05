// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Immutable snapshot of the nearest opponent's observable state, relative to the local player.
 *
 * <p>All positional/velocity fields are stored <em>relative</em> to the local player (opponent
 * minus self) rather than as absolute world coordinates, since relative geometry is what is
 * behaviorally relevant for imitation learning on a 1v1 duel.</p>
 */
public final class OpponentSnapshot {

    /** Opponent X position minus local player X position. */
    @SerializedName("rel_x")
    public final double relX;

    /** Opponent Y position minus local player Y position. */
    @SerializedName("rel_y")
    public final double relY;

    /** Opponent Z position minus local player Z position. */
    @SerializedName("rel_z")
    public final double relZ;

    /** Opponent velocity X minus local player velocity X, in blocks/tick. */
    @SerializedName("rel_vx")
    public final double relVx;

    /** Opponent velocity Y minus local player velocity Y, in blocks/tick. */
    @SerializedName("rel_vy")
    public final double relVy;

    /** Opponent velocity Z minus local player velocity Z, in blocks/tick. */
    @SerializedName("rel_vz")
    public final double relVz;

    /** Opponent's absolute yaw, in degrees. */
    @SerializedName("yaw")
    public final float yaw;

    /** Opponent's absolute pitch, in degrees. */
    @SerializedName("pitch")
    public final float pitch;

    /** Opponent's current health, 0-20. */
    @SerializedName("health")
    public final float health;

    /** Coarse category of whatever item the opponent currently has selected, never {@code null}. */
    @SerializedName("held_item_category")
    public final OpponentHeldItemCategory heldItemCategory;

    /**
     * Creates an immutable opponent snapshot.
     *
     * @param relX opponent X minus local player X
     * @param relY opponent Y minus local player Y
     * @param relZ opponent Z minus local player Z
     * @param relVx opponent velocity X minus local player velocity X
     * @param relVy opponent velocity Y minus local player velocity Y
     * @param relVz opponent velocity Z minus local player velocity Z
     * @param yaw opponent absolute yaw in degrees
     * @param pitch opponent absolute pitch in degrees
     * @param health opponent current health
     * @param heldItemCategory coarse category of the opponent's held item; must not be {@code null}
     */
    public OpponentSnapshot(double relX, double relY, double relZ, double relVx, double relVy, double relVz,
            float yaw, float pitch, float health, OpponentHeldItemCategory heldItemCategory) {
        this.relX = relX;
        this.relY = relY;
        this.relZ = relZ;
        this.relVx = relVx;
        this.relVy = relVy;
        this.relVz = relVz;
        this.yaw = yaw;
        this.pitch = pitch;
        this.health = health;
        this.heldItemCategory = heldItemCategory == null ? OpponentHeldItemCategory.OTHER : heldItemCategory;
    }
}
