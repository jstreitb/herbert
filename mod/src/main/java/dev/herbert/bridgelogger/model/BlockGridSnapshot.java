package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Immutable snapshot of the local block grid sampled around the player for a single tick.
 *
 * <p>The grid is centered on the player's feet block. Cells are flattened into a single array
 * in row-major order with Y as the outermost axis, then Z, then X:</p>
 *
 * <pre>
 * index = (yIndex * depth + zIndex) * width + xIndex
 * </pre>
 *
 * <p>where {@code xIndex} in {@code [0, width)} maps to world offset {@code xIndex - width/2},
 * {@code yIndex} in {@code [0, height)} maps to world offset {@code yIndex - height/2}, and
 * {@code zIndex} in {@code [0, depth)} maps to world offset {@code zIndex - depth/2}, all
 * relative to the player's feet block position (integer division). See the README for a worked
 * example.</p>
 */
public final class BlockGridSnapshot {

    /** Grid width (X axis, blocks) used to produce {@link #cells}. */
    @SerializedName("width")
    public final int width;

    /** Grid height (Y axis, blocks) used to produce {@link #cells}. */
    @SerializedName("height")
    public final int height;

    /** Grid depth (Z axis, blocks) used to produce {@link #cells}. */
    @SerializedName("depth")
    public final int depth;

    /** Fixed documentation string describing where the grid is centered; always {@code "player_feet_centered"}. */
    @SerializedName("origin")
    public final String origin;

    /**
     * Flattened {@link BlockCategory} values, length {@code width * height * depth}, indexed as
     * documented on the class Javadoc. Serialized as the enum's {@code name()} string.
     */
    @SerializedName("cells")
    public final BlockCategory[] cells;

    /**
     * Creates an immutable block grid snapshot.
     *
     * @param width grid width in blocks, must be {@code > 0}
     * @param height grid height in blocks, must be {@code > 0}
     * @param depth grid depth in blocks, must be {@code > 0}
     * @param cells flattened category array of length {@code width * height * depth}; must not be {@code null}
     * @throws IllegalArgumentException if {@code cells.length != width * height * depth}
     */
    public BlockGridSnapshot(int width, int height, int depth, BlockCategory[] cells) {
        if (cells == null || cells.length != width * height * depth) {
            throw new IllegalArgumentException(
                    "cells length must equal width*height*depth (" + width + "*" + height + "*" + depth + ")");
        }
        this.width = width;
        this.height = height;
        this.depth = depth;
        this.origin = "player_feet_centered";
        this.cells = cells;
    }
}
