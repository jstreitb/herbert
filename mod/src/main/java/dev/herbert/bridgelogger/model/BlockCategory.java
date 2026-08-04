package dev.herbert.bridgelogger.model;

/**
 * Coarse categorical classification of a single world block, as sampled into the local block
 * grid captured every tick by {@link dev.herbert.bridgelogger.capture.BlockGridMapper}.
 *
 * <p>This enum intentionally keeps a small, stable vocabulary so the resulting JSONL logs stay
 * easy to consume from the training pipeline (a fixed small categorical space is far easier to
 * one-hot/embed than raw Minecraft block IDs, which also change between versions).</p>
 */
public enum BlockCategory {

    /** Empty space (air) that is not classified as {@link #VOID}. */
    AIR,

    /**
     * A solid, full-cube block that is typical "bridge material" on Hypixel Bridge maps
     * (wool, clay, terracotta, planks, stone-family blocks, etc.) — i.e. a block a player could
     * plausibly stand on or that resembles player-placed bridge material.
     */
    SOLID_BRIDGEABLE,

    /** Water or lava, flowing or stationary. */
    LIQUID,

    /**
     * Air below the configured void threshold Y coordinate — i.e. the "you will die if you fall
     * here" open space beneath a Bridge map's islands. This is a best-effort heuristic since
     * Hypixel does not expose an explicit void block; see {@code voidThresholdY} in the config.
     */
    VOID,

    /** Any other solid block that is not classified as bridgeable (bedrock, glass, fences, stairs, slabs, etc.). */
    OTHER_SOLID
}
