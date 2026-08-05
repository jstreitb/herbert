// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.model;

/**
 * Coarse categorical classification of the item an opponent is currently holding, used in
 * {@link OpponentSnapshot#heldItemCategory}. Mirrors the small, stable vocabulary approach used
 * by {@link BlockCategory} so the training pipeline can treat it as a fixed categorical feature.
 */
public enum OpponentHeldItemCategory {

    /** Any sword item (wood/stone/iron/gold/diamond sword). */
    SWORD,

    /** A bow. */
    BOW,

    /** Any placeable block item (wool, blocks used for bridging/defending, etc.). */
    BLOCKS,

    /** Anything else (empty hand, tools, food, unrecognized items, etc.). */
    OTHER
}
