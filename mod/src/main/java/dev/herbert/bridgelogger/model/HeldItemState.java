// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.model;

import com.google.gson.annotations.SerializedName;

/**
 * Immutable snapshot of what the local player currently has selected in their hotbar.
 */
public final class HeldItemState {

    /** Selected hotbar slot index, 0-8. */
    @SerializedName("hotbar_slot")
    public final int hotbarSlot;

    /**
     * Registry name of the held item/block (e.g. {@code "minecraft:wool"}), or {@code null}
     * if the hotbar slot is empty.
     */
    @SerializedName("item_id")
    public final String itemId;

    /** Remaining stack count of the held item, or 0 if the slot is empty. */
    @SerializedName("count")
    public final int count;

    /**
     * Creates an immutable held-item snapshot.
     *
     * @param hotbarSlot the selected hotbar slot index, 0-8
     * @param itemId registry name of the held item, or {@code null} if empty
     * @param count remaining stack size, or 0 if empty
     */
    public HeldItemState(int hotbarSlot, String itemId, int count) {
        this.hotbarSlot = hotbarSlot;
        this.itemId = itemId;
        this.count = count;
    }
}
