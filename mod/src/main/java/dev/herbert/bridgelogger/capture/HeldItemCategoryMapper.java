// SPDX-License-Identifier: MIT
package dev.herbert.bridgelogger.capture;

import dev.herbert.bridgelogger.model.OpponentHeldItemCategory;
import net.minecraft.item.Item;
import net.minecraft.item.ItemBlock;
import net.minecraft.item.ItemBow;
import net.minecraft.item.ItemSword;

/**
 * Maps a Minecraft {@link Item} to the coarse {@link OpponentHeldItemCategory} vocabulary used
 * when recording an opponent's held item.
 *
 * <p>Kept as a small, pure, independently testable class following the same pattern as
 * {@link BlockGridMapper}: given an {@link Item}, {@link #mapCategory(Item)} always returns the
 * same result with no dependency on a running client.</p>
 */
public final class HeldItemCategoryMapper {

    private HeldItemCategoryMapper() {
        // Static utility class; never instantiated.
    }

    /**
     * Classifies an item into a coarse held-item category.
     *
     * @param item the item to classify, as returned by {@code ItemStack.getItem()}; may be {@code null} (empty hand)
     * @return {@link OpponentHeldItemCategory#SWORD}, {@link OpponentHeldItemCategory#BOW},
     *         {@link OpponentHeldItemCategory#BLOCKS} if the item is a placeable block, or
     *         {@link OpponentHeldItemCategory#OTHER} otherwise (including empty hand). Never {@code null}.
     */
    public static OpponentHeldItemCategory mapCategory(Item item) {
        if (item == null) {
            return OpponentHeldItemCategory.OTHER;
        }
        if (item instanceof ItemSword) {
            return OpponentHeldItemCategory.SWORD;
        }
        if (item instanceof ItemBow) {
            return OpponentHeldItemCategory.BOW;
        }
        if (item instanceof ItemBlock) {
            return OpponentHeldItemCategory.BLOCKS;
        }
        return OpponentHeldItemCategory.OTHER;
    }
}
