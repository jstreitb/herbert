// SPDX-License-Identifier: MIT
'use strict';

/**
 * Classifies an item name into `/mod`'s coarse `HeldItemCategory` enum (`SWORD | BOW | BLOCKS |
 * OTHER`), used for the opponent's held item (see `mod/README.md`'s schema table:
 * `opponent.held_item_category`). Uses `minecraft-data`'s item/block tables rather than a
 * hand-maintained name list where possible, so it stays correct across resource-pack-agnostic
 * vanilla item names without needing per-item maintenance.
 */

function createItemClassifier(mcData) {
  function classify(itemName) {
    if (!itemName) return 'OTHER';
    const name = itemName.toLowerCase();
    if (name.includes('sword')) return 'SWORD';
    if (name.includes('bow') && !name.includes('bowl')) return 'BOW';
    if (mcData.blocksByName && mcData.blocksByName[name]) return 'BLOCKS';
    return 'OTHER';
  }
  return { classify };
}

/** Best-effort mapping from a minecraft-data item name to a Forge-style registry id
 * (`"minecraft:<name>"`), matching the format `/mod` writes for `held_item.item_id` /
 * `input.place_block_type` (see `mod/README.md`'s schema table). This is an approximation: some
 * items/blocks have Forge registry names that differ from their vanilla `minecraft-data` name
 * (rare for vanilla-only blocks, which is all a vanilla/Spigot/PaperMC 1.8.9 server can place
 * anyway), so treat exact string equality against `/nn`-side vocab entries fit from real `/mod`
 * logs as approximate for RL-collected data. */
function toRegistryId(itemName) {
  if (!itemName) return null;
  return `minecraft:${itemName}`;
}

module.exports = { createItemClassifier, toRegistryId };
