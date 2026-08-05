// SPDX-License-Identifier: MIT
'use strict';

/**
 * Local block-grid classification, matching `/mod`'s schema exactly (see `mod/README.md`'s
 * "JSONL schema" section, `block_grid.cells`): a flattened `width*height*depth` array of
 * `AIR | SOLID_BRIDGEABLE | LIQUID | VOID | OTHER_SOLID`, centered on the player's feet block,
 * with `index = (yIndex * depth + zIndex) * width + xIndex` and `xIndex/zIndex` offsets running
 * `-floor(width/2)..floor(width/2)` (similarly for `yIndex`/height) from the feet block.
 *
 * **Heuristic, needs per-server tuning** (same spirit as `/mod`'s own `voidThresholdY` /
 * `bridgeScoreboardTitleMatches` config -- see `mod/README.md`'s "Configuration" table):
 * `SOLID_BRIDGEABLE` vs `OTHER_SOLID` is decided by `OTHER_SOLID_BLOCK_NAMES` below, a denylist
 * of "structural, not meant to be placed/bridged with" block names (bedrock, barriers, portal
 * frames, ...) -- everything else solid is treated as bridgeable. `/mod`'s own exact
 * classification logic isn't published (closed-source heuristic in the Forge mod), so this is an
 * independent best-effort re-implementation, not a verified port; tune `OTHER_SOLID_BLOCK_NAMES`
 * for your own server/map if the RL agent's block-grid observations look wrong in practice.
 */

const Vec3 = require('vec3');

const OTHER_SOLID_BLOCK_NAMES = new Set([
  'bedrock',
  'barrier',
  'portal',
  'end_portal',
  'end_portal_frame',
  'end_gateway',
  'command_block',
  'structure_block',
  'mob_spawner',
]);

const LIQUID_NAME_PATTERN = /water|lava/;

function classifyBlock(block, worldY, voidThresholdY) {
  if (!block || block.name === 'air' || block.boundingBox === 'empty') {
    return worldY <= voidThresholdY ? 'VOID' : 'AIR';
  }
  if (LIQUID_NAME_PATTERN.test(block.name)) {
    return 'LIQUID';
  }
  if (OTHER_SOLID_BLOCK_NAMES.has(block.name)) {
    return 'OTHER_SOLID';
  }
  return 'SOLID_BRIDGEABLE';
}

/**
 * Compute the flattened block-grid cell array for the current tick.
 *
 * @param {import('mineflayer').Bot} bot
 * @param {{width: number, height: number, depth: number, voidThresholdY: number}} config
 * @returns {string[]} length `width*height*depth`, in the exact index order documented above.
 */
function computeBlockGrid(bot, config) {
  const { width, height, depth, voidThresholdY } = config;
  const feetX = Math.floor(bot.entity.position.x);
  const feetY = Math.floor(bot.entity.position.y);
  const feetZ = Math.floor(bot.entity.position.z);
  const halfW = Math.floor(width / 2);
  const halfH = Math.floor(height / 2);
  const halfD = Math.floor(depth / 2);

  const cells = new Array(width * height * depth);
  for (let yIndex = 0; yIndex < height; yIndex++) {
    const worldY = feetY + (yIndex - halfH);
    for (let zIndex = 0; zIndex < depth; zIndex++) {
      const worldZ = feetZ + (zIndex - halfD);
      for (let xIndex = 0; xIndex < width; xIndex++) {
        const worldX = feetX + (xIndex - halfW);
        const block = bot.blockAt(new Vec3(worldX, worldY, worldZ), false);
        const index = (yIndex * depth + zIndex) * width + xIndex;
        cells[index] = classifyBlock(block, worldY, voidThresholdY);
      }
    }
  }
  return cells;
}

module.exports = { computeBlockGrid, classifyBlock };
