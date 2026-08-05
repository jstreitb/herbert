'use strict';

/**
 * Builds one tick's observation record, in the exact JSON shape of `herbert_rl.schema.TickRecordRL`
 * (which is itself field-for-field identical to `/mod`'s per-tick schema, minus the session
 * header -- see `mod/README.md`'s "JSONL schema" section and `herbert_rl/schema.py`'s docstring).
 */

const { computeBlockGrid } = require('./blockGrid');
const { mineflayerYawToDegrees, mineflayerPitchToDegrees } = require('./angles');
const { toRegistryId } = require('./itemClassifier');

/** Finds the nearest other player entity within `rangeBlocks`, mirroring `/mod`'s
 * `opponentTrackingEnabled` behavior (mod/README.md: "record the nearest opponent's relative
 * state each tick"; "null if no opponent is within tracking range (48 blocks)" by default). */
function findOpponentEntity(bot, rangeBlocks) {
  let nearest = null;
  let nearestDist = Infinity;
  for (const id of Object.keys(bot.entities)) {
    const entity = bot.entities[id];
    if (!entity || entity === bot.entity) continue;
    if (entity.type !== 'player') continue;
    if (entity.username === bot.username) continue;
    if (!entity.position || typeof entity.position.distanceTo !== 'function') continue;
    const dist = entity.position.distanceTo(bot.entity.position);
    if (dist <= rangeBlocks && dist < nearestDist) {
      nearest = entity;
      nearestDist = dist;
    }
  }
  return nearest;
}

/**
 * @param {import('mineflayer').Bot} bot
 * @param {object} ctx
 * @param {number} ctx.tick
 * @param {object} ctx.lastActionEcho - the `InputState`-shaped echo of the action just executed.
 * @param {ReturnType<typeof import('./matchState').createMatchStateTracker>} ctx.matchStateTracker
 * @param {ReturnType<typeof import('./itemClassifier').createItemClassifier>} ctx.itemClassifier
 * @param {{width:number,height:number,depth:number,voidThresholdY:number}} ctx.blockGridConfig
 * @param {number} ctx.opponentTrackingRangeBlocks
 */
function buildTickRecord(bot, ctx) {
  const { tick, lastActionEcho, matchStateTracker, itemClassifier, blockGridConfig, opponentTrackingRangeBlocks } = ctx;
  const pos = bot.entity.position;
  const vel = bot.entity.velocity || { x: 0, y: 0, z: 0 };

  const player = {
    x: pos.x,
    y: pos.y,
    z: pos.z,
    vx: vel.x,
    vy: vel.y,
    vz: vel.z,
    yaw: mineflayerYawToDegrees(bot.entity.yaw),
    pitch: mineflayerPitchToDegrees(bot.entity.pitch),
    on_ground: !!bot.entity.onGround,
    sneaking: !!(bot.controlState && bot.controlState.sneak),
    health: typeof bot.health === 'number' ? bot.health : 20.0,
    food: typeof bot.food === 'number' ? bot.food : 20,
  };

  const cells = computeBlockGrid(bot, blockGridConfig);
  const block_grid = {
    width: blockGridConfig.width,
    height: blockGridConfig.height,
    depth: blockGridConfig.depth,
    origin: 'player_feet_centered',
    cells,
  };

  const heldItem = bot.heldItem;
  const held_item = {
    hotbar_slot: typeof bot.quickBarSlot === 'number' ? bot.quickBarSlot : 0,
    item_id: heldItem ? toRegistryId(heldItem.name) : null,
    count: heldItem ? heldItem.count : 0,
  };

  const opponentEntity = findOpponentEntity(bot, opponentTrackingRangeBlocks);
  let opponent = null;
  if (opponentEntity) {
    const oPos = opponentEntity.position;
    const oVel = opponentEntity.velocity || { x: 0, y: 0, z: 0 };
    const heldOpponentItem = opponentEntity.heldItem;
    opponent = {
      rel_x: oPos.x - pos.x,
      rel_y: oPos.y - pos.y,
      rel_z: oPos.z - pos.z,
      rel_vx: oVel.x - vel.x,
      rel_vy: oVel.y - vel.y,
      rel_vz: oVel.z - vel.z,
      yaw: mineflayerYawToDegrees(opponentEntity.yaw || 0),
      pitch: mineflayerPitchToDegrees(opponentEntity.pitch || 0),
      // KNOWN LIMITATION: vanilla 1.8.9 does sync other living entities' health to observing
      // clients at the protocol level, but core Mineflayer does not expose a convenience
      // `entity.health` getter for entities other than the bot itself (confirmed by reading the
      // installed `mineflayer` version's source -- only `bot.health`, from the dedicated
      // `update_health` packet, is populated). Reading it properly means manually decoding the
      // opponent's raw `entity.metadata` at the 1.8.9 `EntityLivingBase` "Health" index, which
      // this bridge does not attempt (too easy to silently misparse and feed the policy wrong
      // data). This therefore reports a constant 20.0 (full health) unless `entity.health`
      // happens to be populated by a future Mineflayer version -- treat `opponent.health` as
      // currently non-informative and do not tune reward/behavior around it until this is
      // implemented properly.
      health: typeof opponentEntity.health === 'number' ? opponentEntity.health : 20.0,
      held_item_category: itemClassifier.classify(heldOpponentItem ? heldOpponentItem.name : null),
    };
  }

  const matchState = matchStateTracker.getMatchState();
  const chat = matchStateTracker.drainChatLines();

  return {
    tick,
    timestamp: new Date().toISOString(),
    player,
    block_grid,
    held_item,
    opponent,
    match: matchState,
    input: lastActionEcho,
    disconnected: false,
    chat,
  };
}

/** A placeholder record emitted when the bot is not currently connected (see
 * `bridge.js`'s reconnect handling) -- every field except `tick`/`disconnected` is a stale/inert
 * placeholder; the Python side must not treat this as a real observation (see
 * `herbert_rl/schema.py::TickRecordRL.disconnected` -- `MatchCoordinator.advance()` special-cases
 * `disconnected` records and skips feature-encoding them entirely). The block grid is still
 * shaped correctly (all `AIR`) as defense in depth, so a bug that *did* accidentally try to
 * feature-encode one fails on meaningless-but-shape-valid data rather than crashing on a shape
 * mismatch. */
function buildDisconnectedRecord(tick, lastActionEcho, blockGridConfig) {
  const { width, height, depth } = blockGridConfig;
  return {
    tick,
    timestamp: new Date().toISOString(),
    player: { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, yaw: 0, pitch: 0, on_ground: false, sneaking: false, health: 0, food: 0 },
    block_grid: { width, height, depth, origin: 'player_feet_centered', cells: new Array(width * height * depth).fill('AIR') },
    held_item: { hotbar_slot: 0, item_id: null, count: 0 },
    opponent: null,
    match: null,
    input: lastActionEcho,
    disconnected: true,
    chat: [],
  };
}

module.exports = { buildTickRecord, buildDisconnectedRecord, findOpponentEntity };
