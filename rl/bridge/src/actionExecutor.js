'use strict';

/**
 * Applies one tick's `ActionCommand` (see `herbert_rl/env/ipc.py`'s module docstring for the
 * wire format) to the bot's movement/look/attack/place state, and returns an `InputState`-shaped
 * echo of what actually happened -- which can differ from the requested intent (e.g. `attack:
 * true` but nothing was in range to swing at meaningfully, or `place: true` but no valid
 * placement target was in the crosshair), exactly mirroring `/mod`'s distinction between
 * requested and recorded input.
 */

const Vec3 = require('vec3');
const {
  mineflayerYawToDegrees,
  mineflayerPitchToDegrees,
  degreesToMineflayerYaw,
  degreesToMineflayerPitch,
  wrapDegrees,
} = require('./angles');
const { toRegistryId } = require('./itemClassifier');

const MAX_PITCH_DEG = 90;
const MIN_PITCH_DEG = -90;
const PLACE_REACH_BLOCKS = 4;

/** Mineflayer's `blockAtCursor`/`Block.face` index -> unit direction vector (down, up,
 * north(-z), south(+z), west(-x), east(+x)). Verified directly against the pinned
 * `prismarine-world`'s `BlockFace` enum (`iterators.js`) for the dependency versions in
 * `package.json` -- re-check this mapping if you bump the `mineflayer`/`prismarine-world`
 * version and placements start landing one block off. */
const FACE_VECTORS = [
  new Vec3(0, -1, 0),
  new Vec3(0, 1, 0),
  new Vec3(0, 0, -1),
  new Vec3(0, 0, 1),
  new Vec3(-1, 0, 0),
  new Vec3(1, 0, 0),
];

function clampPitch(deg) {
  return Math.max(MIN_PITCH_DEG, Math.min(MAX_PITCH_DEG, deg));
}

/**
 * @param {import('mineflayer').Bot} bot
 * @param {{forward:number, strafe:number, jump:boolean, sneak:boolean, delta_yaw:number,
 *   delta_pitch:number, attack:boolean, place:boolean}} action
 * @param {{findOpponentEntity: (bot: any, range: number) => any, opponentTrackingRangeBlocks: number}} ctx
 * @returns {Promise<object>} an `InputState`-shaped echo.
 */
async function applyAction(bot, action, ctx) {
  bot.setControlState('forward', action.forward === 1);
  bot.setControlState('back', action.forward === -1);
  bot.setControlState('right', action.strafe === 1);
  bot.setControlState('left', action.strafe === -1);
  bot.setControlState('jump', !!action.jump);
  bot.setControlState('sneak', !!action.sneak);

  const newYawDeg = wrapDegrees(mineflayerYawToDegrees(bot.entity.yaw) + action.delta_yaw);
  const newPitchDeg = clampPitch(mineflayerPitchToDegrees(bot.entity.pitch) + action.delta_pitch);
  // bot.look() is async; awaited even though `force: true` resolves synchronously in current
  // Mineflayer versions, so a future version that makes the force path genuinely async doesn't
  // silently turn into an unhandled promise rejection here.
  await bot.look(degreesToMineflayerYaw(newYawDeg), degreesToMineflayerPitch(newPitchDeg), true);

  const { attackOccurred, attackTargetType } = tryAttack(bot, action, ctx);
  const placeResult = await tryPlace(bot, action);

  return {
    forward: action.forward,
    strafe: action.strafe,
    jump: !!action.jump,
    sneak: !!action.sneak,
    delta_yaw: action.delta_yaw,
    delta_pitch: action.delta_pitch,
    attack_occurred: attackOccurred,
    attack_target_type: attackTargetType,
    place_occurred: placeResult.placeOccurred,
    place_block_type: placeResult.placeBlockType,
    place_x: placeResult.placeX,
    place_y: placeResult.placeY,
    place_z: placeResult.placeZ,
  };
}

function tryAttack(bot, action, ctx) {
  if (!action.attack) {
    return { attackOccurred: false, attackTargetType: null };
  }
  try {
    const target = ctx.findOpponentEntity(bot, ctx.opponentTrackingRangeBlocks);
    if (target) {
      bot.attack(target);
      return { attackOccurred: true, attackTargetType: 'EntityOtherPlayerMP' };
    }
    bot.swingArm('right');
    return { attackOccurred: true, attackTargetType: null };
  } catch (err) {
    return { attackOccurred: false, attackTargetType: null };
  }
}

async function tryPlace(bot, action) {
  const empty = { placeOccurred: false, placeBlockType: null, placeX: null, placeY: null, placeZ: null };
  if (!action.place) return empty;
  if (!bot.heldItem) return empty;
  try {
    const cursor = typeof bot.blockAtCursor === 'function' ? bot.blockAtCursor(PLACE_REACH_BLOCKS) : null;
    if (!cursor || !cursor.position || cursor.face === undefined || cursor.face === null) {
      return empty;
    }
    const faceVector = FACE_VECTORS[cursor.face];
    if (!faceVector) return empty;
    await bot.placeBlock(cursor, faceVector);
    const placedPos = cursor.position.plus(faceVector);
    return {
      placeOccurred: true,
      placeBlockType: toRegistryId(bot.heldItem.name),
      placeX: placedPos.x,
      placeY: placedPos.y,
      placeZ: placedPos.z,
    };
  } catch (err) {
    return empty;
  }
}

module.exports = { applyAction, clampPitch };
