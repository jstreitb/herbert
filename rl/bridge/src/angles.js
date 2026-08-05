'use strict';

/**
 * Conversion between Mineflayer's internal yaw/pitch (radians) and "vanilla convention" degrees,
 * matching `/mod`'s schema (`mod/README.md`: "player.yaw/pitch | float | no | Degrees, vanilla
 * convention", i.e. the same values Forge's `EntityLivingBase.rotationYaw`/`rotationPitch`
 * expose).
 *
 * **Calibrate this against your own server before trusting aim-related RL behavior.** Mineflayer
 * is commonly documented as using a yaw convention that is the *negation* of Minecraft's raw
 * protocol yaw (pitch is not negated), which is what this module implements -- but that has not
 * been independently re-verified against this specific project's mod build. If bridge-recorded
 * rotations look mirrored/rotated relative to what `/mod` would record for the same physical
 * rotation (e.g. record a short session with `/mod` while also running the bridge as a
 * spectator bot and compare `player.yaw`/`pitch` for the same turn), fix the sign here first --
 * everything downstream (observations, the `delta_yaw`/`delta_pitch` action semantics) inherits
 * whatever convention this module settles on, so a wrong sign is a systematic mirroring, not a
 * silent crash.
 */

function radToDeg(rad) {
  return (rad * 180) / Math.PI;
}

function degToRad(deg) {
  return (deg * Math.PI) / 180;
}

function wrapDegrees(deg) {
  let wrapped = deg % 360;
  if (wrapped >= 180) wrapped -= 360;
  if (wrapped < -180) wrapped += 360;
  return wrapped;
}

function mineflayerYawToDegrees(yawRad) {
  return wrapDegrees(radToDeg(-yawRad));
}

function mineflayerPitchToDegrees(pitchRad) {
  return radToDeg(pitchRad);
}

function degreesToMineflayerYaw(yawDeg) {
  return -degToRad(wrapDegrees(yawDeg));
}

function degreesToMineflayerPitch(pitchDeg) {
  return degToRad(pitchDeg);
}

module.exports = {
  radToDeg,
  degToRad,
  wrapDegrees,
  mineflayerYawToDegrees,
  mineflayerPitchToDegrees,
  degreesToMineflayerYaw,
  degreesToMineflayerPitch,
};
