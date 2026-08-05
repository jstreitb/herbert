// SPDX-License-Identifier: MIT
'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');

const {
  radToDeg,
  degToRad,
  wrapDegrees,
  mineflayerYawToDegrees,
  mineflayerPitchToDegrees,
  degreesToMineflayerYaw,
  degreesToMineflayerPitch,
} = require('../src/angles');

test('radToDeg converts pi to 180', () => {
  assert.ok(Math.abs(radToDeg(Math.PI) - 180) < 1e-9);
});

test('degToRad converts 180 to pi', () => {
  assert.ok(Math.abs(degToRad(180) - Math.PI) < 1e-9);
});

test('radToDeg and degToRad are inverses', () => {
  const original = 42.5;
  assert.ok(Math.abs(degToRad(radToDeg(original)) - original) < 1e-9);
});

test('wrapDegrees leaves an in-range value unchanged', () => {
  assert.equal(wrapDegrees(45), 45);
});

test('wrapDegrees wraps exactly 180 down to -180', () => {
  // Matches the documented [-180, 180) convention: -180 inclusive, 180 exclusive.
  assert.equal(wrapDegrees(180), -180);
});

test('wrapDegrees leaves exactly -180 unchanged', () => {
  assert.equal(wrapDegrees(-180), -180);
});

test('wrapDegrees wraps a value just over 180', () => {
  assert.ok(Math.abs(wrapDegrees(190) - (-170)) < 1e-9);
});

test('wrapDegrees wraps a value just under -180', () => {
  assert.ok(Math.abs(wrapDegrees(-190) - 170) < 1e-9);
});

test('wrapDegrees handles a large cumulative value beyond a single revolution', () => {
  assert.ok(Math.abs(wrapDegrees(720 + 45) - 45) < 1e-9);
});

test('mineflayerYawToDegrees negates and wraps', () => {
  // -90 degrees of mineflayer yaw (radians) should map to +90 vanilla degrees.
  const result = mineflayerYawToDegrees(degToRad(-90));
  assert.ok(Math.abs(result - 90) < 1e-9);
});

test('mineflayerPitchToDegrees does not negate', () => {
  const result = mineflayerPitchToDegrees(degToRad(30));
  assert.ok(Math.abs(result - 30) < 1e-9);
});

test('degreesToMineflayerYaw and mineflayerYawToDegrees round-trip', () => {
  const original = 123.4;
  const roundTripped = mineflayerYawToDegrees(degreesToMineflayerYaw(original));
  assert.ok(Math.abs(roundTripped - original) < 1e-6);
});

test('degreesToMineflayerPitch and mineflayerPitchToDegrees round-trip', () => {
  const original = -45.6;
  const roundTripped = mineflayerPitchToDegrees(degreesToMineflayerPitch(original));
  assert.ok(Math.abs(roundTripped - original) < 1e-6);
});
