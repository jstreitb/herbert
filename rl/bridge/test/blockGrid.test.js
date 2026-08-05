// SPDX-License-Identifier: MIT
'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { classifyBlock } = require('../src/blockGrid');

const VOID_THRESHOLD_Y = 0;

test('classifyBlock treats a null block as air above the void threshold', () => {
  assert.equal(classifyBlock(null, 10, VOID_THRESHOLD_Y), 'AIR');
});

test('classifyBlock treats a null block as void at the threshold (inclusive)', () => {
  assert.equal(classifyBlock(null, VOID_THRESHOLD_Y, VOID_THRESHOLD_Y), 'VOID');
});

test('classifyBlock treats a null block as void below the threshold', () => {
  assert.equal(classifyBlock(null, -5, VOID_THRESHOLD_Y), 'VOID');
});

test('classifyBlock treats an explicit air block the same as a null block', () => {
  assert.equal(classifyBlock({ name: 'air' }, 10, VOID_THRESHOLD_Y), 'AIR');
});

test('classifyBlock treats an empty-bounding-box block as air', () => {
  assert.equal(classifyBlock({ name: 'tallgrass', boundingBox: 'empty' }, 10, VOID_THRESHOLD_Y), 'AIR');
});

test('classifyBlock treats water as liquid', () => {
  assert.equal(classifyBlock({ name: 'water', boundingBox: 'block' }, 64, VOID_THRESHOLD_Y), 'LIQUID');
});

test('classifyBlock treats flowing_lava as liquid', () => {
  assert.equal(classifyBlock({ name: 'flowing_lava', boundingBox: 'block' }, 64, VOID_THRESHOLD_Y), 'LIQUID');
});

test('classifyBlock treats wool as solid bridgeable', () => {
  assert.equal(classifyBlock({ name: 'wool', boundingBox: 'block' }, 64, VOID_THRESHOLD_Y), 'SOLID_BRIDGEABLE');
});

test('classifyBlock treats stone as solid bridgeable', () => {
  assert.equal(classifyBlock({ name: 'stone', boundingBox: 'block' }, 64, VOID_THRESHOLD_Y), 'SOLID_BRIDGEABLE');
});

test('classifyBlock treats bedrock as other solid', () => {
  assert.equal(classifyBlock({ name: 'bedrock', boundingBox: 'block' }, 64, VOID_THRESHOLD_Y), 'OTHER_SOLID');
});

test('classifyBlock treats barrier as other solid', () => {
  assert.equal(classifyBlock({ name: 'barrier', boundingBox: 'block' }, 64, VOID_THRESHOLD_Y), 'OTHER_SOLID');
});

test('classifyBlock is one below the threshold boundary correct (threshold + 1 is air)', () => {
  assert.equal(classifyBlock(null, VOID_THRESHOLD_Y + 1, VOID_THRESHOLD_Y), 'AIR');
});
