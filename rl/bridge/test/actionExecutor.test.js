// SPDX-License-Identifier: MIT
'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { clampPitch } = require('../src/actionExecutor');

test('clampPitch leaves an in-range value unchanged', () => {
  assert.equal(clampPitch(30), 30);
});

test('clampPitch clamps a value above 90 down to 90', () => {
  assert.equal(clampPitch(150), 90);
});

test('clampPitch clamps a value below -90 up to -90', () => {
  assert.equal(clampPitch(-150), -90);
});

test('clampPitch leaves exactly 90 unchanged', () => {
  assert.equal(clampPitch(90), 90);
});

test('clampPitch leaves exactly -90 unchanged', () => {
  assert.equal(clampPitch(-90), -90);
});
