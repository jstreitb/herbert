// SPDX-License-Identifier: MIT
'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { createItemClassifier, toRegistryId } = require('../src/itemClassifier');

function fakeMcData(blockNames = []) {
  const blocksByName = {};
  for (const name of blockNames) {
    blocksByName[name] = {};
  }
  return { blocksByName };
}

test('classify treats a null/empty item name as OTHER', () => {
  const { classify } = createItemClassifier(fakeMcData());
  assert.equal(classify(null), 'OTHER');
  assert.equal(classify(''), 'OTHER');
});

test('classify recognizes any sword by substring, case-insensitively', () => {
  const { classify } = createItemClassifier(fakeMcData());
  assert.equal(classify('diamond_sword'), 'SWORD');
  assert.equal(classify('WOODEN_SWORD'), 'SWORD');
});

test('classify recognizes a bow', () => {
  const { classify } = createItemClassifier(fakeMcData());
  assert.equal(classify('bow'), 'BOW');
});

test('classify does not confuse "bowl" with "bow"', () => {
  const { classify } = createItemClassifier(fakeMcData());
  assert.equal(classify('mushroom_stew_bowl'), 'OTHER');
});

test('classify recognizes a placeable block via minecraft-data', () => {
  const { classify } = createItemClassifier(fakeMcData(['wool']));
  assert.equal(classify('wool'), 'BLOCKS');
});

test('classify falls back to OTHER for a non-block, non-sword, non-bow item', () => {
  const { classify } = createItemClassifier(fakeMcData(['wool']));
  assert.equal(classify('golden_apple'), 'OTHER');
});

test('toRegistryId prefixes with "minecraft:"', () => {
  assert.equal(toRegistryId('wool'), 'minecraft:wool');
});

test('toRegistryId returns null for a null/empty name', () => {
  assert.equal(toRegistryId(null), null);
  assert.equal(toRegistryId(''), null);
});
