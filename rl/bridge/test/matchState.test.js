// SPDX-License-Identifier: MIT
'use strict';

const { test } = require('node:test');
const assert = require('node:assert/strict');

const { createMatchStateTracker } = require('../src/matchState');

const DEFAULT_CONFIG = {
  ownScorePattern: 'You:\\s*(\\d+)',
  opponentScorePattern: 'Them:\\s*(\\d+)',
  elapsedSecondsPattern: '(\\d+:\\d{2})',
  kitPattern: 'Kit:\\s*(.+)',
};

function fakeBot(scoreboardLines = []) {
  return {
    on: () => {},
    scoreboards: {
      sidebar: {
        items: scoreboardLines.map((line) => ({ displayName: line })),
      },
    },
  };
}

test('getMatchState returns null when nothing is parseable', () => {
  const bot = fakeBot(['~~~~~~~~~~~~~~', 'Bridge Duels']);
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  assert.equal(tracker.getMatchState(), null);
});

test('getMatchState returns null when the scoreboard is completely empty', () => {
  const bot = fakeBot([]);
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  assert.equal(tracker.getMatchState(), null);
});

test('getMatchState parses own and opponent scores', () => {
  const bot = fakeBot(['You: 3', 'Them: 1']);
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  const state = tracker.getMatchState();
  assert.equal(state.own_score, 3);
  assert.equal(state.opponent_score, 1);
});

test('getMatchState parses mm:ss elapsed time into seconds', () => {
  const bot = fakeBot(['02:15']);
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  const state = tracker.getMatchState();
  assert.equal(state.elapsed_seconds, 135);
});

test('getMatchState parses the kit name', () => {
  const bot = fakeBot(['Kit: Iron Man']);
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  const state = tracker.getMatchState();
  assert.equal(state.kit, 'Iron Man');
});

test('getMatchState leaves individual fields null when only some things parse', () => {
  const bot = fakeBot(['Kit: Archer']);
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  const state = tracker.getMatchState();
  assert.equal(state.kit, 'Archer');
  assert.equal(state.own_score, null);
  assert.equal(state.opponent_score, null);
  assert.equal(state.elapsed_seconds, null);
});

test('getMatchState handles a missing scoreboards object without throwing', () => {
  const bot = { on: () => {} };
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  assert.equal(tracker.getMatchState(), null);
});

test('drainChatLines returns collected lines and then clears them', () => {
  const bot = fakeBot([]);
  let messageHandler;
  bot.on = (event, handler) => {
    if (event === 'message') messageHandler = handler;
  };
  const tracker = createMatchStateTracker(bot, DEFAULT_CONFIG);
  messageHandler({ toString: () => 'hello' });
  messageHandler({ toString: () => 'world' });
  assert.deepEqual(tracker.drainChatLines(), ['hello', 'world']);
  assert.deepEqual(tracker.drainChatLines(), []);
});

test('message handler does not throw on a malformed message component', () => {
  const bot = fakeBot([]);
  let messageHandler;
  bot.on = (event, handler) => {
    if (event === 'message') messageHandler = handler;
  };
  createMatchStateTracker(bot, DEFAULT_CONFIG);
  assert.doesNotThrow(() => {
    messageHandler({
      toString: () => {
        throw new Error('boom');
      },
    });
  });
});
