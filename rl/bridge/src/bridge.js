#!/usr/bin/env node
'use strict';

/**
 * Herbert /rl bridge entrypoint: connects one bot account to a Minecraft 1.8.9 server and
 * exposes it to the Python RL env over stdin/stdout JSON-lines IPC (see `src/ipc.js` and
 * `herbert_rl/env/ipc.py` for the exact wire protocol). Run two of these (one per side of a
 * duel) via `herbert_rl.env.bridge_process.BridgeProcess` during training, or manually for
 * debugging -- see `README.md`.
 *
 * Tick cadence is request-driven, not free-running: nothing is emitted until Python sends an
 * `action`/`reset` command. On each `action`, the bridge applies it, waits for exactly one
 * Mineflayer `physicsTick` (so the applied controls have actually been simulated), then emits
 * exactly one observation -- a tight one-action-in/one-observation-out loop matching the
 * synchronous request/reply contract `BridgeProcess` expects on the Python side.
 */

const yargs = require('yargs/yargs');
const { hideBin } = require('yargs/helpers');
const mcDataLoader = require('minecraft-data');

const ipc = require('./ipc');
const { ReconnectingBot } = require('./reconnect');
const { createMatchStateTracker } = require('./matchState');
const { createItemClassifier } = require('./itemClassifier');
const { buildTickRecord, buildDisconnectedRecord, findOpponentEntity } = require('./observationBuilder');
const { applyAction } = require('./actionExecutor');

const MC_VERSION = '1.8.9';

const NO_OP_ECHO = {
  forward: 0,
  strafe: 0,
  jump: false,
  sneak: false,
  delta_yaw: 0.0,
  delta_pitch: 0.0,
  attack_occurred: false,
  attack_target_type: null,
  place_occurred: false,
  place_block_type: null,
  place_x: null,
  place_y: null,
  place_z: null,
};

function parseArgs(argv) {
  return yargs(hideBin(argv))
    .option('host', { type: 'string', demandOption: true })
    .option('port', { type: 'number', default: 25565 })
    .option('username', { type: 'string', demandOption: true })
    .option('auth', { type: 'string', default: 'offline', choices: ['offline', 'microsoft', 'mojang'] })
    .option('view-distance', { type: 'number', default: 6 })
    .option('log-level', { type: 'string', default: 'info' })
    .option('block-grid-width', { type: 'number', default: 7 })
    .option('block-grid-height', { type: 'number', default: 3 })
    .option('block-grid-depth', { type: 'number', default: 7 })
    .option('void-threshold-y', { type: 'number', default: 0 })
    .option('opponent-tracking-range', { type: 'number', default: 48 })
    .option('own-score-pattern', { type: 'string', default: 'You:\\s*(\\d+)' })
    .option('opponent-score-pattern', { type: 'string', default: 'Them:\\s*(\\d+)' })
    .option('elapsed-seconds-pattern', { type: 'string', default: '(\\d+:\\d{2})' })
    .option('kit-pattern', { type: 'string', default: 'Kit:\\s*(.+)' })
    .option('reset-grace-period-ms', { type: 'number', default: 3000 })
    .option('reconnect-base-delay-ms', { type: 'number', default: 1000 })
    .option('reconnect-max-delay-ms', { type: 'number', default: 30000 })
    .strict()
    .parse();
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function waitForNextPhysicsTick(bot) {
  return new Promise((resolve) => bot.once('physicsTick', resolve));
}

function main() {
  const args = parseArgs(process.argv);

  const blockGridConfig = {
    width: args.blockGridWidth,
    height: args.blockGridHeight,
    depth: args.blockGridDepth,
    voidThresholdY: args.voidThresholdY,
  };
  const matchConfig = {
    ownScorePattern: args.ownScorePattern,
    opponentScorePattern: args.opponentScorePattern,
    elapsedSecondsPattern: args.elapsedSecondsPattern,
    kitPattern: args.kitPattern,
  };
  const mcData = mcDataLoader(MC_VERSION);
  const itemClassifier = createItemClassifier(mcData);

  let tickCounter = 0;
  let matchStateTracker = null;
  let hasEmittedReady = false;

  const reconnectingBot = new ReconnectingBot(
    {
      host: args.host,
      port: args.port,
      username: args.username,
      auth: args.auth,
      version: MC_VERSION,
      viewDistance: args.viewDistance,
    },
    { baseDelayMs: args.reconnectBaseDelayMs, maxDelayMs: args.reconnectMaxDelayMs }
  );

  reconnectingBot.on('connected', (bot) => {
    matchStateTracker = createMatchStateTracker(bot, matchConfig);
    if (!hasEmittedReady) {
      hasEmittedReady = true;
      ipc.emitReady();
    } else {
      ipc.log('info', 'Bridge reconnected after a prior disconnect.');
    }
  });
  reconnectingBot.on('disconnected', (reason) => {
    matchStateTracker = null;
    ipc.log('warn', `Bridge disconnected: ${reason}`);
  });

  function tickContext() {
    return {
      findOpponentEntity,
      opponentTrackingRangeBlocks: args.opponentTrackingRange,
    };
  }

  async function emitTickFor(bot, lastActionEcho) {
    if (!bot || !matchStateTracker) {
      tickCounter += 1;
      ipc.emitObservation(buildDisconnectedRecord(tickCounter, lastActionEcho, blockGridConfig));
      return;
    }
    await waitForNextPhysicsTick(bot);
    tickCounter += 1;
    const record = buildTickRecord(bot, {
      tick: tickCounter,
      lastActionEcho,
      matchStateTracker,
      itemClassifier,
      blockGridConfig,
      opponentTrackingRangeBlocks: args.opponentTrackingRange,
    });
    ipc.emitObservation(record);
  }

  async function handleAction(action) {
    try {
      const bot = reconnectingBot.getBot();
      if (!bot || !matchStateTracker) {
        await emitTickFor(null, actionToEcho(action));
        return;
      }
      const echo = await applyAction(bot, action, tickContext());
      await emitTickFor(bot, echo);
    } catch (err) {
      ipc.log('error', `Error while handling action: ${err && err.stack}`);
      await emitTickFor(reconnectingBot.getBot(), actionToEcho(action));
    }
  }

  async function handleReset() {
    try {
      tickCounter = 0;
      const bot = reconnectingBot.getBot();
      if (bot) {
        bot.clearControlStates();
      }
      await sleep(args.resetGracePeriodMs);
      ipc.emitResetAck();
      await emitTickFor(reconnectingBot.getBot(), NO_OP_ECHO);
    } catch (err) {
      ipc.log('error', `Error while handling reset: ${err && err.stack}`);
    }
  }

  function handleClose() {
    ipc.log('info', 'Received close command; shutting down.');
    reconnectingBot.stop();
    process.exit(0);
  }

  ipc.listenForCommands({
    onAction: (command) => {
      handleAction(command);
    },
    onReset: () => {
      handleReset();
    },
    onClose: handleClose,
  });

  process.on('uncaughtException', (err) => {
    ipc.emitError(`Uncaught exception: ${err && err.stack}`);
    process.exit(1);
  });
  process.on('unhandledRejection', (reason) => {
    ipc.emitError(`Unhandled rejection: ${reason}`);
  });
}

function actionToEcho(action) {
  return {
    forward: action.forward,
    strafe: action.strafe,
    jump: !!action.jump,
    sneak: !!action.sneak,
    delta_yaw: action.delta_yaw,
    delta_pitch: action.delta_pitch,
    attack_occurred: false,
    attack_target_type: null,
    place_occurred: false,
    place_block_type: null,
    place_x: null,
    place_y: null,
    place_z: null,
  };
}

main();
