'use strict';

/**
 * Wraps `mineflayer.createBot` with automatic exponential-backoff reconnection, so a bridge
 * process survives a server restart / connection drop without the Python side needing to
 * respawn the whole subprocess (see `herbert_rl/env/bridge_process.py`'s docstring: the Python
 * side only ever spawns a bridge process once per training run).
 *
 * Emits:
 *   - `'connected'` (bot) -- on every successful spawn, including reconnects.
 *   - `'disconnected'` (reason: string) -- on every connection loss.
 *   - `'reconnecting'` (delayMs, attempt) -- right before scheduling the next connection attempt.
 */

const EventEmitter = require('events');
const mineflayer = require('mineflayer');
const { log } = require('./ipc');

const DEFAULT_BASE_DELAY_MS = 1000;
const DEFAULT_MAX_DELAY_MS = 30000;

class ReconnectingBot extends EventEmitter {
  constructor(mineflayerOptions, options = {}) {
    super();
    this.mineflayerOptions = mineflayerOptions;
    this.baseDelayMs = options.baseDelayMs || DEFAULT_BASE_DELAY_MS;
    this.maxDelayMs = options.maxDelayMs || DEFAULT_MAX_DELAY_MS;
    this.bot = null;
    this.attempt = 0;
    this.stopped = false;
    this._connect();
  }

  getBot() {
    return this.bot;
  }

  _connect() {
    if (this.stopped) return;
    log(
      'info',
      `Connecting to ${this.mineflayerOptions.host}:${this.mineflayerOptions.port} as ` +
        `${this.mineflayerOptions.username} (attempt ${this.attempt + 1})...`
    );
    let bot;
    try {
      bot = mineflayer.createBot(this.mineflayerOptions);
    } catch (err) {
      this._handleDisconnect(`createBot threw: ${err && err.message}`);
      return;
    }
    this.bot = bot;

    bot.once('spawn', () => {
      this.attempt = 0;
      this.emit('connected', bot);
    });
    bot.once('end', (reason) => this._handleDisconnect(`end: ${reason}`));
    bot.once('error', (err) => this._handleDisconnect(`error: ${err && err.message}`));
    bot.once('kicked', (reason) => this._handleDisconnect(`kicked: ${reason}`));
  }

  _handleDisconnect(reason) {
    if (this.stopped) return;
    if (this.bot) {
      this.bot.removeAllListeners();
      this.bot = null;
    }
    this.emit('disconnected', reason);
    if (this.stopped) return;
    this.attempt += 1;
    const delay = Math.min(this.maxDelayMs, this.baseDelayMs * 2 ** (this.attempt - 1));
    log('warn', `Disconnected (${reason}); reconnecting in ${delay}ms (attempt ${this.attempt}).`);
    this.emit('reconnecting', delay, this.attempt);
    setTimeout(() => this._connect(), delay);
  }

  stop() {
    this.stopped = true;
    if (this.bot) {
      try {
        this.bot.quit();
      } catch (err) {
        // Already disconnected -- nothing to do.
      }
      this.bot = null;
    }
  }
}

module.exports = { ReconnectingBot };
