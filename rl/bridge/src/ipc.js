'use strict';

/**
 * The JSON-lines IPC protocol spoken with the Python side (`herbert_rl/env/ipc.py`). Kept in a
 * single small module so both directions of the wire format live in one place; see that Python
 * module's docstring for the authoritative, byte-for-byte protocol description -- this file must
 * stay in sync with it by hand.
 *
 * Python -> bridge (stdin), one command object per line: {"cmd": "action", ...} | {"cmd":
 * "reset"} | {"cmd": "close"}.
 *
 * Bridge -> Python (stdout), one object per line: either a raw tick observation (has a "tick"
 * key) or a lifecycle event {"event": "ready"|"reset_ack"|"error"|"log", ...}.
 */

const readline = require('readline');

/** Writes one tick observation line to stdout. `record` must already match the Python
 * `TickRecordRL` schema exactly (see observationBuilder.js). */
function emitObservation(record) {
  process.stdout.write(JSON.stringify(record) + '\n');
}

/** Writes one lifecycle event line to stdout. */
function emitEvent(event, fields = {}) {
  process.stdout.write(JSON.stringify({ event, ...fields }) + '\n');
}

function emitReady() {
  emitEvent('ready');
}

function emitResetAck() {
  emitEvent('reset_ack');
}

function emitError(message) {
  emitEvent('error', { message: String(message) });
}

/** Forwards a verbose bridge-side log line to the Python process's logger, instead of writing
 * to stderr, so it's timestamped/leveled consistently with the rest of the training run's logs
 * (see `herbert_rl/env/bridge_process.py::_read_message`, which routes `event: "log"` lines into
 * the Python `logging` module at the given level). */
function log(level, message) {
  emitEvent('log', { level, message: String(message) });
}

/**
 * Starts reading newline-delimited JSON command objects from stdin.
 *
 * @param {object} handlers
 * @param {(action: object) => void} handlers.onAction
 * @param {() => void} handlers.onReset
 * @param {() => void} handlers.onClose
 */
function listenForCommands({ onAction, onReset, onClose }) {
  const rl = readline.createInterface({ input: process.stdin, terminal: false });
  rl.on('line', (line) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let command;
    try {
      command = JSON.parse(trimmed);
    } catch (err) {
      log('warn', `Discarding malformed stdin line (invalid JSON): ${err.message}`);
      return;
    }
    switch (command.cmd) {
      case 'action':
        onAction(command);
        break;
      case 'reset':
        onReset();
        break;
      case 'close':
        onClose();
        break;
      default:
        log('warn', `Discarding stdin line with unknown cmd: ${JSON.stringify(command)}`);
    }
  });
  rl.on('close', () => {
    // Python process exited / closed stdin without an explicit "close" command -- shut down
    // rather than leaving an orphaned Minecraft connection.
    onClose();
  });
  return rl;
}

module.exports = { emitObservation, emitEvent, emitReady, emitResetAck, emitError, log, listenForCommands };
