# Changelog

All notable changes to Herbert will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `/mod` — BridgeLogger: Forge 1.8.9 mod for passive recording of Hypixel Bridge duel gameplay
  - Tick-synchronized capture of player state (position, rotation, health, food, held item)
  - Block environment state tracking within configurable render distance
  - Opponent state tracking (position, rotation, health)
  - Player input/action capture (clicks, attacks, placements, block breaks)
  - Automatic Bridge duel detection via scoreboard heuristics
  - Manual session control via in-game commands (`/herbert start/stop/status`)
  - Direct Discord webhook file-attachment upload with automatic chunking
  - SHA-256 username hashing by default; opt-in raw username logging
  - Dry-run mode for local testing without uploads
  - Comprehensive config system with safe defaults
- `/nn` — Training Pipeline: Python behavioral cloning pipeline for imitation learning
  - Ingest and validate JSONL session files from `/mod`
  - Preprocessing pipeline with tensor caching for fast iteration
  - MLP and GRU policy architectures for state-to-action prediction
  - Training infrastructure with configurable hyperparameters
  - Model evaluation and per-class accuracy reporting
  - Replay inspector for qualitative testing
  - Smoke-test mode for end-to-end validation
- `/rl` — Reinforcement Learning: PPO fine-tuning via self-play on private Minecraft server
  - Node.js/Mineflayer bot clients for automated Bridge duel gameplay
  - Private server integration (1.8.9) for controlled training environment
  - Reward function design for Bridge-specific objectives
  - Policy gradient training with PPO
  - Episode logging and analysis tooling
- `/bot` — Intake bot (private component, not published)
  - Discord.py bot for session validation and data-quality curation
  - Automatic file-attachment recognition
  - JSONL schema validation
  - Duplicate submission detection
  - Data storage and archival
- Documentation
  - Root README with architecture overview and quickstart
  - CONTRIBUTING.md with setup instructions, code quality bar, and issue templates
  - Component-specific READMEs for `/mod`, `/nn`, `/rl`, `/bot`
  - LICENSE (MIT)
  - CODE_OF_CONDUCT.md (Contributor Covenant 2.1)
  - SECURITY.md with vulnerability disclosure policy
  - This CHANGELOG

### Security

- SPDX license headers on all source files
- Secret management via environment variables (no hardcoded tokens/webhooks)
- `.gitignore` excludes `.env` files, `/bot` component, and training artifacts
- Defensive JSONL parsing to prevent resource exhaustion
- Username hashing in session data by default

### Known Limitations

- Bridge detection heuristics are game-client-specific and may require tuning for different
  language/mod configurations
- Models trained on small datasets (few hours of a single player) reflect that player's habits
  narrowly; generalization is limited
- RL training is experimental and scoped to private servers only
- No support for Minecraft versions beyond 1.8.9 (due to Hypixel's version lock)

---

## How to Contribute

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code quality expectations,
and the PR process.

## Version History

- **v0.1.0** (2025-08-05) — Initial release: `/mod`, `/nn`, `/rl` alpha; `/bot` private
