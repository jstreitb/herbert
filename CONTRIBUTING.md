# Contributing to Herbert

Thanks for your interest in Herbert — a community experiment in imitation learning for Hypixel
Bridge duels. This document covers how to contribute in each of the three ways the project
needs help: recording data, running/improving the data pipeline, and writing code.

Herbert is split into three independent components — [`/mod`](mod/), [`/bot`](bot/), and
[`/nn`](nn/) — each with its own toolchain and README. Read the relevant component README
before working on it; this file covers cross-cutting expectations.

## 1. Recording sessions (mod setup)

The easiest way to contribute to Herbert is simply to play Bridge duels on Hypixel with the
BridgeLogger mod installed.

1. Follow the build/installation instructions in [`mod/README.md`](mod/README.md) to get
   BridgeLogger running on a Forge 1.8.9 client.
2. Ask in the community Discord server for the intake channel's webhook URL, and set it in the
   mod's config (`discordWebhookUrl`). If you'd rather not upload yet, leave the mod in
   `dryRun` mode and just log locally to try it out first.
3. Play Bridge duels normally. The mod should auto-detect the start/end of a duel via the
   scoreboard; if detection misfires for your client/language settings, use the manual
   `/herbert start`, `/herbert stop`, and `/herbert status` commands instead, and please open an
   issue describing what the scoreboard looked like so the heuristic can be improved.
4. At the end of a session the mod uploads your log directly to the intake channel's webhook as
   a file attachment (splitting it into a few messages first if it's too large for one) — there
   is no separate submission step and no third-party paste site involved. The intake bot
   recognizes a submission directly by its `.jsonl` file attachment, so a successful upload gets
   the bot's ✅ reaction automatically; see "Submitting data" below if you need to submit a
   session that wasn't uploaded automatically (e.g. one recorded in `dryRun` mode).

Your username is never uploaded in the clear by default — every session header stores a SHA-256
hash of it, and the mod only adds your raw username too if you explicitly opt in when it asks at
the end of each session (see the privacy note in [`mod/README.md`](mod/README.md)).

Before opening a PR against `/mod`, run:

```bash
cd mod
./gradlew build   # compiles and runs the unit test suite (src/test/java)
```

## 2. Submitting data

Automatic uploads land in the intake channel and get the bot's ✅ validation reaction as
described above. To submit a session that wasn't uploaded automatically (e.g. one recorded in
`dryRun` mode, or recovered after a failed upload), post it as a `.jsonl` file attachment in the
intake channel yourself — the bot validates any message carrying one, regardless of who or what
posted it.

## 3. Contributing to the `/nn` pipeline

The training pipeline is a standard Python project — see [`nn/README.md`](nn/README.md) for
environment setup and the full preprocess → train → evaluate walkthrough. Useful ways to
contribute:

- New or improved model architectures (kept small and fast to train — see the constraints in
  the README; this project is intentionally scoped to single-GPU, minutes-to-low-hours
  iteration, not large-scale training).
- Preprocessing/feature engineering improvements.
- Additional evaluation metrics or better qualitative tooling (the replay inspector is a good
  place to start).
- Bug fixes and test coverage.

Before opening a PR against `/nn`, run:

```bash
cd nn
pip install -e ".[dev]"
ruff check .
black --check .
mypy src/
pytest
python -m herbert_nn.smoketest   # confirms the full pipeline still runs end-to-end
```

## Code quality bar

Herbert is open-source and meant to be picked up by contributors who didn't write the original
code. PRs are expected to meet the same bar as the rest of the codebase:

> **Using an AI coding assistant?** [AGENTS.md](AGENTS.md) is the machine-readable version of
> this section — exact formatter/linter settings, the test commands for each component,
> environment gotchas, and the cross-component invariants that fail silently when broken.
> Point your tool at it (`CLAUDE.md`, `GEMINI.md`, `.cursorrules`, and
> `.github/copilot-instructions.md` all redirect there) so its output matches the rest of the
> codebase.

- **Documentation on every public symbol.** Javadoc for Java (`/mod`), docstrings for Python
  (`/nn`, `/bot`) and JSDoc for any JS/TS tooling. If a reviewer has to ask "what does this do,"
  that's a missing docstring, not a question for the PR thread.
- **No stubs, no TODOs in production code.** If something isn't finished, it isn't merged. Open
  an issue instead of leaving a `// TODO` in a shipped path.
- **No magic numbers.** Constants get names and, where the value isn't self-explanatory, a
  comment explaining where it came from (e.g. a tick rate, an HTTP timeout, a default grid
  size).
- **Consistent formatting.** Java code follows standard 4-space-indent conventions; Python code
  is formatted with `black` and linted with `ruff`/`mypy` per the configs in `nn/pyproject.toml`
  and `bot/pyproject.toml`.
- **Defensive parsing around Hypixel-specific formats.** Hypixel's scoreboard/tab-list format
  isn't officially documented and changes without notice. Code that parses it (mostly in
  `/mod`) must degrade gracefully (null/unknown fields) rather than throwing — a parsing bug
  should never crash a contributor's game client.
- **Strict separation of concerns.** `/mod`, `/nn`, and `/bot` must remain independently
  buildable with no shared code. The JSONL schema and the Discord webhook file-attachment upload
  convention are the only contracts between them; changes to either must be reflected in
  `mod/README.md` (the schema is documented there field-by-field, and the upload contract in its
  "Contract with the rest of the Herbert project" section) and coordinated across the components
  that depend on it.
- **Nothing that automates gameplay.** This is a hard line for `/mod`: it is a passive logger.
  PRs that add input injection, packet manipulation, or any form of gameplay automation will be
  rejected regardless of other merits.

## Reporting issues

Bug reports, scoreboard-parsing edge cases, and feature requests are all welcome via the
project's issue tracker. For scoreboard/heuristic issues, include a screenshot or raw text dump
of the scoreboard/tab list that wasn't handled correctly — that's the fastest path to a fix.
