# Herbert

Herbert is an open-source, community-driven experiment in **imitation learning for Hypixel
Bridge duels** (the 1v1 rush/PvP/build gamemode on the [Hypixel](https://hypixel.net) network,
running on Minecraft 1.8.9).

The question Herbert is trying to answer is deliberately modest:

> Can a small, behaviorally-cloned neural network — trained on a single consumer GPU from a
> few hours of one skilled player's logged sessions — learn anything resembling Bridge play?

This is **not** an attempt to build a superhuman or even "good" Bridge bot. It is a feasibility
and learning project: a passive data logger, an offline training pipeline, and a community
data-collection funnel, built to see what a small state-based model can pick up from human
demonstration data. **Nothing in this project automates gameplay, injects input into the game
client, or otherwise interacts with Hypixel's servers beyond normal, passive, human-driven
play.**

## Architecture

Herbert is split into three independent components. They share **no code** — each is a
standalone project with its own toolchain, dependencies, and README. The only things that
connect them are two narrow data contracts:

```
 ┌──────────────┐   JSONL session log    ┌──────────────┐   pastes.dev URL   ┌──────────────┐
 │     /mod     │ ───────────────────►   │  pastes.dev  │ ─────────────────► │     /bot     │
 │ BridgeLogger │   (upload via HTTP)     │  (paste host)│   (Discord webhook) │ intake funnel│
 │ Forge 1.8.9  │                         └──────────────┘                    │  discord.py  │
 └──────────────┘                                                             └──────────────┘
                                                                                      │
                                                                                      │ validated
                                                                                      │ pastes.dev
                                                                                      │ URL, posted
                                                                                      │ in Discord
                                                                                      ▼
                                                                              ┌──────────────┐
                                                                              │      /nn     │
                                                                              │  training     │
                                                                              │  pipeline     │
                                                                              │  (offline,    │
                                                                              │  manual data   │
                                                                              │  download)     │
                                                                              └──────────────┘
```

1. **[`/mod`](mod/) — BridgeLogger.** A Forge 1.8.9 mod that a community member runs while
   playing Bridge duels on Hypixel. It passively records synchronized player state,
   block-environment state, opponent state, and input/action data at a fixed tick rate. At the
   end of a session it uploads the log to [pastes.dev](https://pastes.dev) and posts the
   resulting URL to a shared Discord webhook. It is purely observational: it never sends game
   packets, never simulates input, and never automates any in-game action.

2. **[`/bot`](bot/) — intake bot.** A `discord.py` bot that watches the community's intake
   channel for the URLs the mod posts, validates that each one is a real, well-formed Herbert
   session (fetches the paste, checks the header), and either acknowledges good submissions
   (✅ reaction + thread reply) or silently deletes anything that doesn't match — keeping the
   channel a clean feed of usable data.

3. **[`/nn`](nn/) — training pipeline.** An offline Python pipeline that ingests the collected
   JSONL session files, preprocesses them into cached tensors, and trains small,
   fast-to-iterate behavioral-cloning models (MLP and GRU policies over structured state — not
   raw pixels) to predict a player's next action from the current game state. Runs comfortably
   on a single RTX 3060 (12GB VRAM).

The **JSONL schema** produced by `/mod` (documented field-by-field in
[`mod/README.md`](mod/README.md)) is the only contract between `/mod` and `/nn`. The
**pastes.dev URL format** (`https://pastes.dev/{key}`) is the only contract between `/mod` and
`/bot`. Each component can be built, tested, and evolved independently as long as those two
contracts hold.

**Publication status:** `/mod` and `/nn` are open-source and public, as described above. `/bot`
is intentionally kept **private and unpublished** — it is not part of this public repository, by
project policy, for security reasons (it protects the integrity of the community data-intake
pipeline). Contributors working on `/mod` or `/nn` do not need access to the bot's source. See
the warning at the top of `bot/README.md` for the full explanation, if you have access to it.

## Getting started

What you want to do determines where to start:

- **I want to record gameplay data.** Install the mod — see [`mod/README.md`](mod/README.md)
  for build/installation steps, config reference, and the exact schema it produces. You'll need
  a Discord webhook URL from the community server to have your sessions auto-uploaded (or run
  in `dry-run` mode to log locally without uploading).
- **I want to run or improve the intake bot.** See [`bot/README.md`](bot/README.md) for setup
  (bare Python or Docker), required environment variables, and how the validation logic works.
- **I want to train or experiment with models.** See [`nn/README.md`](nn/README.md) for
  environment setup, the expected data layout, and a full preprocess → train → evaluate
  walkthrough, including a `--smoke-test` mode that validates the whole pipeline in under a
  minute.
- **I want to contribute code.** See [`CONTRIBUTING.md`](CONTRIBUTING.md) for the code quality
  bar, how each component is tested, and how to submit a PR.

## Project status & framing

Herbert is an experimental community project, not a product. Expect the model to be, at best,
a rough approximation of one player's habits learned from a small dataset — see the "what does
success look like" section in [`nn/README.md`](nn/README.md) for how we're framing that. The
project is open to contributions of all sizes: more recorded sessions, mod robustness fixes
(Hypixel's scoreboard format is not officially documented and heuristics will need tuning),
pipeline improvements, or just better documentation.

## License & privacy note

Herbert is community/open-source software. The mod never logs raw usernames — every session
header contains a SHA-256 hash of the recording player's username instead (see the privacy note
in [`mod/README.md`](mod/README.md)). Only join the intake channel and submit data if you're
comfortable with your (anonymized) gameplay being used to train these experimental models.
