# Herbert / rl

The reinforcement-learning fine-tuning component of **Herbert**: takes the behavioral-cloning
checkpoint produced by [`/nn`](../nn/) and fine-tunes it with PPO via self-play, two
Herbert-controlled bots playing automated Bridge duels against each other on a private,
self-hosted Minecraft 1.8.9 server.

**This is the RL phase that comes *after* `/nn`'s imitation-learning baseline.** If you haven't
already trained (or don't yet have) a `/nn` checkpoint, start there first -- see
[`nn/README.md`](../nn/README.md). `/rl` does not duplicate `/nn`'s training pipeline; it only
consumes one of its checkpoints as a warm start.

Like `/nn`, this is a feasibility experiment, not a competitive-bot project -- see "Managing
expectations" below before you get too invested in the results.

## Contents

- [Architecture](#architecture)
- [Environment setup](#environment-setup)
- [Server prerequisites](#server-prerequisites)
- [Walkthrough: fresh clone -> first training run](#walkthrough-fresh-clone---first-training-run)
- [What should TensorBoard look like?](#what-should-tensorboard-look-like)
- [Managing expectations](#managing-expectations)
- [Config reference](#config-reference)
- [CLI entry points](#cli-entry-points)
- [Project layout](#project-layout)
- [Development: lint, type-check, test](#development-lint-type-check-test)
- [Constraints / contract with the rest of the project](#constraints--contract-with-the-rest-of-the-project)

## Architecture

Three layers, spawned/orchestrated by the Python training process (see `start.sh`'s docstring
for why the bridges are spawned *by* Python rather than by a separate shell step):

```
                    ┌─────────────────────────────────────────────────────┐
                    │                  Python (herbert_rl)                 │
                    │                                                       │
  /nn checkpoint ──►│  policy/checkpoint_adapter.py -> policy/sb3_policy.py│
  (.pt)             │       (loads /nn weights into an SB3 ActorCriticPolicy)│
                    │                        │                              │
                    │                        ▼                              │
                    │  train/rollout.py -- custom PPO rollout loop over    │
                    │  env/match_coordinator.py (drives both sides at once)│
                    └───────────┬───────────────────────────┬───────────────┘
                                │ stdin/stdout                │ stdin/stdout
                                │ JSON-lines                  │ JSON-lines
                                ▼                              ▼
                    ┌───────────────────────┐      ┌───────────────────────┐
                    │  rl/bridge (Node.js)   │      │  rl/bridge (Node.js)   │
                    │  Mineflayer bot A      │      │  Mineflayer bot B      │
                    └───────────┬───────────┘      └───────────┬───────────┘
                                │ Minecraft protocol            │ Minecraft protocol
                                ▼                                ▼
                    ┌─────────────────────────────────────────────────────┐
                    │      your private, self-hosted Minecraft 1.8.9       │
                    │      server (NOT Hypixel -- see server/SETUP.md)      │
                    └─────────────────────────────────────────────────────┘
```

- **[`env/`](src/herbert_rl/env/)** -- `HerbertBridgeEnv` (a `gymnasium.Env`), the bridge-process
  IPC client (`bridge_process.py`/`ipc.py`), `RewardFunction`, and `MatchCoordinator` (keeps both
  sides of a duel in lockstep -- see its module docstring for the multi-agent design rationale).
- **[`bridge/`](bridge/)** -- one Mineflayer bot client per side of the duel, Node.js, talking to
  the Python env over stdin/stdout JSON-lines. See [`bridge/README.md`](bridge/README.md) for the
  exact protocol.
- **[`policy/`](src/herbert_rl/policy/)** -- an architectural copy of `/nn`'s
  encoder/GRU/MLP trunk (`backbone.py`), a loader that pulls a `/nn` checkpoint's weights into it
  without importing `herbert_nn` (`checkpoint_adapter.py`), and the custom `stable_baselines3`
  `ActorCriticPolicy` that splices the pretrained aim/discrete-action heads into PPO's action
  head and adds a fresh value head (`sb3_policy.py`).
- **[`train/`](src/herbert_rl/train/)** -- the custom PPO rollout-collection loop
  (`rollout.py`, needed because genuinely simultaneous two-agent self-play doesn't map cleanly
  onto SB3's normal single/vectorized-env rollout collection -- see `env/match_coordinator.py`),
  and the three CLI entry points (`train.py`, `smoketest.py`, `evaluate.py`).

Every one of these design decisions where the task spec allowed more than one reasonable
approach is documented at the point of the decision (module docstrings), not just here --
`env/match_coordinator.py` and `env/spaces.py` in particular are worth reading before touching
either.

## Environment setup

Target Python: **3.10 or 3.11** (matching `/nn`). Target Node.js: **18+**.

```bash
cd rl
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"

cd bridge
npm install
cd ..
```

If you have a CUDA GPU (the target is a single RTX 3060, 12GB), install a CUDA-enabled `torch`
build first from the appropriate PyTorch index before `pip install -e .` -- same note as
`/nn/README.md`.

You'll also need a completed `/nn` run: a checkpoint file (`.pt`) and the `manifest.json` of the
preprocessing cache it was trained against (see [`nn/README.md`](../nn/README.md)'s walkthrough
if you don't have one yet). Keep track of both paths -- you'll need them below.

## Server prerequisites

`/rl` connects to **a private Minecraft 1.8.9 server you control** -- not Hypixel, not any
third-party server. See [`server/SETUP.md`](server/SETUP.md) for the full checklist (server
software, offline-mode auth, automatic match cycling, scoreboard format, JVM flags). Do this
before attempting anything below -- every `/rl` entry point needs a live server to talk to.

## Walkthrough: fresh clone -> first training run

1. **Set up the server.** Follow [`server/SETUP.md`](server/SETUP.md) end to end. Confirm you can
   connect a normal Minecraft client and see a Bridge match auto-cycle at least once.

2. **Install both halves** (Python + Node) per "Environment setup" above.

3. **Sanity-check bridge connectivity** (no Python involved yet):

   ```bash
   cd bridge
   ./scripts/start_both.sh <your-server-host>
   ```

   Confirm both bot accounts join and stay connected, then `Ctrl-C`.

4. **Run the IPC smoke test** (real server, random actions, no checkpoint):

   ```bash
   python -m herbert_rl.train.smoketest \
       --host <your-server-host> \
       --nn-cache-manifest-path /path/to/nn/data/cache/<hash>
   ```

   This should complete 2 episodes and print `SMOKE TEST PASSED`. If it times out waiting for
   episodes to complete, your match-end detection is misconfigured -- see
   `rl/conf/env/default.yaml`'s `match_end` block and `server/SETUP.md` section 4.

5. **Run a real training run:**

   ```bash
   python -m herbert_rl.train.train \
       pretrained_checkpoint_path=/path/to/nn/runs/gru_run1/2026-08-04_12-00-00/best.pt \
       nn_cache_manifest_path=/path/to/nn/data/cache/<hash> \
       env.host=<your-server-host> \
       experiment_name=first_run
   ```

   Or, equivalently, via the orchestration script:

   ```bash
   HOST=<your-server-host> \
   NN_CACHE_MANIFEST_PATH=/path/to/nn/data/cache/<hash> \
   PRETRAINED_CHECKPOINT_PATH=/path/to/nn/runs/gru_run1/2026-08-04_12-00-00/best.pt \
   ./start.sh train experiment_name=first_run
   ```

   Every run writes to `runs/{experiment_name}/{timestamp}/`: the resolved config (`.hydra/`),
   `best.zip`/`last.zip` (SB3 model archives, loadable with `PPO.load(...)`), periodic
   `checkpoint_update_{n}.zip` snapshots, TensorBoard event files, and `metrics.json`.

   **Important:** `env.window_length` (`rl/conf/env/default.yaml`) must match the `/nn`
   checkpoint's family -- `1` for an MLP checkpoint, or the exact `data.window_length` value the
   GRU checkpoint's cache was built with (`nn/conf/data/default.yaml`), or the pretrained
   backbone will see a differently-shaped input than it was trained on.

6. **Watch training:**

   ```bash
   tensorboard --logdir runs
   ```

7. **Evaluate a checkpoint:**

   ```bash
   python -m herbert_rl.train.evaluate \
       --checkpoint runs/first_run/2026-08-05_.../best.zip \
       --host <your-server-host> \
       --nn-cache-manifest-path /path/to/nn/data/cache/<hash> \
       --episodes 10
   ```

## What should TensorBoard look like?

Even in the best case, expect noisy, slow-moving curves -- a few hundred PPO updates on a couple
hours of real-time self-play is a small amount of RL experience by normal standards (see
"Managing expectations"). Signs the loop is *working*, whether or not the agent is getting good:

- `rollout/ep_len_mean` and episode counts per update are non-zero and roughly consistent --
  matches are actually starting, running, and ending via your configured detection, not stalling
  or immediately truncating on disconnect.
- `train/clip_fraction`, `train/approx_kl`, `train/loss` (SB3's own PPO metrics, logged
  automatically every update) look like *any* PPO run -- non-NaN, clip fraction not pinned at 0
  or 1, KL not exploding. If these look broken, the issue is almost certainly hyperparameters
  (`rl/conf/ppo/default.yaml`) or a reward-scale problem (`rl/conf/reward/default.yaml`), not the
  IPC/env plumbing (which the smoke test already covers separately).
- `rollout/reward_*_per_tick` (per-component reward breakdown) shows the four terms firing at
  all -- e.g. `reward_idle_penalty_per_tick` consistently very negative for many updates in a row
  suggests the agent isn't moving at all (check the movement action dims, which have no BC
  warm start -- see "Managing expectations").
- `rollout/ep_rew_mean` is *not* required to trend up for the loop to be "working" -- it trending
  up over enough updates is the actual "did PPO improve on the BC baseline" signal, which is a
  much higher, optional bar. Compare `rollout/ep_rew_mean` from an early update against a late
  one (or use `rl.evaluate` on `checkpoint_update_1.zip` vs `best.zip`) to check for any signal at
  all, rather than expecting a clean, monotonic learning curve.

## Managing expectations

This is an experimental extension of an already-explicitly-modest project (`/nn`'s own README:
*"not an attempt at superhuman (or even good) play"*). Reasons this phase in particular is
unlikely to produce a strong agent, even if everything works exactly as designed:

- **PPO gets very little real experience per wall-clock hour.** Every tick is a real round-trip
  to a live Minecraft server at ~20Hz -- there is no way to run thousands of parallel simulated
  environments the way most PPO benchmarks do. `rl/conf/ppo/default.yaml`'s small `n_steps`
  (256 ticks/side per update) reflects this constraint, not a tuning preference.
- **The reward function is a first guess, not a tuned signal** (see `env/reward.py`'s module
  docstring) -- goal events plus a crude bridge-progress heuristic and an idle penalty is enough
  to give PPO *something* directed to optimize, not a carefully shaped signal known to produce
  good Bridge play.
- **`move_forward`/`strafe` start with no behavioral-cloning warm start at all** (`/nn` never
  modeled movement -- see `nn/README.md`'s "Known limitations"), so PPO has to learn basic
  navigation from scratch on top of everything else, using the same small amount of experience.
- **Self-play against an identical, simultaneously-changing policy is a moving target** -- unlike
  fine-tuning against a fixed opponent, both sides' behavior drift together every update, which
  is a much less stable optimization setting than single-agent PPO.

The goal, per the task this was built for, is to **confirm the self-play training loop runs
end-to-end and observe whether the policy shows any signal of improvement at all** over the BC
baseline -- not to produce a competitive agent. Treat any measurable movement in
`rollout/ep_rew_mean` or `rl.evaluate`'s average reward, in either direction, as more informative
than "did it become good," and expect to spend real time iterating on the reward function and
PPO hyperparameters before drawing conclusions either way.

## Config reference

Configs live under [`conf/`](conf/) and compose via Hydra (`conf/config.yaml` pulls in one file
from each of `env/`, `reward/`, `ppo/`), same convention as `/nn`. Override anything with
`key=value` on the command line, e.g. `ppo.learning_rate=1e-5`.

| Group | File | Key fields |
|---|---|---|
| top-level | `conf/config.yaml` | `experiment_name`, `seed`, `device`, `pretrained_checkpoint_path`, `nn_cache_manifest_path`, `checkpoint_every_n_updates`, `log_every_n_updates` |
| `env` | `conf/env/default.yaml` | `host`/`port`, `username_a`/`username_b`, `window_length` (**must match the checkpoint's family**), `view_distance`, bridge timeouts, `match_end.{score_threshold,chat_patterns}` |
| `reward` | `conf/reward/default.yaml` | `goal_scored`, `goal_conceded`, `bridge_progress`, `idle_penalty`, `idle_speed_threshold`, `bridge_axis`, `own_goal_forward_sign` -- see `env/reward.py`'s docstring, these are starting guesses |
| `ppo` | `conf/ppo/default.yaml` | `n_steps`, `batch_size`, `n_epochs`, `gamma`, `gae_lambda`, `clip_range`, `learning_rate`, `ent_coef`, `vf_coef`, `max_grad_norm`, `num_updates` |

## CLI entry points

Installed as console scripts (`pip install -e .`) and runnable as modules:

| Console script | Module | Purpose |
|---|---|---|
| `herbert-rl-train` | `python -m herbert_rl.train.train` | Full PPO fine-tuning run (Hydra-driven). Requires a real server + a `/nn` checkpoint. |
| `herbert-rl-smoketest` | `python -m herbert_rl.train.smoketest` | End-to-end IPC loop check with random actions. Requires a real server; does **not** require a checkpoint. |
| `herbert-rl-evaluate` | `python -m herbert_rl.train.evaluate` | Load a checkpoint, run N self-play episodes, print win-rate/avg reward. Requires a real server. |

[`start.sh`](start.sh) wraps all three with prerequisite checks and environment-variable-driven
config -- see its header comment for usage and why it doesn't spawn the bridges itself.

## Project layout

```
rl/
  conf/                     Hydra configs (env/reward/ppo groups)
  src/herbert_rl/
    schema.py                Per-tick observation schema (hand-synced copy of /mod's, see docstring)
    constants.py              Feature layout constants (hand-synced copy of /nn's)
    features.py                Streaming single-tick feature encoder (/nn-compatible)
    normalization.py / vocab.py  Hand-synced copies of /nn's Standardizer/CategoricalVocab
    nn_cache.py                 Loads an /nn cache's manifest.json without importing herbert_nn
    env/                         HerbertBridgeEnv, MatchCoordinator, RewardFunction, IPC client
    policy/                      /nn-derived backbone, checkpoint loading, custom SB3 policy
    train/                       Custom PPO rollout loop + train/smoketest/evaluate CLIs
  bridge/                     Mineflayer bot client (Node.js) -- see bridge/README.md
  server/                     SETUP.md (server-side documentation only, no code)
  tests/                      pytest suite (mocked bridge process, no real server needed)
  start.sh                    Orchestration wrapper (see above)
```

## Development: lint, type-check, test

```bash
pytest              # unit tests -- mocked bridge process, no real server needed, CPU-only
ruff check .
black --check .
mypy src/
```

All four are configured in `pyproject.toml`, same as `/nn`. The test suite (see
`tests/test_match_coordinator.py::FakeBridgeProcess`) never talks to a real Mineflayer process or
Minecraft server -- only `rl.smoketest`/`rl.train`/`rl.evaluate` do.

## Constraints / contract with the rest of the project

`/rl` does not modify or import from `/mod`, `/nn`, or `/bot`. Its only contracts with the rest
of the project are:

1. **The JSONL observation schema** (`herbert_rl/schema.py`), kept byte-for-byte compatible with
   `/mod`'s schema (documented in `mod/README.md`) -- so RL-collected trajectories could in
   principle be run through `/nn`'s preprocessing pipeline unmodified.
2. **The `/nn` checkpoint format** (`herbert_nn.training.checkpoint.save_checkpoint`'s on-disk
   shape: `model_state_dict`/`model_cfg`/`data_meta`), read via `torch.load` directly rather than
   by importing `herbert_nn` -- see `policy/checkpoint_adapter.py`'s docstring.
3. **The `/nn` preprocessing cache's `manifest.json` format** (fitted normalization stats and
   vocabularies), read the same way -- see `nn_cache.py`'s docstring.

Every one of these is a hand-maintained copy with an explicit sync-point comment pointing at the
canonical `/nn` source; if `/nn`'s feature layout, schema, or checkpoint format ever changes,
these files need corresponding manual updates.
