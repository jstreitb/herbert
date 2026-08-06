# Herbert / nn

The neural-network component of **Herbert**: a community feasibility experiment asking
*"can a small, behaviorally-cloned neural network, trained on a single consumer GPU from a
few hours of one skilled player's logged Hypixel Bridge duels, learn anything resembling
Bridge play?"*

This is **not** an attempt at superhuman (or even good) play, and it does not touch raw
pixels/video. It trains small, fast-iterating state-based models (an MLP baseline and a GRU
sequence model) over structured per-tick game state logged by the sibling `/mod` component,
targeting a single RTX 3060 (12GB VRAM) with experiments completing in minutes to low hours.

This directory is a self-contained Python project. It does not share code with `/mod` (a
Forge 1.8.9 Minecraft mod) or `/bot` (a Discord intake bot) -- the only thing connecting them
is the JSONL log format `/mod` produces, which this project fully re-models as Pydantic
schemas (see [`src/herbert_nn/schemas/`](src/herbert_nn/schemas/)).

## Contents

- [Environment setup](#environment-setup)
- [Data directory layout](#data-directory-layout)
- [Walkthrough: preprocess -> train -> evaluate -> inspect](#walkthrough-preprocess---train---evaluate---inspect)
- [Smoke test](#smoke-test)
- [Config reference](#config-reference)
- [The schema registry](#the-schema-registry-versioning-the-mod-contract)
- [Project layout](#project-layout)
- [Development](#development-lint-type-check-test)
- [What does success look like?](#what-does-success-look-like)
- [Known limitations / natural next steps](#known-limitations--natural-next-steps)

## Environment setup

Target Python: **3.10 or 3.11** (3.12+ is not currently tested; some pinned dependency
ranges may not yet support it).

```bash
cd nn
python3.11 -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -e ".[dev]"
```

This installs the package in editable mode plus dev tooling (`pytest`, `ruff`, `black`,
`mypy`). If you want [Weights & Biases](https://wandb.ai) logging, also install the optional
extra: `pip install -e ".[wandb]"`.

If you're on a machine with a CUDA GPU and want a CUDA-enabled `torch` build, install it
first from the appropriate PyTorch index (see https://pytorch.org/get-started/locally/)
before running `pip install -e .` -- otherwise pip will resolve whatever default `torch`
wheel your platform ships (CPU-only on many platforms), which still works, just slower.

Verify the install works end-to-end in a few seconds, without any real data:

```bash
python -m herbert_nn.smoketest
```

## Data directory layout

Raw session logs from `/mod` are newline-delimited JSON (`.jsonl`): one header line, then
one line per recorded game tick. Put every session file (one per recording) flat in a single
directory:

```
data/
  raw/
    session_2026-08-01_18-03-11.jsonl
    session_2026-08-02_09-40-02.jsonl
    ...
```

Filenames don't matter -- sessions are identified by the `session_id` (a UUID4) inside each
file's header line, which must be unique across every file in the raw directory. This
project never mutates `data/raw/`; preprocessing reads it and writes normalized tensors to a
separate, content-hashed cache directory (`data/cache/` by default).

`data/`, `runs/`, and `cache/` are all git-ignored (see `.gitignore`) -- they hold your local
recordings/experiments, not project source.

## Walkthrough: preprocess -> train -> evaluate -> inspect

### 1. Preprocess

Parses every `*.jsonl` file under `--raw-dir`, validates it against the BridgeLogger schema,
engineers features, splits sessions into train/val/test, fits normalization statistics and
categorical vocabularies on the **training split only**, and writes a compressed, cached
tensor dataset:

```bash
python -m herbert_nn.preprocess --raw-dir data/raw --cache-dir data/cache --window-length 32
```

Re-running with the same raw files and options reuses the existing cache (it's named by a
hash of the resolved config + schema version + input filenames). Pass `--force-rebuild` to
recompute anyway, or change any option (e.g. `--window-length`) to automatically get a fresh
cache directory. See `python -m herbert_nn.preprocess --help` for every option.

### 2. Train

Training is Hydra-driven; every hyperparameter is overridable from the command line.

```bash
# MLP baseline (default model), from the repo default config:
python -m herbert_nn.train data.raw_dir=data/raw

# GRU sequence model, with a couple of overrides:
python -m herbert_nn.train model=gru data.raw_dir=data/raw \
    training.optimizer.lr=1e-4 training.batch_size=128 experiment_name=gru_run1
```

If no cache matching the current `data.*` config already exists, `train` builds one
automatically (equivalent to step 1). Every run writes to
`runs/{experiment_name}/{timestamp}/`:

```
runs/gru_run1/2026-08-04_12-00-00/
  .hydra/config.yaml       # exact resolved config for this run (written by Hydra)
  best.pt                  # checkpoint with the lowest validation composite loss so far
  last.pt                  # checkpoint from the most recent epoch (includes optimizer state)
  events.out.tfevents.*    # TensorBoard logs
  metrics.json             # final summary: per-epoch history + best/final losses
```

View training curves with:

```bash
tensorboard --logdir runs
```

### 3. Evaluate

Loads a checkpoint (it embeds its own model architecture config and a pointer to the exact
preprocessing cache it was trained against) and scores it on a held-out split:

```bash
python -m herbert_nn.evaluate --checkpoint runs/gru_run1/2026-08-04_12-00-00/best.pt --split test
```

Writes `metrics_test.json` next to the checkpoint (or to `--output-dir` if given): mouse MAE,
per-action accuracy/F1/ROC-AUC for the discrete head, and top-1/top-3 accuracy for block
placement (computed only over ticks with an active placement event).

### 4. Inspect (qualitative replay)

Given one held-out session file and a checkpoint, runs the model tick-by-tick over that
session and writes a CSV (actual vs. predicted per tick) plus a matplotlib figure, for
eyeballing whether the policy's aim/timing/placement look at all like the recorded player's:

```bash
python -m herbert_nn.inspect --session data/raw/session_2026-08-02_09-40-02.jsonl \
    --checkpoint runs/gru_run1/2026-08-04_12-00-00/best.pt --output-dir inspect_out
```

This is a sanity-checking tool, not a metrics tool -- use `evaluate` for aggregate numbers.

## Smoke test

`herbert_nn.smoketest` runs one full epoch over 200 synthetic (randomly generated,
shape-correct) samples through a real, tiny model -- a real forward pass, backward pass,
composite loss, and checkpoint save -- without needing any real recorded session data or a
preprocessing pass. It typically finishes in a few seconds on CPU (comfortably under the
60-second threshold the tool itself warns about) and confirms the whole
model/loss/training-engine wiring is intact:

```bash
python -m herbert_nn.smoketest --model mlp
python -m herbert_nn.smoketest --model gru
```

Run this after `pip install -e .` and after any code changes to the model/loss/training
engine, before committing to a full training run on real data.

## Config reference

Configs live under [`conf/`](conf/) and compose via Hydra's defaults list
(`conf/config.yaml` pulls in one file from each of `data/`, `model/`, `training/`). Override
anything with `key=value` on the command line, e.g. `training.optimizer.lr=1e-4`, or swap a
whole config group, e.g. `model=gru`.

| Group | File(s) | Key fields |
|---|---|---|
| top-level | `conf/config.yaml` | `experiment_name`, `seed`, `device` (`auto`/`cpu`/`cuda`) |
| `data` | `conf/data/default.yaml` | `raw_dir`, `cache_dir`, `window_length`, `window_stride`, `train_ratio`/`val_ratio`/`test_ratio`, `split_seed`, `block_grid_width`/`height`/`depth` (auto-detected if null), `item_type_vocab_size`, `kit_type_vocab_size`, `place_block_type_vocab_size` |
| `model` | `conf/model/mlp.yaml`, `conf/model/gru.yaml` | `family` (`mlp`/`gru`), trunk sizes (`hidden_dims` / `hidden_size`+`num_layers`+`trunk_hidden_dims`), `dropout`, embedding dims |
| `training` | `conf/training/default.yaml` | `batch_size`, `epochs`, `optimizer.lr`/`weight_decay`, `warmup_steps`, `min_lr_ratio`, `grad_accum_steps`, `grad_clip_norm`, `amp`, `early_stopping_patience`/`min_delta`, `loss_weights.{mouse,discrete,block_placement}`, `huber_delta`, `use_wandb`/`wandb_project` |

An example composed config demonstrating Hydra's `experiment` pattern lives at
`conf/experiment/gru_baseline.yaml` -- use it with `python -m herbert_nn.train
+experiment=gru_baseline`.

### Model architecture

Both `MLPPolicy` and `GRUPolicy` ([`src/herbert_nn/models/`](src/herbert_nn/models/)) share:

- A `FeatureEncoder` that embeds every categorical tick field (block-grid cells, hotbar item
  type, hotbar slot, opponent held-item category, match kit type) and concatenates them with
  the continuous feature vector (position deltas, velocities, sin/cos-encoded angles, health,
  presence flags for the best-effort-detected opponent/match-context fields, ...).
- Four output heads on top of the trunk: `MouseHead` (regression: `d_yaw`, `d_pitch`),
  `DiscreteHead` (binary multi-label: `jump`, `sneak`, `attack`, `place`),
  `BlockPlacementHead` (softmax over the fitted block-type vocabulary, loss-masked to ticks
  with an active placement event), and `MovementHead` (regression: `forward`, `strafe`, each
  in `{-1, 0, 1}` -- a regression head rather than a classifier so its weights splice
  directly into `/rl`'s PPO action head the same way `MouseHead`'s already do).

`MLPPolicy` runs the encoder once per single tick. `GRUPolicy` runs the encoder over every
tick in a sliding window, feeds the sequence through a `nn.GRU`, and heads off the final
timestep's hidden state.

## The schema registry: versioning the mod contract

`/mod` emits `.jsonl` files whose lines are described by a `schema_version` semver string in
the header. `herbert_nn.schemas` implements this as a **versioned registry**
([`src/herbert_nn/schemas/registry.py`](src/herbert_nn/schemas/registry.py)): each schema
version gets its own frozen Pydantic-model module (currently only
[`v1_0_0.py`](src/herbert_nn/schemas/v1_0_0.py)), and `SCHEMA_REGISTRY` maps the version
string to its models. All parsing goes through `load_session()` /
`get_models_for_version()`, which dispatch on the header's declared version -- so old
recordings always remain parseable even after the schema evolves.

**To add support for a new version** (e.g. the mod bumps to `"1.1.0"`): create
`src/herbert_nn/schemas/v1_1_0.py` modeled after `v1_0_0.py` with its own models (never edit
`v1_0_0.py` itself), then add one line to `SCHEMA_REGISTRY` in `registry.py`. No other code
needs to change. This is intentionally *not* pre-built for hypothetical future versions --
only `"1.0.0"` is implemented today.

## Project layout

```
nn/
  conf/                   Hydra configs (data/model/training groups + an example experiment)
  src/herbert_nn/
    schemas/               Pydantic models + version registry for the BridgeLogger format
    data/                  feature engineering, normalization, vocabs, splitting, caching,
                            torch Datasets
    models/                FeatureEncoder, MLPPolicy, GRUPolicy, heads, composite loss
    training/               seeding, LR schedule, checkpointing, early stopping, the epoch
                            loop, the full training-run orchestration, the synthetic
                            smoke-test dataset
    eval/                   held-out metrics (sklearn-based)
    preprocess.py           `python -m herbert_nn.preprocess`
    train.py                `python -m herbert_nn.train` (Hydra entry point)
    evaluate.py              `python -m herbert_nn.evaluate`
    inspect.py               `python -m herbert_nn.inspect`
    smoketest.py             `python -m herbert_nn.smoketest`
  tests/                    pytest suite (synthetic fixtures only, no real data needed)
```

## Development: lint, type-check, test

```bash
pytest              # unit + integration tests (all synthetic data, CPU-only)
ruff check .
black --check .
mypy src/
```

All four are configured in `pyproject.toml` (`[tool.pytest.ini_options]`, `[tool.ruff]`,
`[tool.black]`, `[tool.mypy]`) and pass cleanly on a fresh checkout.

## What does success look like?

This is a feasibility experiment, not a benchmark to top -- there's no leaderboard score that
means "done." Treat a run as promising if, after preprocessing a few hours of one player's
sessions and training for a modest number of epochs, the `inspect` replay output shows
*qualitative* signal such as:

- Predicted mouse movement (`d_yaw`/`d_pitch`) roughly tracking the actual player's aim
  trajectory shape, even if not pixel-perfect -- e.g. turning in the right direction when the
  opponent enters/leaves the field of view.
- Predicted `attack` probability rising around ticks where the player actually attacked
  (rather than being uniformly flat/noisy).
- Predicted `place` probability rising in bridging situations (e.g. player moving forward
  over void with a bridging block selected), and `BlockPlacementHead`'s top-1/top-3 predicted
  block type matching what the player actually placed on those ticks.
- Predicted `forward`/`strafe` (`MovementHead`) roughly tracking bridging-vs-retreating
  behavior, measured via `evaluate.py`'s bucketed-accuracy metric (raw regression output
  thresholded to `{-1, 0, 1}`, matching how `/rl` interprets the same action dimensions).
- Validation composite loss decreasing smoothly and diverging from training loss only mildly
  (severe divergence suggests overfitting to one player's idiosyncrasies faster than useful
  patterns -- try more sessions, a smaller model, or stronger loss weighting on the
  under-fit head).

Conversely, a `discrete` head metrics report with `f1 == 0.0` across the board despite decent
accuracy is a common false-positive signal here: many actions (`jump`, `place`) are rare
per-tick events, so a model that always predicts "no" gets high accuracy and zero F1/AUC.
Watch F1 and ROC-AUC, not accuracy, for these.

## Known limitations / natural next steps

- **`input.attack_target_type` and `input.place_{x,y,z}`** are parsed and available in the
  cached data but not modeled as prediction targets. They can't be, in any way that would
  change bot behavior: `/rl`'s bridge command protocol (see
  `rl/src/herbert_rl/env/ipc.py`'s `ActionCommand`) has no field for "which entity type to
  hit" or "which grid cell to place at" -- both are purely observational echoes of what
  happened as a side effect of aim (`MouseHead`) plus the existing `attack`/`place` flags
  (`DiscreteHead`), never independently controllable actions.
- Single-player behavioral cloning will pick up that player's idiosyncrasies as much as
  general "Bridge skill" -- more sessions from more players would be needed to test whether
  learned behavior generalizes.
