# AGENTS.md — Conventions for AI coding agents

This file is the single source of truth for **how code, documentation, and tests are
written in this repository**. It is written for AI coding agents (Claude Code, Cursor,
Copilot, Codex, Aider, …) so that every agent produces work in the same style, with the
same invariants respected, regardless of which tool a contributor uses.

Human contributors: see [CONTRIBUTING.md](CONTRIBUTING.md). This file complements it with
the mechanical detail an agent needs.

**If you are an agent reading this: read the whole file before your first edit.** The
"Hard rules" and "Cross-component invariants" sections describe things that are not
discoverable from any single file, and that are easy to silently break.

---

## 1. What this project is

Herbert is an experiment in teaching a bot to play **Hypixel Bridge duels** by imitation.
Four components form one pipeline:

| Path | Language | Role | Public? |
|---|---|---|---|
| `/mod` | Java 8 (Minecraft Forge 1.8.9) | Records human gameplay to JSONL | Yes |
| `/nn` | Python 3.10/3.11 | Behavioral cloning on that JSONL | Yes |
| `/rl` | Python 3.10/3.11 + Node.js | PPO fine-tuning; `/rl/bridge` is a Mineflayer bot | Yes |
| `/bot` | Python | Private Discord intake bot | **No — see Hard rules** |

**The JSONL session schema is the contract that binds all four.** `/mod` writes it, `/nn`
parses and trains on it, `/rl`'s bridge re-emits the same shapes, `/bot` validates it.
It is documented authoritatively in `mod/README.md` ("JSONL schema"). Treat it as a
published API.

---

## 2. Hard rules

Violating any of these is worse than not doing the task. If a task seems to require it,
stop and ask.

1. **Never commit, print, echo, paste, or otherwise reproduce `mod/webhook.txt`.** It
   contains a live Discord webhook URL. It is gitignored (`mod/.gitignore`). If you see
   its contents in tool output, do not repeat them in your response. The same applies to
   any `.env` file.
2. **Never add `/bot/` to git.** It is excluded via `/bot/` in the root `.gitignore`. It
   holds bot-token-adjacent operational logic and anti-abuse heuristics that lose their
   value if public. `/bot` receives security review only — not the full style/docs
   treatment the public components get. Do not "helpfully" un-ignore it.
3. **Never change the JSONL schema without bumping `schema_version`.** See
   "Cross-component invariants" below for the full ritual. A silent schema change breaks
   every already-collected session file.
4. **Never weaken a lint/type/test gate to make it pass.** Do not add `# type: ignore`,
   `# noqa`, `eslint-disable`, or loosen a `pyproject.toml` rule to get green. Fix the
   code. If a suppression is genuinely correct, it must carry a comment explaining why.
5. **Anchor `.gitignore` directory rules with a leading `/`.** An unanchored `data/`
   matches *every* directory named `data` at any depth — this repo previously lost the
   entire `nn/src/herbert_nn/data/` source package from version control that way. Write
   `/data/`, not `data/`.

---

## 3. Style by language

Match the surrounding code first. Where this file and existing code disagree, prefer the
existing code and mention the discrepancy.

### Java (`/mod`)

- **4-space indent.** No tabs. Lines up to ~120 chars (the codebase runs to ~134 where
  splitting would hurt readability; do not reflow existing lines just to shorten them).
- Braces on the same line (K&R). Always brace `if`/`for`/`while`, even single statements.
- `final` on fields wherever possible; the model classes are immutable value objects.
- Serialized field names use `@SerializedName("snake_case")` — the Java field stays
  `camelCase`, the JSON stays `snake_case`. **Never rename one without the other.**
- Package layout mirrors responsibility: `capture/`, `config/`, `model/`, `serialize/`,
  `session/`, `upload/`, `util/`.
- **Never let logging crash the game.** Every optional capture sub-section (block grid,
  opponent lookup, match context) is individually try/caught so one failure does not
  abort the tick. Preserve this pattern in new capture code.
- No empty catch blocks. Every catch either logs via `FMLLog.log(...)` or carries a
  comment explaining why swallowing is correct.

### Python (`/nn`, `/rl`)

- **Black, line-length 88.** Black is the source of truth for formatting; `E501` is
  ignored in ruff on purpose.
- **Ruff** with `select = ["E", "F", "W", "I", "UP", "B", "C4", "SIM", "D"]`. The `D`
  (pydocstyle) rules are on — docstrings are enforced, not optional.
- **mypy with `disallow_untyped_defs = true`.** Every function gets full annotations,
  including `-> None`. `src/` is checked; `tests/` is excluded.
- `from __future__ import annotations` at the top of every module.
- Prefer narrow types over `Any`. When a library returns a base class you need to narrow,
  use an `isinstance` guard that raises a clear `TypeError` — not a cast or an ignore.
  (See `rl/src/herbert_rl/train/rollout.py` for the established pattern.)
- Constants that define tensor layout or feature order live in `constants.py` and are
  imported everywhere — never re-literal them at a use site.

### JavaScript (`/rl/bridge`)

- **2-space indent, single quotes, semicolons required**, enforced by
  `rl/bridge/.eslintrc.json` (`eslint:recommended` + those three rules).
- CommonJS (`require`/`module.exports`), not ESM. `"sourceType": "script"`.
- Unused args must be prefixed `_` to pass `no-unused-vars`.
- **Zero runtime dependencies beyond what `package.json` already declares.** Tests use
  Node's built-in `node:test` — do not introduce Jest/Mocha/Chai.

### All languages

- **Every source file starts with an SPDX header**, before anything except a shebang:
  - Java/JS: `// SPDX-License-Identifier: MIT`
  - Python: `# SPDX-License-Identifier: MIT`
  - In `rl/bridge/src/bridge.js` the shebang comes first, then SPDX — that order matters
    for parsing.
- Files end with exactly one trailing newline. No trailing whitespace on any line.
- No magic numbers. Name them as constants with a comment explaining the value's origin.
- **No open `TODO`/`FIXME`.** Either do it, or document it as a known limitation in
  [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) with the reasoning.
- No dead code, no commented-out code. Git remembers it.

---

## 4. Documentation

Documentation is treated as part of the implementation, not an afterthought.

- **Every public symbol is documented**: Javadoc on public Java classes/methods/fields,
  docstrings on Python modules/classes/functions, JSDoc on exported JS functions.
- **Comments explain WHY, not WHAT.** `// increment i` is noise. `// Recursive retry had
  no bound and reset the timeout budget each call; a single deadline is shared instead.`
  is the standard. Most existing comments in this repo justify a decision or warn about a
  subtlety — match that register.
- Python docstrings use **Google style** (`Args:` / `Returns:` / `Raises:`). Document
  every raised exception the caller should handle.
- Module-level docstrings explain the module's *role in the pipeline*, not just its
  contents. Read `rl/src/herbert_rl/policy/sb3_policy.py` for the standard being set.
- **When behavior changes, the docstring changes in the same edit.** Several module
  docstrings in `/rl` describe cross-component coupling (which checkpoint keys get
  spliced where); these go stale silently and are actively misleading when they do.
- Update the relevant `README.md` when you change user-visible behavior, CLI flags,
  config keys, or the schema. Each component has its own README; the root
  [README.md](README.md) is the entry point.
- Add a [CHANGELOG.md](CHANGELOG.md) entry (Keep a Changelog format) for user-visible
  changes.

---

## 5. Testing

- **Every new public function gets tests.** Every bug fix gets a regression test that
  fails before the fix.
- Test names describe the behavior and the condition:
  `test_line_parse_rejects_before_parsing_when_line_count_exceeds_max_ticks`. In Python
  tests, the name *is* the documentation (`D103` is disabled for `tests/`).
- Tests must be **deterministic** — seed every RNG, never depend on wall-clock time,
  network, or a live Minecraft server.
- Never write tautological assertions (`assert x == x`). **Verify expected values
  independently** — the SHA-256 vectors in `HashUtilTest.java` were computed with Python's
  `hashlib` before being hardcoded. Do the equivalent.
- Prefer real execution over mocks where feasible; the suites here run in seconds.

### Running the suites

Run every suite your change could affect, and paste real output — never claim green
without having run it.

```bash
# /mod — Java. REQUIRES JDK 8 (ForgeGradle 2.1 will not run on newer JDKs).
cd mod && JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 ./gradlew build

# /nn — Python
cd nn && source .venv/bin/activate
python -m pytest -q
black --check . && mypy src
find src tests -name '*.py' -print0 | xargs -0 ruff check   # see gotcha below

# /rl — Python
cd rl && source .venv/bin/activate
python -m pytest -q
black --check . && mypy src
find src tests -name '*.py' -print0 | xargs -0 ruff check

# /rl/bridge — JavaScript
cd rl/bridge && npm run lint && npm test
```

---

## 6. Environment gotchas

These will waste your time if you do not know them up front.

- **`/mod` needs JDK 8.** `JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64`. ForgeGradle 2.1 /
  Gradle 2.14.1 fail on modern JDKs. The Forge toolchain is already cached locally.
- **`/rl`'s venv must be Python 3.11.** `pyproject.toml` pins `>=3.10,<3.12`; a venv built
  with the system Python (3.14) cannot install the package. Rebuild with
  `python3.11 -m venv .venv && pip install -e ".[dev]"`.
- **`ruff check <directory>/` is unreliable in this environment.** `ruff check src/`
  silently reports "All checks passed" while missing real findings. Always use
  `ruff check .` from the project root, or explicit enumeration:
  `find src tests -name '*.py' -print0 | xargs -0 ruff check`. This has caused real missed
  violations — do not trust the bare-directory form.
- **`black .` and other bare-directory invocations skip gitignored paths.** If a source
  directory is accidentally gitignored, it is silently never formatted or linted. (This
  is how `nn/src/herbert_nn/data/` accumulated formatting drift.)
- `mypy` runs with `ignore_missing_imports = true`. If the real dependencies are not
  installed, library types degrade to `Any` and mypy silently passes over genuine type
  errors. **Install the dev extras before trusting a clean mypy run.**

---

## 7. Cross-component invariants

These couplings are not visible from any single file. Breaking one usually fails
*silently* — no exception, just wrong behavior or lost weights.

### The JSONL schema (`/mod` → `/nn`, `/rl`, `/bot`)

Changing the recorded schema requires **all** of:
1. Bump `SCHEMA_VERSION` in `mod/src/main/java/dev/herbert/bridgelogger/util/HerbertConstants.java`.
2. Update the field table in `mod/README.md`, marking which version added/changed the field.
3. Add a versioned schema module under `nn/src/herbert_nn/schemas/` and register it in
   `SCHEMA_REGISTRY` in `nn/src/herbert_nn/schemas/registry.py`.
4. Check `/rl/src/herbert_rl/schema.py` (a parallel copy) and `/bot`'s validator.

`/nn`'s Pydantic models use `extra="forbid"`. **A new field the mod emits will hard-fail
parsing until the schema module is updated** — this is deliberate, so drift is loud.

### Duplicated code that must be hand-synced

`/rl` may not import `herbert_nn` at runtime (see `rl/src/herbert_rl/constants.py`).
Consequently these are deliberate copies, and a change to one requires the same change to
the other:

- `nn/src/herbert_nn/data/vocab.py` ↔ `rl/src/herbert_rl/vocab.py`
- `nn/src/herbert_nn/data/normalization.py` ↔ `rl/src/herbert_rl/normalization.py`
- `nn/src/herbert_nn/schemas/v1_0_0.py` ↔ `rl/src/herbert_rl/schema.py`
- `_TARGET_KEYS` exists in **both** `nn/src/herbert_nn/data/dataset.py` and
  `nn/src/herbert_nn/training/engine.py`. Adding a target key requires editing both.

### Checkpoint weight loading (`/nn` → `/rl`)

`rl/src/herbert_rl/policy/backbone.py` is an **architectural mirror** of `/nn`'s
`encoder.py` + `mlp.py`/`gru.py` — same submodule attribute names, same layer order, same
shapes. `checkpoint_adapter.py` loads a `/nn` checkpoint into it with
`load_state_dict(strict=False)`.

- If `/nn`'s encoder or trunk architecture changes, **`backbone.py` must be updated by
  hand** or weights are silently dropped (`strict=False` only warns).
- Output heads are *not* part of the backbone. They are extracted by state-dict key
  (`"<name>_head.linear.weight"`) and spliced into specific rows of SB3's `action_net`.
  **The head's Python attribute name in `/nn` is the checkpoint key** — renaming
  `self.movement_head` silently breaks the splice with no error. Row ownership is defined
  by the `*_ACTION_NET_ROWS` slice constants in `checkpoint_adapter.py` and must stay
  consistent with the 8-dim encoding documented in `rl/src/herbert_rl/env/action_wrapper.py`.

### Preprocessing cache invalidation (`/nn`)

The cache directory name is a hash of `PreprocessConfig` + schema version + raw file list.
**If you change the per-tick tensor layout** (add/remove/reorder a feature or target),
bump `feature_schema_version` in `nn/src/herbert_nn/data/config.py`. Otherwise an existing
cache is silently reused and training reads stale tensors — or crashes with a `KeyError`
on the missing key.

---

## 8. Before you report finished

Work through this list. Do not claim completion until each is true.

- [ ] Every affected test suite actually run, with real output reviewed — not assumed.
- [ ] Lint, format, and type checks clean for every language touched (using the reliable
      ruff invocation from §6).
- [ ] New/changed public symbols documented; stale docstrings updated in the same edit.
- [ ] READMEs / CHANGELOG updated if user-visible behavior changed.
- [ ] SPDX header present on every new file; single trailing newline; no trailing whitespace.
- [ ] No new `TODO`, no suppressed warnings, no weakened config.
- [ ] `git status` reviewed — nothing unintended staged, nothing secret, `/bot/` still ignored.
- [ ] Anything you chose *not* to fix is written down in
      [KNOWN_LIMITATIONS.md](KNOWN_LIMITATIONS.md) with the reasoning, not left silent.

**Report honestly.** If tests fail, say so and show the output. If you skipped a step, say
which. Never describe unverified work as verified — a confidently wrong "all green" is the
most expensive thing you can produce here.
