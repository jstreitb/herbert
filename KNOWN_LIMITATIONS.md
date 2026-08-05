# Known Limitations

This document tracks substantive, intentional gaps in Herbert's behavior that
were identified during development and code review but are not fixed in
code, either because the fix requires infrastructure or design work beyond
the current scope, or because the risk is judged low enough not to warrant
one. Nothing here should be surprising in production; if you hit one of
these, this is the expected behavior, not a bug.

## `/rl` training has no automatic recovery from a bridge process crash

If the Mineflayer bridge subprocess dies mid-episode (crashes, is killed, or
the Minecraft server connection is lost in a way `reconnect.js` can't
recover from), `BridgeProcessError` propagates uncaught up through
`collect_rollout` / `run_training` and terminates the whole training run.
Progress since the last checkpoint save is lost. There is no
supervisor/retry loop around a training run today. Restart training from
the last checkpoint (see `rl/README.md` for checkpoint/resume usage) if
this happens.

## RL bridge yaw/pitch sign convention is not independently re-verified

`rl/bridge/src/angles.js` encodes the yaw/pitch sign convention used to
translate between Mineflayer's coordinate system and the action space the
policy predicts. This is documented in-code and in
`rl/bridge/README.md`, but has not been independently re-verified against
a live server as part of this review beyond what the existing unit tests
in `rl/bridge/test/angles.test.js` cover.

## RL bridge always reports full opponent health

`opponent.health` in the RL bridge observation is always `20.0` (full
health). Mineflayer does not expose other players' health through a
convenience getter, and implementing it would require tracking damage
events independently. Documented in-code and in `rl/bridge/README.md`.

## `/mod`'s `BlockGridMapper` and `HeldItemCategoryMapper` have no automated tests

Both classes depend on `net.minecraft.init.Blocks` / `net.minecraft.init.Items`,
which cannot be statically initialized outside a fully bootstrapped
Minecraft/FML environment. Plain JUnit tests against these classes fail
with `NoClassDefFoundError` at class-load time. See `mod/build.gradle`'s
test dependency comment and the Testing section in `mod/README.md`. Every
other class in `/mod` that doesn't touch these Minecraft registries has
unit test coverage.

## `/bot`'s rate limiting only applies to accepted submissions

`herbert_bot`'s per-player rate limit is enforced on the accept path, not
on submissions that get rejected for infrastructure or content reasons. An
attacker with a valid Discord account could flood the intake channel with
malformed or oversized attachments without tripping the rate limit. The
size-limit (`HERWERT_MAX_ATTACHMENT_BYTES`) and early tick-count
short-circuit in `SessionValidator._check_line_parse` bound the *cost* of
each rejected submission, but do not bound submission *frequency*. Since
`/bot` is not part of the public repo, this is accepted as a low-priority
gap rather than fixed now; a moderator can manually rate-limit or ban an
abusive account via standard Discord tooling in the meantime.

## `npm audit` reports 6 moderate vulnerabilities in transitive bridge dependencies

All 6 are in `uuid`, pulled in via `@azure/msal-node` → `prismarine-auth` →
`minecraft-protocol` → `mineflayer`, and relate to Microsoft/online-mode
authentication. The project's bridge always connects with `auth: "offline"`
(see `rl/bridge/README.md`), so this code path is never exercised.
`npm audit fix --force` would force a breaking major-version bump of
`mineflayer` and was not applied. Re-evaluate if the bridge ever needs to
support online-mode auth.

## `/nn`'s modeling limitations

`herbert_nn`'s own known limitations (movement `forward`/`strafe` not
modeled as an RL output head, `attack_target_type` and `place_x,y,z` not
modeled as prediction targets, single-player behavioral-cloning
generalization concerns) are documented in the "Known limitations" section
of `nn/README.md` and are not duplicated here.

## `rl/bridge`'s JS test suite covers pure logic only

The 47 Node.js unit tests in `rl/bridge/test/` cover `angles.js`,
`blockGrid.js`, `itemClassifier.js`, `actionExecutor.js`'s `clampPitch`,
and `matchState.js` (via a fake bot object) — all server-independent, pure
logic. `bridge.js`, `reconnect.js`, and `observationBuilder.js` have no
automated tests, since exercising them requires a real or heavily-mocked
live Mineflayer connection. See the Testing section of
`rl/bridge/README.md`.
