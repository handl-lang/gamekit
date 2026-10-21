# gamekit

The game layer for the mlopt game — extracted from
`handl-lang/graphopt` (plan 0032 phase 4).  Substrate-free by
contract (`gamekit ⊥ eqsat`): the kernel imports nothing outside the
stdlib.

- `packages/gamekit` — `gamekit.engine` (the `Board` protocol +
  `run_episode`/`evaluate`/`train_policy`/`imitate_policy` drivers),
  `gamekit.players` (`LinearPolicy`, the `DomainSpec`-injected
  `LearnedPlayer`, the name→spec `registry`, and the player folders
  `random`/`playbook`/`linear`/`neural`/`distilled`/`gated`),
  `gamekit.scoring` (the referee's value spec as data),
  `gamekit.calibrate` (Platt/isotonic calibration + metrics),
  `gamekit.advantage` (the actor-critic player),
  `gamekit.gamedata` (the game weight/feature/move tables).

Downstream domain repos (`graphopt`, `kernelopt`) consume this as a
pinned git+ssh dep on `packages/gamekit` — the domain brings its own
substrate into the caller's environment; the player folders' `make`
factories bind it lazily.

## Develop

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q
.venv/bin/ruff check packages
.venv/bin/ruff format --check packages
.venv/bin/ty check
.venv/bin/lint-imports
```
