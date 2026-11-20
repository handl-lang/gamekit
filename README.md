# gamekit

The game layer for the mlopt game.  Substrate-free by
contract (`gamekit ⊥ eqsat`): the kernel imports nothing outside the
stdlib.

## Packages

- `packages/gamekit` — `gamekit.engine` (the `Board` protocol +
  `run_episode`/`evaluate`/`train_policy`/`imitate_policy` drivers),
  `gamekit.players` (`LinearPolicy`, the `DomainSpec`-injected
  `LearnedPlayer`, the name→spec `registry`, and the player folders
  `random`/`playbook`/`linear`/`neural`/`distilled`/`gated`),
  `gamekit.scoring` (the referee's value spec as data),
  `gamekit.calibrate` (Platt/isotonic calibration + metrics),
  `gamekit.advantage` (the actor-critic player),
  `gamekit.gamedata` (the game weight/feature/move tables).
- `packages/benchkit` — the benchmark harness:
  `Case`/`Variant`/`Runner`/`Report`,
  the JSONL ledger + resume `Checkpoint`, baseline comparison
  (`benchkit.baselines` contract types — `expectation_gaps` takes the
  caller's suite catalog, no `bench.registry` import), and the report
  renderers (`benchkit.render` — markdown/HTML/slidev/quarto/plots).
  **Not** substrate-free: it declares `torch`, `mlopt-core` and
  `jinja2` — the `gamekit ⊥ eqsat` contract covers only the `gamekit`
  package.  Its only check today is the selftest
  (`.venv/bin/python -m benchkit`); pytest coverage lands with the
  first domain consumer's suite.

Downstream domain repos (`graphopt`, `kernelopt`) consume this as a
pinned git dep on `packages/gamekit` (and `packages/benchkit`
for the harness) — the domain brings its own
substrate into the caller's environment; the player folders' `make`
factories bind it lazily.

## Develop

```sh
uv sync --group dev
```

## Verify

```sh
uv run scripts/gate.py --coverage   # pytest → ruff → ty → lint-imports → coverage floor
```

`scripts/gate.py` is a stdlib runner over the dev toolchain.  The
coverage floor (`[tool.coverage.report] fail_under = 98`) was
re-measured on the kernel suite — tighten, never lower.

## License

MIT.

## Documentation

<https://handl-lang.github.io/docs/gamekit/>
