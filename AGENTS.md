# Agent guide — gamekit

The game layer for the mlopt game, extracted from
`handl-lang/graphopt` (plan 0032 phase 4).  Python 3.13; the shipped
package is **stdlib-only by contract** — the game kernel is
substrate-free (`gamekit ⊥ eqsat`), enforced by the import-linter
contract in `pyproject.toml`.

## Layout

- `packages/gamekit/src/gamekit/` — the kernel:
  `engine` (the `Board` protocol, `Report`, `Trajectory`,
  `run_episode`/`evaluate`/`train_policy`/`imitate_policy`),
  `players._base` (`LinearPolicy`, `DomainSpec`, `LearnedPlayer`,
  `GatedPlayer`, `PlayHist`, the composable feature parts +
  `PARTS`/`LEGALS`/`RANKS` registries), `players/registry`
  (`PlayerSpec`, the name→spec table, `register_player` — the
  downstream-domain seam, `make`),
  `players/{random,playbook,linear,neural,distilled,gated}/` — each
  folder ships its `PlayerSpec`s + a `make` arm factory that binds
  domain machinery **lazily** (the domain repo brings
  `mlopt_discovery`/`mlopt_torch`/`torch` into the caller's env; in
  this repo those calls decline honestly — tested),
  `scoring` (`ScoreGate`/`ScoringSpec`/`score` — the referee's value
  function as data), `calibrate` (Platt/isotonic + metrics),
  `advantage` (`AdvantagePlayer` — actor-critic), `gamedata`
  (weight/feature/move tables).
- `tests/` — the kernel suite: a hand-rolled toy board drives every
  driver path; domain-bound coverage (players seated on real boards)
  stayed in graphopt.

## Substrate consumers

`handl-lang/graphopt` and `handl-lang/kernelopt` pin this repo as a
git+ssh dep on `packages/gamekit`.  `register_player(spec, make)` is
the plugin seam: a fresh name must supply `make`; a name matching
the shipped roster unions `boards` into the shipped seat.

## Verification

```sh
uv sync --group dev
.venv/bin/python -m pytest tests/ -q          # the suite
.venv/bin/ruff check packages                  # lint (shipped source)
.venv/bin/ruff format --check packages         # formatting
.venv/bin/ty check                             # typecheck — 0 errors
.venv/bin/lint-imports                         # substrate-free contract
coverage run -m pytest tests/ -q && coverage report -m   # floor in pyproject
```

- `ty` is pinned to the parent's locked 0.0.84 (parity with graphopt;
  unpin once the floor re-measures clean on newer ty).
- Coverage floor re-measured on the fresh kernel suite — see
  `[tool.coverage.report] fail_under`; tighten, never lower.
- The `mlopt_discovery.{engine,scoring,advantage,players,lawdata.game}`
  shims live in graphopt — never import them here.

<!-- myco:agent-awareness -->

## Agent awareness

- Session start / any work discussion: run `myco status --prompt`
  before proposing next steps — know the board first.
- User describes new work → offer to capture it (`myco todo add`,
  `issue new`, `proposal new`). Don't wait to be asked.
- Picking up a task: `myco resume <ID>` is the briefing.
- Wrapping up: `myco handoff <ID>` anything resumable; keep statuses
  honest (claimed → in-progress → blocked/done).

<!-- /myco:agent-awareness -->
