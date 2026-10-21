::: warning Vendored slice — do not edit here
Synced from `graphopt:docs/public_api.md` @ `fb00089` — the phase-1
frozen manifest (`API_VERSION = "0.1.0"`). This page is the **gamekit
slice**: the game-layer rows of the contract, landed at the plan-0032
phase-4 split. The substrate rows are
[eqsat's](/eqsat/reference/frozen-api) and the domain rows are
[graphopt's](/graphopt/reference/frozen-api).
:::

# The frozen public API — the gamekit (game) slice

Plan 0032 lane A landed the in-repo extraction first; the phase-4 split
moved the package to `handl-lang/gamekit`. Canonical paths are
`gamekit.*`; the old `mlopt_discovery.{engine,scoring,advantage,players,
lawdata.game}` shims live in graphopt and are not part of this contract.

## Game contracts

| Contract | Canonical location | Frozen surface |
|---|---|---|
| `Board` protocol | `gamekit.engine` | `Board` (`observe() -> state` / `step(action) -> (state, report)`; `state.done` terminates), `Report` (floor: `reward`, `terminal`), `Trajectory` (`.reports`, `.total`, `.terminal`), `run_episode(board, player, budget)`, `evaluate(arms, cases, board_of, *, budget)`, `train_policy(player, cases, board_of, *, budget)` |
| Player registry | `gamekit.players.registry` | `PlayerSpec` (`name`/`kind`/`boards`/`schema`/`train`/`artifact`/`record`/`variant`), `BOARDS`, `EPISODE_BOARDS`, `names()`, `spec(name)`, `specs()`, `for_board(board)`, `module(name)`, **`make(name, domain, *, seed=0, ctx=None) -> (i -> player)`** — `ValueError` on an unplayed board, never a silent run |
| Player machinery | `gamekit.players` | `LinearPolicy`, `DomainSpec` (incl. `to_record`/`build_from_record`), `LearnedPlayer`, `PlayHist`, the feature parts `Bias`/`StateBroadcast`/`PlayRates`/`OpOneHot`/`ActionFeats`/`RelFeats`, `compose_featurizer`, `register_part`/`register_legal`/`register_rank`, `spec_from_record`, `PARTS`/`LEGALS`/`RANKS` |
| Scoring spec | `gamekit.scoring` | `ScoreGate{axis, threshold, direction}`, `ScoringSpec{gates, weights, references}`, `GateVerdict`, `ScoreVerdict`, `score(spec, measured) -> ScoreVerdict`, `AXES` (`"speed"`, `"memory"`, `"launch"`, `"params"`), `DIRECTIONS` — the referee's value function **as data**; see below |
| Substrate-free kernel | import-linter contract | `gamekit.engine`, `gamekit.scoring`, `gamekit.calibrate`, `gamekit.advantage`, `gamekit.gamedata`, `gamekit.players._base`, `gamekit.players.registry` import nothing outside the stdlib — the `gamekit ⊥ eqsat` edge, enforced (see `pyproject.toml`) |

## The scoring spec — the referee seam

**The value spec is gamekit's; the measurement is the gauntlet's.**

```python
from gamekit.scoring import ScoreGate, ScoringSpec, score

spec = ScoringSpec(
    gates=(
        ScoreGate("speed", 0.95),     # delivered latency ≤ 0.95 × ref
        ScoreGate("params", 1.0),     # parameter bytes at most the ref
        ScoreGate("win_rate", 0.5, direction="at_least"),
    ),
    weights={"speed": 1.0, "memory": 0.25},
    references={"memory": 4.0e8},     # ungated axes need a normaliser
)
verdict = score(spec, measured)       # measured: {axis: value} — the
                                      # gauntlet's emission shape
verdict.passed                        # every gate cleared?
verdict.value                         # Σ w·(measurement/bar) — nan
                                      # when a weighted axis is absent
verdict.gates                         # per-gate GateVerdicts
verdict.missing                       # axes the measurement lacked
```

The two sides meet only through a plain `Mapping[str, float]`:
measurement code never imports this module, and this module imports no
measurement machinery (the "game kernel is substrate-free" import
contract).

## Boundary notes

- **`imitate_policy`** (`gamekit.engine`) is a documented driver that
  was never added to `__all__` — reachable but unlisted. It is **not**
  frozen; promote it deliberately or treat it as internal.
- **Registry contents vs registry shape**: `PlayerSpec` rows and the
  `gamedata` weight/feature/move tables are *data* — the names and
  access functions are frozen, the rows are each domain repo's
  business.

The full manifest — including the substrate and domain rows — is
[`docs/public_api.md`](https://github.com/handl-lang/graphopt/blob/main/docs/public_api.md)
upstream.
