# gamekit — the game layer

<!-- provenance: docs: + gamekit:README.md@417d550 -->

**gamekit** is the game layer of the mlopt stack, extracted from
`graphopt` at the plan-0032 phase-4 repo split (`research-mlopt` →
`graphopt` → three public repos). Discovery is a game — observe a
state, play a move, get a report, repeat until done — and gamekit names
that shape once and puts the drivers, the players, and the referee's
value spec over it.

The kernel is **substrate-free by contract** (`gamekit ⊥ eqsat`,
enforced by an import-linter contract): the shipped package imports
nothing outside the stdlib. A domain repo brings its own substrate into
the caller's environment; the player factories bind it lazily and
decline honestly when it is absent.

## The shape

```python
class Board(Protocol):
    def observe(self) -> state: ...                  # state.done ends
    def step(self, action) -> (state, report): ...   # report: reward + terminal
```

One `Board` protocol, every domain. The drivers (`run_episode`,
`evaluate`, `train_policy`) ride on it; the player registry maps
name → spec so a CLI enumerates arms from one table; `ScoringSpec`
declares the referee's value function *as data*.

## What's inside

| module | role |
|---|---|
| `gamekit.engine` | the `Board` protocol, `Report`, `Trajectory`, `run_episode`/`evaluate`/`train_policy`/`imitate_policy` |
| `gamekit.players` | `LinearPolicy`, `DomainSpec`, `LearnedPlayer`, `GatedPlayer`, the composable feature parts, the `random`/`playbook`/`linear`/`neural`/`distilled`/`gated` folders, the name→spec `registry` |
| `gamekit.scoring` | `ScoreGate`/`ScoringSpec`/`score` — the referee's value spec as data |
| `gamekit.calibrate` | Platt/isotonic score calibration + metrics |
| `gamekit.advantage` | the actor-critic player |
| `gamekit.gamedata` | the game weight/feature/move tables |

## How it's consumed

`graphopt` pins `packages/gamekit` as a git dependency — `uv sync` in a
`graphopt` clone resolves it; the domain registry
(`mlopt_discovery.play`) seats the boards. Working on the kernel
itself:

```sh
git clone https://github.com/handl-lang/gamekit
cd gamekit
uv sync --group dev
.venv/bin/python -m pytest tests/ -q     # the kernel suite (a toy board)
```

## Read next

- [The substrate-free kernel](./explanation/the-kernel) — why the game
  layer imports nothing, and how the referee seam stays honest.
- Reference: [the `Board` protocol](./reference/board) ·
  [players & the registry](./reference/players) ·
  [the frozen API slice](./reference/frozen-api).
- The cross-cutting account of the game — boards, players, the
  learned-policy loop — is [the game](/graphopt/explanation/the-game)
  on the graphopt side.
- Source: [`handl-lang/gamekit`](https://github.com/handl-lang/gamekit).
