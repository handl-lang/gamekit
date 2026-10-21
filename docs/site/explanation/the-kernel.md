# The substrate-free kernel

<!-- provenance: gamekit:packages/gamekit/src/gamekit@417d550 + gamekit:AGENTS.md -->

The game kernel knows nothing about e-graphs, tensors, or laws. That is
a *contract*, not an accident: `gamekit.engine`, `gamekit.scoring`,
`gamekit.calibrate`, `gamekit.advantage`, `gamekit.gamedata`, and
`gamekit.players` import nothing outside the stdlib — a `gamekit ⊥
eqsat` edge enforced by the repo's import-linter contract.

## Why the wall exists

- **A board is a protocol, not a library.** `observe() -> state` and
  `step(action) -> (state, report)` is all the drivers need. Anything
  with two methods can be a board — including a hand-rolled toy that
  exercises every driver path in the kernel's own suite.
- **Players bind the domain lazily.** Each player folder ships a `make`
  arm factory that imports `mlopt_discovery`/domain machinery *inside*
  the factory — so `import gamekit.players` stays stdlib-only, and in
  this repo (where no domain substrate is installed) the factories
  decline honestly instead of failing at import.
- **`register_player(spec, make)` is the plugin seam.** A downstream
  domain seats its boards through the registry: a fresh name supplies
  `make`; a name matching the shipped roster unions `boards` into the
  shipped seat.

## The referee seam

The split that keeps the game honest runs through gamekit's edge:

- **the value spec is gamekit's** — `ScoreGate`/`ScoringSpec` declare
  the axes and thresholds (speed, memory, launch, params, win-rate) as
  *data*; `score(spec, measured)` consumes a plain
  `Mapping[str, float]` and returns gates cleared, the weighted value,
  and the axes the measurement lacked;
- **the measurement is the gauntlet's** — the measured numbers come from
  the admission machinery on the substrate/domain side
  ([the gauntlet](/graphopt/explanation/the-gauntlet)).

The two sides meet only through that mapping: measurement code never
imports the scoring module, and the scoring module imports no
measurement machinery. Calibration (`gamekit.calibrate` — Platt and
isotonic fits plus metrics) tunes *scores*, never verdicts.

## Where the rest lives

The episode machinery, the player roster, and the measured records are
in [the reference](../reference/frozen-api); the cross-cutting
narrative — why a game at all, the learned-policy loop, the meta-game —
is [the game](/graphopt/explanation/the-game) on the graphopt side.
