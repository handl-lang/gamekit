# gamekit — reference

<!-- provenance: docs: -->

The game-layer contract surface, pinned against the phase-1 frozen
manifest (`graphopt:docs/public_api.md`).

- [The `Board` protocol](./board) — `observe`/`step`, `Report`,
  `Trajectory`, and the three drivers.
- [Players & the registry](./players) — `PlayerSpec`, the name→spec
  table, the `make` seam, and the shipped roster.
- [The frozen API slice](./frozen-api) — the game rows of the manifest,
  vendored verbatim.

The domain registry (`mlopt_discovery.play.DOMAINS`) is a
[graphopt](/graphopt/reference/frozen-api) contract — the `Domain`
*shape* is frozen, the board contents are domain data.
