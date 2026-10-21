# Players & the registry

<!-- provenance: gamekit:packages/gamekit/src/gamekit/players@417d550 + graphopt:docs/public_api.md@fb00089 -->

A player is a folder in `gamekit/players/` — a docstring carrying the
measured record, a `SPECS` tuple of `PlayerSpec` rows, and a `make` arm
factory. The registry (`gamekit.players.registry`) maps name → spec so
drivers and CLIs enumerate arms from one table.

## The roster

| player | kind |
|---|---|
| `random` | chance — the paired baseline every eval runs against |
| `playbook` | scripted — the hand-written moves |
| `linear` (`linear-greedy`, `linear-cold`) | the trained linear prior — a featurizer-injected `LinearPolicy` |
| `neural` | the neural arm |
| `distilled` | the distilled contraction-ordering player |
| `gated` | the `GatedPlayer` — an isolated stop channel with a `forced_action` hook |

## `PlayerSpec` — the registration contract

`name` / `kind` (`chance` | `scripted` | `linear` | `neural` |
`distilled`) / `boards` / `schema` / `train` / `artifact` / `record` /
`variant`. `boards` is the honest capability statement — list only what
`make` accepts; `record` is the one-line measured claim that earns the
seat; `variant` carries decode/weights flags the factory reads.

## The registry surface

`names()`, `spec(name)`, `specs()`, `for_board(board)`, `module(name)`,
`BOARDS`/`EPISODE_BOARDS`, and
**`make(name, domain, *, seed=0, ctx=None) -> (i -> player)`** — raises
`ValueError` on an unplayed board, never a silent run. `register_player`
is the plugin seam downstream domains use to seat new arms. Loading is
lazy: folders import on first `spec()`/`make()`.

## The machinery

`gamekit.players._base` — `LinearPolicy`, `DomainSpec` (incl.
`to_record`/`build_from_record`), `LearnedPlayer`, `GatedPlayer`,
`PlayHist`, the composable feature parts (`Bias`, `StateBroadcast`,
`PlayRates`, `OpOneHot`, `ActionFeats`, `RelFeats`,
`compose_featurizer`), and the `register_part`/`register_legal`/
`register_rank` + `PARTS`/`LEGALS`/`RANKS` registries.

## Recipes & records

- [Register a player](/graphopt/how-to/register-a-player) — the full
  recipe (the `make` contract, the `_MODULES` table, the measured
  record).
- [The player leaderboard](/graphopt/reference/leaderboard) — the
  published measured records.

Source:
[`gamekit/players/`](https://github.com/handl-lang/gamekit/tree/main/packages/gamekit/src/gamekit/players).
