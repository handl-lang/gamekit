# Players & the registry

<!-- provenance: gamekit:packages/gamekit/src/gamekit/players@450731f + graphopt:docs/public_api.md@ed89ead -->

A player is a folder in `gamekit/players/` — a docstring carrying the
measured record, a `SPECS` tuple of `PlayerSpec` rows, and a `make`
arm factory. The registry (`gamekit.players.registry`) maps
name → spec so drivers and CLIs enumerate arms from one table. Loading
is lazy: folders import on first `spec()`/`make()`, so
`import gamekit.players` stays stdlib-only.

## `PlayerSpec` — the registration contract

```python
@dataclass(frozen=True)
class PlayerSpec:
    name: str
    kind: str          # "chance" | "scripted" | "linear" | "neural"
                       # | "distilled" | "gated"
    boards: tuple[str, ...]
    schema: str = ""       # the feature vocabulary it scores on
    train: str = ""        # the entry point that trained the weights
    artifact: str = ""     # where the weights live
    record: str = ""       # the one-line measured claim
    variant: Mapping[str, Any] = {}  # decode/weights flags `make` reads
```

`boards` is the honest capability statement — list only what `make`
accepts (`make` raises `ValueError` on anything else). `variant`
carries flags like `{"greedy": True}` (argmax decode),
`{"cold": True}` (no weights), `{"argmax": True}` (single-pass).

## The board names

```python
EPISODE_BOARDS = ("meta", "joint", "torch", "search", "gen", "open",
                  "player")          # the play.DOMAINS keys
BOARDS = (*EPISODE_BOARDS, "contraction")   # + the ordering board
```

Declared here (not imported) so the player folders stay importable
without pulling the domain layer. Downstream domain repos seat their
own boards through `register_player` — a registered board is not an
`EPISODE_BOARDS` member, it is an extension seat on named players.

## The shipped roster

| spec name | folder | kind | boards |
|---|---|---|---|
| `random` | `random` | chance | every board — the paired-draw floor |
| `playbook` | `playbook` | scripted | the episode boards — each board's hand-known-good line |
| `greedy` | `playbook` | scripted | `meta`/`joint`/`torch`/`gen`/`contraction` — immediate-reward / cheapest-pair |
| `linear` | `linear` | linear | the episode boards — the featurizer-injected `LinearPolicy` prior |
| `linear-greedy` | `linear` | linear | episode boards — `variant={"greedy": True}` argmax decode |
| `linear-cold` | `linear` | linear | episode boards — `variant={"cold": True}`, the untrained control |
| `neural` | `neural` | neural | `open` — `RuleValueNet`, trained in-run on `ctx["train"]` rows |
| `distilled` | `distilled` | distilled | `contraction` — the sampled contraction-ordering policy |
| `distilled-argmax` | `distilled` | distilled | `contraction` — `variant={"argmax": True}` single pass |
| `gated` | `gated` | gated | `open` — the isolated stop channel via `forced_action` |
| `open-gated` | `gated` | gated | `open` — `variant={"baked": True}` plays `gamedata.OPEN_GATED_PAYLOAD` from shipped data |

## The registry surface

```python
names() -> tuple[str, ...]             # registered spec names, sorted
spec(name) -> PlayerSpec               # ValueError on unknown
specs() -> dict[str, PlayerSpec]       # the whole roster
for_board(board) -> list[PlayerSpec]   # everyone that can play it
module(name) -> ModuleType             # import the folder (lazy)
make(name, domain, *, seed=0, ctx=None) -> (i -> player)
register_player(spec, make=None) -> None
```

`make` returns the arm factory `i -> player`. On an *episode* board
the player is a `state -> action | None` callable as `run_episode`
expects; on `contraction` it is a `(tensors, sizes) -> cost` callable
— the ordering game scores final contraction cost, not episode reward.
A factory may also write back into `ctx` (`ctx["score_fn"]` — the
page-ordering scorer the leaderboard prices with average precision).
An unplayed board raises `ValueError` — the caller maps that to a
skipped row, never a silent run.

`register_player(spec, make=None)` is the plugin seam downstream
domain repos use to seat arms the folder roster cannot know about:

- **matching name** — the external spec's `boards` union into the
  shipped seat; spec-driven factories (`linear` reads
  `DOMAINS[domain].spec`, `random` is uniform) then bind the new board
  with no code here. An explicit `make` claims exactly the boards the
  shipped spec did not already play.
- **fresh name** — the spec registers outright and `make` is
  **required** (same signature the folder `make`s use:
  `(domain, *, spec=..., seed=..., ctx=...)`).

Re-registering a name replaces its external seat — plugin imports stay
idempotent.

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
