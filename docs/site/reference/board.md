# The `Board` protocol

<!-- provenance: gamekit:packages/gamekit/src/gamekit/engine.py@450731f + graphopt:docs/public_api.md@ed89ead -->

A **board** is one optimization game as an environment — observe a
state, play a move, get a report, repeat until done. The contract is
two methods and stays deliberately thin so anything can sit under the
drivers.

## The protocol

```python
class Board(Protocol):
    def observe(self) -> Any: ...                        # the state
    def step(self, action) -> tuple[Any, Report]: ...    # state', report
```

| name | contract |
|---|---|
| `Board.observe()` | return the read-only state snapshot; `state.done` ends the episode |
| `Board.step(action)` | apply the action; return `(state', report)` |
| `Report` | `action`/`reward`/`terminal` are the floor — plus `applied`, `note`, `cost`, `certificate_ok`, `detail` for boards with richer reports |
| `Trajectory` | `.reports`, `.total` (the summed return), `.terminal` |

## The drivers

```python
run_episode(board, player, budget) -> Trajectory
evaluate(arms, cases, board_of, *, budget) -> dict[str, list[dict]]
train_policy(player, cases, board_of, *, budget) -> list[float]
```

A player is any callable `state -> action | None`; returning `None`
ends the episode early. Two optional channels ride on the player
object:

- **`forced_action`** — a side channel consulted *before* the main
  policy each step (the `GatedPlayer`'s isolated stop head — the quit
  call never rides the ranking softmax).
- **`finish_episode(total, rewards=...)`** — the hook `train_policy`
  calls after each episode for the REINFORCE update; a player without
  it is eval-only.

`imitate_policy(player, expert, cases, board_of, *, budget)` exists as
a documented driver but is **not** in `__all__` — reachable, unfrozen.

## Writing a board

The recipe — state dataclass, `Report`, smoke-test through
`run_episode`, then seat it through the domain registry — is
[add a board](/graphopt/how-to/add-a-board) on the graphopt side (the
registry is the domain's; the protocol is this repo's).

Source: [`gamekit/engine.py`](https://github.com/handl-lang/gamekit/blob/main/packages/gamekit/src/gamekit/engine.py).
