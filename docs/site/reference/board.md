# The `Board` protocol

<!-- provenance: gamekit:packages/gamekit/src/gamekit/engine.py@417d550 + graphopt:docs/public_api.md@fb00089 -->

A **board** is one optimization game as an environment — observe a
state, play a move, get a report, repeat until done. The contract is two
methods and stays deliberately thin so anything can sit under the
drivers.

## The protocol

```python
class Board(Protocol):
    def observe(self) -> state: ...
    def step(self, action) -> tuple[state, Report]: ...
```

| name | contract |
|---|---|
| `Board.observe()` | return the read-only state snapshot; `state.done` ends the episode |
| `Board.step(action)` | apply the action; return `(state', report)` |
| `Report` | the floor — `reward` (the learning signal) and `terminal`; boards may return richer reports |
| `Trajectory` | `.reports`, `.total` (the summed return), `.terminal` |

## The drivers

```python
run_episode(board, player, budget) -> Trajectory
evaluate(arms, cases, board_of, *, budget)      # compare players on identical cases
train_policy(player, cases, board_of, *, budget)  # REINFORCE over episode returns
```

A player is any callable `state -> action | None`; returning `None`
ends the episode early. `imitate_policy` exists as a documented driver
but is **not** in `__all__` — reachable, unfrozen.

## Writing a board

The recipe — state dataclass, `Report`, smoke-test through
`run_episode`, then seat it through the domain registry — is
[add a board](/graphopt/how-to/add-a-board) on the graphopt side (the
registry is the domain's; the protocol is this repo's).

Source: [`gamekit/engine.py`](https://github.com/handl-lang/gamekit/blob/main/packages/gamekit/src/gamekit/engine.py).
