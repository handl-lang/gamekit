"""Shared toy domain for the gamekit suite.

The kernel is substrate-free, so the fixtures are a hand-rolled toy
board — a counter environment that satisfies the ``Board`` protocol
and exercises every driver path without domain machinery.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from gamekit.engine import Report
from gamekit.players import (
    ActionFeats,
    Bias,
    DomainSpec,
    OpOneHot,
    StateBroadcast,
)


@dataclass(frozen=True)
class ToyAction:
    """A move — the ``.op`` attribute is what the players read."""

    op: str


@dataclass
class ToyState:
    """The observed state — ``done`` and ``steps`` are the contract."""

    steps: int
    limit: int
    pool: tuple = ()

    @property
    def done(self) -> bool:
        return self.steps >= self.limit


class ToyBoard:
    """A counting board: ``limit`` steps of +1 reward, then done.

    ``legal`` enumerates the two toy moves until the limit; the board
    reports terminal cost/cert on the last step like the arenas do.
    """

    def __init__(self, limit: int = 3, moves=("a", "b")) -> None:
        self._state = ToyState(0, limit)
        self._moves = tuple(moves)

    def observe(self) -> ToyState:
        return self._state

    def legal(self, state: ToyState) -> list[ToyAction]:
        if state.done:
            return []
        return [ToyAction(op) for op in self._moves]

    def step(self, action) -> tuple[ToyState, Report]:
        self._state = ToyState(self._state.steps + 1, self._state.limit)
        term = self._state.done
        return self._state, Report(
            action=action,
            reward=1.0,
            terminal=term,
            cost=2.5 if term else float("nan"),
            certificate_ok=True if term else None,
        )


def toy_legal(state: ToyState) -> list[ToyAction]:
    """The legal-move enumerator a ``DomainSpec`` binds."""
    if state.done:
        return []
    return [ToyAction("a"), ToyAction("b")]


def _toy_state_feats(state: ToyState) -> dict[str, float]:
    return {"st:steps": float(state.steps)}


def toy_spec(weights=None) -> DomainSpec:
    """A minimal ``DomainSpec`` over the toy board's moves."""
    return DomainSpec.build(
        features=("bias", "st:steps", "op:a", "op:b"),
        legal=toy_legal,
        parts=(
            Bias(),
            StateBroadcast(_toy_state_feats),
            OpOneHot(frozenset({"op:a", "op:b"})),
        ),
        weights=weights,
    )


@pytest.fixture
def board() -> ToyBoard:
    return ToyBoard()
