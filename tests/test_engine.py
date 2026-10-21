"""The Board protocol and the episode drivers — toy-board coverage."""

from __future__ import annotations

import pytest

from gamekit.engine import (
    Board,
    Report,
    Trajectory,
    evaluate,
    imitate_policy,
    run_episode,
    train_policy,
)

from .conftest import ToyAction, ToyBoard, ToyState


class PickFirst:
    """A plain callable player — always the first legal move."""

    def __call__(self, state):
        return state and ToyAction("a")


def test_report_floor_fields():
    r = Report(action="x", reward=1.5)
    assert r.reward == 1.5 and r.terminal is False and r.applied
    assert r.certificate_ok is None and r.detail == {}


def test_trajectory_total_and_terminal():
    t = Trajectory([Report("a", 1.0), Report("b", 2.0, terminal=True)])
    assert t.total == 3.0
    assert t.terminal is t.reports[-1]
    assert Trajectory([]).terminal is None
    assert Trajectory([Report("a", 1.0)]).terminal is None


def test_board_protocol_runtime_checkable():
    assert isinstance(ToyBoard(), Board)


def test_run_episode_plays_until_done():
    traj = run_episode(ToyBoard(limit=3), PickFirst(), budget=10)
    assert len(traj.reports) == 3
    assert traj.total == 3.0
    assert traj.terminal is not None and traj.terminal.terminal


def test_run_episode_budget_caps():
    traj = run_episode(ToyBoard(limit=10), PickFirst(), budget=2)
    assert len(traj.reports) == 2


def test_run_episode_none_action_ends_early():
    calls = []

    def quitter(state):
        calls.append(state.steps)
        return None if state.steps >= 1 else ToyAction("a")

    traj = run_episode(ToyBoard(limit=5), quitter, budget=10)
    assert len(traj.reports) == 1
    assert calls == [0, 1]


def test_run_episode_forced_action_side_channel():
    class Gated:
        def __init__(self):
            self.fired = False

        def forced_action(self, state):
            if self.fired or state.steps < 1:
                return None
            self.fired = True
            return ToyAction("stop")

        def __call__(self, state):
            return ToyAction("a")

    traj = run_episode(ToyBoard(limit=4), Gated(), budget=10)
    assert [r.action.op for r in traj.reports] == ["a", "stop", "a", "a"]


def test_train_policy_calls_finish_episode():
    seen = []

    class P:
        def __call__(self, state):
            return ToyAction("a")

        def finish_episode(self, total, rewards=None):
            seen.append((total, rewards))

    totals = train_policy(P(), [0, 1], lambda c: ToyBoard(limit=2), budget=5)
    assert totals == [2.0, 2.0]
    assert seen == [(2.0, [1.0, 1.0]), (2.0, [1.0, 1.0])]


def test_train_policy_frozen_arm_has_no_hook():
    totals = train_policy(
        PickFirst(), [0], lambda c: ToyBoard(limit=2), budget=5
    )
    assert totals == [2.0]


def test_imitate_policy_counts_shown_and_imitated():
    class P:
        def __init__(self):
            self.got = []

        def imitate(self, state, action):
            self.got.append(action.op)
            return action.op != "b"  # 'b' leaves the legal set

    p = P()

    def expert(i):
        return lambda state: ToyAction("b" if i else "a")

    out = imitate_policy(p, expert, [0, 1], lambda c: ToyBoard(limit=1), budget=3)
    assert out == {"shown": 2, "imitated": 1}
    assert p.got == ["a", "b"]


def test_evaluate_paired_arms_rows():
    def mk(_i):
        return PickFirst()

    out = evaluate(
        {"first": mk, "also": mk},
        [("case-a",), ("case-b",)],
        lambda c: ToyBoard(limit=2),
        budget=5,
    )
    assert set(out) == {"first", "also"}
    row = out["first"][0]
    assert row["case"] == "case-a"
    assert row["steps"] == 2 and row["reward"] == 2.0
    assert row["cost"] == 2.5 and row["cert"] is True


def test_evaluate_case_index_fallback():
    out = evaluate(
        {"a": lambda i: PickFirst()},
        ["bare"],
        lambda c: ToyBoard(limit=1),
        budget=3,
    )
    assert out["a"][0]["case"] == 0
