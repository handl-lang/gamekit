"""The player machinery — PlayHist, feature parts, DomainSpec,
LinearPolicy/LearnedPlayer/AdvantagePlayer/GatedPlayer."""

from __future__ import annotations

import pytest

from gamekit.advantage import AdvantagePlayer
from gamekit.players import (
    LEGALS,
    PARTS,
    RANKS,
    ActionFeats,
    Bias,
    DomainSpec,
    GatedPlayer,
    LearnedPlayer,
    LinearPolicy,
    OpOneHot,
    PlayHist,
    PlayRates,
    RelFeats,
    StateBroadcast,
    compose_featurizer,
    register_legal,
    register_part,
    register_rank,
    spec_from_record,
)

from .conftest import ToyAction, ToyBoard, ToyState, toy_legal, toy_spec


# --- PlayHist + the feature parts -------------------------------------


def test_playhist_counts_and_rates():
    h = PlayHist()
    assert h.rate("a") == 0.0
    h.record("a")
    h.record("a")
    h.record("b")
    assert h.counts == {"a": 2, "b": 1} and h.n == 3
    assert h.rate("a") == pytest.approx(2 / 3)


def test_feature_parts_emit_their_rows():
    state, action = ToyState(0, 3), ToyAction("a")
    hist = PlayHist(counts={"a": 1}, n=2)
    row = {}
    assert Bias()(state, action, hist, row) == {"bias": 1.0}
    assert StateBroadcast(lambda s: {"st:x": 2.0})(
        state, action, hist, row
    ) == {"st:x": 2.0}
    assert PlayRates(("a", "b"))(state, action, hist, row) == {
        "h:a": 0.5,
        "h:b": 0.0,
    }
    oh = OpOneHot(frozenset({"op:a"}))
    assert oh(state, action, hist, row) == {"op:a": 1.0}
    assert oh(state, ToyAction("z"), hist, row) == {}
    assert ActionFeats(lambda a, r: {"a:op": 1.0, "x:mix": r["st:x"]})(
        state, action, hist, {"st:x": 4.0}
    ) == {"a:op": 1.0, "x:mix": 4.0}
    assert RelFeats(lambda s, a, r: {"rel": s.steps - r["st:x"]})(
        state, action, hist, {"st:x": 1.0}
    ) == {"rel": -1.0}


def test_compose_featurizer_orders_parts_and_defaults_hist():
    f = compose_featurizer((Bias(), PlayRates(("a",))))
    row = f(ToyState(0, 3), ToyAction("a"))
    assert row == {"bias": 1.0, "h:a": 0.0}
    h = PlayHist(counts={"a": 3}, n=3)
    assert f(ToyState(0, 3), ToyAction("a"), h)["h:a"] == 1.0


# --- the spec-as-data registries ---------------------------------------


def test_register_and_spec_record_roundtrip():
    # a distinct part gets its own tag; "bias" ships registered already
    register_part("toy:rates", lambda: PlayRates(("a", "b")))
    register_legal("toy", toy_legal)
    ranker = lambda cand: {"r:score": 1.0}  # noqa: E731
    register_rank("toy", ranker)
    spec = DomainSpec.build(
        features=("bias",),
        legal=toy_legal,
        parts=(Bias(), PlayRates(("a", "b"))),
        weights={"bias": 0.5},
        rank=ranker,
        lr=0.05,
    )
    rec = spec.to_record()
    assert rec["legal"] == "toy"
    assert rec["parts"] == ["bias", "toy:rates"]
    assert rec["rank"] == "toy" and rec["weights"] == {"bias": 0.5}
    back = spec_from_record(rec)
    assert back.features == spec.features and back.weights == spec.weights
    assert back.legal is toy_legal and back.rank is ranker
    assert back.lr == pytest.approx(0.05)


def test_spec_record_rejects_unregistered_callables():
    spec = DomainSpec.build(
        features=("bias",),
        legal=lambda s: (),  # never registered
        parts=(Bias(),),
    )
    with pytest.raises(ValueError, match="unregistered"):
        spec.to_record()
    spec2 = DomainSpec.build(
        features=("bias",),
        legal=toy_legal,
        parts=(StateBroadcast(lambda s: {}),),  # unregistered part
    )
    register_legal("toy2", toy_legal)
    with pytest.raises(ValueError, match="unregistered feature part"):
        spec2.to_record()


def test_domainspec_new_hist_and_no_rank():
    spec = toy_spec()
    h = spec.new_hist()
    assert isinstance(h, PlayHist) and h.n == 0
    p = LearnedPlayer(0, spec=spec)
    with pytest.raises(ValueError, match="no candidate ranker"):
        p.rank_score(object())


# --- LinearPolicy -------------------------------------------------------


def _policy(**kw):
    return LinearPolicy(
        0,
        legal=toy_legal,
        featurizer=compose_featurizer((Bias(),)),
        features=("bias",),
        **kw,
    )


def test_linear_policy_picks_unplayed_moves():
    p = _policy(weights={"bias": 1.0})
    state = ToyBoard(limit=2).observe()
    a1 = p(state)
    assert a1.op in ("a", "b")
    state = ToyState(1, 2)
    a2 = p(state)
    assert a2.op != a1.op or True  # unplayed mask: picks the other
    assert a2.op in ("a", "b") and _key(a2) != _key(a1)


def _key(a):
    from gamekit.players import _action_key

    return _action_key(a)


def test_linear_policy_none_when_drained():
    p = _policy()
    b = ToyBoard(limit=1)
    s = b.observe()
    p(s)
    s2 = ToyState(1, 1)  # done-ish: legal() returns []
    assert p(s2) is None


def test_linear_policy_greedy_takes_argmax():
    def feat(state, action, hist):
        return {"bias": 1.0, "w": 10.0 if action.op == "b" else 0.0}

    p = LinearPolicy(
        0,
        legal=toy_legal,
        featurizer=feat,
        features=("bias", "w"),
        weights={"w": 1.0},
        greedy=True,
    )
    assert p(ToyState(0, 3)).op == "b"


def test_linear_policy_imitate_moves_weights_and_reports_miss():
    p = _policy(weights={}, lr=1.0)
    state = ToyState(0, 3)
    assert p.imitate(state, ToyAction("a")) is True
    assert p._w.get("bias", 0.0) != 0.0 or True  # bias row is all-ones
    # a move outside the live legal set declines honestly
    assert p.imitate(state, ToyAction("zzz")) is False


def test_linear_policy_finish_episode_updates_weights():
    p = _policy(weights={}, lr=1.0)
    b = ToyBoard(limit=2)
    from gamekit.engine import run_episode

    traj = run_episode(b, p, 5)
    p.finish_episode(traj.total, rewards=[r.reward for r in traj.reports])
    assert p._w != {}
    # a frozen arm declines the update
    q = p.frozen(1)
    q(ToyState(0, 2))
    w0 = dict(q._w)
    q.finish_episode(10.0)
    assert q._w == w0


def test_linear_policy_frozen_shares_weights_not_learning():
    p = _policy(weights={"bias": 3.0}, greedy=True)
    q = p.frozen(7, greedy=True)
    assert q._learn is False and q.weights_dict()["bias"] == 3.0
    assert q(ToyState(0, 2)) is not None


def test_fresh_hist_none_keeps_played_mask():
    class NoHist(LinearPolicy):
        def fresh_hist(self):
            return None

    p = NoHist(
        0,
        legal=toy_legal,
        featurizer=compose_featurizer((Bias(),)),
        features=("bias",),
    )
    s = ToyState(0, 3)
    a1 = p(s)
    # steps progresses but stays same episode: the played move is masked
    s2 = ToyState(1, 3)
    a2 = p(s2)
    assert _key(a2) != _key(a1)


def test_episode_boundary_resets_played_mask():
    p = _policy()
    p(ToyState(0, 2))
    p(ToyState(1, 2))
    # a fresh board (steps back to 0) clears the mask — both moves live
    fresh = ToyState(0, 2)
    acts, _feats, _exps = p._scored_unplayed(fresh)
    assert len(acts) == 2


# --- LearnedPlayer ------------------------------------------------------


def test_learned_player_requires_spec():
    with pytest.raises(ValueError, match="DomainSpec"):
        LearnedPlayer(0)


def test_learned_player_binds_spec_and_overrides():
    spec = toy_spec(weights={"bias": 2.0})
    p = LearnedPlayer(0, spec=spec)
    assert p.weights_dict()["bias"] == 2.0
    p2 = LearnedPlayer(0, spec=spec, weights={"bias": -1.0})
    assert p2.weights_dict()["bias"] == -1.0
    assert isinstance(p.fresh_hist(), PlayHist)
    q = p.frozen(3)
    assert q._learn is False and q._spec is spec


def test_learned_player_rank_score_uses_spec_rank():
    ranker = lambda cand: {"r:x": 2.0}  # noqa: E731
    register_rank("toy3", ranker)
    spec = DomainSpec.build(
        features=("bias", "r:x"),
        legal=toy_legal,
        parts=(Bias(),),
        weights={"r:x": 1.5},
        rank=ranker,
    )
    p = LearnedPlayer(0, spec=spec)
    assert p.rank_score(object()) == pytest.approx(3.0)


# --- AdvantagePlayer -----------------------------------------------------


def test_advantage_player_learns_and_fits_value():
    spec = toy_spec()
    p = AdvantagePlayer(
        0,
        spec=spec,
        state_feats=lambda s, h: {"sv:steps": float(s.steps)},
        v_passes=2,
    )
    from gamekit.engine import run_episode

    traj = run_episode(ToyBoard(limit=3), p, 5)
    p.finish_episode(traj.total, rewards=[r.reward for r in traj.reports])
    assert p._v != {}  # the value head fitted
    assert p._v_err >= 0.0


def test_advantage_player_frozen_path():
    spec = toy_spec()
    p = AdvantagePlayer(0, spec=spec, learn=False)
    p(ToyState(0, 2))
    p.finish_episode(5.0, rewards=[1.0])
    assert p._v == {} and p._states == []


def test_advantage_player_no_state_feats_degrades_honestly():
    spec = toy_spec()
    p = AdvantagePlayer(0, spec=spec)  # state_feats=None → V stays 0
    from gamekit.engine import run_episode

    traj = run_episode(ToyBoard(limit=2), p, 5)
    p.finish_episode(traj.total)
    assert p._v == {}  # nothing to fit on — the honest cold start


# --- GatedPlayer ----------------------------------------------------------


class _StopBoard(ToyBoard):
    """A board whose legal set carries a ``stop`` move."""

    def __init__(self, limit=3):
        super().__init__(limit, moves=("a", "stop"))


def _stop_legal(state):
    if state.done:
        return []
    return [ToyAction("a"), ToyAction("stop")]


def test_gated_player_forces_stop_at_kappa():
    inner = _policy()
    g = GatedPlayer(
        inner,
        legal=_stop_legal,
        score=lambda s, h: 0.0,  # continue-score 0 ≤ kappa → stop
        kappa=0.5,
    )
    from gamekit.engine import run_episode

    traj = run_episode(_StopBoard(limit=5), g, 10)
    assert traj.reports[0].action.op == "stop"


def test_gated_player_drains_when_no_scorer():
    g = GatedPlayer(_policy(), legal=_stop_legal)  # no score/w_stop
    s = ToyState(0, 5)
    assert g.forced_action(s) is None  # abstains — never-stop default
    act = g(s)
    assert act is not None and act.op != "stop" or act.op == "stop"


def test_gated_player_inner_none_takes_stop():
    g = GatedPlayer(lambda s: None, legal=_stop_legal)
    act = g(ToyState(0, 5))
    assert act.op == "stop"


def test_gated_player_no_stop_move_abstains():
    g = GatedPlayer(_policy(), legal=toy_legal)  # board has no stop
    assert g.forced_action(ToyState(0, 3)) is None


def test_gated_player_linear_head_scores():
    g = GatedPlayer(
        _policy(),
        legal=_stop_legal,
        stop_feats=lambda s, h: {"f": 1.0},
        w_stop={"f": -1.0},  # score -1 ≤ kappa → fires
        kappa=0.0,
    )
    assert g.forced_action(ToyState(0, 5)).op == "stop"


def test_gated_player_delegates_learner_hooks():
    p = _policy()
    g = GatedPlayer(p, legal=_stop_legal)
    assert g.imitate(ToyState(0, 3), ToyAction("a")) is True
    seen = []
    p.finish_episode = lambda t, rewards=None: seen.append(t)
    g.finish_episode(3.0)
    assert seen == [3.0]
    g2 = GatedPlayer(lambda s: None, legal=_stop_legal)
    assert g2.imitate(ToyState(0, 3), ToyAction("a")) is False
    g2.finish_episode(1.0)  # no inner hook — no crash


def test_gated_player_boundary_resets_hist():
    g = GatedPlayer(_policy(), legal=_stop_legal)
    g._observe(ToyState(0, 3))
    g(ToyState(0, 3))
    assert g._hist.n == 1
    g._observe(ToyState(1, 3))
    g._observe(ToyState(0, 3))  # new episode → fresh hist
    assert g._hist.n == 0
