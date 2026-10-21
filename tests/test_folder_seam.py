"""The player folders' lazy domain binding — exercised through stub
``sys.modules`` seats.

The folders' ``make`` factories bind domain machinery lazily — in
production the domain repo imports ``mlopt_discovery.play.DOMAINS``
and friends.  These tests install stub modules for exactly that
surface, so the factory code paths run for real (a spec-driven
``LearnedPlayer`` over the toy spec, a scripted arm over stub
classes) without a cross-repo dependency.
"""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest

from gamekit.players.registry import make

from .conftest import ToyAction, ToyState, toy_legal, toy_spec


def _mod(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    return m


@pytest.fixture
def domain_env(monkeypatch):
    """Seat stub domain modules in ``sys.modules`` for the test."""
    spec0 = toy_spec()
    dom = SimpleNamespace(spec=spec0, legal=toy_legal)

    class Action:
        def __init__(self, op, params=None):
            self.op = op
            self.params = params or {}

    class GreedyPlayer:
        def __call__(self, state):
            return ToyAction("a")

    class ScriptedPlayer(GreedyPlayer):
        pass

    class GreedyEditor(GreedyPlayer):
        pass

    def _random_policy(legal):
        class R:
            def __init__(self, seed):
                self.seed = seed

            def __call__(self, state):
                return legal(state)[0] if legal(state) else None

        return R

    def _scripted_search():
        return lambda i: GreedyPlayer()

    meta_arena = _mod(
        "mlopt_discovery.meta_arena",
        GreedyPlayer=GreedyPlayer,
        ScriptedPlayer=ScriptedPlayer,
        Action=Action,
    )
    play = _mod(
        "mlopt_discovery.play",
        DOMAINS={"meta": dom, "open": dom, "search": dom, "player": dom},
        _random_policy=_random_policy,
        _scripted_search=_scripted_search,
    )
    proposal = _mod(
        "mlopt_discovery.proposal",
        _heuristic_score=lambda cand: 1.0,
    )
    player_arena = _mod(
        "mlopt_discovery.player_arena", GreedyEditor=GreedyEditor
    )
    discovery = _mod("mlopt_discovery")
    torch_stub = _mod("torch")

    class _Pol:
        def cost(self, tensors, sizes):
            return 1.5

        def best_order(self, tensors, sizes, *, samples, seed, temperature):
            return ([0, 1], 2.5)

    def greedy(tensors, sizes):
        return 3.25

    contraction = _mod(
        "mlopt_torch.contraction_policy",
        SAMPLE_TEMP=1.0,
        load_contraction_policy=lambda path, device="cpu": _Pol(),
        greedy=greedy,
        pair_cost=lambda x, y, s: 1.0,
        merge_tensors=lambda ts, a, b: [
            t for k, t in enumerate(ts) if k not in (a, b)
        ]
        + [frozenset(ts[a] ^ ts[b])],
        random_bond_network=lambda n, seed=0: ([], {}),
    )
    mlopt_torch = _mod("mlopt_torch")

    def compute_features(lhs):
        return {"f": 1.0}

    class RuleSample:
        def __init__(self, *args):
            self.args = args

    def rule_vector(rule):
        return {"r": 1.0}

    def input_vector(feats, vec):
        return [1.0, 2.0]

    def train_rule_value(samples, *, epochs, hidden, device, seed):
        class Net:
            def eval(self):
                return self

            def __call__(self, x):
                return [0.75]

        return Net()

    mlopt_core = _mod("mlopt_core")
    features = _mod("mlopt_core.features", compute_features=compute_features)
    trajectories = _mod(
        "mlopt_core.trajectories", RuleSample=RuleSample, rule_vector=rule_vector
    )
    learned_policy = _mod(
        "mlopt_torch.learned_policy",
        input_vector=input_vector,
        train_rule_value=train_rule_value,
    )
    # torch surface the neural scorer uses
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, *a):
            return False

    torch_stub.tensor = lambda xs, dtype=None: xs
    torch_stub.float32 = "float32"
    torch_stub.no_grad = _NoGrad

    mods = {
        "mlopt_discovery": discovery,
        "mlopt_discovery.play": play,
        "mlopt_discovery.meta_arena": meta_arena,
        "mlopt_discovery.proposal": proposal,
        "mlopt_discovery.player_arena": player_arena,
        "mlopt_torch": mlopt_torch,
        "mlopt_torch.contraction_policy": contraction,
        "mlopt_torch.learned_policy": learned_policy,
        "mlopt_core": mlopt_core,
        "mlopt_core.features": features,
        "mlopt_core.trajectories": trajectories,
        "torch": torch_stub,
    }
    for k, v in mods.items():
        monkeypatch.setitem(sys.modules, k, v)
    return SimpleNamespace(
        spec=spec0, action=Action, greedy_player=GreedyPlayer
    )


def test_random_make_binds_domain_legal(domain_env):
    arm = make("random", "meta", seed=2)
    player = arm(0)
    assert player(ToyState(0, 3)).op == "a"


def test_linear_make_builds_learned_arm(domain_env):
    ctx = {}
    arm = make("linear", "meta", seed=0, ctx=ctx)
    player = arm(0)
    assert player(ToyState(0, 3)).op in ("a", "b")


def test_linear_cold_blanks_weights(domain_env):
    from gamekit.players.linear import make as lmake
    from gamekit.players.registry import spec

    arm = lmake("meta", spec=spec("linear-cold"), seed=0, ctx={})
    assert callable(arm)


def test_linear_open_deposits_score_fn(domain_env):
    ctx = {}
    make("linear", "open", seed=0, ctx=ctx)
    assert callable(ctx["score_fn"])


def test_linear_requires_domainspec(domain_env):
    from gamekit.players.linear import make as lmake
    from gamekit.players.registry import spec

    import mlopt_discovery.play as play

    play.DOMAINS["bare"] = SimpleNamespace(spec=None, legal=toy_legal)
    with pytest.raises(ValueError, match="ships no DomainSpec"):
        lmake("bare", spec=spec("linear"), seed=0, ctx={})


def test_gated_make_builds_gated_player(domain_env):
    ctx = {}
    arm = make("gated", "open", seed=0, ctx=ctx)
    player = arm(0)
    from gamekit.players import GatedPlayer

    assert isinstance(player, GatedPlayer)
    assert callable(ctx["score_fn"])


def test_gated_make_cfg_binds_channel(domain_env):
    from gamekit.players.gated import make as gmake
    from gamekit.players.registry import spec

    cfg = {
        "gated": {
            "weights": {"bias": -1.0},
            "kappa": 0.25,
            "score": lambda s, h: 0.5,
            "rank_weights": {"bias": 1.0},
        }
    }
    arm = gmake("open", spec=spec("gated"), seed=0, ctx=cfg)
    assert callable(arm)


def test_neural_requires_train_rows(domain_env):
    with pytest.raises(ValueError, match=r"ctx\['train'\]"):
        make("neural", "open", seed=0, ctx={})


def test_neural_make_trains_and_scores(domain_env):
    cand = SimpleNamespace(
        lhs=object(),
        as_rule=lambda: SimpleNamespace(name="r1"),
        label="x",
    )
    outcome = SimpleNamespace(base_cost=10.0, cand_cost=4.0)
    ctx = {"train": [(cand, outcome)], "epochs": 1}
    arm = make("neural", "open", seed=0, ctx=ctx)
    assert callable(ctx["score_fn"])
    player = arm(0)
    state = SimpleNamespace(pending=[0], pool=[cand])
    act = player(state)
    assert act.op == "evaluate"
    state2 = SimpleNamespace(pending=[], pool=[cand])
    assert player(state2).op == "stop"


def test_distilled_make_binds_policy(domain_env):
    arm = make("distilled", "contraction", seed=0, ctx={"samples": 4})
    player = arm(2)
    assert player([], {}) == 2.5


def test_distilled_argmax_uses_cost(domain_env):
    arm = make("distilled-argmax", "contraction", seed=0)
    assert arm(0)([], {}) == 1.5


def test_playbook_greedy_arms(domain_env):
    for dom in ("meta", "search", "player", "contraction"):
        arm = make("playbook", dom) if dom != "contraction" else make(
            "greedy", "contraction"
        )
        assert callable(arm), dom
        player = arm(0)
        assert callable(player) or player is not None


def test_playbook_open_drain_deposits_score_fn(domain_env):
    ctx = {}
    arm = make("playbook", "open", ctx=ctx)
    assert callable(ctx["score_fn"])
    player = arm(0)
    cand = SimpleNamespace()
    state = SimpleNamespace(pending=[0], pool=[cand])
    act = player(state)
    assert act.op == "evaluate"
    empty = SimpleNamespace(pending=[], pool=[])
    assert player(empty).op == "stop"


def test_playbook_contraction_arm_calls_greedy(domain_env):
    arm = make("greedy", "contraction")
    assert arm(0)([frozenset({0})], {0: 1}) == 3.25


def test_random_contraction_arm_contracts(domain_env):
    arm = make("random", "contraction", seed=1)
    cost = arm(0)(
        [frozenset({0, 1}), frozenset({1, 2})], {0: 2, 1: 3, 2: 4}
    )
    assert isinstance(cost, float)
