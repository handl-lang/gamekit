"""The player registry — roster, the downstream ``register_player``
seam, and the honest-degradation contract of the folder factories."""

from __future__ import annotations

import pytest

from gamekit.players import registry
from gamekit.players.registry import (
    BOARDS,
    EPISODE_BOARDS,
    PlayerSpec,
    for_board,
    make,
    module,
    names,
    register_player,
    spec,
    specs,
)


def test_roster_names_and_specs():
    ns = names()
    for expected in (
        "random",
        "playbook",
        "linear",
        "neural",
        "distilled",
        "gated",
    ):
        assert any(n.startswith(expected) for n in ns), (expected, ns)
    all_specs = specs()
    assert set(all_specs) >= set(ns)
    for s in all_specs.values():
        assert isinstance(s, PlayerSpec)
        assert set(s.boards) <= set(BOARDS)


def test_boards_vocabulary():
    assert "contraction" in BOARDS
    assert "meta" in EPISODE_BOARDS
    assert BOARDS == (*EPISODE_BOARDS, "contraction")


def test_spec_lookup_and_unknown_errors():
    s = spec("random")
    assert s.name == "random" and s.kind == "chance"
    with pytest.raises(ValueError, match="unknown player"):
        spec("no-such-player")
    with pytest.raises(ValueError, match="unknown player"):
        module("no-such-player")


def test_module_resolves_folder_and_spec_names():
    m = module("random")
    assert m.__name__ == "gamekit.players.random"
    # a spec alias (e.g. linear-cold) resolves to its folder too
    cold = next(n for n in names() if n.startswith("linear"))
    assert module(cold).__name__.startswith("gamekit.players.")


def test_for_board_lists_seated_players():
    seated = for_board("contraction")
    assert any(s.name == "random" for s in seated)
    assert for_board("no-such-board") == []


def test_make_refuses_unplayed_board():
    with pytest.raises(ValueError, match="does not play board"):
        make("random", "no-such-board")


def test_register_player_fresh_name_requires_make():
    with pytest.raises(ValueError, match="must supply its own make"):
        register_player(
            PlayerSpec(name="fresh-x", kind="linear", boards=("meta",))
        )


def test_register_player_fresh_name_with_make():
    calls = []

    def mk(domain, *, spec, seed, ctx):
        calls.append((domain, spec.name, seed))
        return lambda i: None

    try:
        register_player(
            PlayerSpec(name="fresh-y", kind="linear", boards=("meta",)),
            make=mk,
        )
        assert "fresh-y" in names()
        assert spec("fresh-y").boards == ("meta",)
        arm = make("fresh-y", "meta", seed=3)
        assert callable(arm)
        assert calls == [("meta", "fresh-y", 3)]
    finally:
        registry._EXTERNAL.pop("fresh-y", None)


def test_register_player_unions_boards_into_shipped_seat():
    shipped_boards = spec("random").boards
    try:
        register_player(
            PlayerSpec(name="random", kind="chance", boards=("kernel",))
        )
        merged = spec("random")
        assert set(merged.boards) == set(shipped_boards) | {"kernel"}
    finally:
        registry._EXTERNAL.pop("random", None)
    assert spec("random").boards == shipped_boards


def test_register_player_make_claims_only_new_boards():
    """An external ``make`` seats the boards the shipped spec didn't —
    the shipped folder keeps its own domains."""
    calls = []

    def mk(domain, *, spec, seed, ctx):
        calls.append(domain)
        return lambda i: None

    try:
        register_player(
            PlayerSpec(name="random", kind="chance", boards=("kernel",)),
            make=mk,
        )
        make("random", "kernel", seed=0)
        assert calls == ["kernel"]
        # a shipped board still routes to the folder factory, not ext
        calls.clear()
        with pytest.raises(Exception):
            make("random", "meta")  # folder make → domain seam
        assert calls == []
    finally:
        registry._EXTERNAL.pop("random", None)


def test_folder_makes_decline_honestly_without_domains():
    """The lazy domain binding: ``make`` builds the arm factory (the
    substrate is bound at *call* time), and invoking the arm without
    the domain's packages raises ``ImportError`` — honest degradation,
    never a silent wrong player."""
    arm = make("random", "contraction", seed=1)
    assert callable(arm)
    player = arm(0)
    tensors = [frozenset({0, 1}), frozenset({1, 2})]
    sizes = {0: 2, 1: 3, 2: 4}
    with pytest.raises(ModuleNotFoundError, match="mlopt_torch"):
        player(tensors, sizes)


def test_random_contraction_arm_is_seeded_factory():
    a = make("random", "contraction", seed=5)
    b = make("random", "contraction", seed=5)
    # factories bind the seed at make time — callables are independent
    assert callable(a) and callable(b)
    with pytest.raises(ModuleNotFoundError):
        a(0)([frozenset({0, 1})], {0: 1, 1: 1})


def test_domain_bound_makes_raise_without_substrate():
    """Folders whose board needs ``mlopt_discovery``/``mlopt_torch``
    decline at the lazy import inside ``make`` — or inside the arm's
    first call — never a silent wrong player."""
    with pytest.raises((ImportError, KeyError, ModuleNotFoundError)):
        make("linear", "meta", seed=0)
    with pytest.raises((ImportError, KeyError, ModuleNotFoundError)):
        make("playbook", "meta", seed=0)
