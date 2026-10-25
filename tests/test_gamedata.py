"""The game-side content tables — shapes, not domain semantics."""

from __future__ import annotations

import gamekit.gamedata as gamedata
from gamekit.gamedata import features, moves, weights


def test_gamedata_package_surface():
    for name in gamedata.__all__:
        assert getattr(gamedata, name, None) is not None


def test_moves_table_is_nonempty_vocabulary():
    for name in moves.__all__:
        table = getattr(moves, name)
        assert table is not None, name


def test_features_tables_shapes():
    # the schema tuples
    for name in ("ARENA_FEATURES", "META_ARENA_FEATURES", "ARENA_HISTORY_STATS"):
        table = getattr(features, name)
        assert isinstance(table, tuple) and table, name
        assert all(isinstance(x, str) for x in table), name
    # the hash-bucket width tables
    for name in ("ARENA_HASH_BUCKETS", "META_ARENA_HASH_BUCKETS"):
        table = getattr(features, name)
        assert isinstance(table, dict) and table, name
        assert all(
            isinstance(k, str) and isinstance(v, int) and v > 0
            for k, v in table.items()
        ), name
    # the move vocabulary the schemas size off
    assert isinstance(moves.ARENA_MOVE_ORDER, tuple)
    assert all(isinstance(m, str) for m in moves.ARENA_MOVE_ORDER)


def test_weights_tables_are_numeric():
    seen = 0
    for name in weights.__all__:
        table = getattr(weights, name)
        assert table is not None, name
        if isinstance(table, dict):
            for k, v in table.items():
                seen += 1
                assert isinstance(k, str)
                assert isinstance(v, (int, float)) or isinstance(v, dict)
    assert seen > 0


def test_open_gated_payload_is_pure_data():
    """The ``open-gated`` seat's payload is the serializable term:
    JSON-clean leaves, spec blocks as records, the rank weights the
    shipped open prior's own table (referenced, not duplicated)."""
    import json

    from gamekit.gamedata import OPEN_GATED_PAYLOAD

    rec = json.loads(json.dumps(OPEN_GATED_PAYLOAD))
    rank, stop = rec["rank"], rec["stop"]
    assert rank["weights"] == weights.OPEN_GAME_PLAYER_WEIGHTS
    assert OPEN_GATED_PAYLOAD["rank"]["weights"] is (
        weights.OPEN_GAME_PLAYER_WEIGHTS
    )
    for block in (rank["spec"], stop["spec"]):
        assert block["legal"] == "open" and block["rank"] == "open"
        assert "bias" in block["features"]
        assert block["weights"] is None
    assert stop["kappa"] == 0.0 and stop["scorer"] is None
    assert all(isinstance(v, float) for v in stop["weights"].values())
