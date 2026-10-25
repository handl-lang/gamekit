"""``gated`` — the isolated stop channel: ranking + a side-channel quit.

The delivery the eight-round stop-policy loop kept converging on
(``docs/results_stop_isolated.md``): the ``stop`` row corrupts the
ranking whenever it shares the evaluate softmax, and every
weights-table fold of a working gate lands a degenerate point — so
the decision is delivered *outside* the softmax instead.  The player
is a :class:`~gamekit.players.GatedPlayer`: the inner ranker (the
domain's learned arm, its ``legal`` filtered to non-``stop`` moves)
proposes the ordering exactly as ``linear`` does, and the
``forced_action`` hook the engine consults first carries the quit
decision — a ``w_stop · φ_stop`` head or an injected score callable
compared against the calibrated break-even ``kappa``.

Two seats:

* ``gated`` — the mechanism: ``ctx["gated"]`` feeds the channel's
  config, in either form:

  * the *flat* in-process form ``{spec, weights, kappa, score,
    rank_weights}`` — a live ``DomainSpec`` and/or scorer callable;
  * the *serialized payload* form ``{rank: {spec, weights}, stop:
    {spec, weights, kappa, scorer, scorer_params}}`` — every leaf a
    literal: spec blocks are :meth:`DomainSpec.to_record` records
    reified through the ``parts``/``legal``/``rank`` tags the domain
    registers, and ``stop.scorer`` names a
    :data:`gamekit.players.SCORERS` entry (the same registry idiom)
    when the gate rides a callable.

  Absent the config the channel is inert and the arm drains — the
  honest never-stop control a registered player must still play.
* ``open-gated`` — the shipped binding: ``variant={"baked": True}``
  plays :data:`gamekit.gamedata.OPEN_GATED_PAYLOAD` (the
  ``vhead:baked`` bake candidate — incumbent spec records, the
  shipped open prior ranking, the ridge value head at κ=0) from
  shipped data alone; a ``ctx["gated"]`` payload still overrides.

Measured record (``docs/results_stop_isolated.md``, round 9 — the
memoized-verdict protocol): the channel reproduces the analytic
break-even gates and the calibrated head arms bit-for-bit against
their wrapper-gate twins, with the ranking softmax never seeing the
quit decision.  ``open-gated``'s record is round 10's
(``docs/results_stop_gate.md``): held-out **0.2893 reward/ep, 0.2101
AP, 26 stops** — the first serializable arm to clear both bake bars
honestly.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from gamekit.gamedata import OPEN_GATED_PAYLOAD
from gamekit.players._base import (
    SCORERS,
    GatedPlayer,
    LearnedPlayer,
    spec_from_record,
)
from gamekit.players.registry import PlayerSpec

__all__ = ["SPECS", "make"]

SPECS: tuple[PlayerSpec, ...] = (
    PlayerSpec(
        name="gated",
        kind="gated",
        boards=("open",),
        schema=(
            "ranker: the domain's DomainSpec row; stop row: "
            "ctx['gated']'s spec (or record) — the channel's schema"
        ),
        train="— (scorer + κ are the payload; ctx['gated'] binds them)",
        artifact="ctx['gated'] payload — flat config or {rank, stop} record",
        record=(
            "the isolated-delivery arm — the quit decision rides the "
            "engine's forced_action channel, never the ranking "
            "softmax (docs/results_stop_isolated.md)"
        ),
    ),
    PlayerSpec(
        name="open-gated",
        kind="gated",
        boards=("open",),
        schema=(
            "the incumbent open spec — spec record literals over "
            "the shipped schema (rank + stop rows alike)"
        ),
        train=(
            "bench/drivers/policy/stop_gate.py — the vhead:baked arm "
            "(shipped-prior ranker + ridge stop head, round 10)"
        ),
        artifact=(
            "gamedata.OPEN_GATED_PAYLOAD — w_stop literals + κ=0 "
            "over OPEN_GAME_PLAYER_WEIGHTS"
        ),
        record=(
            "open 0.2893 reward/ep held-out, AP 0.2101, 26 stops — "
            "the first serializable gate clearing both bake bars "
            "(docs/results_stop_gate.md)"
        ),
        variant={"baked": True},
    ),
)

#: The baked seat's shipped payload per board — ``variant.baked``
#: binds it when ``ctx["gated"]`` carries no override.
_BAKED_PAYLOADS: dict[str, dict] = {"open": OPEN_GATED_PAYLOAD}


def _named_scorer(name: str) -> Any:
    """Resolve a payload's ``stop.scorer`` tag — honest on unknown."""
    fn = SCORERS.get(name)
    if fn is None:
        raise ValueError(
            f"unknown gated scorer {name!r} — registered: "
            f"{', '.join(sorted(SCORERS)) or '—'}"
        )
    return fn


def make(
    domain: str,
    *,
    spec: PlayerSpec,
    seed: int = 0,
    ctx: dict | None = None,
) -> Callable[[int], Any]:
    """Return the gated arm factory for *domain* — ``i -> player``.

    ``ctx['gated']`` carries the channel's config — the flat
    ``{weights, kappa, score, spec, rank_weights}`` form (live
    ``DomainSpec``/scorer) or the serialized ``{rank, stop}``
    payload (spec records, weight literals, κ, a named scorer); the
    payload decodes exactly the term the measurement played — a
    ``json.dumps`` round-trip — so the seat's numbers are the
    artifact's numbers.  ``variant.baked`` binds the board's shipped
    payload (:data:`_BAKED_PAYLOADS`) when ctx carries none.
    Unconfigured, the channel never fires early — the arm drains
    like a no-stop ``linear`` (the never-stop control the
    registered seat owes).  ``ctx['score_fn']`` deposits the
    candidate-only page ranker.
    """
    from mlopt_discovery.play import DOMAINS  # ty: ignore[unresolved-import] — lazy domain seam (absent here by design)

    ctx = {} if ctx is None else ctx
    spec0 = DOMAINS[domain].spec
    if spec0 is None:
        raise ValueError(f"domain {domain!r} ships no DomainSpec")
    cfg = dict(ctx.get("gated") or {})
    if not cfg and spec.variant.get("baked"):
        # the shipped seat plays the serialized artifact itself —
        # the same JSON round-trip the measurement proved
        cfg = json.loads(json.dumps(_BAKED_PAYLOADS[domain]))
    if "rank" in cfg or "stop" in cfg:
        # the serialized payload form — spec blocks are records
        rank = cfg.get("rank") or {}
        stop = cfg.get("stop") or {}
        rec = rank.get("spec")
        rspec = spec0 if rec is None else spec_from_record(rec)
        w_rank = rank.get("weights", spec0.weights)
        rec = stop.get("spec")
        sspec = rspec if rec is None else spec_from_record(rec)
        w_stop = stop.get("weights")
        kappa = float(stop.get("kappa", 0.0))
        score = None
        scorer = stop.get("scorer")
        if scorer is not None:
            score = _named_scorer(scorer)(
                sspec, stop.get("scorer_params") or {}, w_rank
            )
    else:
        # the flat in-process form — live spec / scorer objects
        rspec = spec0
        sspec = cfg.get("spec", spec0)
        w_stop = cfg.get("weights")
        kappa = float(cfg.get("kappa", 0.0))
        score = cfg.get("score")
        w_rank = cfg.get("rank_weights", spec0.weights)
    legal = rspec.legal

    def _eval_only(state: Any) -> tuple:
        """Return the ranker's pick set — every non-stop legal move."""
        return tuple(
            a for a in legal(state) if getattr(a, "op", None) != "stop"
        )

    def _stop_move(state: Any) -> Any:
        return next(
            (
                a
                for a in legal(state)
                if getattr(a, "op", None) == "stop"
            ),
            None,
        )

    def _stop_feats(state: Any, hist: Any) -> dict:
        return sspec.featurize(state, _stop_move(state), hist)

    def factory(i: int) -> GatedPlayer:
        inner = LearnedPlayer(
            seed + 40_000 + i,
            spec=rspec,
            weights=w_rank,
            legal=_eval_only,
            learn=False,
        )
        return GatedPlayer(
            inner,
            legal=legal,
            stop_feats=_stop_feats,
            w_stop=w_stop,
            kappa=kappa,
            score=score,
        )

    if domain == "open":
        ctx["score_fn"] = LearnedPlayer(
            seed, spec=rspec, weights=w_rank
        ).rank_score
    return factory
