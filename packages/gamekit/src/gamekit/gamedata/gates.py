"""The gate payloads — the ``gated`` seats' shipped bindings.

A gated player (:class:`gamekit.players.GatedPlayer`) is
*parameterized* by a payload — the pure-data ``ctx["gated"]`` term
``{rank: {spec: <DomainSpec record>, weights}, stop: {spec: <record>,
weights, kappa, scorer, scorer_params}}`` — and like the weight
tables a measured binding is content, so it lives here.  Every leaf
is a literal: spec blocks are :meth:`DomainSpec.to_record` records
(callables reify through the ``parts``/``legal``/``rank`` tags the
domain registers), weight blocks are ``{feature: w}`` tables, κ a
float, ``scorer`` a named :data:`gamekit.players.SCORERS` entry.
"""

from __future__ import annotations

from .weights import OPEN_GAME_PLAYER_WEIGHTS

__all__ = ["OPEN_GATED_PAYLOAD"]

#: The incumbent open spec as a ``DomainSpec`` record — the shipped
#: schema (:data:`mlopt_discovery.open_game.OPEN_GAME_FEATURES`),
#: parts, enumerator and ranker by tag, ``weights=None`` (the
#: weight block rides in ``rank.weights``/``stop.weights``).  Both
#: the ranker and the stop row featurize through it — the
#: ``vhead:baked`` bake paired the *shipped prior's* rank table
#: with a value head fit on the incumbent's stop rows.
_OPEN_SPEC_RECORD: dict = {
    "features": [
        "bias",
        "st:pending_frac",
        "st:seen_frac",
        "st:useful_rate",
        "st:true_rate",
        "h:evaluate",
        "h:stop",
        "op:evaluate",
        "op:stop",
        "a:depth",
        "a:metas",
        "a:ops",
        "a:score",
        "a:score_rank",
        "a:score_rel",
        "a:form:comm",
        "a:form:assoc",
        "a:form:rotate",
        "a:form:rekey",
        "a:form:factor-left",
        "a:form:factor-right",
        "a:form:unary-hom",
        "a:form:pull-left",
        "a:form:pull-right",
        "a:form:unary-swap",
        "a:form:involution",
        "a:form:unary-inverse",
        "a:form:unary-absorb",
        "a:form:self-op",
        "a:form:idempotent",
        "a:form:lit-self",
        "a:form:const-op",
        "a:form:id-right",
        "a:form:id-left",
        "a:form:annihilate-right",
        "a:form:annihilate-left",
        "a:form:cancel-self",
        "a:form:cancel-unary-right",
        "a:form:cancel-unary-left",
        "a:form:hom-cross",
        "a:form:bin-unary-right",
        "a:form:bin-unary-left",
        "a:form:unary-of-lit",
        "x:score:depth",
        "x:score:ops",
    ],
    "legal": "open",
    "parts": [
        "bias",
        "open:state",
        "open:hist",
        "open:onehot",
        "open:action",
        "open:score-rank",
        "open:score-rel",
        "open:score-x-depth",
        "open:score-x-ops",
    ],
    "weights": None,
    "rank": "open",
    "lr": 0.05,
}

#: The ``open-gated`` seat's baked payload — the ``vhead:baked`` arm
#: of ``bench/drivers/policy/stop_gate.py`` (round 10,
#: ``docs/results_stop_gate.md``), the first serializable
#: out-of-softmax gate to clear both bake bars honestly: held-out
#: **0.2893 reward/ep, 0.2101 AP, 26 stops** (14.56 evals/ep — real
#: gating, not the never-stop confound).
#:
#: * ``rank`` — the incumbent spec record scoring candidate moves
#:   under the shipped open prior
#:   (:data:`OPEN_GAME_PLAYER_WEIGHTS` — referenced, not copied:
#:   the same trained snapshot the ``linear`` seat plays);
#: * ``stop`` — the ridge value head as ``w_stop`` literals over the
#:   incumbent spec's stop rows, κ = 0.0 (the head's own
#:   continue-value zero), no named scorer — the pure
#:   ``w·φ_stop ≤ κ`` linear path.
#:
#: The head weights are the round-10 literals (the driver's
#: ``results.heads["vhead:baked"].weights`` — ridge fit on the
#: train-page drains, 4-decimal snapshot).  Regenerate by re-running
#: the driver and pasting the emitted weights; the seat itself is
#: ``players/gated`` (``variant={"baked": True}`` binds this table
#: when ``ctx["gated"]`` carries no override).
OPEN_GATED_PAYLOAD: dict = {
    "rank": {
        "spec": _OPEN_SPEC_RECORD,
        "weights": OPEN_GAME_PLAYER_WEIGHTS,
    },
    "stop": {
        "spec": _OPEN_SPEC_RECORD,
        "weights": {
            "st:pending_frac": 0.1437,
            "st:seen_frac": -0.1437,
            "st:useful_rate": -1.437,
            "st:true_rate": 1.351,
            "h:evaluate": -0.1428,
            "bias": 0.2228,
        },
        "kappa": 0.0,
        "scorer": None,
        "scorer_params": None,
    },
}
