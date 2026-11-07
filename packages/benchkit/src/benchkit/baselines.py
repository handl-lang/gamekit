"""The baseline-adapter contract — the types a limits baseline binds.

A baseline is a folder that exports ``SPEC`` (a :class:`BaselineSpec`
row — the measured record that earned it a seat) and::

    build(ctx: BaselineCtx, *, spec: BaselineSpec) -> Callable

returning the arm to time — or raising :class:`SkipBaseline` with the
reason when it cannot run the case/environment.

This module is the *types only* half of the contract: the name→spec
enumeration machinery (``specs``/``names``/``for_case``/``availability``/
``build``) stays with the domain that owns the roster — graphopt's
``bench.limits.registry`` ships it; a domain with one baseline folder
needs only these three names for the folder's ``SPEC``/``build``
signature to stay a drop-in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "BaselineCtx",
    "BaselineSpec",
    "SkipBaseline",
]


class SkipBaseline(Exception):
    """A baseline declining a case — the row records *why*."""


@dataclass(frozen=True)
class BaselineSpec:
    """One baseline's registration — its seat on the roster.

    *cases* is the honest capability statement: the discovery-corpus
    case names the ``build`` accepts (``()`` = every case).
    *requires* names modules that must import for the arm to exist;
    *needs_cuda* gates CUDA-only kernels; *symbolic* says the arm
    has a dag_cost-pricable form (``False`` = measured-only, the
    fast lane reports it but cannot price it).  *record* is the
    one-line measured record the folder docstring argues at length.
    """

    name: str
    kind: str  # "eager" | "compile" | "order" | "kernel" | "impl" | "carrier"
    title: str = ""
    cases: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    needs_cuda: bool = False
    symbolic: bool = True
    note: str = ""
    record: str = ""


@dataclass
class BaselineCtx:
    """Everything a ``build`` may read — the case, its fixture, the run."""

    case: Any  # the domain's discovery-case object
    model: Any  # the spelled "before" module
    feed: Any  # tensor or positional-args tuple
    sr: Any  # the search() result (extracted term inside)
    sink: Any  # the verify + lower ports
    source: Any  # the IR exporter
    device: str = "cpu"
    lane: str = "fast"
