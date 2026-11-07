"""Dashboard renderer — a single index page across every suite.

Scans a results directory for canonical report JSON, (re)writes each
suite's HTML page, and emits ``index.html`` grouping suites by
category with their verdict badges.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from benchkit.report import Report

_ENV = Environment(
    loader=PackageLoader("benchkit", "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def _intent(suite: str, intent_of) -> str:
    try:
        return intent_of(suite)
    except Exception:
        return "other"


def _intent_order(intent_order) -> tuple[str, ...]:
    return tuple(intent_order) or ("other",)


def build_dashboard(
    results_dir: str | Path,
    outdir: str | Path | None = None,
    title: str = "mlopt benchmarks",
    *,
    intent_of=None,
    intent_order=None,
) -> Path:
    """Build ``index.html`` from ``*.json`` reports in ``results_dir``.

    *intent_of* is an optional ``suite -> category`` lookup (graphopt
    passes its registry's ``intent``); without one every suite groups
    under ``"other"``.  *intent_order* optionally names the category
    order; the default is ``("other",)``.
    """
    intent_of = intent_of or (lambda _suite: "other")
    intent_order = intent_order or ("other",)
    results_dir = Path(results_dir)
    outdir = Path(outdir) if outdir else results_dir

    reports: list[Report] = []
    for jpath in sorted(results_dir.glob("*.json")):
        if jpath.name == "ledger.jsonl":
            continue
        try:
            reports.append(Report.from_json(jpath))
        except Exception:
            continue

    buckets: dict[str, list[dict]] = {}
    for report in reports:
        page = outdir / f"{report.suite}.html"
        if not page.exists():
            report.to_html(page)
        buckets.setdefault(_intent(report.suite, intent_of), []).append(
            {
                "suite": report.suite,
                "title": report.title or report.suite,
                "ncells": len(report.cells),
                "when": report.env.get("timestamp_utc", "?"),
                "page": page.name,
                "findings": [
                    {
                        "verdict": f.verdict.value,
                        "badge": f.verdict.badge,
                    }
                    for f in report.findings
                ],
            }
        )

    env = reports[0].env if reports else {}
    groups = [
        (cat, buckets[cat])
        for cat in _intent_order(intent_order)
        if cat in buckets
    ]
    html = _ENV.get_template("dashboard.html.j2").render(
        title=title,
        groups=groups,
        nsets=len(reports),
        generated=datetime.now(UTC).isoformat(timespec="seconds"),
        git_sha=env.get("git_sha", "?"),
        git_dirty=env.get("git_dirty", False),
        device=env.get("device", "?"),
    )
    index = outdir / "index.html"
    index.write_text(html)
    return index
