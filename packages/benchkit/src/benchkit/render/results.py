"""Results document — the pinned baselines, rendered into one page.

`docs/results.md` is generated from `bench/baselines/*.json` so the
README can quote *findings* while every number keeps its provenance
(device, git sha, timestamp) and links back to the committed baseline.
"""

from __future__ import annotations

from pathlib import Path


def _provenance(env: dict, name: str) -> str:
    """One line naming the machine a baseline was measured on."""
    when = env.get("timestamp_utc", "?")
    dev = env.get("device_name", "?")
    sha = env.get("git_sha", "?")
    dirty = " (dirty)" if env.get("git_dirty") else ""
    return (
        f"Measured {when} on {dev} (git `{sha}`{dirty}) — "
        f"baseline [`bench/baselines/{name}`]"
        f"(../bench/baselines/{name})."
    )


def render_results_doc(
    baselines_dir: str | Path,
    title: str = "mlopt — measured results",
    *,
    spec_lookup=None,
    docs_dir: str | Path | None = None,
) -> str:
    """Render every pinned baseline as one Markdown document.

    *spec_lookup* is an optional ``name -> spec`` catalog lookup
    (graphopt passes ``bench.registry.get``; the spec needs ``title``,
    ``question`` and ``expects`` attributes).  Without one every
    baseline is rendered under its file stem and the question/expect
    lines are omitted — benchkit has no suite catalog of its own.

    When *docs_dir* is given, two hand-maintained files are appended
    verbatim, in order, if present: ``results_tooling.md``
    (tooling-level measurements) and ``results_curated.md`` (curated
    sections that are not derivable from a pinned baseline — e.g. the
    machine-derived structure tables).  Editing them there keeps the
    generated file reproducible; hand-editing the generated file
    breaks the doc-sync test.
    """
    from benchkit.report import Report

    lines = [
        f"# {title}",
        "",
        "Generated from the pinned baselines in `bench/baselines/` —",
        "regenerate with `python -m bench results --out docs/results.md`.",
        "Each entry states what the suite asked, what it expected, what",
        "it measured, and the machine it measured on.  The full",
        "per-suite reports (timings, plots, raw metrics) are produced by",
        "`python -m bench report <baseline.json>`.",
        "",
    ]
    for path in sorted(Path(baselines_dir).glob("*.json")):
        if spec_lookup is not None:
            try:
                spec = spec_lookup(path.stem)
            except KeyError:
                continue
        else:
            spec = None
        report = Report.from_json(path)
        lines += [
            f"## {getattr(spec, 'title', None) or path.stem}",
            "",
        ]
        question = getattr(spec, "question", None)
        if question:
            lines += [f"*{question}*", ""]
        expects = getattr(spec, "expects", None)
        if expects:
            lines += [f"**Expected** — {expects}", ""]
        if report.findings:
            lines += [
                "| verdict | finding | headline |",
                "|---|---|---|",
            ]
            for f in report.findings:
                lines.append(
                    f"| **{f.verdict.badge}** | {f.claim} | {f.headline} |"
                )
            lines.append("")
        else:
            lines += [
                "> no findings recorded — baseline is not golden",
                "",
            ]
        lines += [_provenance(report.env, path.name), ""]
    body = "\n".join(lines).rstrip() + "\n"
    if docs_dir is not None:
        docs = Path(docs_dir)
        for name in ("results_tooling.md", "results_curated.md"):
            trailer = docs / name
            if trailer.exists():
                body += "\n" + trailer.read_text().rstrip() + "\n"
    return body
