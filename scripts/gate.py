#!/usr/bin/env python3
"""Repo gate — stdlib runner over the uv dev toolchain.

Usage: `uv run scripts/gate.py` (uv syncs the dev group first).

Steps run only when their tool/config exists:
  pytest → ruff check → ruff format --check → ty check → lint-imports.
`--coverage` adds `coverage run -m pytest` + `coverage report -m` instead of
the plain pytest step (for repos with a `[tool.coverage.report] fail_under`).
"""

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_BIN = ROOT / ".venv" / "bin"
PYPROJECT = ROOT / "pyproject.toml"


def tool(name: str) -> str | None:
    candidate = VENV_BIN / name
    if candidate.exists():
        return str(candidate)
    return shutil.which(name)


def src_dirs() -> list[str]:
    dirs = [d for d in ("src", "packages") if (ROOT / d).is_dir()]
    return dirs or ["."]


def main() -> int:
    coverage = "--coverage" in sys.argv
    pyproject = PYPROJECT.read_text() if PYPROJECT.is_file() else ""
    steps: list[tuple[str, list[str]]] = []

    if (ROOT / "tests").is_dir():
        pytest = tool("pytest") or tool("python")
        if pytest:
            if coverage and tool("coverage"):
                steps.append(("coverage", [tool("coverage"), "run", "-m", "pytest", "tests/", "-q"]))
                steps.append(("coverage-report", [tool("coverage"), "report", "-m"]))
            else:
                steps.append(("pytest", [pytest, "-m", "pytest", "tests/", "-q"]))

    ruff = tool("ruff")
    if ruff:
        for d in src_dirs():
            steps.append((f"ruff check {d}", [ruff, "check", d]))
            steps.append((f"ruff format {d}", [ruff, "format", "--check", d]))

    ty = tool("ty")
    if ty:
        steps.append(("ty", [ty, "check"]))

    lint_imports = tool("lint-imports")
    if lint_imports and ("importlinter" in pyproject or "import-linter" in pyproject):
        steps.append(("lint-imports", [lint_imports]))

    failed = []
    for name, argv in steps:
        print(f"$ {' '.join(argv)}")
        if subprocess.run(argv, cwd=ROOT).returncode != 0:
            failed.append(name)

    if failed:
        print(f"gate: FAILED — {', '.join(failed)}", file=sys.stderr)
        return 1
    print(f"gate: ok ({len(steps)} steps)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
