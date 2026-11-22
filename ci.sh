#!/usr/bin/env bash
# bupha entry point (runner.kind = "script", cmd = "./ci.sh").
# The repo's verification ladder (AGENTS.md): uv sync dev group →
# ruff lint+format → ty → vulture → the import-linter contract → the
# suite under the pyproject coverage floor.
#
# `ty` is scoped to packages/gamekit on purpose: `uv sync --group dev`
# installs only the root + gamekit (benchkit's torch/jinja2/plotly/
# polars deps stay out — the CUDA torch stack does not fit this host's
# disk), so an unscoped `ty check` reports unresolved benchkit imports.
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

uv sync --group dev
.venv/bin/ruff check packages
.venv/bin/ruff format --check packages
.venv/bin/ty check packages/gamekit
.venv/bin/vulture packages
.venv/bin/lint-imports
.venv/bin/coverage run --source=gamekit -m pytest tests/ -q
.venv/bin/coverage report -m
