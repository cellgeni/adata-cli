"""Comparative benchmark for adata-cli, against anndata and scanpy.

Report-only, and deliberately separate from `tests/`. The tests measure
operation counts and gate merges; this measures wall time and peak RSS, runs
on tags, and never fails a build -- timing on a shared runner is too noisy to
gate on.

    uv run python -m benchmarks.run --tier smoke
    uv run python -m benchmarks.report results.json
"""
