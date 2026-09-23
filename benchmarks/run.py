"""Run the benchmark and write results.json.

Not a pytest module. Each contender has to run in its own process for peak
RSS to mean anything, and the whole thing takes tens of minutes at the `ci`
tier -- neither of which belongs in a suite that gates merges. The guards
that *do* gate merges are in `tests/test_performance.py`, and they measure
operation counts rather than time.

    uv run python -m benchmarks.run --tier smoke
    uv run python -m benchmarks.run --tier ci --out results.json

Baselines run from environments built once, up front, rather than through
`uv run --with`. `reference_stores.py` uses the latter to build fixtures,
where cost is irrelevant; here the first invocation would resolve and download
several hundred megabytes of wheels straight into the measured wall time, and
uv's own memory into the measured peak.
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks import datasets
from benchmarks._measure import (
    DEFAULT_MEMORY_LIMIT,
    Measurement,
    drop_page_cache,
    measure,
)
from benchmarks.cases import CASES, Case, Contender

#: Packages each baseline environment needs.
ENVIRONMENTS: Dict[str, List[str]] = {
    "anndata": ["anndata", "scipy", "pandas", "h5py", "zarr"],
    "scanpy": ["scanpy", "anndata", "scipy", "pandas", "h5py", "zarr"],
}


def build_environments(root: Path, wanted: List[str]) -> Dict[str, Path]:
    """Create one venv per baseline and return its interpreter."""
    interpreters: Dict[str, Path] = {}
    for name in wanted:
        venv = root / name
        python = venv / "bin" / "python"
        if not python.exists():
            print(f"[env] building {name}", flush=True)
            subprocess.run(["uv", "venv", str(venv)], check=True, capture_output=True)
            subprocess.run(
                ["uv", "pip", "install", "--python", str(python), *ENVIRONMENTS[name]],
                check=True,
            )
        interpreters[name] = python
    return interpreters


def _versions(interpreters: Dict[str, Path]) -> Dict[str, str]:
    """Record exactly what was compared, so a table is interpretable later."""
    found: Dict[str, str] = {}
    try:
        found["adata-cli"] = subprocess.run(
            ["adata", "--version"], capture_output=True, text=True, timeout=120
        ).stdout.strip()
    except Exception:  # pragma: no cover
        found["adata-cli"] = "unknown"
    for name, python in interpreters.items():
        code = (
            "import importlib.metadata as m;"
            f"print(m.version({name!r}))"
        )
        try:
            found[name] = subprocess.run(
                [str(python), "-c", code], capture_output=True, text=True, timeout=120
            ).stdout.strip()
        except Exception:  # pragma: no cover
            found[name] = "unknown"
    return found


def _run_contender(
    case: Case,
    contender: Contender,
    inputs: List[Path],
    workdir: Path,
    interpreters: Dict[str, Path],
    *,
    timeout_s: float,
    scripts_dir: Path,
) -> Measurement:
    workdir.mkdir(parents=True, exist_ok=True)
    output = workdir / f"{case.name}{case.output_suffix}"
    outdir = workdir / f"{case.name}-out"

    # The harness never overwrites, so clear any previous run first.
    for stale in (output, outdir):
        if stale.is_dir():
            shutil.rmtree(stale)
        elif stale.exists():
            stale.unlink()

    substitutions = {
        "output": str(output),
        "outdir": str(outdir),
        **{f"input{i}": str(p) for i, p in enumerate(inputs)},
    }

    if contender.argv is not None:
        command = [part.format(**substitutions) for part in contender.argv]
        target = outdir if "{outdir}" in " ".join(contender.argv) else output
    else:
        safe = f"{case.name}-{contender.label}".replace(" ", "_").replace("(", "").replace(")", "")
        script = scripts_dir / f"{safe}.py"
        script.write_text(contender.script or "")
        # A script that makes a directory is handed the directory; everything
        # else is handed the single output path.
        target = outdir if "makedirs" in (contender.script or "") else output
        command = [
            str(interpreters[contender.env]), str(script), *map(str, inputs), str(target)
        ]

    # Read-only cases produce nothing to size.
    watched = None if (case.output_suffix == "" and target == output) else target
    return measure(command, output=watched, timeout_s=timeout_s)


def run(
    tier_name: str,
    *,
    out: Path,
    work: Path,
    only: Optional[List[str]] = None,
    repeats: int = 3,
    timeout_s: float = 900.0,
    drop_cache: bool = True,
) -> Dict:
    tier = datasets.TIERS[tier_name]
    work.mkdir(parents=True, exist_ok=True)
    fixtures = work / "fixtures"
    scripts = work / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)

    selected = [c for c in CASES if only is None or c.name in only]
    needed = sorted(
        {c.env for case in selected for c in case.contenders if c.script}
    )
    interpreters = build_environments(work / "envs", needed)

    print(f"[data] building {tier.name} fixtures ({tier.n_obs} x {tier.n_var})",
          flush=True)
    main_inputs = datasets.build(fixtures, tier, count=2)
    var_heavy_inputs = datasets.build(fixtures, datasets.VAR_HEAVY, count=2)

    cache_dropped = drop_page_cache() if drop_cache else False

    results: List[Dict] = []
    for case in selected:
        pool = var_heavy_inputs if case.var_heavy else main_inputs
        inputs = pool[: case.inputs]
        print(f"[case] {case.name}", flush=True)
        for contender in case.contenders:
            if contender.unsupported:
                results.append(
                    {
                        "case": case.name,
                        "contender": contender.label,
                        "status": "n/a",
                        "note": contender.unsupported,
                    }
                )
                print(f"    {contender.label}: n/a ({contender.unsupported})")
                continue

            runs: List[Measurement] = []
            for _ in range(repeats):
                if drop_cache:
                    drop_page_cache()
                runs.append(
                    _run_contender(
                        case,
                        contender,
                        inputs,
                        work / "out",
                        interpreters,
                        timeout_s=timeout_s,
                        scripts_dir=scripts,
                    )
                )
                if runs[-1].status != "ok":
                    break  # a failure repeats identically; do not pay for it twice

            best = min(runs, key=lambda m: m.wall_s)
            record = {
                "case": case.name,
                "contender": contender.label,
                # The fastest run, because a slower one differs only by what
                # else the machine was doing.
                **best.as_dict(),
                "runs": len(runs),
                "wall_s_all": [m.wall_s for m in runs],
            }
            results.append(record)
            print(
                f"    {contender.label}: {best.status} "
                f"{best.wall_s:.2f}s  {best.maxrss_bytes / 1e6:.0f} MB",
                flush=True,
            )

    payload = {
        "tier": tier.name,
        "tier_shape": [tier.n_obs, tier.n_var, tier.density],
        "var_heavy_shape": [datasets.VAR_HEAVY.n_obs, datasets.VAR_HEAVY.n_var],
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ref": _git_ref(),
        "platform": f"{platform.system()} {platform.machine()}",
        "python": sys.version.split()[0],
        "memory_limit_bytes": DEFAULT_MEMORY_LIMIT,
        "page_cache_dropped": cache_dropped,
        "repeats": repeats,
        "versions": _versions(interpreters),
        "results": results,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nwrote {out}")
    return payload


def _git_ref() -> str:
    for args in (["git", "describe", "--tags", "--exact-match"],
                 ["git", "rev-parse", "--short", "HEAD"]):
        try:
            done = subprocess.run(args, capture_output=True, text=True, timeout=30)
            if done.returncode == 0 and done.stdout.strip():
                return done.stdout.strip()
        except Exception:  # pragma: no cover
            pass
    return "unknown"


def main() -> int:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="smoke", choices=sorted(datasets.TIERS))
    parser.add_argument("--out", type=Path, default=Path("results.json"))
    parser.add_argument("--work", type=Path, default=Path(".bench"))
    parser.add_argument("--case", action="append", dest="only")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--keep-cache",
        action="store_true",
        help="do not try to drop the page cache between runs",
    )
    args = parser.parse_args()
    run(
        args.tier,
        out=args.out,
        work=args.work,
        only=args.only,
        repeats=args.repeats,
        timeout_s=args.timeout,
        drop_cache=not args.keep_cache,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
