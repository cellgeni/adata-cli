"""Turn results.json into the markdown that gets published.

Two sinks, and the docs page is the authoritative one: artifacts expire, and
a single absolute number with nothing to compare it to is close to
uninterpretable. `docs/benchmarks/<ref>.json` keeps the series, and
`docs/BENCHMARKS.md` renders the latest run plus a history table. Both are in
git and both are already served by GitHub Pages, so no extra machinery.

    uv run python -m benchmarks.report results.json
    uv run python -m benchmarks.report results.json --docs docs --publish
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from benchmarks.cases import by_name


def _bytes(n: int) -> str:
    if not n:
        return "-"
    for unit, scale in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= scale:
            return f"{n / scale:.1f} {unit}"
    return f"{n} B"


def _cell(record: Dict) -> tuple:
    """(wall, rss, output) as they should read in the table."""
    status = record.get("status")
    if status == "n/a":
        return "n/a", "n/a", "n/a"
    if status == "timeout":
        # This is how the 0.5.1 hang presented: still running, output stuck at
        # the size it reached in the first second. Say so, rather than
        # reporting a large time as though the run had completed.
        return (
            f"**timeout** (>{record['wall_s']:.0f} s)",
            _bytes(record["maxrss_bytes"]),
            f"{_bytes(record['output_bytes'])} and not growing",
        )
    if status == "oom":
        return "**out of memory**", "hit the ceiling", "-"
    if status == "failed":
        return "**failed**", _bytes(record["maxrss_bytes"]), "-"

    output = _bytes(record["output_bytes"])
    if record.get("output_files", 0) > 1:
        # Zarr: a directory of small chunks looks far larger under `du` than
        # the bytes it holds, so the file count goes alongside.
        output += f" in {record['output_files']} files"
    return f"{record['wall_s']:.2f} s", _bytes(record["maxrss_bytes"]), output


def render(payload: Dict, previous: Optional[Dict] = None) -> str:
    cases = by_name()
    before = (
        {(r["case"], r["contender"]): r for r in previous["results"]}
        if previous
        else {}
    )

    lines: List[str] = []
    # H2, because this is embedded under the page's own H1 by `build_page`.
    # The framing lives in `page_template.md`; repeating it here would print
    # it twice on the published page.
    lines.append(f"## Results — `{payload['ref']}`")
    lines.append("")

    n_obs, n_var, density = payload["tier_shape"]
    vh_obs, vh_var = payload["var_heavy_shape"]
    cache = (
        "dropped between runs"
        if payload.get("page_cache_dropped")
        else "**warm** (could not be dropped; reads are served from RAM, "
        "which understates the streaming advantage)"
    )
    lines.append("| | |")
    lines.append("|---|---|")
    lines.append(f"| Tier | `{payload['tier']}`, {n_obs:,} obs x {n_var:,} var, "
                 f"{density:.0%} dense CSR |")
    lines.append(f"| Var-heavy shape | {vh_obs:,} obs x {vh_var:,} var |")
    lines.append(f"| Page cache | {cache} |")
    lines.append(f"| Address-space ceiling | "
                 f"{_bytes(payload['memory_limit_bytes'])} per process |")
    lines.append(f"| Repeats | {payload['repeats']} (fastest shown) |")
    lines.append(f"| Platform | {payload['platform']}, Python "
                 f"{payload['python']} |")
    versions = ", ".join(f"{k} {v}" for k, v in payload["versions"].items())
    lines.append(f"| Versions | {versions} |")
    lines.append(f"| Generated | {payload['generated']} |")
    lines.append("")

    grouped: Dict[str, List[Dict]] = {}
    for record in payload["results"]:
        grouped.setdefault(record["case"], []).append(record)

    for name, records in grouped.items():
        case = cases.get(name)
        lines.append(f"### `{name}`")
        lines.append("")
        if case:
            lines.append(f"{case.question}")
            lines.append("")
        lines.append("| Contender | Wall time | Peak RSS | Output | vs previous |")
        lines.append("|---|---|---|---|---|")
        for record in records:
            wall, rss, output = _cell(record)
            lines.append(
                f"| {record['contender']} | {wall} | {rss} | {output} | "
                f"{_delta(record, before.get((name, record['contender'])))} |"
            )
            note = record.get("note") or (
                record.get("stderr_tail") if record.get("status") == "n/a" else None
            )
            if note:
                lines.append(f"| | *{note}* | | | |")
        lines.append("")

    lines.append("---")
    lines.append("")
    lines.append(
        "Rows marked `n/a` are operations the baseline does not offer; that "
        "is a result, not a gap in the measurement. Where a baseline is "
        "faster, the row stands as it is -- the trade this tool makes is "
        "memory for time, and hiding the cost would make the table useless."
    )
    lines.append("")
    lines.append(
        "**Output size** is the file as the filesystem reports it, which for "
        "HDF5 includes allocation slack. On a small tier that slack can be "
        "several times the stored bytes and the column says more about the "
        "writer's allocation strategy than about the data; at `ci` and above "
        "it is noise. Zarr stores report a file count alongside, because a "
        "directory of small chunks measures much larger than it holds."
    )
    return "\n".join(lines) + "\n"


def _delta(record: Dict, previous: Optional[Dict]) -> str:
    """Wall time and peak RSS against the last published run."""
    if not previous or record.get("status") != "ok" or previous.get("status") != "ok":
        return "-"
    parts = []
    for key, label in (("wall_s", "time"), ("maxrss_bytes", "RSS")):
        old, new = previous.get(key), record.get(key)
        if not old:
            continue
        change = (new - old) / old
        if abs(change) < 0.10:  # within run-to-run noise on a shared runner
            continue
        parts.append(f"{label} {change:+.0%}")
    return ", ".join(parts) if parts else "no change"


def history_table(docs: Path) -> str:
    """One row per published run, newest first."""
    runs = []
    for path in sorted((docs / "benchmarks").glob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # pragma: no cover
            continue
        rows = {(r["case"], r["contender"]): r for r in payload["results"]}
        key = ("concat-inner", "adata-cli")
        headline = rows.get(key, {})
        runs.append(
            (
                payload.get("generated", ""),
                payload.get("ref", path.stem),
                payload.get("tier", "?"),
                headline.get("wall_s"),
                headline.get("maxrss_bytes"),
                path.name,
            )
        )
    if not runs:
        return ""

    lines = [
        "### History",
        "",
        "`concat-inner` on adata-cli, run by run. Full results for each are "
        "in [`docs/benchmarks/`](benchmarks/).",
        "",
        "| Run | Ref | Tier | Wall time | Peak RSS | Raw |",
        "|---|---|---|---|---|---|",
    ]
    for generated, ref, tier, wall, rss, filename in sorted(runs, reverse=True):
        lines.append(
            f"| {generated} | `{ref}` | {tier} | "
            f"{f'{wall:.2f} s' if wall else '-'} | "
            f"{_bytes(rss) if rss else '-'} | [json](benchmarks/{filename}) |"
        )
    return "\n".join(lines) + "\n"


#: The standing prose of the docs page, with `<!-- results -->` marking where
#: the tables go. Kept as a file rather than inline so the words that explain
#: the numbers live in one place and survive every republish -- an earlier
#: version overwrote the whole page with bare tables, which would have thrown
#: away the framing on the first tag.
PAGE_TEMPLATE = Path(__file__).with_name("page_template.md")

#: What the template shows before any run has happened.
NOT_YET_RUN = (
    "## Results\n\nNo run has been published yet. The next tag fills this in; "
    "until then, produce one locally with the commands below.\n"
)


def build_page(body: str) -> str:
    """Wrap rendered results in the page's standing explanation."""
    template = PAGE_TEMPLATE.read_text()
    if "<!-- results -->" not in template:  # pragma: no cover - template edited
        return template.rstrip() + "\n\n" + body
    return template.replace("<!-- results -->", body.strip())


def publish(payload: Dict, results: Path, docs: Path) -> Path:
    """Copy the raw results in and rewrite the docs page.

    Only the results section is replaced; everything explaining what the
    numbers mean comes from `page_template.md` and is reinstated every time.
    """
    store = docs / "benchmarks"
    store.mkdir(parents=True, exist_ok=True)
    ref = payload["ref"].replace("/", "-")
    shutil.copyfile(results, store / f"{ref}.json")

    previous = _previous(store, skip=f"{ref}.json")
    page = docs / "BENCHMARKS.md"
    page.write_text(
        build_page(render(payload, previous) + "\n" + history_table(docs))
    )
    return page


def _previous(store: Path, *, skip: str) -> Optional[Dict]:
    candidates = sorted(
        (p for p in store.glob("*.json") if p.name != skip),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        return None
    try:
        return json.loads(candidates[-1].read_text())
    except Exception:  # pragma: no cover
        return None


def main() -> int:  # pragma: no cover - CLI entry
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--docs", type=Path, default=Path("docs"))
    parser.add_argument(
        "--publish",
        action="store_true",
        help="write docs/benchmarks/<ref>.json and rewrite docs/BENCHMARKS.md",
    )
    parser.add_argument("--out", type=Path, help="also write the markdown here")
    args = parser.parse_args()

    payload = json.loads(args.results.read_text())
    if args.publish:
        page = publish(payload, args.results, args.docs)
        print(f"wrote {page}")
        text = page.read_text()
    else:
        text = render(payload, _previous(args.docs / "benchmarks", skip=""))
        print(text)
    if args.out:
        args.out.write_text(text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
