"""The benchmark harness has to be trustworthy before its numbers are.

Only the measurement is tested here, not the cases. If peak RSS were
attributed to the wrong process every table the project publishes would be
wrong, and the failure would be invisible -- the numbers would still look
plausible. Everything else in `benchmarks/` is reporting, and a mistake there
shows up the moment anyone reads the page.
"""

from __future__ import annotations

import json
import sys

import pytest

from benchmarks._measure import Measurement, _looks_like_oom, measure


#: A child that allocates nothing should cost about what a bare interpreter
#: costs. Generous, because that depends on the build; far below the ballast
#: below, which is the point.
FLOOR_CEILING = 60 * 1024**2

#: Allocated by the parent before measuring, to make an inherited floor
#: impossible to miss.
BALLAST = 300 * 1024**2


def test_peak_rss_is_the_child_s_own_and_carries_no_floor():
    """Two ways this number can lie, both of which it has.

    `RUSAGE_CHILDREN` is a running maximum over every child a process has
    reaped, so a 400 MB case makes every later one look like 400 MB. `wait4`
    fixes that.

    Worse, and what CI caught: on Linux a forked child inherits its parent's
    resident pages, and `execve` folds that into the `maxrss` `wait4` reports.
    A child of a fat parent could not appear small however little it used.
    Measured under python:3.12-slim before the fix -- parent at 329.6 MB, a
    no-op child reported 326.4 MB. `posix_spawn` reported 329.5 MB, so
    dropping `preexec_fn` would not have helped; measuring from a shim
    reported 8.1 MB.

    macOS resets the high-water mark at exec and shows none of this, so this
    is mostly a Linux guard -- which is the platform the benchmark runs on.

    The test holds ballast in the parent for its whole duration so that an
    inherited floor cannot hide.
    """
    ballast = bytearray(BALLAST)
    ballast[::4096] = b"\x01" * len(ballast[::4096])  # make it resident

    baseline = measure([sys.executable, "-c", "pass"])
    big = measure([sys.executable, "-c", "x = bytearray(400 * 1024 * 1024)"])
    small = measure([sys.executable, "-c", "x = bytearray(1024)"])

    assert {baseline.status, big.status, small.status} == {"ok"}

    # No floor: a child that allocates nothing looks like nothing, even though
    # this process is holding 300 MB.
    assert baseline.maxrss_bytes < FLOOR_CEILING, (
        f"a no-op child reported {baseline.maxrss_bytes / 1e6:.0f} MB while "
        f"this process held {BALLAST / 1e6:.0f} MB of ballast. The measured "
        "peak is inheriting the parent's resident pages, so every benchmark "
        "row would be floored at roughly the runner's own footprint."
    )

    # Isolation: the small child resembles the baseline, not the big one that
    # ran before it.
    assert big.maxrss_bytes > 300 * 1024**2, big
    assert small.maxrss_bytes < baseline.maxrss_bytes * 2, (
        f"peak RSS leaked between children: {small.maxrss_bytes / 1e6:.0f} MB "
        f"for a 1 KB allocation, against a {baseline.maxrss_bytes / 1e6:.0f} MB "
        f"baseline, after a {big.maxrss_bytes / 1e6:.0f} MB child"
    )

    del ballast


def test_a_hang_is_reported_as_a_timeout_with_its_output_size(tmp_path):
    """How the 0.5.1 hang presented: still running, output never growing.

    A harness that recorded this as a slow success would have made the
    reported bug look like ordinary slowness.
    """
    output = tmp_path / "partial.bin"
    command = [
        sys.executable,
        "-c",
        f"open({str(output)!r}, 'wb').write(b'x' * 1024); import time; time.sleep(60)",
    ]
    result = measure(command, output=output, timeout_s=2.0)

    assert result.status == "timeout"
    assert result.wall_s >= 2.0
    assert result.output_bytes == 1024, "output size at the kill must be recorded"


def test_a_failure_against_the_memory_ceiling_is_recorded_as_oom():
    """A baseline that cannot cope is a result, not a crashed job.

    Skipped on macOS, which refuses RLIMIT_AS; the ceiling only has to hold
    on the Linux runner where the benchmark actually runs.
    """
    if not sys.platform.startswith("linux"):
        pytest.skip("RLIMIT_AS is not enforceable on this platform")

    result = measure(
        [sys.executable, "-c", "x = bytearray(8 * 1024**3)"],
        memory_limit=512 * 1024**2,
    )
    assert result.status == "oom", result


@pytest.mark.parametrize(
    "text",
    ["MemoryError", "numpy: Unable to allocate 4.0 GiB", "std::bad_alloc"],
)
def test_out_of_memory_is_recognised_however_it_is_phrased(text):
    assert _looks_like_oom(text)


def test_an_ordinary_failure_is_not_mistaken_for_an_oom():
    result = measure([sys.executable, "-c", "raise ValueError('nope')"])
    assert result.status == "failed"
    assert "ValueError" in result.stderr_tail


def test_the_harness_refuses_to_overwrite_an_existing_output(tmp_path):
    """A stale output would be sized and reported as this run's work.

    Checked in the parent, not the shim, so it surfaces as an exception the
    caller must fix rather than as a quiet "failed" row.
    """
    existing = tmp_path / "already.h5ad"
    existing.write_bytes(b"old")
    with pytest.raises(FileExistsError):
        measure([sys.executable, "-c", "pass"], output=existing)


def test_a_missing_binary_is_a_result_not_a_crash():
    """`h5ls` ships with the HDF5 tools and is often simply absent."""
    result = measure(["/nonexistent-binary-for-tests"])
    assert result.status == "n/a"
    assert "not installed" in result.stderr_tail


def test_maxrss_is_normalised_to_bytes():
    """`ru_maxrss` is KiB on Linux and bytes on macOS; the field is bytes.

    A missing conversion would be a silent 1024x error in one direction on
    one platform, which is exactly the kind of thing nobody notices in a
    table of plausible-looking numbers.
    """
    result = measure([sys.executable, "-c", "pass"])
    assert isinstance(result, Measurement)
    # Any CPython start-up is well over 1 MB and well under 4 GB.
    assert 1024**2 < result.maxrss_bytes < 4 * 1024**3, result.maxrss_bytes


def test_publishing_keeps_the_page_prose():
    """A republish must not reduce the docs page to bare tables.

    `publish` rewrites `docs/reference/benchmarks.md` in full on every tag. An earlier
    version wrote only the rendered results, which would have thrown away
    everything explaining what the numbers mean the first time it ran.
    """
    from benchmarks.report import PAGE_TEMPLATE, build_page

    template = PAGE_TEMPLATE.read_text()
    assert "<!-- results -->" in template, (
        "the results marker is gone from page_template.md, so results would "
        "be appended rather than placed"
    )

    page = build_page("## Results\n\nsome tables\n")
    assert "some tables" in page
    assert "Peak RSS is the headline" in page, "the framing was dropped"
    assert "Running it yourself" in page, "the trailing sections were dropped"
    assert page.count("\n# ") + page.startswith("# ") == 1, (
        "the published page must have exactly one H1"
    )


def test_publishing_writes_the_reference_page_the_site_serves(tmp_path):
    """The page moved from `docs/BENCHMARKS.md` into `docs/reference/` when
    the docs became a Just the Docs site; a publish that still wrote the old
    path would leave the sidebar's page frozen at the last release. Its link
    to each run's JSON must be absolute: with `permalink: pretty` the page is
    served at `/reference/benchmarks/`, where `benchmarks/x.json` 404s.
    """
    from benchmarks.report import publish

    payload = {
        "ref": "9.9.9", "tier": "smoke", "tier_shape": [1, 1, 0.05],
        "var_heavy_shape": [1, 1], "page_cache_dropped": True,
        "memory_limit_bytes": 1, "repeats": 1, "platform": "x",
        "python": "3", "versions": {}, "generated": "now",
        "results": [{
            "case": "concat-inner", "contender": "adata-cli", "status": "ok",
            "wall_s": 1.0, "maxrss_bytes": 100_000_000,
            "output_bytes": 1, "output_files": 1,
        }],
    }
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload))

    page = publish(payload, results, tmp_path / "docs")

    assert page == tmp_path / "docs" / "reference" / "benchmarks.md"
    assert (tmp_path / "docs" / "benchmarks" / "9.9.9.json").exists()
    text = page.read_text()
    assert text.startswith("---\ntitle: Benchmarks\nparent: Reference\n"), (
        "without front matter the page drops out of the sidebar"
    )
    assert "redirect_from:\n  - /BENCHMARKS.html" in text
    assert (
        "[json](https://github.com/cellgeni/adata-cli/blob/main/docs/"
        "benchmarks/9.9.9.json)" in text
    )


def test_the_wide_fixture_holds_the_same_matrix_in_wider_dtypes(tmp_path):
    """`recast-narrow` measures nothing unless its input is actually wide.

    Until 0.7.0 the dtype case narrowed the main fixture, which is already
    float32 with int32 indices: a rewrite that changed nothing.
    """
    ad = pytest.importorskip("anndata")
    h5py = pytest.importorskip("h5py")
    from benchmarks import datasets

    tiny = datasets.Tier("tiny", 50, 40, 0.1, "test")
    narrow = ad.read_h5ad(datasets.build(tmp_path, tiny, count=1)[0])

    for variant, compression in datasets.WIDE_COMPRESSION.items():
        (path,) = datasets.build_wide(tmp_path, tiny, variant)
        with h5py.File(path, "r") as handle:
            assert handle["X/data"].dtype == "float64"
            assert handle["X/indices"].dtype == "int64"
            assert handle["X/indptr"].dtype == "int64"
            assert handle["X/data"].compression == compression

        wide = ad.read_h5ad(path)
        assert (wide.X != narrow.X).nnz == 0
        assert list(wide.obs_names) == list(narrow.obs_names)
        assert list(wide.var.columns) == list(narrow.var.columns)


def test_a_size_case_reports_input_to_output():
    from benchmarks.report import render

    record = {
        "case": "recast-narrow", "contender": "adata-cli", "status": "ok",
        "wall_s": 1.0, "maxrss_bytes": 100_000_000,
        "input_bytes": 600_000_000, "output_bytes": 300_000_000,
        "output_files": 1,
    }
    payload = {
        "ref": "test", "tier": "smoke", "tier_shape": [1, 1, 0.05],
        "var_heavy_shape": [1, 1], "page_cache_dropped": True,
        "memory_limit_bytes": 1, "repeats": 1, "platform": "x",
        "python": "3", "versions": {}, "generated": "now",
        "results": [record],
    }
    text = render(payload)
    assert "Input → output" in text
    assert "600.0 MB → 300.0 MB (-50%)" in text


def _run(ref, tier="ci"):
    """The smallest payload `render` accepts, named and tiered."""
    return {
        "ref": ref, "tier": tier, "tier_shape": [1, 1, 0.05],
        "var_heavy_shape": [1, 1], "page_cache_dropped": True,
        "memory_limit_bytes": 1, "repeats": 1, "platform": "x",
        "python": "3", "versions": {}, "generated": "now",
        "results": [{
            "case": "concat-inner", "contender": "adata-cli", "status": "ok",
            "wall_s": 1.0, "maxrss_bytes": 100_000_000,
            "output_bytes": 1, "output_files": 1,
        }],
    }


def test_vs_previous_is_the_release_just_before_whatever_the_file_times(tmp_path):
    """Up to 0.17.4 "vs previous" took the file with the newest mtime. A fresh
    checkout gives every file about the same one, in whatever order git
    wrote them, so the published 0.17.0 page compared with 0.8.0.
    """
    import os

    from benchmarks.report import _previous

    refs = ["0.6.0", "0.8.0", "0.9.0", "0.10.0", "0.16.2", "0.17.0", "0.17.1"]
    for age, ref in enumerate(refs):
        path = tmp_path / f"{ref}.json"
        path.write_text(json.dumps(_run(ref)))
        # Newest mtime on the oldest release: the order the bug obeyed.
        os.utime(path, (1e9 - age, 1e9 - age))

    def previous(ref):
        found = _previous(tmp_path, _run(ref))
        return found and found["ref"]

    assert previous("0.17.1") == "0.17.0"
    assert previous("0.17.0") == "0.16.2"
    assert previous("0.10.0") == "0.9.0", "versions compare as numbers"
    assert previous("0.17.2") == "0.17.1", "a release not yet stored"
    assert previous("0.6.0") is None, "the first release has nothing before it"


def test_a_run_that_is_not_a_release_compares_with_the_newest_release(tmp_path):
    from benchmarks.report import _previous

    for ref in ["0.9.0", "0.10.0", "8d779d6"]:
        (tmp_path / f"{ref}.json").write_text(json.dumps(_run(ref)))

    assert _previous(tmp_path, _run("8d779d6"))["ref"] == "0.10.0"
    # A stored dispatch is not a release, so no release compares with it.
    assert _previous(tmp_path, _run("0.11.0"))["ref"] == "0.10.0"


def test_vs_previous_only_compares_runs_at_the_same_tier(tmp_path):
    from benchmarks.report import _previous

    (tmp_path / "0.9.0.json").write_text(json.dumps(_run("0.9.0", "ci")))
    (tmp_path / "0.10.0.json").write_text(json.dumps(_run("0.10.0", "smoke")))

    assert _previous(tmp_path, _run("0.11.0", "ci"))["ref"] == "0.9.0"
    assert _previous(tmp_path, _run("0.11.0", "large")) is None


def test_every_published_run_compares_with_the_release_before_it():
    """F085, checked against the runs in docs/benchmarks/, with packaging's
    version ordering as the oracle. A dispatch with `publish` stores a run
    named by its commit; that is no release, and compares with the newest."""
    import re
    from pathlib import Path

    from packaging.version import Version

    from benchmarks.report import _previous

    store = Path(__file__).parents[1] / "docs" / "benchmarks"
    runs = {path.stem: json.loads(path.read_text()) for path in store.glob("*.json")}
    releases = {ref: run for ref, run in runs.items() if re.fullmatch(r"\d+\.\d+\.\d+", ref)}
    assert len(releases) >= 2
    for ref, payload in runs.items():
        earlier = [
            other for other, run in releases.items()
            if run["tier"] == payload["tier"]
            and (ref not in releases or Version(other) < Version(ref))
        ]
        expected = max(earlier, key=Version) if earlier else None
        found = _previous(store, payload)
        assert (found and found["ref"]) == expected, ref


def test_a_stored_dispatch_does_not_break_the_comparison(tmp_path):
    """A `Benchmark` dispatch with `publish` commits `docs/benchmarks/<sha>.json`.
    The check above once fed that name to Version() and failed on main."""
    import shutil
    from pathlib import Path

    store = Path(__file__).parents[1] / "docs" / "benchmarks"
    for path in store.glob("*.json"):
        shutil.copy(path, tmp_path / path.name)
    (tmp_path / "8d779d6.json").write_text(json.dumps(_run("8d779d6")))

    from benchmarks.report import _previous

    newest = max(
        (p.stem for p in store.glob("*.json")),
        key=lambda ref: tuple(int(part) for part in ref.split(".")),
    )
    assert _previous(tmp_path, _run("8d779d6"))["ref"] == newest
    assert _previous(tmp_path, _run("99.0.0"))["ref"] == newest


def test_the_page_names_the_run_it_compares_with(tmp_path):
    from benchmarks.report import publish

    docs = tmp_path / "docs"
    (docs / "benchmarks").mkdir(parents=True)
    for ref in ["0.16.2", "0.17.0"]:
        (docs / "benchmarks" / f"{ref}.json").write_text(json.dumps(_run(ref)))
    payload = _run("0.17.1")
    payload["results"][0]["wall_s"] = 2.0
    results = tmp_path / "results.json"
    results.write_text(json.dumps(payload))

    text = publish(payload, results, docs).read_text()

    assert "| vs previous | `0.17.0` |" in text
    assert "| adata-cli | 2.00 s | 100.0 MB | 1 B | time +100% |" in text
