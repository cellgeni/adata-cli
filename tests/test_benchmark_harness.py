"""The benchmark harness has to be trustworthy before its numbers are.

Only the measurement is tested here, not the cases. If peak RSS were
attributed to the wrong process every table the project publishes would be
wrong, and the failure would be invisible -- the numbers would still look
plausible. Everything else in `benchmarks/` is reporting, and a mistake there
shows up the moment anyone reads the page.
"""

from __future__ import annotations

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

    `publish` rewrites `docs/BENCHMARKS.md` in full on every tag. An earlier
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
