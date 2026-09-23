"""Run one command as a child process and report what it cost.

Peak RSS is the number that matters here. adata-cli exists so that memory is
set by chunk size rather than by input size, and a comparison reporting only
wall time would misrepresent it -- for anything that fits in RAM, loading the
whole thing is usually faster.

Two things have to be right for the number to mean anything, and both were
wrong at some point.

**Use `os.wait4`, not `resource.getrusage`.** `RUSAGE_CHILDREN` is a running
maximum over every child the process has ever reaped: run a 400 MB case and
then a 17 MB one and it still reports 400 MB. `wait4` returns rusage for one
specific child.

**Fork the child from a small process.** On Linux a forked child inherits its
parent's resident pages, and `execve` folds that pre-exec high-water mark into
the accumulated `maxrss` that `wait4` reports. So a child of a fat parent can
never appear small. Measured under python:3.12-slim:

    parent 14.7 MB  ->  no-op child   11.8 MB
    parent 329.6 MB ->  no-op child  326.4 MB   <- the parent's RSS, not the child's
    parent 329.6 MB ->  via shim       8.1 MB

macOS resets the high-water mark at exec and shows none of this, which is why
it went unnoticed locally and failed on CI. It matters because `run.py`
imports anndata, pandas and numpy to build fixtures in the same process that
measures, so every contender would have been floored at ~200 MB and the
tables would have read "everything costs about the same".

`posix_spawn` does not help -- the middle row above is 329.5 MB that way too.
The fix is the shim: `measure()` re-invokes this file as a subprocess, and
that freshly-exec'd interpreter, about 8 MB, is what forks the command being
measured.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

#: Address-space ceiling for every child, in bytes.
#:
#: Cases where the in-memory baseline cannot cope are the whole point of the
#: comparison, but an uncontained OOM on a hosted runner kills the runner
#: agent and the job ends with no report at all. With a ceiling the baseline
#: dies with a MemoryError and a non-zero exit, which is a *result* -- recorded
#: as `oom` at a limit we can state.
#:
#: Linux only -- macOS refuses RLIMIT_AS, so a local run is unbounded and an
#: `oom` row cannot be reproduced there.
DEFAULT_MEMORY_LIMIT = 12 * 1024**3

#: `ru_maxrss` is KiB on Linux and bytes on macOS. Nothing in the man pages
#: says so; it is simply what each kernel does.
_MAXRSS_SCALE = 1024 if sys.platform.startswith("linux") else 1

#: How often to check whether the child has exited. Fine enough that it adds
#: no meaningful error to a run measured in seconds.
POLL_INTERVAL_S = 0.01


@dataclass
class Measurement:
    """What one run of one contender cost."""

    wall_s: float
    maxrss_bytes: int
    exit_code: int
    status: str  # "ok" | "failed" | "oom" | "timeout"
    output_bytes: int = 0
    output_files: int = 0
    stderr_tail: str = ""

    def as_dict(self) -> Dict:
        return asdict(self)


def _tree_size(path: Path) -> tuple:
    """Apparent bytes and file count for a file or a directory.

    Zarr stores are directories of many small chunks. Reporting only `du`
    would make them look several times larger than the data they hold, since
    every chunk rounds up to a filesystem block -- so the file count is
    reported alongside, and both go in the table.
    """
    if not path.exists():
        return 0, 0
    if path.is_file():
        return path.stat().st_size, 1
    total = 0
    count = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
            count += 1
    return total, count


def drop_page_cache() -> bool:
    """Try to drop the page cache, so a read is not served from RAM.

    A fixture written seconds ago is entirely in cache, which systematically
    understates how much streaming helps. GitHub-hosted runners have
    passwordless sudo, so this usually works there and usually does not
    locally; the report says which happened rather than pretending.
    """
    try:
        subprocess.run(
            ["sudo", "-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"],
            check=True,
            capture_output=True,
            timeout=30,
        )
        return True
    except Exception:
        return False


def measure(
    command: Sequence[str],
    *,
    output: Optional[Path] = None,
    timeout_s: float = 900.0,
    memory_limit: Optional[int] = DEFAULT_MEMORY_LIMIT,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Measurement:
    """Run `command` from a small shim, and report what it cost.

    The shim is this same file, re-invoked. It exists so that the process
    which forks `command` is a bare interpreter rather than whatever imported
    anndata -- see the module docstring for the measurement it fixes.
    """
    # Checked here rather than in the shim: a stale output would be sized and
    # reported as this run's work, which is a bug in the caller and should be
    # loud rather than turned into a "failed" row.
    if output is not None and output.exists():
        raise FileExistsError(f"{output} exists; the harness never overwrites.")

    argv = [sys.executable, str(Path(__file__).resolve()), "--timeout", str(timeout_s)]
    if output is not None:
        argv += ["--output", str(output)]
    if memory_limit is None:
        argv += ["--no-memory-limit"]
    else:
        argv += ["--memory-limit", str(memory_limit)]
    if cwd is not None:
        argv += ["--cwd", str(cwd)]
    for key, value in (env or {}).items():
        argv += ["--env", f"{key}={value}"]
    argv += ["--", *command]

    done = subprocess.run(argv, capture_output=True, text=True)
    try:
        return Measurement(**json.loads(done.stdout))
    except (json.JSONDecodeError, TypeError, ValueError):
        # The shim itself failed. Report it rather than crashing the run, and
        # keep enough of its output to diagnose.
        return Measurement(
            wall_s=0.0,
            maxrss_bytes=0,
            exit_code=done.returncode,
            status="failed",
            stderr_tail="measurement shim failed:\n"
            + "\n".join((done.stderr or done.stdout).strip().splitlines()[-6:]),
        )


def _measure_here(
    command: Sequence[str],
    *,
    output: Optional[Path] = None,
    timeout_s: float = 900.0,
    memory_limit: Optional[int] = DEFAULT_MEMORY_LIMIT,
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
) -> Measurement:
    """Run `command` in this process's own child. Only the shim calls this."""

    def limit() -> None:  # pragma: no cover - runs in the forked child
        if memory_limit is None:
            return
        try:
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))
        except (ValueError, OSError):
            # macOS refuses RLIMIT_AS outright. The ceiling only has to hold
            # on the runner, which is Linux; raising here would abort the
            # spawn and lose the measurement entirely.
            pass

    if output is not None and output.exists():
        raise FileExistsError(f"{output} exists; the harness never overwrites.")

    full_env = {**os.environ, **(env or {})}

    if shutil.which(command[0]) is None and not Path(command[0]).exists():
        return Measurement(
            wall_s=0.0, maxrss_bytes=0, exit_code=127, status="n/a",
            stderr_tail=f"{command[0]} is not installed",
        )

    # Output goes to temporary files, not pipes. A pipe would have to be
    # drained with communicate(), and communicate() reaps the child -- after
    # which wait4 has nothing to report and the per-child rusage is lost.
    with tempfile.TemporaryFile("w+") as out_f, tempfile.TemporaryFile("w+") as err_f:
        started = time.perf_counter()
        process = subprocess.Popen(
            list(command),
            stdout=out_f,
            stderr=err_f,
            preexec_fn=limit,
            env=full_env,
            cwd=str(cwd) if cwd else None,
            text=True,
        )

        status = "ok"
        deadline = started + timeout_s
        while True:
            pid, wait_status, usage = os.wait4(process.pid, os.WNOHANG)
            if pid != 0:
                process.returncode = (
                    -os.WTERMSIG(wait_status)
                    if os.WIFSIGNALED(wait_status)
                    else os.WEXITSTATUS(wait_status)
                )
                maxrss = usage.ru_maxrss
                exit_code = process.returncode
                break
            if time.perf_counter() > deadline:
                process.kill()
                _, wait_status, usage = os.wait4(process.pid, 0)
                process.returncode = -9
                maxrss = usage.ru_maxrss
                exit_code = -9
                status = "timeout"
                break
            time.sleep(POLL_INTERVAL_S)

        err_f.seek(0)
        stderr = err_f.read()

    wall_s = time.perf_counter() - started
    stderr = stderr or ""

    if status == "ok" and exit_code != 0:
        status = "oom" if _looks_like_oom(stderr) else "failed"

    size, files = _tree_size(output) if output is not None else (0, 0)
    return Measurement(
        wall_s=round(wall_s, 3),
        maxrss_bytes=int(maxrss) * _MAXRSS_SCALE,
        exit_code=exit_code,
        status=status,
        output_bytes=size,
        output_files=files,
        stderr_tail="\n".join(stderr.strip().splitlines()[-6:]),
    )


def _looks_like_oom(stderr: str) -> bool:
    """Did the child die against the address-space ceiling?

    Under RLIMIT_AS an allocation failure surfaces as MemoryError, or as one
    of numpy's or HDF5's own phrasings of the same thing.
    """
    markers = (
        "MemoryError",
        "Unable to allocate",
        "bad_alloc",
        "Cannot allocate memory",
        "out of memory",
    )
    return any(m in stderr for m in markers)


def main(argv: List[str]) -> int:  # pragma: no cover - runs as the shim
    """Measure the command after `--` and print one JSON object.

    This is the shim `measure()` re-invokes; it is also usable by hand:

        python benchmarks/_measure.py --timeout 60 -- adata view f.h5ad
    """
    parser = argparse.ArgumentParser(description="Measure one command.")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--memory-limit", type=int, default=DEFAULT_MEMORY_LIMIT)
    parser.add_argument("--no-memory-limit", action="store_true")
    parser.add_argument("--cwd", type=Path)
    parser.add_argument("--env", action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("no command given after --")

    extra = dict(pair.split("=", 1) for pair in args.env)
    result = _measure_here(
        command,
        output=args.output,
        timeout_s=args.timeout,
        memory_limit=None if args.no_memory_limit else args.memory_limit,
        env=extra or None,
        cwd=args.cwd,
    )
    print(json.dumps(result.as_dict()))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
