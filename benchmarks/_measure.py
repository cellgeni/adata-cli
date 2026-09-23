"""Run one command as a child process and report what it cost.

Peak RSS is the number that matters here. adata-cli exists so that memory is
set by chunk size rather than by input size, and a comparison reporting only
wall time would misrepresent it -- for anything that fits in RAM, loading the
whole thing is usually faster.

Measuring it correctly needs `os.wait4`, not `resource.getrusage`.
`RUSAGE_CHILDREN` is a running maximum over every child the process has ever
reaped: run a 400 MB case and then a 17 MB one and it still reports 400 MB.
`wait4` returns rusage for one specific child. (Checked on this machine:
wait4 gives 435 MB then 17 MB where RUSAGE_CHILDREN stays at 435 MB.)
"""

from __future__ import annotations

import json
import os
import resource
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
    """Run `command`, returning its wall time, peak RSS and output size."""

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


def main(argv: List[str]) -> int:  # pragma: no cover - CLI entry
    """Measure a command given after `--`, printing JSON."""
    if "--" not in argv:
        print("usage: python -m benchmarks._measure [--output P] -- CMD...")
        return 2
    split = argv.index("--")
    head, command = argv[:split], argv[split + 1 :]
    output = None
    if "--output" in head:
        output = Path(head[head.index("--output") + 1])
    print(json.dumps(measure(command, output=output).as_dict(), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
