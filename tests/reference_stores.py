"""Build reference stores with real, pinned anndata releases.

The point is to check compatibility against what anndata actually wrote at
each version, rather than against this repo's idea of the format. Each store
is produced by running `tests/fixtures/write_reference_store.py` inside an
environment `uv` assembles for that release, so nothing here depends on the
versions installed for the test suite itself.

Stores are built once per session and cached, since assembling six
environments is slow the first time and free afterwards.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

WRITER = Path(__file__).parent / "fixtures" / "write_reference_store.py"


@dataclass(frozen=True)
class Release:
    """An anndata release to build fixtures with."""

    #: Requirement string passed to `uv --with`.
    spec: str
    #: Short label used in test ids.
    label: str
    #: Era-appropriate pins. Modern pandas makes string columns a type the
    #: older releases cannot write, so each release needs the stack it shipped
    #: against rather than today's.
    extras: tuple = ()
    #: Interpreter to run under. pandas 1.x has no wheels for 3.12, so the
    #: oldest releases build against 3.11.
    python: str = "3.12"


#: anndata 0.8 introduced the encoding-type/encoding-version scheme, so it is
#: the oldest release whose output this tool claims to read. 0.11 is where the
#: index became a nullable-string-array group, and 0.12 is where Zarr v3
#: arrived -- the two changes that broke the CLI.
RELEASES: List[Release] = [
    Release(
        "anndata==0.8.0", "0.8",
        extras=("pandas<2", "numpy<2", "zarr<3"), python="3.11",
    ),
    Release(
        "anndata==0.9.2", "0.9",
        extras=("pandas<2", "numpy<2", "zarr<3"), python="3.11",
    ),
    Release("anndata==0.10.9", "0.10", extras=("pandas<3", "numpy<2", "zarr<3")),
    Release("anndata==0.11.4", "0.11", extras=("pandas<3", "zarr<3")),
    Release("anndata==0.12.2", "0.12", extras=("pandas<3", "zarr>=3")),
    Release("anndata~=0.13.3", "0.13", extras=("zarr>=3",)),
]


class ReferenceUnavailable(RuntimeError):
    """Raised when a store could not be produced for a release."""


def uv_available() -> bool:
    return shutil.which("uv") is not None


def offline() -> bool:
    """Honour the usual opt-out for tests that need to reach the network."""
    return os.environ.get("ADATA_SKIP_VERSION_FIXTURES", "").strip() not in ("", "0")


def must_build() -> bool:
    """Whether a store that fails to build should fail the test.

    Locally a broken environment is a nuisance and skipping is reasonable. In
    CI it is the whole point of the job: turning a failed build into a skip
    would let a bad release pin, or a fixture script that no longer runs
    anywhere, leave the job green having checked nothing.
    """
    override = os.environ.get("ADATA_REQUIRE_VERSION_FIXTURES", "").strip()
    if override not in ("", "0"):
        return True
    return os.environ.get("CI", "").strip().lower() in ("1", "true")


def build(release: Release, fmt: str, out_dir: Path) -> Path:
    """Write one reference store, returning its path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv", "run", "--no-project", "--python", release.python,
        "--with", release.spec, "--with", "scipy",
    ]
    for extra in release.extras:
        cmd += ["--with", extra]
    cmd += [str(WRITER), str(out_dir), fmt]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    target = out_dir / f"reference.{fmt}"
    if result.returncode != 0 or not target.exists():
        raise ReferenceUnavailable(
            f"{release.spec} ({fmt}) could not be built:\n"
            f"{result.stdout[-2000:]}\n{result.stderr[-2000:]}"
        )
    return target


class ReferenceCache:
    """Builds each (release, format) store once and remembers failures."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._built: Dict[str, Path] = {}
        self._failed: Dict[str, str] = {}

    def get(self, release: Release, fmt: str) -> Optional[Path]:
        key = f"{release.label}-{fmt}"
        if key in self._built:
            return self._built[key]
        if key in self._failed:
            raise ReferenceUnavailable(self._failed[key])

        try:
            path = build(release, fmt, self.root / key)
        except Exception as exc:  # noqa: BLE001 - reported to the test as a skip
            self._failed[key] = str(exc)
            raise ReferenceUnavailable(str(exc)) from exc

        self._built[key] = path
        return path
