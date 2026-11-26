"""Build reference stores with real, pinned anndata releases.

The point is to check compatibility against what anndata actually wrote at
each version, rather than against this repo's idea of the format. Each store
is produced by running `tests/fixtures/write_reference_store.py` inside an
environment `uv` assembles for that release, so nothing here depends on the
versions installed for the test suite itself.

Stores are built once per session and cached, since assembling seven
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
    #: The stack the release is run on, pinned exactly. Modern pandas makes
    #: string columns a type the older releases cannot write, so each needs
    #: the stack it shipped against rather than today's; and a floating pin
    #: could change what the job checks, or turn it red, with no change here.
    extras: tuple = ()
    #: Interpreter to run under. pandas 1.x has no wheels for 3.12, so the
    #: oldest releases build against 3.11, and 0.7 against 3.10.
    python: str = "3.12"
    #: Whether the release can open what adata writes. anndata 0.7 predates
    #: the encoding scheme adata writes (categoricals as groups and the
    #: rest), so its stores are read, never written for it.
    reads_back: bool = True


#: anndata 0.7 wrote no encoding attributes at the root, and its categoricals
#: as codes with an object reference: the oldest layout this tool reads. 0.8
#: introduced the encoding-type/encoding-version scheme adata writes, so it is
#: the oldest release that reads what adata writes. 0.11 is the first that can
#: read a nullable-string-array (it does not write one: anndata writes them
#: from 0.13 with pandas 3), 0.12 the first that reads a `null` and Zarr v3.
RELEASES: List[Release] = [
    Release(
        "anndata==0.7.8", "0.7",
        extras=("scipy==1.9.3", "pandas==1.3.5", "numpy==1.23.5",
                "zarr==2.10.3", "h5py==3.7.0", "numcodecs==0.13.1"),
        python="3.10", reads_back=False,
    ),
    Release(
        "anndata==0.8.0", "0.8",
        extras=("scipy==1.17.1", "pandas==1.5.3", "numpy==1.26.4",
                "zarr==2.18.7", "h5py==3.16.0"),
        python="3.11",
    ),
    Release(
        "anndata==0.9.2", "0.9",
        extras=("scipy==1.17.1", "pandas==1.5.3", "numpy==1.26.4",
                "zarr==2.18.7", "h5py==3.16.0"),
        python="3.11",
    ),
    Release(
        "anndata==0.10.9", "0.10",
        extras=("scipy==1.17.1", "pandas==2.3.3", "numpy==1.26.4",
                "zarr==2.18.7", "h5py==3.16.0"),
    ),
    Release(
        "anndata==0.11.4", "0.11",
        extras=("scipy==1.18.1", "pandas==2.3.3", "numpy==2.5.4",
                "zarr==2.18.7", "h5py==3.16.0"),
    ),
    Release(
        "anndata==0.12.2", "0.12",
        extras=("scipy==1.18.1", "pandas==2.3.3", "numpy==2.5.4",
                "zarr==3.4.2", "h5py==3.16.0"),
    ),
    Release(
        "anndata==0.13.4", "0.13",
        extras=("scipy==1.18.1", "pandas==3.1.0", "numpy==2.5.4",
                "zarr==3.4.2", "h5py==3.16.0"),
    ),
]

#: The readers the external-reader tests compare against, at exact versions,
#: for the same reason as the releases' stacks above. Move them deliberately.
ORACLES: Dict[str, str] = {
    "anndata": "anndata==0.13.4",
    "geopandas": "geopandas==1.2.0",
    "loompy": "loompy==3.0.8",
    "mudata": "mudata==0.4.1",
    "pandas": "pandas==3.1.0",
    "pyarrow": "pyarrow==26.0.0",
    "scanpy": "scanpy==1.12.5",
    "scipy": "scipy==1.18.1",
    "squidpy": "squidpy==1.8.4",
    "tables": "tables==3.11.1",
}


def oracle(package: str) -> str:
    """`package` at its pinned version; a requirement with its own pin as given."""
    return ORACLES.get(package, package)


def with_args(release: Release) -> List[str]:
    """The `uv run --with` arguments that assemble `release`'s environment."""
    args = ["--with", release.spec]
    for extra in release.extras:
        args += ["--with", extra]
    return args


#: Stores anndata 0.7.8 wrote (`write_reference_store.py`, in the 0.7
#: release's environment above), committed so the unit suite reads a real
#: legacy layout without the network. The Zarr store is zipped.
LEGACY_FIXTURES = Path(__file__).parent / "fixtures" / "anndata-0.7.8"


def legacy_store(fmt: str, dest: Path) -> Path:
    """A fresh copy, under `dest`, of the store anndata 0.7.8 wrote as `fmt`."""
    import zipfile

    dest.mkdir(parents=True, exist_ok=True)
    if fmt == "zarr":
        with zipfile.ZipFile(LEGACY_FIXTURES / "reference.zarr.zip") as archive:
            archive.extractall(dest)
        return dest / "reference.zarr"
    target = dest / "reference.h5ad"
    shutil.copyfile(LEGACY_FIXTURES / "reference.h5ad", target)
    return target


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
    cmd = ["uv", "run", "--no-project", "--python", release.python, *with_args(release)]
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
