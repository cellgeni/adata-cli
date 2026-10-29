"""Write outputs beside their destination, and move them into place at the end.

A command that fails half way must not leave a half-written store where the
output should be: the next command to open it would read whatever got written
as if it were the whole thing. Writing to a temporary sibling and renaming it
only on success means the output either appears whole or not at all.

An existing output is renamed aside (`out.adata-old.zarr`) before its
successor is renamed in, and deleted only once every output is in place. If a
rename fails part way, the outputs already swapped are put back. A run killed
between the two renames leaves the old data under the `adata-old` name, never
nowhere, and the next run refuses to start until that has been dealt with.

Two runs writing the same output would race past that check, so each output
is claimed first with a lock file (`out.h5ad.adata-lock`) that only one run
can create. A scheduler's SIGTERM (or a closed terminal's SIGHUP) is turned
into an exception where it lands, so it gets the same cleanup as a failure.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

from adata.util.path import is_zarr_path


def _sibling(path: Path, tag: str) -> Path:
    """`out.zarr` -> `out.<tag>.zarr`, keeping the suffix that picks the backend."""
    return path.with_name(f"{path.stem}.{tag}{path.suffix}")


def _temp_for(path: Path) -> Path:
    temp = _sibling(path, "adata-tmp")
    if path.suffix != ".zarr" and is_zarr_path(path):
        # A Zarr store named without the suffix. A temporary named the same
        # way would not exist yet, and so be opened as HDF5. Only a Zarr
        # store: any other directory, such as an MTX folder, keeps its name.
        temp = temp.with_name(f"{temp.name}.zarr")
    return temp


def lock_for(path: Path) -> Path:
    """The lock file claiming `path` while a run writes it."""
    return path.with_name(f"{path.name}.adata-lock")


def _remove(path: Path) -> None:
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    elif path.exists() or path.is_symlink():
        path.unlink()


def _exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _discard(path: Path, settle: float = 0.5) -> None:
    """Remove a failed write's temporary, and keep it removed.

    zarr writes chunks from its own event-loop thread, and an exception in
    ours (a failed chunk, or SIGTERM landing mid-write) does not stop the
    writes already in flight: they can recreate part of a store while it is
    being deleted. A directory is therefore removed until it stays gone.
    """
    if not (path.is_dir() and not path.is_symlink()):
        _remove(path)
        return
    deadline = time.monotonic() + settle
    while True:
        try:
            _remove(path)
        except OSError:  # a file appeared under rmtree; go round again
            if time.monotonic() >= deadline:
                raise
        time.sleep(0.05)
        if not _exists(path) or time.monotonic() >= deadline:
            break
    _remove(path)


class Terminated(SystemExit):
    """A termination signal, raised where it arrived so cleanup runs.

    A SystemExit, so commands' `except Exception` lets it through and the
    process ends with the shell's code for the signal (143 for SIGTERM).
    """


class _Signals:
    """Turn SIGTERM and SIGHUP into `Terminated` for the life of a write.

    Python's default for both is to die on the spot, skipping every
    `finally`; a scheduler's SIGTERM then left the temporaries behind for
    the next run to refuse. A signal that is ignored or already handled is
    left alone (`nohup` ignores SIGHUP), and so is everything off the main
    thread, where Python cannot install handlers.

    While `deferred` is set -- the swap, and the cleanup after a failure --
    a signal is held and raised once that is done, so an interruption never
    stops half way through putting things back.
    """

    NAMES = ("SIGTERM", "SIGHUP")

    def __init__(self) -> None:
        self.deferred = False
        self.pending: Optional[int] = None
        self._previous: List[Tuple[int, object]] = []

    def _handle(self, signum: int, frame: object) -> None:
        if self.deferred:
            self.pending = signum
            return
        raise Terminated(128 + signum)

    def __enter__(self) -> "_Signals":
        if threading.current_thread() is not threading.main_thread():
            return self
        for name in self.NAMES:
            signum = getattr(signal, name, None)
            if signum is None:  # pragma: no cover - Windows has no SIGHUP
                continue
            if signal.getsignal(signum) is not signal.SIG_DFL:
                continue
            self._previous.append((signum, signal.signal(signum, self._handle)))
        return self

    def __exit__(self, *exc: object) -> None:
        for signum, previous in reversed(self._previous):
            signal.signal(signum, previous)
        self._previous.clear()

    @contextmanager
    def hold(self) -> Iterator[None]:
        """Hold signals for the duration; raise one that came, after."""
        self.deferred = True
        try:
            yield
        finally:
            self.deferred = False
        if self.pending is not None:
            raise Terminated(128 + self.pending)


def _claim(locks: Sequence[Path]) -> List[Path]:
    """Create every lock file, or none: another run holds one of them."""
    held: List[Path] = []
    stamp = f"{socket.gethostname()} {os.getpid()}\n".encode()
    try:
        for lock in locks:
            try:
                fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                raise FileExistsError(
                    f"'{lock}' exists: another adata run is writing "
                    f"'{lock.name[: -len('.adata-lock')]}', or one was killed "
                    "before it could clean up. If none is running, remove the "
                    "lock (and any .adata-tmp or .adata-old file beside it) "
                    "and try again."
                ) from None
            held.append(lock)
            try:
                os.write(fd, stamp)
            finally:
                os.close(fd)
    except BaseException:
        _release(held)
        raise
    return held


def _release(locks: Sequence[Path]) -> None:
    for lock in locks:
        try:
            lock.unlink()
        except FileNotFoundError:
            pass


def _fsync_tree(path: Path) -> None:
    """Push a file, or every file in a directory tree, out to the disk.

    Only the bytes matter: a directory that cannot be opened for fsync (some
    filesystems refuse) is skipped, but a file that fails to sync is an
    error, because its contents may never reach the disk.
    """
    if path.is_dir() and not path.is_symlink():
        for root, dirs, files in os.walk(path):
            for name in files:
                _fsync_file(Path(root) / name)
            _fsync_dir(Path(root))
    elif path.exists():
        _fsync_file(path)


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _fsync_dir(path: Path) -> None:
    try:
        fd = os.open(path, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform without directory handles
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - filesystem refuses directory fsync
        pass
    finally:
        os.close(fd)


def _keep_permissions(source: Path, target: Path) -> None:
    """Give `target` the mode and group of the `source` it replaces.

    A new file gets the umask's mode, so replacing a store the owner had
    made private (640) quietly made it world-readable (644). For a Zarr
    store, the top directory's.
    """
    try:
        st = source.stat()
    except OSError:
        return
    try:
        os.chmod(target, st.st_mode & 0o7777)
    except OSError:  # pragma: no cover - not ours to change
        pass
    try:
        os.chown(target, -1, st.st_gid)
    except (OSError, AttributeError):  # pragma: no cover - not in that group
        pass


def _leftovers(paths: Sequence[Path]) -> None:
    """Refuse to start while any earlier run's temporary or backup exists.

    Every one is named at once, so a split over many groups is cleared up in
    one go rather than one rerun per leftover.
    """
    found = [p for p in paths if _exists(p)]
    if not found:
        return
    names = ", ".join(f"'{p}'" for p in found)
    if len(found) == 1:
        message = f"{names} already exists, probably left by an interrupted run."
    else:
        message = (
            f"{len(found)} files from an interrupted run already exist: {names}."
        )
    raise FileExistsError(
        f"{message} Check what {'it holds' if len(found) == 1 else 'they hold'}, "
        f"remove {'it' if len(found) == 1 else 'them'}, and try again."
    )


@contextmanager
def staged_outputs(outputs: Sequence[Path]) -> Iterator[List[Path]]:
    """Yield temporary paths to write `outputs` to; move them into place after.

    On success each output is swapped in: an existing one is set aside first
    and deleted only once every new output is in place. On failure the
    temporary paths are deleted and the outputs are left as they were.

    A replaced output passes its permissions on to its successor, which is
    synced to disk before the old one is deleted.
    """
    outputs = [Path(p) for p in outputs]
    temps = [_temp_for(p) for p in outputs]
    backups = [_sibling(p, "adata-old") for p in outputs]

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    with _Signals() as signals:
        locks = _claim([lock_for(p) for p in outputs])
        try:
            _leftovers(temps + backups)

            try:
                yield temps
            except BaseException:
                with signals.hold():
                    for temp in temps:
                        _discard(temp)
                raise

            with signals.hold():
                _swap(outputs, temps, backups)
        finally:
            _release(locks)


def _swap(outputs: List[Path], temps: List[Path], backups: List[Path]) -> None:
    replacing = [_exists(output) for output in outputs]
    try:
        for output, temp, had_output in zip(outputs, temps, replacing):
            if had_output:
                _keep_permissions(output, temp)
                # The old copy is deleted below; until the new one is on
                # disk, a power cut would leave neither.
                _fsync_tree(temp)
    except BaseException:
        for temp in temps:
            _remove(temp)
        raise

    swapped: List[Tuple[Path, Path, bool]] = []
    try:
        for output, temp, backup, had_output in zip(
            outputs, temps, backups, replacing
        ):
            if had_output:
                output.rename(backup)
            try:
                temp.rename(output)
            except BaseException:
                if had_output:
                    backup.rename(output)
                raise
            swapped.append((output, backup, had_output))
    except BaseException:
        # Put back what was already swapped, newest first, and drop the
        # temporaries that never made it.
        for output, backup, had_output in reversed(swapped):
            _remove(output)
            if had_output:
                backup.rename(output)
        for temp in temps:
            _remove(temp)
        raise

    if any(replacing):
        for parent in {output.parent for output in outputs}:
            _fsync_dir(parent)
    for _, backup, had_output in swapped:
        if had_output:
            _remove(backup)


# ---------------------------------------------------------------------------
# free space


def disk_size(path: Path) -> int:
    """Bytes held by a file, or by every file in a directory tree."""
    path = Path(path)
    if path.is_dir():
        total = 0
        for root, _, files in os.walk(path):
            for name in files:
                try:
                    total += os.lstat(os.path.join(root, name)).st_size
                except OSError:  # pragma: no cover - removed while walking
                    pass
        return total
    try:
        return path.stat().st_size
    except OSError:
        return 0


def _human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1000
    return f"{n:,.1f} TB"  # pragma: no cover - unreachable


def check_free_space(output: Path, needed: float, *, check: bool = True) -> None:
    """Refuse to start an output that the disk it goes on cannot hold.

    A full disk part way through an HDF5 write can abort the process inside
    the HDF5 library, where no exception reaches us and nothing is cleaned
    up. Checking first is the only defence. `needed` is an estimate, so
    `check=False` (`--no-space-check`) skips it.
    """
    if not check or needed <= 0:
        return
    directory = Path(output).absolute().parent
    while not directory.exists() and directory != directory.parent:
        directory = directory.parent
    try:
        free = shutil.disk_usage(directory).free
    except OSError:  # pragma: no cover - nothing to measure
        return
    if free < needed:
        raise OSError(
            f"Not enough free space for '{output}': it needs about "
            f"{_human(needed)} and {directory} has {_human(free)} free. Free "
            "some space or write elsewhere; if the estimate is wrong, pass "
            "--no-space-check."
        )
