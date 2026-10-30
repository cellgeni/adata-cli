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
can create. A lock left by a run that was killed on this host is taken over;
one from another host, whose run cannot be checked, is refused. A
scheduler's SIGTERM (or a closed terminal's SIGHUP) is turned into an
exception where it lands, so it gets the same cleanup as a failure.

An output that is a symlink is written through: the file or store it points
at is replaced, and the link kept.
"""

from __future__ import annotations

import os
import shutil
import signal
import socket
import threading
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple, Union

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
                # Even over a failure in the body: the run was asked to stop,
                # and its exit code should say so.
                signum, self.pending = self.pending, None
                raise Terminated(128 + signum)


def _pid_namespace() -> str:
    """This process's PID namespace, where Linux names one; "-" elsewhere.

    Two containers can share a hostname but not their process ids, and a
    pid looked up in the wrong namespace looks dead.
    """
    try:
        return os.readlink("/proc/self/ns/pid")
    except OSError:
        return "-"


def _stamp() -> str:
    return f"{socket.gethostname()} {os.getpid()} {_pid_namespace()}\n"


def _holder(lock: Path) -> Optional[Tuple[str, int, str]]:
    """The host, process id and PID namespace a lock names, if readable."""
    try:
        fields = lock.read_text().split()
        host, pid = fields[0], int(fields[1])
        return host, pid, fields[2] if len(fields) > 2 else "?"
    except (OSError, ValueError, IndexError):
        return None


def _dead(holder: Optional[Tuple[str, int, str]]) -> bool:
    """Is the run that made a lock gone? Knowable only where it ran: on this
    host, in this PID namespace."""
    if holder is None or os.name == "nt":
        return False
    host, pid, namespace = holder
    if host != socket.gethostname() or namespace != _pid_namespace():
        return False
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return True
    except OSError:  # alive, as another user
        return False
    return False


def _take_over(lock: Path, holder: Tuple[str, int, str]) -> bool:
    """Remove a dead run's lock, unless another run got to it first.

    Renamed aside first, so two runs taking over the same stale lock cannot
    both delete it: the one that finds a fresh lock under its name instead
    puts it back and gives way.
    """
    aside = lock.with_name(f"{lock.name}.{os.getpid()}.stale")
    try:
        os.rename(lock, aside)
    except FileNotFoundError:
        return True
    if _holder(aside) == holder:
        aside.unlink()
        return True
    try:
        os.link(aside, lock)
    except OSError:  # pragma: no cover - a third run claimed it meanwhile
        pass
    aside.unlink()
    return False


def _refusal(lock: Path, holder: Optional[Tuple[str, int, str]]) -> FileExistsError:
    output = lock.name[: -len(".adata-lock")]
    who = (
        f"another adata run (process {holder[1]} on {holder[0]})"
        if holder
        else "another adata run"
    )
    return FileExistsError(
        f"'{lock}' exists: {who} is writing '{output}', or was killed before "
        "it could clean up. If that run is no longer alive, delete the lock "
        "and try again."
    )


def _claim(locks: Sequence[Path]) -> List[Path]:
    """Create every lock file, or none: another run holds one of them."""
    held: List[Path] = []
    stamp = _stamp().encode()
    try:
        for lock in locks:
            for attempt in range(2):
                try:
                    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    break
                except FileExistsError:
                    holder = _holder(lock)
                    if not (attempt == 0 and _dead(holder) and _take_over(lock, holder)):
                        raise _refusal(lock, holder) from None
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
    """Push a file, or a directory tree, out to the disk.

    A file that fails to sync is an error: its contents may never reach the
    disk. A Zarr store can hold millions of chunk files, and an fsync each
    is a round trip per file on a network filesystem, so a tree is flushed
    with one `sync()` where there is one.
    """
    if path.is_dir() and not path.is_symlink():
        if _sync_filesystem(path):
            return
        for root, dirs, files in os.walk(path):  # pragma: no cover - Windows
            for name in files:
                _fsync_file(Path(root) / name)
            _fsync_dir(Path(root))
    elif path.exists():
        _fsync_file(path)


def _sync_filesystem(path: Path) -> bool:
    """Flush the filesystem `path` is on; False where there is no way to.

    Linux's syncfs() waits for that filesystem alone. sync() waits for every
    mount on the node -- other jobs' writes, a hung network mount -- so it
    is the fallback only where syncfs() does not exist (macOS, where sync()
    does not wait anyway).
    """
    try:
        import ctypes

        syncfs = getattr(ctypes.CDLL(None, use_errno=True), "syncfs", None)
    except (OSError, AttributeError):  # pragma: no cover - no libc to ask
        syncfs = None
    if syncfs is not None:  # pragma: no cover - Linux
        fd = os.open(path, os.O_RDONLY)
        try:
            if syncfs(fd) != 0:
                error = ctypes.get_errno()
                raise OSError(error, os.strerror(error), str(path))
        finally:
            os.close(fd)
        return True
    if hasattr(os, "sync"):
        os.sync()
        return True
    return False  # pragma: no cover - Windows


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
    if source.is_dir() != target.is_dir():
        return  # a file's mode would lock a directory's owner out of it
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
    outputs = [_follow(Path(p)) for p in outputs]
    temps = [_temp_for(p) for p in outputs]
    backups = [_sibling(p, "adata-old") for p in outputs]

    for output in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)

    with _Signals() as signals:
        locks: List[Path] = []
        try:
            with signals.hold():
                locks = _claim([lock_for(p) for p in outputs])
            _leftovers(temps + backups)

            try:
                yield temps
                # Before signals are held: a sync can take a while, and a
                # SIGTERM during it should still clean up.
                _settle(outputs, temps)
            except BaseException:
                with signals.hold():
                    for temp in temps:
                        _discard(temp)
                raise

            with signals.hold():
                _swap(outputs, temps, backups)
        finally:
            with signals.hold():
                _release(locks)


def _follow(path: Path) -> Path:
    """Where an output that is a symlink really is.

    Renaming the link aside and the new store into its place made the link
    a private copy, and left the store it pointed at, which others read
    through the link, unchanged.
    """
    return Path(os.path.realpath(path)) if path.is_symlink() else path


def _settle(outputs: List[Path], temps: List[Path]) -> None:
    """Make each new output ready to replace an existing one."""
    for output, temp in zip(outputs, temps):
        if _exists(output):
            _keep_permissions(output, temp)
            # The old copy is deleted once this is in place; until the new
            # one is on disk, a power cut would leave neither.
            _fsync_tree(temp)


def _swap(outputs: List[Path], temps: List[Path], backups: List[Path]) -> None:
    replacing = [_exists(output) for output in outputs]
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
    for output, backup, had_output in swapped:
        if not had_output:
            continue
        try:
            _remove(backup)
        except OSError as error:
            # Every output is in place: the run succeeded. The old copy
            # will block the next run, which says what to do with it.
            warnings.warn(
                f"Wrote '{output}', but could not delete the old copy at "
                f"'{backup}': {error}. Delete it before the next run.",
                stacklevel=4,
            )


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


def check_free_space(
    output: Path,
    needed: Union[float, Callable[[], float]],
    *,
    check: bool = True,
) -> None:
    """Refuse to start an HDF5 output that the disk it goes on cannot hold.

    A full disk part way through an HDF5 write can abort the process inside
    the HDF5 library, where no exception reaches us and nothing is cleaned
    up. Checking first is the only defence. A Zarr store that runs out of
    space fails with an ordinary error and is cleaned up, so it is not
    checked, and the source is not measured for it: `needed` may be a
    function, called only when the check runs. It is an estimate, so
    `check=False` (`--no-space-check`) skips it.
    """
    output = _follow(Path(output))
    if not check or output.suffix == ".zarr" or is_zarr_path(output):
        return
    if callable(needed):
        needed = needed()
    if needed <= 0:
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
