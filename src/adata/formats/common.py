from __future__ import annotations

import io
import os
import sys
from contextlib import contextmanager
from typing import Any, Iterator

import numpy as np

from adata.storage import is_dataset, is_group
from adata.util.path import norm_path


def _get_encoding_type(group: Any) -> str:
    enc = group.attrs.get("encoding-type", "")
    if isinstance(enc, bytes):
        enc = enc.decode("utf-8")
    return str(enc)


def _resolve(root: Any, obj: str) -> Any:
    obj = norm_path(obj)
    if obj not in root:
        raise KeyError(f"'{obj}' not found in the file.")
    return root[obj]


def _check_json_exportable(h5obj: Any, max_elements: int, path: str = "") -> None:
    if is_dataset(h5obj):
        if h5obj.shape == ():
            return
        n = int(np.prod(h5obj.shape)) if h5obj.shape else 0
        if n > max_elements:
            obj_name = getattr(h5obj, "name", "<object>")
            raise ValueError(
                f"Cannot export to JSON: '{path or obj_name}' has {n} elements "
                f"(max {max_elements}). Use --max-elements to increase limit."
            )
        return

    if is_group(h5obj):
        enc = _get_encoding_type(h5obj)
        if enc in ("csr_matrix", "csc_matrix"):
            obj_name = getattr(h5obj, "name", "<object>")
            raise ValueError(
                f"Cannot export to JSON: '{path or obj_name}' is a sparse matrix. "
                "Export it as .mtx instead."
            )

        for key in h5obj.keys():
            child = h5obj[key]
            child_path = f"{path}/{key}" if path else key
            if is_group(child) or is_dataset(child):
                _check_json_exportable(child, max_elements=max_elements, path=child_path)


def _silence_stdout() -> None:
    """Point stdout at /dev/null, so the flush at exit has nowhere to fail."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
        os.close(devnull)
    except (OSError, ValueError, io.UnsupportedOperation):
        pass  # not a real file (a test's capture): nothing will flush it


@contextmanager
def stdout_reader_may_stop() -> Iterator[None]:
    """End an export quietly when what reads its stdout stops early.

    `adata export ... | head` closes the pipe after a few lines; the next
    write then raises BrokenPipeError, which was reported as an error with
    exit 1 although the reader had all it asked for.
    """
    try:
        yield
        sys.stdout.flush()
    except BrokenPipeError:
        _silence_stdout()
