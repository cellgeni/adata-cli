"""`ls` -- list the contents of any HDF5 or Zarr store.

Unlike `view`, this makes no AnnData assumptions: there is no n_obs x n_var
header and no requirement that `obs`/`var` exist, so it works on `.loom` files
and arbitrary `.h5` stores as well as AnnData ones.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Iterator, Optional, Tuple

from rich.console import Console
from rich.tree import Tree

from adata.core.info import format_type_info, get_entry_type
from adata.storage import is_dataset, is_group, open_store
from adata.util.path import norm_path


def walk(obj: Any, prefix: str = "", depth: Optional[int] = None) -> Iterator[Tuple[str, Any]]:
    """Yield ``(path, element)`` for every member below `obj`, depth-first."""
    if not is_group(obj):
        return
    for key in sorted(obj.keys()):
        child = obj[key]
        path = f"{prefix}/{key}" if prefix else key
        yield path, child
        if is_group(child) and (depth is None or depth > 1):
            next_depth = None if depth is None else depth - 1
            yield from walk(child, path, next_depth)


def _describe(obj: Any, long: bool) -> str:
    """Render the trailing annotation for one entry."""
    if not long:
        return ""
    info = get_entry_type(obj)
    bits = [format_type_info(info)]
    shape = getattr(obj, "shape", None)
    if shape:
        bits.append(f"[dim]{tuple(shape)}[/]")
    if is_dataset(obj) and getattr(obj, "dtype", None) is not None:
        bits.append(f"[dim]{obj.dtype}[/]")
    if info["encoding"]:
        bits.append(f"[dim]{info['encoding']}[/]")
    return " " + " ".join(bits)


def list_store(
    file: Path,
    console: Console,
    entry_path: Optional[str] = None,
    depth: Optional[int] = None,
    long: bool = False,
    plain: bool = False,
) -> None:
    """Print the structure of a store, as a tree or as bare paths.

    `plain` emits one path per line with no markup, so the output pipes
    cleanly into grep or xargs.
    """
    with open_store(file, "r", require_anndata=False) as store:
        root = store.root
        if entry_path:
            entry_path = norm_path(entry_path)
            if entry_path not in root:
                raise KeyError(f"'{entry_path}' not found in the store.")
            root = root[entry_path]

        if plain:
            _print_plain(root, entry_path or "", depth)
            return

        _print_tree(root, console, str(file), entry_path, depth, long)


def _print_plain(root: Any, prefix: str, depth: Optional[int]) -> None:
    if is_dataset(root):
        sys.stdout.write(f"{prefix}\n")
        return
    for path, _ in walk(root, prefix, depth):
        sys.stdout.write(f"{path}\n")


def _print_tree(
    root: Any,
    console: Console,
    label: str,
    entry_path: Optional[str],
    depth: Optional[int],
    long: bool,
) -> None:
    root_label = f"{label}:{entry_path}" if entry_path else label
    tree = Tree(f"[bold]{root_label}[/]")

    if is_dataset(root):
        tree.add(f"[bright_white]{entry_path or label}[/]{_describe(root, long)}")
        console.print(tree)
        return

    nodes = {"": tree}
    for path, obj in walk(root, "", depth):
        parent_path, _, name = path.rpartition("/")
        parent = nodes.get(parent_path, tree)
        if is_group(obj):
            nodes[path] = parent.add(
                f"[bold yellow]{name}/[/]{_describe(obj, long)}"
            )
        else:
            parent.add(f"[bright_white]{name}[/]{_describe(obj, long)}")

    console.print(tree)
