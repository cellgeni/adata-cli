from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.tree import Tree

from adata.core.info import axis_len, format_type_info, get_entry_type
from adata.storage import is_dataset, is_group, open_store

# Preferred display order for top-level keys
KEY_ORDER = ["X", "obs", "var", "obsm", "varm", "layers", "obsp", "varp", "uns"]


def _sort_keys(keys: list) -> list:
    """Sort keys according to KEY_ORDER, with unknown keys at the end."""
    order_map = {k: i for i, k in enumerate(KEY_ORDER)}
    return sorted(keys, key=lambda k: (order_map.get(k, len(KEY_ORDER)), k))


def show_info(
    file: Path,
    console: Console,
    show_types: bool = False,
    depth: Optional[int] = None,
    entry_path: Optional[str] = None,
    out_console: Optional[Console] = None,
) -> None:
    """
    Show high-level information about the .h5ad file.
    Args:
        file (Path): Path to the .h5ad file
        console (Console): Rich console for output
        show_types (bool): Show detailed type information for each entry
        depth (Optional[int]): Maximum recursion depth for type display (only with show_types=True)
        entry_path (Optional[str]): Specific entry path to inspect (e.g., 'obsm/X_pca')
        out_console (Optional[Console]): Where results go. Defaults to stdout so
            that output pipes cleanly; `console` keeps status and errors.
    """
    out = out_console if out_console is not None else Console()
    with open_store(file, "r") as store:
        f = store.root
        # If a specific path is requested, show detailed info for that object
        if entry_path:
            _show_object_info(f, entry_path, out)
            return

        # Get n_obs and n_var
        n_obs = axis_len(f, "obs")
        n_var = axis_len(f, "var")
        out.print(
            "[bold cyan]An object with n_obs × n_var: "
            f"{n_obs if n_obs is not None else '?'} × "
            f"{n_var if n_var is not None else '?'}[/]"
        )

        if show_types:
            _show_types_tree(f, out, root_label=str(file), depth=depth)
        else:
            # List top-level keys and their sub-keys (original behavior)
            for key in _sort_keys(list(f.keys())):
                obj = f[key]
                # Only process Groups, skip Datasets like X
                if is_group(obj):
                    sub_keys = [
                        k
                        for k in obj.keys()
                        if k not in ("_index", "__categories", "obs_names", "var_names")
                    ]
                    if sub_keys and key != "X":
                        out.print(
                            f"\t[bold yellow]{key}:[/]\t"
                            + ", ".join(f"[bright_white]{sub}[/]" for sub in sub_keys)
                        )


def _show_types_tree(
    f: Any, console: Console, root_label: str, depth: Optional[int] = None
) -> None:
    """Show a tree view with type information for all entries.

    Recursion depth by group:
        - obs/var: top level only (no children)
        - X: top level only
        - obsm/obsp/varm/varp/layers: 1 level (show matrices)
        - uns: 2 levels deep
    """
    tree = Tree(f"[bold]{root_label}[/]")

    # Define max depth for each top-level group
    max_depth_map = {
        "obs": 0,
        "var": 0,
        "X": 0,
        "obsm": 1,
        "obsp": 1,
        "varm": 1,
        "varp": 1,
        "layers": 1,
        "uns": 2,
    }

    def add_node(
        parent_tree: Tree,
        name: str,
        obj: Any,
        current_depth: int,
        max_depth: int,
    ) -> None:
        info = get_entry_type(obj)
        type_str = format_type_info(info)

        if is_dataset(obj):
            shape_str = f"[dim]{obj.shape}[/]" if obj.shape else ""
            node_text = f"[bright_white]{name}[/] {shape_str} {type_str}"
            parent_tree.add(node_text)
        else:
            # Group
            node_text = f"[bold yellow]{name}/[/] {type_str}"
            subtree = parent_tree.add(node_text)

            # Recurse only if within allowed depth
            if current_depth < max_depth:
                for child_name in sorted(obj.keys()):
                    if child_name in ("_index", "__categories"):
                        continue
                    child_obj = obj[child_name]
                    add_node(
                        subtree, child_name, child_obj, current_depth + 1, max_depth
                    )

    # Add top-level items in preferred order
    for key in _sort_keys(list(f.keys())):
        obj = f[key]
        # Skip empty groups
        if is_group(obj):
            children = [
                k
                for k in obj.keys()
                if k not in ("_index", "__categories", "obs_names", "var_names")
            ]
            if not children:
                continue
        max_depth = (
            depth if depth is not None else max_depth_map.get(key, 1)
        )  # default to 1 level for unknown groups
        add_node(tree, key, obj, current_depth=0, max_depth=max_depth)

    console.print(tree)


def _show_object_info(f: Any, entry_path: str, console: Console) -> None:
    """Show detailed info for a specific object path."""
    # Normalize path
    entry_path = entry_path.strip().lstrip("/")

    if entry_path not in f:
        console.print(f"[bold red]Error:[/] '{entry_path}' not found in the file.")
        return

    entry = f[entry_path]
    info = get_entry_type(entry)

    console.print(f"\n[bold cyan]Path:[/] {entry_path}")
    console.print(f"[bold cyan]Type:[/] {info['type']}")

    if info["encoding"]:
        console.print(f"[bold cyan]Encoding:[/] {info['encoding']}")

    if info["shape"]:
        console.print(f"[bold cyan]Shape:[/] {info['shape']}")

    if info["dtype"]:
        console.print(f"[bold cyan]Dtype:[/] {info['dtype']}")

    console.print(f"[bold cyan]Details:[/] {info['details']}")

    # Show attributes if any
    if entry.attrs:
        console.print(f"\n[bold cyan]Attributes:[/]")
        for k, v in entry.attrs.items():
            v_str = v.decode("utf-8") if isinstance(v, bytes) else str(v)
            if len(v_str) > 80:
                v_str = v_str[:77] + "..."
            console.print(f"  [dim]{k}:[/] {v_str}")

    # If it's a group, show children
    if is_group(entry):
        children = [
            k
            for k in entry.keys()
            if k not in ("_index", "__categories", "obs_names", "var_names")
        ]
        if children:
            console.print(f"\n[bold cyan]Children:[/]")
            for child_name in sorted(children):
                child_entry = entry[child_name]
                child_info = get_entry_type(child_entry)
                type_str = format_type_info(child_info)
                console.print(f"  [bright_white]{child_name}[/] {type_str}")
