"""`create` -- write a new, empty AnnData store.

Gives `import` something to write into. The result is a complete AnnData
object in its own right: root attributes, obs and var dataframes with real
indices, and the optional mapping groups, so anndata can open it before a
single matrix has been attached.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.console import Console

from adata.elements.write import ensure_anndata_skeleton, write_dataframe_header
from adata.storage import open_store


def _read_names(path: Path) -> List[str]:
    """Read a newline-delimited name list, ignoring blank lines."""
    names = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"'{path}' contains no names.")
    if len(set(names)) != len(names):
        raise ValueError(f"'{path}' contains duplicate names.")
    return names


def _resolve_axis(
    axis: str, n: Optional[int], names_file: Optional[Path]
) -> List[str]:
    """Determine an axis's index, from an explicit list or a generated range."""
    if names_file is not None:
        names = _read_names(names_file)
        if n is not None and n != len(names):
            raise ValueError(
                f"--n-{axis} is {n} but '{names_file}' has {len(names)} names."
            )
        return names

    if n is None:
        raise ValueError(f"Provide either --n-{axis} or --{axis}-names.")
    if n < 0:
        raise ValueError(f"--n-{axis} must not be negative.")

    width = len(str(max(n - 1, 0)))
    prefix = "cell" if axis == "obs" else "gene"
    return [f"{prefix}_{i:0{width}d}" for i in range(n)]


def create_store(
    output: Path,
    console: Console,
    n_obs: Optional[int] = None,
    n_var: Optional[int] = None,
    obs_names: Optional[Path] = None,
    var_names: Optional[Path] = None,
    zarr_format: Optional[int] = None,
    force: bool = False,
) -> None:
    """Create an empty AnnData store at `output`.

    Names come from `obs_names`/`var_names` when given, otherwise they are
    generated as `cell_0000`-style labels wide enough for the axis length.
    """
    if output.exists() and not force:
        raise FileExistsError(
            f"'{output}' already exists. Pass --force to overwrite it."
        )

    obs_index = _resolve_axis("obs", n_obs, obs_names)
    var_index = _resolve_axis("var", n_var, var_names)

    if output.exists() and force:
        import shutil

        shutil.rmtree(output) if output.is_dir() else output.unlink()

    with open_store(output, "w", zarr_format=zarr_format) as store:
        root = store.root
        write_dataframe_header(root, "obs", obs_index, [])
        write_dataframe_header(root, "var", var_index, [])
        ensure_anndata_skeleton(root)

    console.print(
        f"[green]Created[/] {output} "
        f"({len(obs_index)} obs x {len(var_index)} var, empty)"
    )
