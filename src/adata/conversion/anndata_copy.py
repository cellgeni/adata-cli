"""Convert between AnnData backends: h5ad, Zarr v2 and Zarr v3.

A copy of the store, element by element, streamed. `subset` has always been
able to do this along the way; this is the same copy without the selection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.elements.write import ensure_anndata_skeleton
from adata.storage import (
    COMPRESSIONS,
    copy_attrs,
    copy_tree,
    ensure_anndata_root_attrs,
    open_store,
)


def _backend(fmt: str) -> str:
    return "zarr" if fmt == "zarr" else "hdf5"


class AnnDataCopy(Converter):
    options = frozenset({"--compression"})

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        backend = _backend(dst_format)
        if (
            options.compression is not None
            and options.compression not in COMPRESSIONS[backend]
        ):
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )

        with open_store(source, "r", require_anndata=False) as src:
            source_zarr_format = src.zarr_format
            n_elements = len(list(src.root.keys()))

        zarr_format: Optional[int] = None
        if dst_format == "zarr":
            # A v2 store stays v2 unless asked; anything else becomes v3,
            # zarr-python's default and anndata's.
            zarr_format = options.zarr_format or source_zarr_format or 3

        target = f"zarr v{zarr_format}" if zarr_format else dst_format
        origin = f"zarr v{source_zarr_format}" if source_zarr_format else src_format
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"{origin} → {target}, {n_elements} top-level "
                f"{'element' if n_elements == 1 else 'elements'}"
            ),
            detail={"zarr_format": zarr_format, "compression": options.compression},
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail = plan.detail
        with open_store(plan.source, "r", require_anndata=False) as src, open_store(
            staged[0], "w", zarr_format=detail["zarr_format"]
        ) as dst:
            backend = _backend(plan.dst_format)
            copy_attrs(src.root.attrs, dst.root.attrs, target_backend=backend)
            ensure_anndata_root_attrs(dst.root)
            for key in src.root.keys():
                with console.status(f"[cyan]Copying {key}...[/]"):
                    copy_tree(
                        src.root[key], dst.root, key,
                        compression=detail["compression"],
                    )
            ensure_anndata_skeleton(dst.root)
