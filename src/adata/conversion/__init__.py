"""Format conversion: what `adata convert` reads and writes.

AnnData is the hub. Every other format has an importer (to AnnData), an
exporter (from AnnData), or both, and moving between two AnnData backends is
a copy of the store. Nothing converts directly between two other formats:
that would be a converter per pair rather than per format, and going through
an h5ad costs one extra command.

A new format is one `Format` entry in `FORMATS`, with the converters that
read and write it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, FrozenSet, List, Optional

import h5py

from adata.conversion import mtx, tenx
from adata.conversion.anndata_copy import AnnDataCopy
from adata.conversion.base import ConversionPlan, Converter, Format, Options
from adata.storage import has_valid_anndata_root_attrs, is_zarr_path

__all__ = [
    "ConversionPlan",
    "Converter",
    "FORMATS",
    "Format",
    "Options",
    "check_options",
    "converter_for",
    "detect_format",
    "output_format",
]


def _sniff_h5ad(path: Path) -> bool:
    """An HDF5 file with an AnnData root, or one structured like it.

    anndata 0.7 wrote no root attributes, so `obs` and `var` groups are taken
    as enough on their own.
    """
    if path.is_dir() or not h5py.is_hdf5(path):
        return False
    with h5py.File(path, "r") as handle:
        if has_valid_anndata_root_attrs(handle):
            return True
        return all(
            isinstance(handle.get(key), h5py.Group) for key in ("obs", "var")
        )


FORMATS: Dict[str, Format] = {
    "h5ad": Format(
        name="h5ad",
        description="AnnData, HDF5 backend",
        extensions=(".h5ad",),
        anndata=True,
        sniff=_sniff_h5ad,
    ),
    "zarr": Format(
        name="zarr",
        description="AnnData, Zarr backend (v2 or v3)",
        extensions=(".zarr",),
        anndata=True,
        sniff=is_zarr_path,
    ),
    "10x-h5": Format(
        name="10x-h5",
        description="10x Genomics feature-barcode matrix (Cell Ranger v2 or v3)",
        extensions=(".h5",),
        sniff=tenx.sniff,
        importer=tenx.TenxImport(),
        exporter=tenx.TenxExport(),
    ),
    # A folder, so no extension: chosen as output with `--to 10x-mtx`.
    "10x-mtx": Format(
        name="10x-mtx",
        description="10x Genomics / STARsolo Matrix Market folder",
        extensions=(),
        sniff=mtx.sniff,
        importer=mtx.MtxImport(),
        exporter=mtx.MtxExport(),
    ),
}

_ANNDATA_COPY = AnnDataCopy()

#: Flags every conversion accepts, whatever its formats.
COMMON_OPTIONS: FrozenSet[str] = frozenset({"--force", "--overwrite"})


def _names() -> str:
    return ", ".join(FORMATS)


def detect_format(path: Path) -> str:
    """Name the format of an existing input: by content, then by extension.

    Content comes first because extensions lie: Cell Ranger, loompy and
    plenty of hand-rolled pipelines all write `.h5`. The AnnData check runs
    last, since its fallback -- `obs` and `var` groups -- is the weakest test.
    Only AnnData falls back to the extension: another format's content check
    is the final word, so a plain HDF5 file named `.h5` is not taken for 10x.
    """
    path = Path(path)
    for fmt in FORMATS.values():
        if not fmt.anndata and fmt.sniff is not None and fmt.sniff(path):
            return fmt.name
    for fmt in FORMATS.values():
        if fmt.anndata and fmt.sniff is not None and fmt.sniff(path):
            return fmt.name
    suffix = path.suffix.lower()
    for fmt in FORMATS.values():
        if (fmt.anndata or fmt.sniff is None) and suffix in fmt.extensions:
            return fmt.name
    if suffix in (".mtx", ".gz") and path.name.lower().endswith((".mtx", ".mtx.gz")):
        raise ValueError(
            f"'{path.name}' is a single Matrix Market file, which `convert` "
            "does not read on its own: give the folder holding it with its "
            "barcodes and features, or use `adata import sparse`."
        )
    raise ValueError(
        f"Cannot tell what format '{path}' is. Pass --from, one of: {_names()}."
    )


def output_format(path: Path, requested: Optional[str] = None) -> str:
    """The format to write to `path`: `requested`, else from its extension.

    A request that contradicts a known extension is refused rather than
    obeyed, because the extension is what picks the backend: `--to zarr -o
    out.h5ad` would otherwise write HDF5 while claiming to write Zarr.
    """
    path = Path(path)
    if requested is not None and requested not in FORMATS:
        raise ValueError(f"Unknown format {requested!r}; use one of: {_names()}.")

    suffix = path.suffix.lower()
    by_extension = next(
        (fmt.name for fmt in FORMATS.values() if suffix in fmt.extensions), None
    )
    if requested is None:
        if by_extension is None:
            raise ValueError(
                f"Cannot tell the output format from '{path.name}'. "
                f"Use a known extension or pass --to, one of: {_names()} "
                "(--to 10x-mtx for a folder)."
            )
        return by_extension

    if by_extension is not None and by_extension != requested:
        extensions = FORMATS[requested].extensions
        raise ValueError(
            f"--to {requested} contradicts the output's extension, which means "
            f"{by_extension}. Name the output "
            + (
                f"with one of: {', '.join(extensions)}."
                if extensions
                else "as a folder, without an extension."
            )
        )
    if (requested == "zarr") != (suffix == ".zarr"):
        raise ValueError(
            "A Zarr output needs a name ending in .zarr, and only a Zarr "
            "output may have one."
        )
    return requested


def converter_for(src_format: str, dst_format: str) -> Converter:
    """The converter for a pair of formats, or a refusal that says why not."""
    for name in (src_format, dst_format):
        if name not in FORMATS:
            raise ValueError(f"Unknown format {name!r}; use one of: {_names()}.")
    src, dst = FORMATS[src_format], FORMATS[dst_format]

    if src.anndata and dst.anndata:
        return _ANNDATA_COPY
    if dst.anndata:
        if src.importer is None:
            raise ValueError(f"{src_format} can be written but not read.")
        return src.importer
    if src.anndata:
        if dst.exporter is None:
            raise ValueError(f"{dst_format} can be read but not written.")
        return dst.exporter
    raise ValueError(
        f"Cannot convert {src_format} to {dst_format} directly; convert it to "
        f".h5ad first, then from that to {dst_format}."
    )


def check_options(
    converter: Converter, options: Options, src_format: str, dst_format: str
) -> None:
    """Refuse a flag the conversion would ignore, rather than ignoring it."""
    accepted = set(COMMON_OPTIONS) | set(converter.options)
    if dst_format == "zarr":
        accepted.add("--zarr-format")
    unused: List[str] = sorted(set(options.given) - accepted)
    if unused:
        raise ValueError(
            f"{', '.join(unused)} {'does' if len(unused) == 1 else 'do'} not "
            f"apply when converting {src_format} to {dst_format}."
        )
