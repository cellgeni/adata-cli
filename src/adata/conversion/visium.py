"""Space Ranger output for standard Visium: counts, spot positions, images.

A Space Ranger `outs/` folder holds the 10x matrix and, in `spatial/`, where
each spot sits on the slide (`tissue_positions.csv`, or Space Ranger 1's
`tissue_positions_list.csv` with no header), the scale factors between the
full-resolution image and the two PNGs beside them, and the PNGs.

The AnnData produced is laid out as `squidpy.read.visium` lays it out, which
scanpy and squidpy plot from:

* obs: `in_tissue`, `array_row`, `array_col`;
* `obsm["spatial"]`: each spot's full-resolution pixel, `[col, row]` (x, y);
* `uns["spatial"][library_id]`: `images` (`hires`, `lowres`, the PNGs'
  uint8 pixels), `scalefactors` (the JSON as it is) and `metadata`
  (`chemistry_description`, `software_version`).

The matrix is read as a 10x HDF5 file is (`tenx`), so features are named by
gene id and every feature type is kept, where squidpy names them by symbol
and keeps Gene Expression only. The positions list every spot on the slide,
a few thousand however large the data, and are matched to the matrix's
barcodes as they are scanned.

Visium HD is read too, one bin size at a time (`--bin-size`, 8 µm by
default): each `binned_outputs/square_NNNum/` folder is laid out as above,
with its positions as Parquet, which pyarrow reads (the `spatial` extra).
"""

from __future__ import annotations

import json
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import h5py
import numpy as np
from rich.console import Console

from adata.conversion import tenx
from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import STRING_STEP, Target, describe_import, write_obs_index, write_var
from adata.conversion.spatial import BarcodeJoin, Join, Sink, table_blocks, numbers, write_image, require_numeric, write_rows
from adata.core.subset import fan_out_sparse_matrix
from adata.elements.read import decode_str_array
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_dense,
    write_mapping,
    write_scalar,
)
from adata.formats.json_data import write_json_value
from adata.formats.parquet import ParquetTable, open_table, require_pyarrow
from adata.formats.table import TableScan, column_values, create_columns, scan_table
from adata.formats.textio import open_binary
from adata.storage import create_dataset, is_zarr_path, open_store
from adata.elements import spec

#: The columns of `tissue_positions.csv`, which Space Ranger 1 wrote without
#: a header, in this order.
POSITION_COLUMNS = (
    "barcode", "in_tissue", "array_row", "array_col",
    "pxl_row_in_fullres", "pxl_col_in_fullres",
)
_POSITION_FILES = ("tissue_positions.csv", "tissue_positions_list.csv", "tissue_positions.parquet")
#: The bin size an HD output is read at unless --bin-size says otherwise:
#: the one 10x recommends for analysis.
DEFAULT_BIN_UM = 8
_IMAGES = ("hires", "lowres")
_FILTERED, _RAW = "filtered_feature_bc_matrix", "raw_feature_bc_matrix"


def _is_hd(folder: Path) -> bool:
    """A Visium HD `outs/`, whose bins sit one folder per size below it."""
    return (folder / "binned_outputs").is_dir()


def _bin_sizes(outs: Path) -> Dict[int, Path]:
    """Each bin size an HD output holds, in µm, and its folder."""
    sizes = {}
    for folder in (outs / "binned_outputs").glob("square_*um"):
        digits = folder.name[len("square_"):-len("um")]
        if digits.isdigit() and folder.is_dir():
            sizes[int(digits)] = folder
    return dict(sorted(sizes.items()))


def _matrix_files(folder: Path, name: str) -> List[Path]:
    """`<name>.h5` in the folder, or one prefixed with the sample name, as
    10x's downloads are (`V1_Mouse_Brain_filtered_feature_bc_matrix.h5`)."""
    return sorted(
        p for p in folder.glob(f"*{name}.h5")
        if p.name == f"{name}.h5" or p.name.endswith(f"_{name}.h5")
    )


def _locate(path: Path) -> Optional[Path]:
    """The Space Ranger output folder at `path` or in its `outs/`."""
    for folder in (path, path / "outs"):
        if not folder.is_dir():
            continue
        if _is_hd(folder):
            return folder
        spatial = folder / "spatial"
        if (spatial / "scalefactors_json.json").is_file() and any(
            (spatial / name).is_file() for name in _POSITION_FILES
        ) and (
            _matrix_files(folder, _FILTERED) or _matrix_files(folder, _RAW)
            or (folder / _FILTERED).is_dir() or (folder / _RAW).is_dir()
        ):
            return folder
    return None


def sniff(path: Path) -> bool:
    """A Space Ranger output folder (or the run folder holding its `outs/`),
    a Visium HD one included, or one of its bin folders."""
    path = Path(path)
    return path.is_dir() and not is_zarr_path(path) and _locate(path) is not None


@dataclass
class _Plan:
    matrix: Any
    matrix_path: Path
    targets: List[Target]
    #: The CSV positions' scan, or the Parquet ones' footer.
    positions: Union[TableScan, ParquetTable]
    join: Join
    bin_um: Optional[int]
    library_id: str
    scalefactors: Dict[str, Any]
    images: Dict[str, Path]
    filtered_barcodes: bool
    zarr_format: Optional[int]


def _bundle(source: Path, options: Options) -> tuple:
    """(the outs folder, the matrix file) for this input."""
    if source.is_file():
        for flag in ("--unfiltered", "--bin-size"):
            if flag in options.given:
                raise ValueError(
                    f"{flag} picks among the matrices of a Space Ranger output "
                    f"folder; {source.name} is a matrix already. Give the folder."
                )
        return source.parent, source
    outs = _locate(source)
    if outs is None:
        raise ValueError(
            f"{source} is not a Space Ranger output folder: it needs spatial/ "
            "(tissue positions and scalefactors_json.json) beside a "
            "*_feature_bc_matrix.h5."
        )
    if _is_hd(outs):
        sizes = _bin_sizes(outs)
        size = DEFAULT_BIN_UM if options.bin_size is None else options.bin_size
        if size not in sizes:
            raise ValueError(
                f"This Visium HD output has no {size} µm bins; it has "
                f"{', '.join(f'{s} µm' for s in sizes) or 'none'}. Pick one with --bin-size."
            )
        outs = sizes[size]
    elif options.bin_size is not None:
        raise ValueError(
            "--bin-size picks among a Visium HD output's bins; this is "
            "standard Visium, or one bin size already."
        )
    name = _RAW if options.unfiltered else _FILTERED
    found = _matrix_files(outs, name)
    if len(found) > 1:
        raise ValueError(
            f"{outs} holds several {name}.h5 files ({', '.join(p.name for p in found)}); "
            "give the one to read with --from visium."
        )
    if not found:
        if (outs / name).is_dir():
            raise ValueError(
                f"{outs} has {name}/ as a Matrix Market folder but no {name}.h5, "
                f"which this reads. Convert the folder with --from 10x-mtx, "
                "without the positions and images."
            )
        raise ValueError(f"{outs} has no {name}.h5.")
    return outs, found[0]


def _has_header(path: Path) -> bool:
    with open_binary(path) as fh:
        first = fh.readline().decode("utf-8-sig", "replace")
    return first.split(",", 1)[0].strip() == "barcode"


def _library_id(attrs: Dict[str, Any], outs: Path, options: Options) -> str:
    if options.library_id is not None:
        if not options.library_id or "/" in options.library_id:
            raise ValueError(
                f"--library-id {options.library_id!r} cannot be a key of uns: it "
                "must be non-empty, with no '/'."
            )
        return options.library_id
    ids = [
        v.decode("utf-8", "replace") if isinstance(v, bytes) else str(v)
        for v in np.atleast_1d(attrs.get("library_ids", []))
    ]
    if len(ids) > 1:
        raise ValueError(
            f"This matrix holds {len(ids)} libraries ({', '.join(ids)}), as "
            "spaceranger aggr writes; their spots share no one image. Convert "
            "each library's own output."
        )
    if ids:
        return ids[0]
    return outs.parent.name if outs.name == "outs" else outs.name


def _bin_um(scalefactors: Dict[str, Any]) -> Optional[int]:
    """An HD bin folder's bin size, from its scale factors."""
    size = scalefactors.get("bin_size_um")
    return int(round(float(size))) if size is not None else None


def _scan_parquet(path: Path, joiner: BarcodeJoin) -> ParquetTable:
    """Visium HD's positions: the footer, then the barcodes, matched as read."""
    require_pyarrow("Visium HD positions")
    table = open_table(path)
    missing = [c for c in POSITION_COLUMNS if c not in table.types]
    if missing:
        raise ValueError(f"{path.name} has no {', '.join(missing)} column.")
    text = [c for c in POSITION_COLUMNS[1:] if not _numeric_arrow(table.types[c])]
    if text:
        raise ValueError(f"{path.name}: {', '.join(text)} {'is' if len(text) == 1 else 'are'} not numeric.")
    # Space Ranger marks these not null; an edited file may not be, and a
    # null cast to int64 would be a made-up number. Missing pixels are NaN.
    gaps = {c: n for c, n in table.null_counts(POSITION_COLUMNS[:4]).items() if n}
    if gaps:
        raise ValueError(
            f"{path.name} has missing values in "
            + ", ".join(f"{c} ({n:,})" for c, n in gaps.items()) + "."
        )
    for _, block in table.blocks(["barcode"]):
        joiner(block["barcode"])
    return table


def _numeric_arrow(type_name: str) -> bool:
    return type_name.startswith(("int", "uint", "float", "double", "halffloat"))


class VisiumImport(Converter):
    options = tenx.TenxImport.options | frozenset(
        {"--unfiltered", "--library-id", "--no-images", "--bin-size"}
    )

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        source = Path(source)
        outs, matrix_path = _bundle(source, options)
        spatial = outs / "spatial"
        position_file = next((spatial / n for n in _POSITION_FILES if (spatial / n).is_file()), None)
        scalefactor_file = spatial / "scalefactors_json.json"
        lacking = [
            what for what, present in (
                ("tissue positions (spatial/tissue_positions.csv or .parquet)", position_file is not None),
                ("spatial/scalefactors_json.json", scalefactor_file.is_file()),
            ) if not present
        ]
        if lacking:
            raise ValueError(f"{outs} has no {' and no '.join(lacking)} beside the matrix.")

        with h5py.File(matrix_path, "r") as handle:
            if not tenx.sniff(matrix_path):
                raise ValueError(f"{matrix_path.name} is not a 10x HDF5 matrix.")
            matrix, outputs, targets, warnings = tenx.plan_matrix_import(handle, output, options)
            library_id = _library_id(matrix.attrs, outs, options)
            group = handle[matrix.group]
            joiner = BarcodeJoin(group["barcodes"])
            if position_file.suffix == ".parquet":
                positions = _scan_parquet(position_file, joiner)
            else:
                positions = scan_table(
                    position_file,
                    index="barcode",
                    names=None if _has_header(position_file) else POSITION_COLUMNS,
                    on_index=joiner,
                )
                require_numeric(positions, POSITION_COLUMNS[1:])
            filtered_barcodes = "filtered_barcodes" in group
            join = joiner.finish()
        join.check(f"spatial/{position_file.name}")

        scalefactors = json.loads(scalefactor_file.read_text())
        bin_um = _bin_um(scalefactors) if position_file.suffix == ".parquet" else None
        images: Dict[str, Path] = {}
        if not options.no_images:
            for name in _IMAGES:
                path = spatial / f"tissue_{name}_image.png"
                if path.is_file():
                    images[name] = path
                else:
                    warnings.append(f"No spatial/{path.name}; uns['spatial'] will have no {name} image.")

        described = describe_import(matrix.n_barcodes, matrix.n_features, matrix.feature_types, targets)
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=(
                f"Space Ranger (Visium{f' HD, {bin_um} µm bins' if bin_um else ''}) → "
                f"{dst_format}, {described}, positions from "
                f"{position_file.name}, library {library_id!r}"
                + (f", images {', '.join(images)}" if images else "")
            ),
            warnings=warnings,
            detail=_Plan(
                matrix=matrix,
                matrix_path=matrix_path,
                targets=targets,
                positions=positions,
                join=join,
                bin_um=bin_um,
                library_id=library_id,
                scalefactors=scalefactors,
                images=images,
                filtered_barcodes=filtered_barcodes,
                zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
            ),
        )

    def write(self, plan: ConversionPlan, staged: Sequence[Path], console: Console) -> None:
        detail: _Plan = plan.detail
        matrix = detail.matrix

        with h5py.File(detail.matrix_path, "r") as handle, ExitStack() as stack:
            group = handle[matrix.group]
            roots = [
                stack.enter_context(open_store(path, "w", zarr_format=detail.zarr_format)).root
                for path in staged
            ]
            for root, target in zip(roots, detail.targets):
                with console.status("[cyan]Writing obs, var and the spot positions...[/]"):
                    _write_obs(root, group, detail)
                    write_var(root, target)
                    tenx._write_uns(root, matrix)
                    _write_spatial_uns(root, detail)

            with console.status("[cyan]Copying X...[/]"):
                fan_out_sparse_matrix(
                    tenx._AsCsr(group, (matrix.n_barcodes, matrix.n_features)),
                    "X",
                    [(root, None, target.keep) for root, target in zip(roots, detail.targets)],
                )


def _write_obs(root: Any, group: h5py.Group, detail: _Plan) -> None:
    """obs (the barcodes, and the positions' columns), and `obsm["spatial"]`."""
    barcodes = group["barcodes"]
    n = barcodes.shape[0]
    obs = write_obs_index(
        root, n,
        (
            [str(v) for v in decode_str_array(np.asarray(barcodes[lo : lo + STRING_STEP]))]
            for lo in range(0, n, STRING_STEP)
        ),
    )
    ensure_anndata_skeleton(root)

    pixels = ("pxl_col_in_fullres", "pxl_row_in_fullres")  # x, y
    if isinstance(detail.positions, ParquetTable):
        order = _write_parquet_positions(root, obs, detail, pixels)
    else:
        order = _write_csv_positions(root, obs, detail, pixels)
    if detail.filtered_barcodes:
        write_dense(obs, "filtered_barcodes", np.asarray(group["filtered_barcodes"][...]).astype(bool))
        order.append("filtered_barcodes")
    set_column_order(obs, order)


def _write_parquet_positions(root: Any, obs: Any, detail: _Plan, pixels: Sequence[str]) -> List[str]:
    """Visium HD's positions: ints in obs, float pixels in `obsm["spatial"]`.

    A second read of the file, of these columns only: the barcodes were
    matched in the first.
    """
    n = detail.join.n_out
    names = list(POSITION_COLUMNS[1:4])
    columns = {}
    for name in names:
        columns[name] = create_dataset(obs, name, shape=(n,), dtype=np.int64)
        spec.set_encoding(columns[name], spec.ARRAY)
    spatial = create_dataset(root["obsm"], "spatial", shape=(n, 2), dtype=np.float64)
    spec.set_encoding(spatial, spec.ARRAY)
    sinks = [Sink(dest, lambda block, name=name: block[name].astype(np.int64)) for name, dest in columns.items()]
    sinks.append(Sink(
        spatial, lambda block: np.stack([block[p].astype(np.float64) for p in pixels], axis=1)
    ))
    write_rows(detail.positions.blocks([*names, *pixels]), detail.join, sinks, root)
    return names


def _write_csv_positions(root: Any, obs: Any, detail: _Plan, pixels: Sequence[str]) -> List[str]:
    n = detail.join.n_out
    scan = detail.positions
    kinds = {c.name: scan.dtype_of(c) for c in scan.columns}
    columns, lookups = create_columns(obs, scan, n, exclude=pixels)
    dtype = np.int64 if all(kinds[p] == "int64" for p in pixels) else np.float64
    spatial = create_dataset(root["obsm"], "spatial", shape=(n, 2), dtype=dtype)
    spec.set_encoding(spatial, spec.ARRAY)

    by_name = {c.name: c for c in scan.columns}
    sinks = [
        Sink(
            dest,
            lambda block, column=by_name[name]: column_values(
                scan, column, block[column.name], lookups.get(column.name, {})
            ),
        )
        for name, dest in columns.items()
    ]
    sinks.append(Sink(spatial, lambda block: numbers(scan, pixels, block, dtype)))
    write_rows(table_blocks(scan), detail.join, sinks, root)
    return list(columns)


def _write_spatial_uns(root: Any, detail: _Plan) -> None:
    """`uns["spatial"][library_id]`, as squidpy writes it.

    A PNG decodes only whole, so each image is read as it is written and
    let go before the next: Visium HD's hires image is 6,000 pixels on its
    longest side, some 100 MB of pixels.
    """
    library = write_mapping(write_mapping(root["uns"], "spatial"), detail.library_id)
    if detail.images:
        holder = write_mapping(library, "images")
        for name, path in detail.images.items():
            write_image(holder, name, path)
    write_json_value(library, "scalefactors", detail.scalefactors)
    metadata = write_mapping(library, "metadata")
    for key in ("chemistry_description", "software_version"):
        value = detail.matrix.attrs.get(key)
        if value is not None:
            write_scalar(
                metadata, key,
                value.decode("utf-8", "replace") if isinstance(value, (bytes, np.bytes_)) else str(value),
            )
