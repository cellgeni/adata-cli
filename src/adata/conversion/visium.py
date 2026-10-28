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
barcodes as they are scanned. Space Ranger's own `analysis/` beside the
matrix, when there is one, adds its graph-based clusters to obs
(`graphclust`) and its UMAP and PCA to obsm, as for Xenium.

Visium HD is read too, one bin size at a time (`--bin-size`, 8 µm by
default): each `binned_outputs/square_NNNum/` folder is laid out as above,
with its positions as Parquet, which pyarrow reads (the `spatial` extra).
Space Ranger writes no `analysis/` for the 2 µm bins.

Or as cells (`--cells`, or the folder given itself): Space Ranger 4
segments the tissue into cells in `segmented_outputs/`, with a cells x
features matrix, each cell's outline and its nucleus's as GeoJSON polygons
in full-resolution pixels, the images and scale factors, and its own
`analysis/`. Each cell's centroid goes in `obsm["spatial"]`, where squidpy
plots spots, its area and its nucleus's in obs (`cell_area`,
`nucleus_area`, in µm² as Xenium gives them), and the clusters, UMAP and
PCA as for Xenium. The outlines themselves are not kept.
"""

from __future__ import annotations

import json
from contextlib import ExitStack
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import h5py
import numpy as np
from rich.console import Console

from adata.conversion import tenx
from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import STRING_STEP, Target, describe_import, write_obs_index, write_var
from adata.conversion.spatial import (
    Analysis,
    BarcodeJoin,
    Join,
    Sink,
    check_analysis,
    numbers,
    require_numeric,
    scan_analysis,
    table_blocks,
    write_analysis,
    write_image,
    write_rows,
)
from adata.core.subset import fan_out_sparse_matrix
from adata.elements.read import decode_str_array
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_dense,
    write_mapping,
    write_scalar,
)
from adata.formats.geojson import polygon_stats
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
#: Space Ranger 4's segmented cells: the folder, its matrices and polygons.
_SEGMENTED = "segmented_outputs"
_CELL_FILTERED, _CELL_RAW = "filtered_feature_cell_matrix", "raw_feature_cell_matrix"
_CELL_POLYGONS, _NUCLEUS_POLYGONS = "cell_segmentations.geojson", "nucleus_segmentations.geojson"
_CELL_BARCODE = re.compile(r"cellid_(\d+)-\d+")
#: Polygons' measures handed to the writer at a time.
_POLYGON_STEP = 1 << 16


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


def _is_segmented(folder: Path) -> bool:
    """A `segmented_outputs/` folder: cell polygons beside a cell matrix."""
    return (
        (folder / _CELL_POLYGONS).is_file()
        and (folder / "spatial" / "scalefactors_json.json").is_file()
        and any(
            _matrix_files(folder, name) or (folder / name).is_dir()
            for name in (_CELL_FILTERED, _CELL_RAW)
        )
    )


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
        if _is_hd(folder) or _is_segmented(folder):
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
    a Visium HD one included, or one of its bin folders, or its segmented
    cells."""
    path = Path(path)
    return path.is_dir() and not is_zarr_path(path) and _locate(path) is not None


@dataclass
class _Plan:
    matrix: Any
    matrix_path: Path
    targets: List[Target]
    library_id: str
    scalefactors: Dict[str, Any]
    images: Dict[str, Path]
    filtered_barcodes: bool
    zarr_format: Optional[int]
    #: The CSV positions' scan, or the Parquet ones' footer; None for cells.
    positions: Optional[Union[TableScan, ParquetTable]] = None
    join: Optional[Join] = None
    bin_um: Optional[int] = None
    #: Segmented cells: the cells' polygons, their nuclei's, and analysis/.
    cells: Optional["_Polygons"] = None
    nuclei: Optional["_Polygons"] = None
    analysis: Analysis = field(default_factory=Analysis)


@dataclass
class _Polygons:
    """A GeoJSON's polygons, measured, in file order, matched to the matrix."""

    path: Path
    join: Join
    #: In µm².
    area: np.ndarray
    #: Centroids, [x, y] in full-resolution pixels; None for nuclei.
    spatial: Optional[np.ndarray] = None

    def blocks(self) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        for lo in range(0, self.area.size, _POLYGON_STEP):
            hi = min(lo + _POLYGON_STEP, self.area.size)
            block = {"area": self.area[lo:hi]}
            if self.spatial is not None:
                block["spatial"] = self.spatial[lo:hi]
            yield hi - lo, block


@dataclass
class _Bundle:
    folder: Path
    matrix: Path
    #: Space Ranger 4's segmented cells, rather than spots or bins.
    cells: bool = False
    #: Bins were read from an output that has segmented cells too.
    cells_beside: bool = False


def _bundle(source: Path, options: Options) -> _Bundle:
    """The folder read, and its matrix file, for this input."""
    if source.is_file():
        cells = source.name.endswith("_cell_matrix.h5") or (source.parent / _CELL_POLYGONS).is_file()
        # --cells only says what a cell matrix is already.
        for flag in ("--unfiltered", "--bin-size") + (() if cells else ("--cells",)):
            if flag in options.given:
                raise ValueError(
                    f"{flag} picks among the matrices of a Space Ranger output "
                    f"folder; {source.name} is a matrix already. Give the folder."
                )
        return _Bundle(source.parent, source, cells=cells)
    outs = _locate(source)
    if outs is None:
        raise ValueError(
            f"{source} is not a Space Ranger output folder: it needs spatial/ "
            "(tissue positions and scalefactors_json.json) beside a "
            "*_feature_bc_matrix.h5."
        )
    if options.cells and options.bin_size is not None:
        raise ValueError("--cells reads the segmented cells and --bin-size a size of bins; give one.")
    cells = _is_segmented(outs)
    if options.cells and not cells:
        if _is_hd(outs) and _is_segmented(outs / _SEGMENTED):
            outs, cells = outs / _SEGMENTED, True
        else:
            raise ValueError(
                f"{outs} has no segmented cells: --cells reads the "
                f"{_SEGMENTED}/ folder Space Ranger 4 writes for Visium HD."
            )
    cells_beside = False
    if cells:
        if options.bin_size is not None:
            raise ValueError(f"--bin-size picks among a Visium HD output's bins; {outs.name} holds cells.")
    elif _is_hd(outs):
        cells_beside = _is_segmented(outs / _SEGMENTED)
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
    if cells:
        name = _CELL_RAW if options.unfiltered else _CELL_FILTERED
    else:
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
    return _Bundle(outs, found[0], cells=cells, cells_beside=cells_beside)


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
        {"--unfiltered", "--library-id", "--no-images", "--bin-size", "--cells"}
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
        bundle = _bundle(source, options)
        if bundle.cells:
            return self._plan_cells(source, bundle, output, src_format, dst_format, options)
        outs, matrix_path = bundle.folder, bundle.matrix
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
            analysis = scan_analysis(outs / "analysis", group["barcodes"])
        join.check(f"spatial/{position_file.name}")
        check_analysis(analysis, warnings, "bins" if position_file.suffix == ".parquet" else "spots")

        scalefactors = json.loads(scalefactor_file.read_text())
        bin_um = _bin_um(scalefactors) if position_file.suffix == ".parquet" else None
        images = _images(spatial, options, warnings)

        found = analysis.found()
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
                + (f", analysis: {', '.join(found)}" if found else "")
                + (f", images {', '.join(images)}" if images else "")
                + (" (segmented cells too: --cells)" if bundle.cells_beside else "")
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
                analysis=analysis,
            ),
        )

    def _plan_cells(
        self,
        source: Path,
        bundle: _Bundle,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        """Space Ranger 4's segmented cells, measured and matched to the matrix
        here, so that a refusal comes before anything is written."""
        folder, matrix_path = bundle.folder, bundle.matrix
        spatial = folder / "spatial"
        scalefactor_file = spatial / "scalefactors_json.json"
        lacking = [
            what for what, present in (
                (_CELL_POLYGONS, (folder / _CELL_POLYGONS).is_file()),
                ("spatial/scalefactors_json.json", scalefactor_file.is_file()),
            ) if not present
        ]
        if lacking:
            raise ValueError(f"{folder} has no {' and no '.join(lacking)} beside the matrix.")
        scalefactors = json.loads(scalefactor_file.read_text())
        microns_per_pixel = scalefactors.get("microns_per_pixel")
        if not isinstance(microns_per_pixel, (int, float)) or isinstance(microns_per_pixel, bool) or microns_per_pixel <= 0:
            raise ValueError(
                "spatial/scalefactors_json.json has no microns_per_pixel, which "
                "gives the cells' areas in µm²."
            )

        with h5py.File(matrix_path, "r") as handle:
            if not tenx.sniff(matrix_path):
                raise ValueError(f"{matrix_path.name} is not a 10x HDF5 matrix.")
            matrix, outputs, targets, warnings = tenx.plan_matrix_import(handle, output, options)
            library_id = _library_id(matrix.attrs, folder, options)
            group = handle[matrix.group]
            ids = _cell_ids(group["barcodes"], matrix_path)
            cells = _measure(folder / _CELL_POLYGONS, ids, microns_per_pixel, centroids=True)
            nuclei = None
            if (folder / _NUCLEUS_POLYGONS).is_file():
                nuclei = _measure(folder / _NUCLEUS_POLYGONS, ids, microns_per_pixel, centroids=False)
            del ids
            analysis = scan_analysis(folder / "analysis", group["barcodes"])
            filtered_barcodes = "filtered_barcodes" in group

        cells.join.check(f"{_CELL_POLYGONS} (by cell_id)")
        if nuclei is None:
            warnings.append(f"No {_NUCLEUS_POLYGONS}; obs will have no nucleus_area.")
        else:
            nuclei.join.check(f"{_NUCLEUS_POLYGONS} (by cell_id)", allow_missing=True)
            if nuclei.join.n_missing:
                warnings.append(
                    f"{nuclei.join.n_missing:,} cells have no nucleus in "
                    f"{_NUCLEUS_POLYGONS}; their nucleus_area is left empty."
                )
        check_analysis(analysis, warnings)
        images = _images(spatial, options, warnings)

        found = analysis.found()
        described = describe_import(matrix.n_barcodes, matrix.n_features, matrix.feature_types, targets)
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=(
                f"Space Ranger (Visium HD, segmented cells) → {dst_format}, {described}, "
                f"centroids and areas from {_CELL_POLYGONS}, library {library_id!r}"
                + (f", analysis: {', '.join(found)}" if found else "")
                + (f", images {', '.join(images)}" if images else "")
            ),
            warnings=warnings,
            detail=_Plan(
                matrix=matrix,
                matrix_path=matrix_path,
                targets=targets,
                library_id=library_id,
                scalefactors=scalefactors,
                images=images,
                filtered_barcodes=filtered_barcodes,
                zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
                cells=cells,
                nuclei=nuclei,
                analysis=analysis,
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
                with console.status(
                    f"[cyan]Writing obs, var and the {'cells' if detail.cells else 'spot positions'}...[/]"
                ):
                    if detail.cells is not None:
                        _write_cell_obs(root, group, detail)
                    else:
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


def _images(spatial: Path, options: Options, warnings: List[str]) -> Dict[str, Path]:
    images: Dict[str, Path] = {}
    if not options.no_images:
        for name in _IMAGES:
            path = spatial / f"tissue_{name}_image.png"
            if path.is_file():
                images[name] = path
            else:
                warnings.append(f"No spatial/{path.name}; uns['spatial'] will have no {name} image.")
    return images


def _cell_ids(barcodes: Any, path: Path) -> np.ndarray:
    """The integer id in each of the matrix's `cellid_<id>-<n>` barcodes,
    which is what the polygons are keyed by."""
    n = barcodes.shape[0]
    ids = np.empty(n, dtype=np.int64)
    for lo in range(0, n, STRING_STEP):
        for i, barcode in enumerate(decode_str_array(np.asarray(barcodes[lo : lo + STRING_STEP])), lo):
            match = _CELL_BARCODE.fullmatch(str(barcode))
            if match is None:
                raise ValueError(
                    f"{path.name}: barcode {str(barcode)!r} does not name a segmented "
                    "cell, as Space Ranger's cellid_000000001-1 does."
                )
            ids[i] = int(match.group(1))
    return ids


def _measure(path: Path, ids: np.ndarray, microns_per_pixel: float, *, centroids: bool) -> _Polygons:
    """Each polygon's area, and centroid if asked, in one read of the file,
    matched to the matrix's cells by id as they are read. What is kept is a
    few numbers a polygon, as a spot's position is: the file is some
    hundreds of bytes a polygon, and its outlines are not kept."""
    joiner = BarcodeJoin(ids)
    points, areas = [], []
    for block_ids, x, y, area in polygon_stats(path):
        joiner(block_ids)
        if centroids:
            points.append(np.stack([x, y], axis=1))
        areas.append(area * microns_per_pixel ** 2)
    join = joiner.finish()
    return _Polygons(
        path, join,
        area=np.concatenate(areas) if areas else np.zeros(0),
        spatial=(np.concatenate(points) if points else np.zeros((0, 2))) if centroids else None,
    )


def _write_index(root: Any, group: h5py.Group) -> Any:
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
    return obs


def _write_cell_obs(root: Any, group: h5py.Group, detail: _Plan) -> None:
    """obs (the barcodes, the areas, the clusters), `obsm["spatial"]` (the
    cells' centroids) and the analysis projections."""
    obs = _write_index(root, group)
    n = detail.cells.join.n_out
    spatial = create_dataset(root["obsm"], "spatial", shape=(n, 2), dtype=np.float64)
    spec.set_encoding(spatial, spec.ARRAY)
    area = create_dataset(obs, "cell_area", shape=(n,), dtype=np.float64)
    spec.set_encoding(area, spec.ARRAY)
    write_rows(
        detail.cells.blocks(), detail.cells.join,
        [
            Sink(spatial, lambda block: block["spatial"]),
            Sink(area, lambda block: block["area"]),
        ],
        root,
    )
    order = ["cell_area"]
    if detail.nuclei is not None:
        nucleus = create_dataset(obs, "nucleus_area", shape=(n,), dtype=np.float64)
        spec.set_encoding(nucleus, spec.ARRAY)
        write_rows(
            detail.nuclei.blocks(), detail.nuclei.join,
            [Sink(nucleus, lambda block: block["area"], np.nan)],
            root,
        )
        order.append("nucleus_area")
    order += write_analysis(root, obs, detail.analysis)
    if detail.filtered_barcodes:
        write_dense(obs, "filtered_barcodes", np.asarray(group["filtered_barcodes"][...]).astype(bool))
        order.append("filtered_barcodes")
    set_column_order(obs, order)


def _write_obs(root: Any, group: h5py.Group, detail: _Plan) -> None:
    """obs (the barcodes, the positions' columns, the clusters),
    `obsm["spatial"]` and the analysis projections."""
    obs = _write_index(root, group)

    pixels = ("pxl_col_in_fullres", "pxl_row_in_fullres")  # x, y
    if isinstance(detail.positions, ParquetTable):
        order = _write_parquet_positions(root, obs, detail, pixels)
    else:
        order = _write_csv_positions(root, obs, detail, pixels)
    order += write_analysis(root, obs, detail.analysis)
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
