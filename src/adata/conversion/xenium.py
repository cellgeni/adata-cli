"""Xenium Onboard Analysis output: counts, cells, and Xenium's own analysis.

A Xenium output bundle holds the cells x features matrix
(`cell_feature_matrix.h5`, a 10x matrix), a table of the cells
(`cells.csv.gz`: centroid, counts per kind of probe, cell and nucleus area),
and, in `analysis/`, the clustering, UMAP and PCA Xenium Explorer shows. It
is recognised by `experiment.xenium`.

The AnnData produced:

* obs: the cell table's columns, typed as pandas reads them, indexed by
  `cell_id` (the matrix's barcodes); `graphclust`, Xenium's graph-based
  clusters, as a categorical;
* `obsm["spatial"]`: each cell's centroid, `[x, y]` in microns, as
  spatialdata-io places it; `obsm["X_umap"]` and `obsm["X_pca"]` from
  `analysis/`.

The cell table can hold a million rows. It is matched to the matrix's
barcodes while it is scanned, and Xenium writes both in the same order, so
it is then written straight through; a table in another order goes through
`core.scatter`. Boundaries, transcripts and morphology images are not read.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import h5py
import numpy as np
from rich.console import Console

from adata.conversion import tenx
from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import STRING_STEP, Target, describe_import, write_obs_index, write_var
from adata.conversion.spatial import BarcodeJoin, Join, Sink, write_rows
from adata.core.subset import fan_out_sparse_matrix
from adata.elements import spec
from adata.elements.read import decode_str_array
from adata.elements.write import create_categorical, ensure_anndata_skeleton, set_column_order
from adata.formats.table import TableScan, column_values, create_columns, drop_unused, scan_table
from adata.storage import create_dataset, is_zarr_path, open_store
from adata.util.names import natural_sorted

_CENTROIDS = ("x_centroid", "y_centroid")
_CELL_TABLES = ("cells.csv.gz", "cells.csv")

#: Xenium's analysis outputs: (path under analysis/, where it goes, value columns).
_CLUSTERS = Path("clustering/gene_expression_graphclust/clusters.csv")
_PROJECTIONS = {
    "X_umap": Path("umap/gene_expression_2_components/projection.csv"),
    "X_pca": Path("pca/gene_expression_10_components/projection.csv"),
}


def sniff(path: Path) -> bool:
    """A Xenium output bundle: `experiment.xenium` beside the cell matrix."""
    path = Path(path)
    return (
        path.is_dir()
        and not is_zarr_path(path)
        and (path / "experiment.xenium").is_file()
        and ((path / "cell_feature_matrix.h5").is_file() or (path / "cell_feature_matrix").is_dir())
    )


@dataclass
class _Table:
    """A table beside the matrix, scanned and matched to its barcodes."""

    scan: TableScan
    join: Join


@dataclass
class _Plan:
    matrix: Any
    matrix_path: Path
    targets: List[Target]
    cells: _Table
    clusters: Optional[_Table]
    projections: Dict[str, _Table] = field(default_factory=dict)
    zarr_format: Optional[int] = None


def _bundle(source: Path) -> Path:
    bundle = source.parent if source.is_file() else source
    if not (bundle / "cell_feature_matrix.h5").is_file():
        if (bundle / "cell_feature_matrix").is_dir():
            raise ValueError(
                f"{bundle} has the cell matrix as a Matrix Market folder only, not "
                "cell_feature_matrix.h5, which this reads. Convert "
                "cell_feature_matrix/ with --from 10x-mtx, without the cell table."
            )
        raise ValueError(f"{bundle} has no cell_feature_matrix.h5.")
    return bundle


def _scan(path: Path, index: str, barcodes: Any) -> _Table:
    joiner = BarcodeJoin(barcodes)
    scan = scan_table(path, index=index, on_index=joiner)
    return _Table(scan, joiner.finish())


class XeniumImport(Converter):
    options = tenx.TenxImport.options

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        source = Path(source)
        bundle = _bundle(source)
        matrix_path = bundle / "cell_feature_matrix.h5"
        cell_file = next((bundle / n for n in _CELL_TABLES if (bundle / n).is_file()), None)
        if cell_file is None:
            parquet = " It has cells.parquet only, which is not read yet." if (
                bundle / "cells.parquet"
            ).is_file() else ""
            raise ValueError(f"{bundle} has no cells.csv.gz.{parquet}")

        analysis = bundle / "analysis"
        warnings: List[str] = []
        with h5py.File(matrix_path, "r") as handle:
            if not tenx.sniff(matrix_path):
                raise ValueError(f"{matrix_path.name} is not a 10x HDF5 matrix.")
            matrix, outputs, targets, warnings = tenx.plan_matrix_import(handle, output, options)
            barcodes = handle[matrix.group]["barcodes"]
            cells = _scan(cell_file, "cell_id", barcodes)
            clusters = None
            projections: Dict[str, _Table] = {}
            if analysis.is_dir():
                if (analysis / _CLUSTERS).is_file():
                    clusters = _scan(analysis / _CLUSTERS, "Barcode", barcodes)
                for key, path in _PROJECTIONS.items():
                    if (analysis / path).is_file():
                        projections[key] = _scan(analysis / path, "Barcode", barcodes)

        missing = [c for c in _CENTROIDS if c not in cells.scan.header]
        if missing:
            raise ValueError(f"{cell_file.name} has no {', '.join(missing)} column.")
        cells.join.check(cell_file.name)
        if cells.join.n_extra:
            warnings.append(
                f"{cells.join.n_extra:,} cells of {cell_file.name} are not in the "
                "matrix; they are left out."
            )

        found = []
        for label, table in [("clusters", clusters), *projections.items()]:
            if table is None:
                continue
            found.append(label)
            table.join.check(f"analysis {label}", allow_missing=True)
            if table.join.n_missing:
                warnings.append(
                    f"{table.join.n_missing:,} cells have no {label} in analysis/; "
                    "they are left empty."
                )
        if not analysis.is_dir():
            if (bundle / "analysis.tar.gz").is_file():
                warnings.append(
                    "Xenium's analysis is still packed in analysis.tar.gz: unpack it "
                    "beside it to bring its clusters, UMAP and PCA in."
                )
            else:
                warnings.append("No analysis/ folder: no clusters, UMAP or PCA to bring in.")

        described = describe_import(matrix.n_barcodes, matrix.n_features, matrix.feature_types, targets)
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=(
                f"Xenium → {dst_format}, {described}, cells from {cell_file.name}"
                + (f", analysis: {', '.join(found)}" if found else "")
            ),
            warnings=warnings,
            detail=_Plan(
                matrix=matrix,
                matrix_path=matrix_path,
                targets=targets,
                cells=cells,
                clusters=clusters,
                projections=projections,
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
                with console.status("[cyan]Writing obs, var and the cell table...[/]"):
                    obs = _write_obs_index(root, group)
                    ensure_anndata_skeleton(root)
                    columns = _write_cells(root, obs, detail.cells)
                    if detail.clusters is not None:
                        _write_clusters(root, obs, detail.clusters)
                        columns.append("graphclust")
                    set_column_order(obs, columns)
                    for key, table in detail.projections.items():
                        _write_projection(root, key, table)
                    write_var(root, target)
                    tenx._write_uns(root, matrix)

            with console.status("[cyan]Copying X...[/]"):
                fan_out_sparse_matrix(
                    tenx._AsCsr(group, (matrix.n_barcodes, matrix.n_features)),
                    "X",
                    [(root, None, target.keep) for root, target in zip(roots, detail.targets)],
                )


def _write_obs_index(root: Any, group: h5py.Group) -> Any:
    barcodes = group["barcodes"]
    n = barcodes.shape[0]
    return write_obs_index(
        root, n,
        (
            [str(v) for v in decode_str_array(np.asarray(barcodes[lo : lo + STRING_STEP]))]
            for lo in range(0, n, STRING_STEP)
        ),
    )


def _write_cells(root: Any, obs: Any, table: _Table) -> List[str]:
    """The cell table's columns into obs, its centroids into `obsm["spatial"]`."""
    scan, join = table.scan, table.join
    n = join.n_out
    columns, lookups = create_columns(obs, scan, n, exclude=_CENTROIDS)
    spatial = create_dataset(root["obsm"], "spatial", shape=(n, 2), dtype=np.float64)
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
    sinks.append(Sink(
        spatial,
        lambda block: np.stack(
            [np.asarray(block[c], dtype=np.float64) for c in _CENTROIDS], axis=1
        ),
    ))
    write_rows(scan, join, sinks, root)

    if join.n_extra:
        # Categories only the left-out cells used go, as anndata's
        # subsetting drops them.
        for name in lookups:
            used = _used_codes(columns[name], len(lookups[name]), n)
            if not used.all():
                drop_unused(obs, name, used, n, block_rows=STRING_STEP)
    return list(columns)


def _used_codes(codes: Any, n_categories: int, n: int) -> np.ndarray:
    used = np.zeros(n_categories, dtype=bool)
    for lo in range(0, n, STRING_STEP):
        block = np.asarray(codes[lo : lo + STRING_STEP], dtype=np.int64)
        used[block[block >= 0]] = True
    return used


def _write_clusters(root: Any, obs: Any, table: _Table) -> None:
    """Xenium's graph-based clusters, `obs["graphclust"]`, a categorical."""
    scan, join = table.scan, table.join
    column = next((c for c in scan.columns if c.name == "Cluster"), None)
    if column is None:
        raise ValueError(f"{scan.path.name} has no Cluster column.")
    categories = natural_sorted(column.seen or set())
    lookup = {c: i for i, c in enumerate(categories)}
    codes = create_categorical(obs, "graphclust", categories, join.n_out)
    write_rows(
        scan, join,
        [Sink(codes, lambda block: np.array([lookup.get(v, -1) for v in block["Cluster"]], dtype=np.int64), -1)],
        root,
    )


def _write_projection(root: Any, key: str, table: _Table) -> None:
    """An analysis projection (UMAP, PCA) into obsm, NaN for a cell it lacks."""
    scan, join = table.scan, table.join
    names = [c.name for c in scan.columns]
    out = create_dataset(root["obsm"], key, shape=(join.n_out, len(names)), dtype=np.float64)
    spec.set_encoding(out, spec.ARRAY)
    write_rows(
        scan, join,
        [Sink(out, lambda block: np.stack([np.asarray(block[c], dtype=np.float64) for c in names], axis=1), np.nan)],
        root,
    )
