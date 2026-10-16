"""CellBender `remove-background` output.

CellBender writes the denoised counts as a 10x v3 matrix, readable as any
other (see `tenx`), plus its latent variables:

* 0.3 and later: `/droplet_latents` (per analysed droplet: cell probability,
  size, background fraction, efficiency, the gene-expression encoding),
  `/global_latents` (the ambient profile and fitted constants) and
  `/metadata` (learning curve, analysed barcodes, settings).
* 0.2: the same kinds of value inside the matrix group itself, named
  `latent_cell_probability`, `latent_scale`, `latent_RT_efficiency`,
  `latent_gene_encoding`, `ambient_expression`, and so on. The matrix group
  is `/background_removed` when CellBender's input was Cell Ranger 2.

The AnnData uses the names CellBender's own loaders give, so code written
against them keeps working: per-droplet latents become obs columns (NaN for
droplets CellBender did not analyse) plus a `cellbender_analyzed` flag, the
encoding `obsm`, per-feature globals `var`, and everything else
`uns["cellbender"]`. With `--raw`, the Cell Ranger counts CellBender started
from are added as `layers["cellranger"]`, and the output's own as
`layers["cellbender"]`, as `load_anndata_from_input_and_output` does.

Which latent belongs to which barcode follows CellBender's own rule. In the
full `output.h5`, every input barcode is a row and `barcode_indices_for_latents`
places the analysed ones. In `output_filtered.h5` there is one latent row per
barcode -- but 0.3 and later still store every analysed droplet's index, so
the index is used only when its length matches the latents and it points
inside the file.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import h5py
import numpy as np
from rich.console import Console

from adata.conversion import tenx
from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import (
    STRING_STEP,
    Target,
    write_obs_index,
    write_var,
)
from adata.core.csr_build import CsrAppender, count_keys, scratch_group, write_runs
from adata.core.recast import Matrix
from adata.core.subset import fan_out_sparse_matrix
from adata.elements import spec
from adata.elements.read import decode_str_array
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_dense,
    write_mapping,
    write_scalar,
    write_string_array,
)
from adata.formats.mtx import EntryMap, iter_matrix_entries
from adata.storage import create_dataset, is_zarr_group, open_store

#: Datasets of a 0.2 matrix group that are the matrix, not a latent.
_MATRIX_KEYS = frozenset({
    "barcodes", "data", "indices", "indptr", "shape", "features", "genes",
    "gene_names", "genome", "filtered_barcodes",
})

_INDEX = "barcode_indices_for_latents"

#: The cell-probability column, by CellBender version.
_PROBABILITY = ("cell_probability", "latent_cell_probability")

#: Rows of obsm written per block.
_ROW_STEP = 1 << 16


def _matrix_group(handle: h5py.File) -> Optional[str]:
    if "matrix" in handle:
        return "matrix"
    if "background_removed" in handle:
        return "background_removed"
    return None


def _has_latents(handle: h5py.File) -> bool:
    if isinstance(handle.get("droplet_latents"), h5py.Group):
        return True
    name = _matrix_group(handle)
    return name is not None and any(
        key in handle[name] for key in ("latent_cell_probability", _INDEX)
    )


def sniff(path: Path) -> bool:
    if not tenx.sniff(path):
        return False
    with h5py.File(path, "r") as handle:
        return _has_latents(handle)


# ---------------------------------------------------------------------------
# the latents


@dataclass
class _Latents:
    version: str
    #: obs column -> dataset path, one value per latent row.
    per_droplet: Dict[str, str]
    #: obsm key -> dataset path, one row per latent row.
    embeddings: Dict[str, str]
    #: var column -> dataset path, one value per feature.
    per_feature: Dict[str, str]
    #: uns key -> dataset path.
    other: Dict[str, str]
    n_latent: int
    #: The file row of each latent row; None when there is one per row.
    rows: Optional[np.ndarray]
    probability: Optional[str] = None
    n_analyzed: int = 0


def _datasets(group: h5py.Group, prefix: str) -> Iterator[Tuple[str, str, h5py.Dataset]]:
    for key, item in group.items():
        if isinstance(item, h5py.Dataset):
            yield key, f"{prefix}/{key}", item


def _read_latents(handle: h5py.File, group: str, n_barcodes: int, n_features: int) -> _Latents:
    """Classify every latent dataset, and work out which row each belongs to."""
    per_droplet: Dict[str, str] = {}
    embeddings: Dict[str, str] = {}
    per_feature: Dict[str, str] = {}
    other: Dict[str, str] = {}
    lengths: Dict[str, int] = {}
    index_path: Optional[str] = None

    def place_global(key: str, path: str, item: h5py.Dataset) -> None:
        if item.ndim == 1 and item.shape[0] == n_features and item.dtype.kind in "iuf":
            per_feature[key] = path
        else:
            other[key] = path

    if isinstance(handle.get("droplet_latents"), h5py.Group):
        version = "0.3"
        for key, path, item in _datasets(handle["droplet_latents"], "droplet_latents"):
            if key == _INDEX:
                index_path = path
            elif item.ndim == 2:
                embeddings[key] = path
                lengths[key] = item.shape[0]
            elif item.ndim == 1:
                per_droplet[key] = path
                lengths[key] = item.shape[0]
            else:
                other[key] = path
        if isinstance(handle.get("global_latents"), h5py.Group):
            for key, path, item in _datasets(handle["global_latents"], "global_latents"):
                place_global(key, path, item)
        # Metadata is uns whatever its length: `features_analyzed_inds` has
        # one entry per feature when every feature was analysed, and is
        # still not a property of each feature.
        if isinstance(handle.get("metadata"), h5py.Group):
            for key, path, _ in _datasets(handle["metadata"], "metadata"):
                other[key if key not in other else f"metadata_{key}"] = path
    else:
        version = "0.2"
        for key, path, item in _datasets(handle[group], group):
            if key in _MATRIX_KEYS:
                continue
            if key == _INDEX:
                index_path = path
            elif key.startswith("latent_") and item.ndim == 2:
                embeddings[key] = path
                lengths[key] = item.shape[0]
            elif key.startswith("latent_") and item.ndim == 1:
                per_droplet[key] = path
                lengths[key] = item.shape[0]
            else:
                place_global(key, path, item)

    sizes = set(lengths.values())
    if len(sizes) > 1:
        raise ValueError(
            "CellBender's per-droplet latents differ in length: "
            + ", ".join(f"{k} {n}" for k, n in lengths.items()) + "."
        )
    n_latent = sizes.pop() if sizes else 0

    index = None if index_path is None else np.asarray(handle[index_path][...], dtype=np.int64)
    n_analyzed = int(index.size) if index is not None else n_latent
    rows: Optional[np.ndarray]
    if index is not None and index.size == n_latent and (
        n_latent == 0 or (index.min() >= 0 and index.max() < n_barcodes)
    ):
        if np.unique(index).size != index.size:
            raise ValueError(f"{_INDEX} repeats a barcode.")
        rows = index
    elif n_latent in (0, n_barcodes):
        rows = None
    else:
        raise ValueError(
            f"Cannot tell which barcode each of CellBender's {n_latent:,} latent "
            f"rows belongs to: the matrix has {n_barcodes:,} barcodes, and "
            f"{_INDEX} {'is missing' if index is None else f'has {index.size:,} entries'}."
        )

    probability = next((name for name in _PROBABILITY if name in per_droplet), None)
    return _Latents(
        version, per_droplet, embeddings, per_feature, other, n_latent, rows,
        probability, n_analyzed,
    )


def _spread(values: np.ndarray, rows: Optional[np.ndarray], n: int) -> np.ndarray:
    """Per-latent-row values as per-barcode ones: NaN where none was inferred."""
    if rows is None:
        return values.astype(np.float64, copy=False)
    full = np.full(n, np.nan)
    full[rows] = values
    return full


# ---------------------------------------------------------------------------
# --raw


@dataclass
class _Raw:
    path: Path
    group: str
    n_barcodes: int
    dtype: np.dtype
    #: For each raw barcode, its row in the output, or -1.
    to_output: np.ndarray
    #: Whether output rows follow the raw file's order.
    in_order: bool


def _barcode_blocks(
    dataset: h5py.Dataset, rows: Optional[np.ndarray] = None
) -> Iterator[np.ndarray]:
    """The barcodes, or those at `rows` (ascending), a block of the column at a time.

    Rows are picked out of contiguous reads. h5py's fancy indexing pays per
    index selected: 65,536 scattered rows read some 500 times slower than a
    slice of the whole column.
    """
    n = dataset.shape[0]
    for lo in range(0, n, STRING_STEP):
        hi = min(lo + STRING_STEP, n)
        if rows is None:
            yield decode_str_array(np.asarray(dataset[lo:hi]))
            continue
        first, last = np.searchsorted(rows, (lo, hi))
        if first < last:
            yield decode_str_array(np.asarray(dataset[lo:hi])[rows[first:last] - lo])


def _plan_raw(
    path: Path, options: Options, matrix: Any, barcodes: h5py.Dataset, out_rows: Optional[np.ndarray]
) -> _Raw:
    """Check the raw file against the output, and map its barcodes to output rows."""
    with h5py.File(path, "r") as handle:
        if not tenx.sniff(path):
            raise ValueError(f"--raw {path.name} is not a 10x HDF5 matrix.")
        raw = tenx._read_matrix(handle, options.genome)
        same = (
            raw.n_features == matrix.n_features
            and np.array_equal(raw.ids, matrix.ids)
            and np.array_equal(raw.names, matrix.names)
        )
        if same and raw.feature_types is not None and matrix.feature_types is not None:
            same = np.array_equal(raw.feature_types, matrix.feature_types)
        if not same:
            raise ValueError(
                f"--raw {path.name} does not list the same features, in the same "
                "order, as the CellBender output; it must be the matrix CellBender "
                "was run on."
            )

        n_out = barcodes.shape[0] if out_rows is None else out_rows.size
        raw_barcodes = handle[raw.group]["barcodes"]
        to_output = np.full(raw.n_barcodes, -1, dtype=np.int64)

        # The full output.h5 lists the raw file's barcodes in its own order:
        # compare block by block before building a lookup of them all.
        if out_rows is None and raw.n_barcodes == barcodes.shape[0] and all(
            np.array_equal(a, b)
            for a, b in zip(_barcode_blocks(raw_barcodes), _barcode_blocks(barcodes))
        ):
            to_output[:] = np.arange(raw.n_barcodes)
        elif n_out:
            # The output's barcodes, sorted, to look each raw block up in:
            # a numpy array, rather than a dict whose per-string overhead is
            # several times the barcode itself.
            wanted = np.concatenate(list(_barcode_blocks(barcodes, out_rows)))
            order = np.argsort(wanted, kind="stable")
            ranked = wanted[order]
            start = 0
            for block in _barcode_blocks(raw_barcodes):
                at = np.minimum(np.searchsorted(ranked, block), ranked.size - 1)
                hit = ranked[at] == block
                to_output[start : start + block.size][hit] = order[at[hit]]
                start += block.size
        found = int((to_output >= 0).sum())
        if found != n_out:
            raise ValueError(
                f"{n_out - found:,} of the output's {n_out:,} barcodes are not in "
                f"--raw {path.name}; it must be the matrix CellBender was run on."
            )
        mapped = to_output[to_output >= 0]
        return _Raw(
            path=path,
            group=raw.group,
            n_barcodes=raw.n_barcodes,
            dtype=handle[raw.group]["data"].dtype,
            to_output=to_output,
            in_order=bool(np.all(np.diff(mapped) > 0)),
        )


def _write_raw(
    raw: _Raw, roots: Sequence[Any], targets: Sequence[Target], n_out: int,
    n_features: int, compression: Optional[str],
) -> None:
    """layers["cellranger"]: the raw counts of each output's barcodes and features."""
    emap = _entry_map(n_out, n_features, targets)
    destinations = [
        [CsrAppender(
            root["layers"], "cellranger", shape=(n_out, emap.target_cols[t]),
            data_dtype=raw.dtype, compression=compression,
        )]
        for t, root in enumerate(roots)
    ]
    with h5py.File(raw.path, "r") as handle:
        group = handle[raw.group]
        source = Matrix(
            tenx._AsCsr(group, (raw.n_barcodes, n_features)), spec.CSR_MATRIX,
            (raw.n_barcodes, n_features), group["data"].dtype,
        )
        def blocks():
            for rows, cols, values in iter_matrix_entries(source, transpose=False):
                out = raw.to_output[rows]
                kept = out >= 0
                yield emap.apply(out[kept], cols[kept], values[kept])

        grouped = raw.in_order and emap.n_targets == 1
        # Entries per output row, which size the buckets. Without a feature
        # filter that is each raw row's length; with one, only a counting
        # pass knows, and an estimate from widths alone can put a target's
        # concentrated entries all in one bucket.
        counts = np.zeros(emap.n_targets * n_out, dtype=np.int64)
        if grouped:
            pass  # streamed straight through: no buckets to size
        elif emap.owner is None:
            indptr = np.asarray(group["indptr"][...], dtype=np.int64)
            hit = raw.to_output >= 0
            counts[raw.to_output[hit]] = np.diff(indptr)[hit]
        else:
            for keys, _, _ in blocks():
                count_keys(counts, keys)

        write_runs(
            blocks(),
            n_rows=n_out,
            n_targets=emap.n_targets,
            destinations=destinations,
            scratch_parent=roots[0],
            grouped=grouped,
            canonical=False,
            counts=counts,
            value_dtype=raw.dtype,
            chunk=1 << 20,
        )
    for appenders in destinations:
        for appender in appenders:
            appender.finish()


def _entry_map(n_rows: int, n_features: int, targets: Sequence[Target]) -> EntryMap:
    if len(targets) == 1 and targets[0].keep is None:
        return EntryMap(n_rows=n_rows, n_cols=n_features)
    owner = np.full(n_features, -1, dtype=np.int64)
    local = np.zeros(n_features, dtype=np.int64)
    widths = []
    for t, target in enumerate(targets):
        keep = np.arange(n_features) if target.keep is None else target.keep
        owner[keep] = t
        local[keep] = np.arange(keep.size)
        widths.append(int(keep.size))
    return EntryMap(
        n_rows=n_rows, n_cols=n_features, owner=owner, local=local,
        n_targets=len(targets), target_cols=widths,
    )


# ---------------------------------------------------------------------------
# import


@dataclass
class _Plan:
    matrix: Any
    targets: List[Target]
    latents: _Latents
    out_rows: Optional[np.ndarray]
    raw: Optional[_Raw]
    zarr_format: Optional[int]
    filtered_barcodes: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)


class CellbenderImport(Converter):
    options = tenx.TenxImport.options | frozenset({"--raw", "--min-cell-probability"})

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        with h5py.File(source, "r") as handle:
            matrix, outputs, targets, warnings = tenx.plan_matrix_import(
                handle, output, options
            )
            latents = _read_latents(
                handle, matrix.group, matrix.n_barcodes, matrix.n_features
            )

            out_rows: Optional[np.ndarray] = None
            if options.min_cell_probability is not None:
                if not 0 <= options.min_cell_probability <= 1:
                    raise ValueError("--min-cell-probability must lie between 0 and 1.")
                if latents.probability is None:
                    raise ValueError(
                        "This CellBender file records no cell probabilities to "
                        "filter on."
                    )
                values = np.asarray(handle[latents.per_droplet[latents.probability]][...])
                probability = _spread(values, latents.rows, matrix.n_barcodes)
                with np.errstate(invalid="ignore"):
                    out_rows = np.flatnonzero(probability >= options.min_cell_probability)

            raw = None
            if options.raw is not None:
                raw = _plan_raw(
                    Path(options.raw), options, matrix,
                    handle[matrix.group]["barcodes"], out_rows,
                )

        n_out = matrix.n_barcodes if out_rows is None else out_rows.size
        described = (
            f"{matrix.n_barcodes:,} droplets ({latents.n_analyzed:,} analysed) x "
            f"{matrix.n_features:,} features"
        )
        if out_rows is not None:
            described += (
                f", keeping {n_out:,} with cell probability ≥ "
                f"{options.min_cell_probability:g}"
            )
        if len(outputs) > 1:
            described += f", into {len(outputs)} outputs by feature type"
        if raw is not None:
            described += f", with the raw counts of {raw.path.name}"
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=f"CellBender {latents.version} → {dst_format}, {described}",
            warnings=warnings,
            detail=_Plan(
                matrix=matrix,
                targets=targets,
                latents=latents,
                out_rows=out_rows,
                raw=raw,
                zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
            ),
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Plan = plan.detail
        matrix, latents, out_rows = detail.matrix, detail.latents, detail.out_rows
        n_out = matrix.n_barcodes if out_rows is None else out_rows.size

        with h5py.File(plan.source, "r") as handle, ExitStack() as stack:
            group = handle[matrix.group]
            roots = [
                stack.enter_context(
                    open_store(path, "w", zarr_format=detail.zarr_format)
                ).root
                for path in staged
            ]
            per_feature = {
                key: np.asarray(handle[path][...]) for key, path in latents.per_feature.items()
            }
            if "features_analyzed_inds" in latents.other:
                # As CellBender's anndata_from_h5 adds it.
                analysed = np.zeros(matrix.n_features, dtype=bool)
                analysed[np.asarray(handle[latents.other["features_analyzed_inds"]][...])] = True
                per_feature["cellbender_analyzed"] = analysed
            with console.status("[cyan]Writing obs, var and the latents...[/]"):
                for root, target in zip(roots, detail.targets):
                    _write_obs(root, handle, group, latents, matrix.n_barcodes, out_rows)
                    columns = dict(target.columns)
                    for key, values in per_feature.items():
                        columns[key] = values if target.keep is None else values[target.keep]
                    write_var(root, replace(target, columns=columns))
                    ensure_anndata_skeleton(root)
                    _write_embeddings(root, handle, latents, matrix.n_barcodes, out_rows)
                    tenx._write_uns(root, matrix)
                    if matrix.group == "background_removed":
                        # CellBender's group, not a genome.
                        del root["uns"]["cellranger"]["genome"]
                    _write_uns(root, handle, latents)

            # One read of X, for X and, with --raw, layers["cellbender"].
            entries = [(root, out_rows, target.keep) for root, target in zip(roots, detail.targets)]
            if detail.raw is not None:
                entries += [
                    (root["layers"], out_rows, target.keep, "cellbender")
                    for root, target in zip(roots, detail.targets)
                ]
            with console.status("[cyan]Copying X...[/]"):
                fan_out_sparse_matrix(
                    tenx._AsCsr(group, (matrix.n_barcodes, matrix.n_features)), "X", entries
                )

            if detail.raw is not None:
                with console.status(f"[cyan]Reading the raw counts of {detail.raw.path.name}...[/]"):
                    _write_raw(
                        detail.raw, roots, detail.targets, n_out, matrix.n_features,
                        None if is_zarr_group(roots[0]) else "gzip",
                    )


def _write_obs(
    root: Any, handle: h5py.File, group: h5py.Group, latents: _Latents,
    n_barcodes: int, out_rows: Optional[np.ndarray],
) -> None:
    n = n_barcodes if out_rows is None else out_rows.size
    blocks = ([str(v) for v in block] for block in _barcode_blocks(group["barcodes"], out_rows))
    obs = write_obs_index(root, n, blocks)

    analyzed = np.zeros(n_barcodes, dtype=bool)
    if latents.rows is None:
        analyzed[:] = latents.n_latent == n_barcodes
    else:
        analyzed[latents.rows] = True
    columns = ["cellbender_analyzed"]
    write_dense(obs, "cellbender_analyzed", analyzed if out_rows is None else analyzed[out_rows])
    for key, path in latents.per_droplet.items():
        values = _spread(np.asarray(handle[path][...]), latents.rows, n_barcodes)
        write_dense(obs, key, values if out_rows is None else values[out_rows])
        columns.append(key)
    set_column_order(obs, columns)


def _write_embeddings(
    root: Any, handle: h5py.File, latents: _Latents, n_barcodes: int,
    out_rows: Optional[np.ndarray],
) -> None:
    """Each embedding, a block of output rows at a time, NaN where none was inferred.

    Latent rows come in CellBender's order, analysed droplets by count, and
    output rows in the file's, so one block of output draws on latents from
    anywhere. Reading those rows by fancy indexing is some 500 times slower
    than a slice, so the latents are read in slices and sorted, by the output
    block they belong to, into a scratch copy that each block then reads in
    one piece.
    """
    if not latents.embeddings:
        return
    n_out = n_barcodes if out_rows is None else out_rows.size
    position = np.arange(n_barcodes) if out_rows is None else np.full(n_barcodes, -1)
    if out_rows is not None:
        position[out_rows] = np.arange(n_out)
    # Each latent row's output row, or -1.
    target = (
        position[: latents.n_latent] if latents.rows is None else position[latents.rows]
    )
    kept = np.flatnonzero(target >= 0)
    block_of = target[kept] // _ROW_STEP
    by_block = np.argsort(block_of, kind="stable")
    n_blocks = -(-n_out // _ROW_STEP)
    starts = np.concatenate(([0], np.cumsum(np.bincount(block_of, minlength=n_blocks))))
    # The output row of each scratch row.
    placed = target[kept][by_block]

    with scratch_group(root, "__embedding_scratch__") as scratch:
        for key, path in latents.embeddings.items():
            source = handle[path]
            width = source.shape[1]
            dtype = source.dtype if source.dtype.kind == "f" else np.float64
            sorted_rows = create_dataset(
                scratch, key, shape=(kept.size, width), dtype=dtype,
                chunks=(min(max(kept.size, 1), _ROW_STEP), width),
            )
            cursor = starts[:-1].copy()
            for lo in range(0, latents.n_latent, _ROW_STEP):
                hi = min(lo + _ROW_STEP, latents.n_latent)
                here = target[lo:hi]
                have = np.flatnonzero(here >= 0)
                if have.size == 0:
                    continue
                values = np.asarray(source[lo:hi])[have]
                blocks = here[have] // _ROW_STEP
                order = np.argsort(blocks, kind="stable")
                values, blocks = values[order], blocks[order]
                edges = np.flatnonzero(np.diff(blocks)) + 1
                for run, b in zip(np.split(values, edges), blocks[np.r_[0, edges]]):
                    sorted_rows[cursor[b] : cursor[b] + len(run)] = run
                    cursor[b] += len(run)

            out = create_dataset(
                root["obsm"], key, shape=(n_out, width), dtype=dtype,
                chunks=(min(max(n_out, 1), _ROW_STEP), width),
            )
            for b in range(n_blocks):
                lo = b * _ROW_STEP
                block = np.full((min(_ROW_STEP, n_out - lo), width), np.nan, dtype=dtype)
                a, z = starts[b], starts[b + 1]
                if a < z:
                    block[placed[a:z] - lo] = sorted_rows[a:z]
                out[lo : lo + block.shape[0]] = block
            spec.set_encoding(out, spec.ARRAY)


def _write_uns(root: Any, handle: h5py.File, latents: _Latents) -> None:
    """Everything else CellBender recorded, as `uns["cellbender"]`."""
    meta = write_mapping(root["uns"], "cellbender")
    write_scalar(meta, "version", latents.version)
    for key, path in latents.other.items():
        value = np.asarray(handle[path][...])
        if value.ndim == 1 and value.size == 1:
            # CellBender wraps each scalar setting in a one-element array.
            value = value.reshape(())
        if value.ndim == 0:
            item = value.item()
            write_scalar(meta, key, item.decode("utf-8", "replace") if isinstance(item, bytes) else item)
        elif value.dtype.kind in "SOU":
            write_string_array(meta, key, decode_str_array(value))
        else:
            write_dense(meta, key, value)
