"""Parse Biosciences split-pipe output.

split-pipe writes, for the combined run and for each sample it was told of:

    <run>/all-sample/DGE_filtered/{count_matrix.mtx, all_genes.csv, cell_metadata.csv}
    <run>/all-sample/DGE_unfiltered/   (every barcode, not only cells)
    <run>/<sample>/DGE_filtered/ ...

(`all-well` rather than `all-sample` in older versions.) The matrix is
cells x genes, the other way round from 10x, so each file row is a row of X.
`all_genes.csv` holds `gene_id,gene_name,genome`, and `cell_metadata.csv`
starts with `bc_wells`, the barcode, followed by the sample, the species and
per-cell counts.

The result matches Parse's scanpy recipe -- `read_mtx` plus `pd.read_csv`
of both tables, indexed by `bc_wells` and `gene_name` -- except that
features are named by gene id unless `--var-names gene_symbols` is given,
and integer counts stay integers.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import (
    Target,
    check_import_options,
    plan_targets,
    write_var,
)
from adata.core.csr_build import CsrAppender
from adata.elements.write import ensure_anndata_skeleton
from adata.formats.mtx import EntryMap, MtxHeader, read_header, read_into, scan, value_dtype
from adata.formats.table import TableScan, read_small_table, scan_table, write_table
from adata.storage import COMPRESSIONS, open_store
from adata.util.names import unique_names
from adata.util.path import is_zarr_path

_FILTERED, _UNFILTERED = "DGE_filtered", "DGE_unfiltered"

#: The combined folder, by split-pipe version.
_COMBINED = ("all-sample", "all-well")

_INDEX = "bc_wells"


# ---------------------------------------------------------------------------
# finding the files


@dataclass
class _Dge:
    path: Path
    matrix: Path
    genes: Path
    cells: Path

    @property
    def sample(self) -> str:
        return self.path.parent.name

    @property
    def filtered(self) -> bool:
        return self.path.name != _UNFILTERED


def _dge(folder: Path) -> Optional[_Dge]:
    """The DGE folder `folder` is, or None."""
    if not folder.is_dir() or is_zarr_path(folder):
        return None
    matrix = next(
        (p for p in (folder / "count_matrix.mtx", folder / "count_matrix.mtx.gz") if p.is_file()),
        None,
    )
    genes, cells = folder / "all_genes.csv", folder / "cell_metadata.csv"
    if matrix is None or not genes.is_file() or not cells.is_file():
        return None
    return _Dge(folder, matrix, genes, cells)


def _sample(folder: Path, which: str) -> Optional[_Dge]:
    """A sample folder's DGE folder of kind `which`, or None."""
    return _dge(folder / which) if folder.is_dir() else None


def _is_sample(folder: Path) -> bool:
    return any(_sample(folder, which) is not None for which in (_FILTERED, _UNFILTERED))


def _children(path: Path) -> List[Path]:
    return sorted(p for p in path.iterdir() if p.is_dir()) if path.is_dir() else []


def sniff(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir() or is_zarr_path(path):
        return False
    # A folder of several runs is recognised too, so `locate` can say to
    # convert them one at a time rather than "unknown format".
    return (
        _dge(path) is not None
        or _is_sample(path)
        or any(
            _is_sample(child) or any(_is_sample(g) for g in _children(child))
            for child in _children(path)
        )
    )


def locate(path: Path, options: Options) -> List[_Dge]:
    """The DGE folders to convert: one, or one per sample."""
    path = Path(path)
    which = _UNFILTERED if options.unfiltered else _FILTERED
    picking = options.sample is not None or options.all_samples

    found = _dge(path)
    if found is not None:
        if options.unfiltered or picking:
            flag = "--unfiltered" if options.unfiltered else (
                "--sample" if options.sample is not None else "--all-samples"
            )
            raise ValueError(
                f"{path.name} is a DGE folder already, so {flag} has nothing to "
                "choose from; give the split-pipe output folder instead."
            )
        return [found]

    if _is_sample(path):
        if picking:
            raise ValueError(
                f"{path.name} is a single sample's folder; give the split-pipe "
                "output folder to choose samples."
            )
        return [_need(path, which)]

    samples = {child.name: child for child in _children(path) if _is_sample(child)}
    if not samples:
        runs = [child.name for child in _children(path) if any(
            _is_sample(grandchild) for grandchild in _children(child)
        )]
        if runs:
            raise ValueError(
                f"{path} holds {len(runs)} split-pipe runs ({', '.join(runs[:5])}"
                f"{', ...' if len(runs) > 5 else ''}); convert one at a time."
            )
        raise ValueError(
            f"{path} is not split-pipe output: it needs count_matrix.mtx, "
            "all_genes.csv and cell_metadata.csv, in DGE_filtered/ or "
            "DGE_unfiltered/ under each sample."
        )

    named = [name for name in samples if name not in _COMBINED]
    if options.all_samples:
        if not named:
            raise ValueError(
                f"{path.name} holds only the combined {', '.join(samples)} folder; "
                "split-pipe was run without samples."
            )
        return [_need(samples[name], which) for name in named]
    if options.sample is not None:
        if options.sample not in samples:
            raise ValueError(
                f"No sample {options.sample!r}; this run has: {', '.join(samples)}."
            )
        return [_need(samples[options.sample], which)]
    combined = next((name for name in _COMBINED if name in samples), None)
    if combined is None and len(samples) == 1:
        combined = next(iter(samples))
    if combined is None:
        raise ValueError(
            f"{path.name} holds {len(samples)} samples ({', '.join(samples)}) and "
            "no combined all-sample folder; pick one with --sample, or write "
            "each with --all-samples."
        )
    return [_need(samples[combined], which)]


def _need(sample: Path, which: str) -> _Dge:
    found = _sample(sample, which)
    if found is None:
        other = _UNFILTERED if which == _FILTERED else _FILTERED
        hint = (
            f"; it has {other}/" + (" (pass --unfiltered)" if other == _UNFILTERED else "")
            if _sample(sample, other) is not None else ""
        )
        raise ValueError(f"{sample.name} has no readable {which}/{hint}.")
    return found


# ---------------------------------------------------------------------------
# import


@dataclass
class _Item:
    dge: _Dge
    header: MtxHeader
    cells: TableScan
    rows: Optional[np.ndarray]
    target: Target
    emap: EntryMap


@dataclass
class _Import:
    items: List[_Item]
    compression: Optional[str]
    zarr_format: Optional[int]


class ParseImport(Converter):
    options = frozenset({
        "--genome", "--var-names", "--make-unique", "--compression",
        "--sample", "--all-samples", "--unfiltered",
    })

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        check_import_options(options)
        if options.sample is not None and options.all_samples:
            raise ValueError(
                "--sample picks one sample and --all-samples writes every one; "
                "pass one or the other."
            )
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )

        dges = locate(Path(source), options)
        stems = unique_names([dge.sample for dge in dges])
        items: List[_Item] = []
        outputs: List[Path] = []
        warnings: List[str] = []
        for dge in dges:
            path = output
            if options.all_samples:
                path = output.with_name(f"{output.stem}.{stems[dge.sample]}{output.suffix}")
            item, notes = _plan_item(dge, path, options)
            items.append(item)
            outputs.append(path)
            warnings += [f"{dge.sample}: {n}" if options.all_samples else n for n in notes]

        first = items[0]
        n_cells = first.cells.n_rows if first.rows is None else first.rows.size
        kept = first.target.keep
        described = (
            f"{n_cells:,} cells x "
            f"{first.header.n_cols if kept is None else kept.size:,} genes"
        )
        if options.genome is not None:
            described += f" of {options.genome}"
        if len(items) > 1:
            described = f"{len(items)} samples, one output each ({', '.join(d.sample for d in dges)})"
        else:
            described += f", {first.header.nnz:,} entries"
        kind = "filtered" if first.dge.filtered else "unfiltered"
        where = "" if len(items) > 1 else f"{first.dge.sample}, "
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=f"Parse split-pipe ({where}{kind}) → {dst_format}, {described}",
            warnings=warnings,
            detail=_Import(
                items=items,
                compression=options.compression or ("gzip" if backend == "hdf5" else None),
                zarr_format=(options.zarr_format or 3) if dst_format == "zarr" else None,
            ),
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Import = plan.detail

        # Pass 1 of every matrix, before any output exists: a file that
        # cannot be read must not leave a store holding obs and var alone.
        scans = []
        for item in detail.items:
            with console.status(f"[cyan]Checking {_label(item)}...[/]"):
                scans.append(scan(item.header, item.emap))

        for item, scanned, path in zip(detail.items, scans, staged):
            with ExitStack() as stack:
                root = stack.enter_context(
                    open_store(path, "w", zarr_format=detail.zarr_format)
                ).root
                with console.status(f"[cyan]Writing obs and var of {item.dge.sample}...[/]"):
                    write_table(root, "obs", item.cells, rows=item.rows)
                    write_var(root, item.target)
                    ensure_anndata_skeleton(root)
                appender = CsrAppender(
                    root, "X",
                    shape=(item.emap.n_rows, item.emap.target_cols[0]),
                    data_dtype=value_dtype(item.header, scanned),
                    compression=detail.compression,
                )
                with console.status(f"[cyan]Reading {_label(item)}...[/]"):
                    summed = read_into(item.header, item.emap, scanned, [[appender]], root)
                    appender.finish()
                if summed:
                    console.print(
                        f"[yellow]Warning:[/] {summed:,} duplicate entries in "
                        f"{_label(item)} were summed, as scipy does."
                    )


def _label(item: _Item) -> str:
    return f"{item.dge.sample}/{item.dge.path.name}/{item.dge.matrix.name}"


def _plan_item(dge: _Dge, output: Path, options: Options):
    """Everything about one DGE folder, checked before any output exists."""
    header = read_header(dge.matrix)
    if header.field not in ("integer", "real", "pattern"):
        raise ValueError(f"{dge.matrix.name} holds {header.field} values, which AnnData cannot.")

    columns, rows = read_small_table(dge.genes)
    for name in ("gene_id", "gene_name"):
        if name not in columns:
            raise ValueError(
                f"{dge.genes.name} has no {name!r} column; it has: {', '.join(columns)}."
            )
    if len(rows) != header.n_cols:
        raise ValueError(
            f"{dge.genes.name} lists {len(rows):,} genes, but {dge.matrix.name} "
            f"has {header.n_cols:,} columns."
        )
    field = {name: np.array([row[i] for row in rows], dtype=object) for i, name in enumerate(columns)}
    names = field["gene_name"]
    var = {"gene_ids": field["gene_id"]}
    var.update({k: v for k, v in field.items() if k not in ("gene_id", "gene_name")})

    select = None
    keep = np.ones(header.n_cols, dtype=bool)
    if options.genome is not None:
        genomes = field.get("genome")
        if genomes is None:
            raise ValueError(f"{dge.genes.name} records no genome per gene to filter on.")
        keep = genomes == options.genome
        if not keep.any():
            available = list(dict.fromkeys(genomes.tolist()))
            raise ValueError(
                f"No genes of genome {options.genome!r}; this run has: {', '.join(available)}."
            )
        select = ("species", options.genome)

    cells = scan_table(dge.cells, index=_INDEX, select=select)
    if cells.n_rows != header.n_rows:
        raise ValueError(
            f"{dge.cells.name} lists {cells.n_rows:,} cells, but {dge.matrix.name} "
            f"has {header.n_rows:,} rows."
        )
    picked = None
    if select is not None:
        picked = cells.selected
        if picked.size == 0:
            raise ValueError(
                f"No cell in {dge.cells.name} has species {options.genome!r}."
            )

    _, (target,), notes = plan_targets(
        output, names, var, None, keep, options, noun="run"
    )
    row_map = None
    if picked is not None:
        row_map = np.full(header.n_rows, -1, dtype=np.int64)
        row_map[picked] = np.arange(picked.size)
    emap = _entry_map(
        picked.size if picked is not None else header.n_rows, header.n_cols, target, row_map
    )
    return _Item(dge, header, cells, picked, target, emap), notes


def _entry_map(
    n_rows: int, n_genes: int, target: Target, row_map: Optional[np.ndarray]
) -> EntryMap:
    if target.keep is None:
        return EntryMap(n_rows=n_rows, n_cols=n_genes, row_map=row_map)
    owner = np.full(n_genes, -1, dtype=np.int64)
    local = np.zeros(n_genes, dtype=np.int64)
    owner[target.keep] = 0
    local[target.keep] = np.arange(target.keep.size)
    return EntryMap(
        n_rows=n_rows, n_cols=n_genes, owner=owner, local=local,
        target_cols=[int(target.keep.size)], row_map=row_map,
    )
