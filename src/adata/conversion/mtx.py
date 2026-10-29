"""10x Genomics and STARsolo Matrix Market folders.

A folder holding `matrix.mtx`, `barcodes.tsv` and `features.tsv` -- each
plain or gzipped -- in one of the shapes these tools write:

* Cell Ranger 3 and later (`filtered_feature_bc_matrix/`): features.tsv with
  id, name and feature type.
* Cell Ranger 2 (`filtered_gene_bc_matrices/<genome>/`): `genes.tsv` with id
  and name, one subfolder per genome.
* STARsolo (`Solo.out/Gene/filtered/`, `GeneFull/`, ...): the same files,
  uncompressed, with two or three feature columns.
* STARsolo velocyto (`Solo.out/Velocyto/raw/`): `spliced.mtx`,
  `unspliced.mtx` and `ambiguous.mtx` over one set of barcodes and features,
  read into layers of the same names, with X a copy of spliced -- the layout
  scvelo expects.
* Any of these with every file named under one prefix, as GEO serves them
  (`GSM123_matrix.mtx.gz`, `GSM123_barcodes.tsv.gz`, ...).

The matrix is stored genes x cells, so each file column is read as a row of
X. With `--var-names gene_symbols`, the result has the names and columns
`scanpy.read_10x_mtx(gex_only=False)` gives, except that integer counts stay
integers -- as in the 10x HDF5 import -- where scanpy gives float32; by
default features are named by gene id instead.

Export writes a Cell Ranger 3 folder: gzipped files, three feature columns,
entries in barcode order.
"""

from __future__ import annotations

import json
import re
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rich.console import Console

from adata.conversion.base import ConversionPlan, Converter, Options
from adata.conversion.features import (
    STRING_STEP,
    Target,
    cellranger_meta,
    check_import_options,
    describe_import,
    plan_matrix_source,
    plan_targets,
    resolve,
    write_obs_index,
    write_var,
)
from adata.core.csr_build import CsrAppender
from adata.core.recast import describe
from adata.elements.read import element_len, read_str_all, read_str_chunk, resolve_index
from adata.elements.write import (
    ensure_anndata_skeleton,
    set_column_order,
    write_mapping,
    write_scalar,
)
from adata.formats.mtx import (
    EntryMap,
    MtxHeader,
    check_sparse,
    count_entries,
    iter_matrix_entries,
    mtx_field,
    read_header,
    read_into,
    scan,
    value_dtype,
    write_entries,
    write_header,
)
from adata.formats.textio import (
    count_lines,
    create_binary,
    iter_first_column,
    read_tsv,
)
from adata.storage import COMPRESSIONS, GZIP_LEVEL, is_dataset, open_store
from adata.util.path import is_zarr_path

#: The velocyto matrices STARsolo writes, in the order they become layers.
VELOCYTO = ("spliced", "unspliced", "ambiguous")

#: What `--compression` may be for an exported folder.
EXPORT_COMPRESSIONS = ("gzip", "none")

_METADATA = "%metadata_json:"


# ---------------------------------------------------------------------------
# finding the files


def _file(folder: Path, name: str) -> Optional[Path]:
    """`name` or `name.gz` in `folder`, whichever exists."""
    for candidate in (folder / name, folder / f"{name}.gz"):
        if candidate.is_file():
            return candidate
    return None


@dataclass
class _Folder:
    path: Path
    prefix: str
    matrices: Dict[str, Path]
    barcodes: Path
    features: Path
    legacy: bool
    genome: Optional[str] = None
    ignored: List[str] = field(default_factory=list)

    @property
    def velocyto(self) -> bool:
        return "spliced" in self.matrices


def _prefixes(folder: Path) -> List[str]:
    """The name prefixes of every barcodes file in `folder` ("" for none)."""
    found = []
    for path in folder.iterdir():
        for suffix in ("barcodes.tsv", "barcodes.tsv.gz"):
            if path.name.endswith(suffix) and path.is_file():
                found.append(path.name[: -len(suffix)])
    return sorted(set(found))


def _read_folder(folder: Path) -> Optional[_Folder]:
    """The matrix folder `folder` is, or None if it is not one.

    Refuses a folder that is one in more than one way -- two samples under
    different prefixes, or a matrix and velocyto matrices side by side --
    rather than guessing which was meant.
    """
    if not folder.is_dir() or is_zarr_path(folder):
        return None
    candidates = []
    for prefix in _prefixes(folder):
        barcodes = _file(folder, f"{prefix}barcodes.tsv")
        features = _file(folder, f"{prefix}features.tsv")
        legacy = features is None
        if legacy:
            features = _file(folder, f"{prefix}genes.tsv")
        matrices = {
            name: path
            for name in ("matrix", *VELOCYTO)
            if (path := _file(folder, f"{prefix}{name}.mtx")) is not None
        }
        if barcodes is None or features is None or not matrices:
            continue
        candidates.append(_Folder(folder, prefix, matrices, barcodes, features, legacy))

    if not candidates:
        return None
    if len(candidates) > 1:
        names = ", ".join(repr(c.prefix or "(none)") for c in candidates)
        raise ValueError(
            f"{folder} holds the files of {len(candidates)} matrices, under the "
            f"prefixes {names}. Move each into a folder of its own."
        )
    found = candidates[0]
    if "matrix" in found.matrices and found.velocyto:
        raise ValueError(
            f"{folder} holds both matrix.mtx and velocyto's spliced.mtx; "
            "convert them from separate folders."
        )
    if not found.velocyto and set(found.matrices) - {"matrix"}:
        raise ValueError(
            f"{folder} holds {', '.join(sorted(found.matrices))}.mtx but no "
            "spliced.mtx, which a velocyto folder needs."
        )
    used = {p.name for p in found.matrices.values()}
    found.ignored = sorted(
        p.name for p in folder.iterdir()
        if p.name.startswith(found.prefix)
        and (p.name.endswith(".mtx") or p.name.endswith(".mtx.gz"))
        and p.name not in used
    )
    return found


#: A run's matrix folder by what it holds, not by genome: STARsolo's
#: `filtered/` and `raw/`, Cell Ranger's `filtered_feature_bc_matrix/` and so on.
_RUN_FOLDER = re.compile(r"(filtered|raw)(_\w+_bc_matri(x|ces))?")


def _subfolders(path: Path) -> Tuple[Dict[str, _Folder], List[str]]:
    """The matrix folders in `path`: Cell Ranger 2's, one per genome, by
    genome; and the names of those that are a run's filtered or raw matrix,
    which are no genome."""
    genomes, runs = {}, []
    for child in sorted(path.iterdir()) if path.is_dir() else []:
        folder = _read_folder(child)
        if folder is None:
            continue
        if _RUN_FOLDER.fullmatch(child.name):
            runs.append(child.name)
        else:
            folder.genome = child.name
            genomes[child.name] = folder
    return genomes, runs


def sniff(path: Path) -> bool:
    path = Path(path)
    if not path.is_dir() or is_zarr_path(path):
        return False
    try:
        return _read_folder(path) is not None or any(_subfolders(path))
    except ValueError:
        # An ambiguous folder is still a matrix folder; `locate` says why.
        return True


def locate(path: Path, genome: Optional[str]) -> _Folder:
    folder = _read_folder(path)
    if folder is not None:
        if genome is not None:
            raise ValueError(
                "This folder records no genome per feature to filter on."
            )
        return folder

    genomes, runs = _subfolders(path)
    if not genomes and runs:
        raise ValueError(
            f"{path.name} holds {' and '.join(f'{r}/' for r in runs)}: a run's "
            f"{'matrices' if len(runs) > 1 else 'matrix'}, not one folder per "
            f"genome. Give the folder to convert, such as {path.name}/{runs[0]}."
        )
    if not genomes:
        raise ValueError(
            f"{path} is not a 10x or STARsolo matrix folder: it needs "
            "matrix.mtx, barcodes.tsv and features.tsv (or genes.tsv), each "
            "plain or gzipped."
        )
    if genome is not None:
        if genome not in genomes:
            raise ValueError(
                f"No features of genome {genome!r}; this folder has: "
                f"{', '.join(genomes)}."
            )
        return genomes[genome]
    if len(genomes) > 1:
        raise ValueError(
            f"This folder holds {len(genomes)} genomes ({', '.join(genomes)}); "
            "pick one with --genome."
        )
    return next(iter(genomes.values()))


def _metadata(header: MtxHeader) -> Dict[str, Any]:
    """Cell Ranger's `%metadata_json` comment, if the matrix has one."""
    for comment in header.comments:
        if comment.startswith(_METADATA):
            try:
                value = json.loads(comment[len(_METADATA):])
            except ValueError:
                return {}
            return value if isinstance(value, dict) else {}
    return {}


# ---------------------------------------------------------------------------
# import


@dataclass
class _Import:
    folder: _Folder
    headers: Dict[str, MtxHeader]
    n_barcodes: int
    targets: List[Target]
    emap: EntryMap
    compression: Optional[str]
    zarr_format: Optional[int]
    metadata: Dict[str, Any]


class MtxImport(Converter):
    options = frozenset(
        {"--genome", "--var-names", "--make-unique", "--feature-type",
         "--split-feature-types", "--compression"}
    )

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        check_import_options(options)
        backend = "zarr" if dst_format == "zarr" else "hdf5"
        if options.compression is not None and options.compression not in COMPRESSIONS[backend]:
            raise ValueError(
                f"--compression {options.compression} is not available for "
                f"{dst_format}; use one of: {', '.join(COMPRESSIONS[backend])}."
            )

        folder = locate(Path(source), options.genome)
        headers = {name: read_header(path) for name, path in folder.matrices.items()}
        shapes = {(h.n_rows, h.n_cols) for h in headers.values()}
        if len(shapes) > 1:
            raise ValueError(
                "The velocyto matrices differ in shape: "
                + ", ".join(f"{n} {h.n_rows} x {h.n_cols}" for n, h in headers.items())
                + "."
            )
        (n_features, n_barcodes), = shapes

        rows = read_tsv(folder.features)
        widths = {len(row) for row in rows}
        if not rows or min(widths) < 2 or len(widths) > 1:
            raise ValueError(
                f"{folder.features.name} should have the same 2 or more "
                "tab-separated columns (id, name, and perhaps type) on every line."
            )
        if len(rows) != n_features:
            raise ValueError(
                f"{folder.features.name} lists {len(rows):,} features, but the "
                f"matrix has {n_features:,} rows."
            )
        barcodes = count_lines(folder.barcodes)
        if barcodes != n_barcodes:
            raise ValueError(
                f"{folder.barcodes.name} lists {barcodes:,} barcodes, but the "
                f"matrix has {n_barcodes:,} columns."
            )

        width = widths.pop()
        ids = np.array([row[0] for row in rows], dtype=object)
        names = np.array([row[1] for row in rows], dtype=object)
        columns: Dict[str, np.ndarray] = {"gene_ids": ids}
        types: Optional[np.ndarray] = None
        if width >= 3 and not folder.legacy:
            types = np.array([row[2] for row in rows], dtype=object)
            columns["feature_types"] = types
        if types is None and (options.feature_type is not None or options.split_feature_types):
            raise ValueError(
                f"{folder.features.name} records no feature types; "
                "--feature-type and --split-feature-types need a third column."
            )

        keep = np.ones(n_features, dtype=bool)
        outputs, targets, warnings = plan_targets(
            output, names, columns, types, keep, options, noun="folder"
        )
        if width > 3:
            warnings.append(
                f"{folder.features.name} has {width} columns; only the first "
                "three (id, name, type) are read, as scanpy reads them."
            )
        if folder.ignored:
            warnings.append(
                f"Not read: {', '.join(folder.ignored)}. Only "
                f"{', '.join(sorted(p.name for p in folder.matrices.values()))} "
                "belong to this layout."
            )

        emap = self._entry_map(n_barcodes, n_features, targets)
        zarr_format = (options.zarr_format or 3) if dst_format == "zarr" else None
        compression = options.compression or ("gzip" if backend == "hdf5" else None)
        detail = _Import(
            folder=folder,
            headers=headers,
            n_barcodes=n_barcodes,
            targets=targets,
            emap=emap,
            compression=compression,
            zarr_format=zarr_format,
            metadata=_metadata(next(iter(headers.values()))),
        )

        kind = (
            "STARsolo velocyto" if folder.velocyto
            else "Cell Ranger 2" if folder.legacy
            else "10x v3" if types is not None
            else "STARsolo"
        )
        entries = sum(h.nnz for h in headers.values())
        described = describe_import(n_barcodes, n_features, types, targets)
        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=outputs,
            summary=(
                f"10x mtx ({kind}) → {dst_format}, {described}, "
                f"{entries:,} entries"
            ),
            warnings=warnings,
            detail=detail,
        )

    @staticmethod
    def _entry_map(n_barcodes: int, n_features: int, targets: List[Target]) -> EntryMap:
        if len(targets) == 1 and targets[0].keep is None:
            return EntryMap(n_rows=n_barcodes, n_cols=n_features, transpose=True)
        owner = np.full(n_features, -1, dtype=np.int64)
        local = np.zeros(n_features, dtype=np.int64)
        widths = []
        for t, target in enumerate(targets):
            keep = np.arange(n_features) if target.keep is None else target.keep
            owner[keep] = t
            local[keep] = np.arange(keep.size)
            widths.append(int(keep.size))
        return EntryMap(
            n_rows=n_barcodes, n_cols=n_features, transpose=True, owner=owner,
            local=local, n_targets=len(targets), target_cols=widths,
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        detail: _Import = plan.detail
        folder, emap = detail.folder, detail.emap

        # Pass 1, before any output exists: a file that cannot be read must
        # not leave a store holding obs and var and nothing else.
        scans = {}
        for name, header in detail.headers.items():
            with console.status(f"[cyan]Checking {header.path.name}...[/]"):
                scans[name] = scan(header, emap)

        with ExitStack() as stack:
            roots = [
                stack.enter_context(
                    open_store(path, "w", zarr_format=detail.zarr_format)
                ).root
                for path in staged
            ]
            with console.status("[cyan]Writing obs and var...[/]"):
                for root, target in zip(roots, detail.targets):
                    obs = write_obs_index(
                        root, detail.n_barcodes,
                        iter_first_column(folder.barcodes, STRING_STEP),
                    )
                    set_column_order(obs, [])
                    write_var(root, target)
                    ensure_anndata_skeleton(root)
                    _write_uns(root, folder, detail.metadata)

            for name, header in detail.headers.items():
                scanned = scans[name]
                dtype = value_dtype(header, scanned)
                destinations = []
                for t, root in enumerate(roots):
                    shape = (detail.n_barcodes, emap.target_cols[t])
                    places = []
                    if name in ("matrix", "spliced"):
                        places.append((root, "X"))
                    if name != "matrix":
                        places.append((root["layers"], name))
                    destinations.append([
                        CsrAppender(
                            parent, key, shape=shape, data_dtype=dtype,
                            compression=detail.compression,
                        )
                        for parent, key in places
                    ])
                with console.status(f"[cyan]Reading {header.path.name}...[/]"):
                    summed = read_into(header, emap, scanned, destinations, roots[0])
                    for appenders in destinations:
                        for appender in appenders:
                            appender.finish()
                if summed:
                    console.print(
                        f"[yellow]Warning:[/] {summed:,} duplicate entries in "
                        f"{header.path.name} were summed, as scipy does."
                    )


def _write_uns(root: Any, folder: _Folder, metadata: Dict[str, Any]) -> None:
    """What an export needs to write the folder back, and the run's metadata."""
    meta = write_mapping(root["uns"], "cellranger")
    write_scalar(meta, "layout", "mtx")
    if folder.genome is not None:
        write_scalar(meta, "genome", folder.genome)
    values = {
        key: value for key, value in metadata.items()
        if isinstance(value, (str, int, float)) and not isinstance(value, bool)
    }
    if values:
        attrs = write_mapping(meta, "attrs")
        for key, value in values.items():
            write_scalar(attrs, key, value)


# ---------------------------------------------------------------------------
# export


def _check_text(values: Sequence[str], what: str) -> None:
    """Refuse what the TSV readers would not read back as written.

    A tab or a line break splits the field. pandas (and so scanpy) and R
    take a field that starts with a double quote for a quoted one, and read
    on to the next quote, across tabs and lines; and both skip a blank line,
    which is what an empty barcode makes.
    """
    for value in values:
        if "\t" in value or "\n" in value or "\r" in value:
            raise ValueError(
                f"A {what} contains a tab or a line break, which would break "
                f"the TSV file: {value!r}."
            )
        if value.startswith('"'):
            raise ValueError(
                f"A {what} starts with a double quote, which pandas and R read "
                f"as the start of a quoted field: {value!r}."
            )
        if not value and what == "barcode":
            raise ValueError(
                "A barcode is empty: its line would be blank, which pandas and "
                "R skip, so every barcode after it would move up a row."
            )


class MtxExport(Converter):
    options = frozenset(
        {"--layer", "--gene-ids-column", "--gene-names-column",
         "--feature-types-column", "--compression"}
    )

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        compression = options.compression or "gzip"
        if compression not in EXPORT_COMPRESSIONS:
            raise ValueError(
                f"--compression {compression}: a 10x mtx folder is either "
                f"gzipped or not; use one of: {', '.join(EXPORT_COMPRESSIONS)}."
            )
        warnings: List[str] = []
        matrix = plan_matrix_source(source, options, warnings)
        features = matrix.features
        _check_text(features.ids, "feature id")
        _check_text(features.names, "feature name")
        _check_text(features.types, "feature type")
        mtx_field(matrix.dtype)

        return ConversionPlan(
            source=source,
            src_format=src_format,
            dst_format=dst_format,
            outputs=[output],
            summary=(
                f"{src_format} → 10x mtx folder, {matrix.n_obs:,} barcodes x "
                f"{matrix.n_var:,} features from {matrix.path}"
                + ("" if compression == "gzip" else ", uncompressed")
            ),
            warnings=warnings,
            detail={"matrix": matrix, "gzipped": compression == "gzip"},
        )

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        source = plan.detail["matrix"]
        gzipped: bool = plan.detail["gzipped"]
        ext = ".gz" if gzipped else ""
        folder = staged[0]
        folder.mkdir()

        features = source.features
        with create_binary(folder / f"features.tsv{ext}", gzipped=gzipped, level=GZIP_LEVEL) as fh:
            for start in range(0, len(features.ids), STRING_STEP):
                stop = start + STRING_STEP
                fh.write("".join(
                    f"{i}\t{n}\t{t}\n"
                    for i, n, t in zip(
                        features.ids[start:stop],
                        features.names[start:stop],
                        features.types[start:stop],
                    )
                ).encode("utf-8"))

        with open_store(plan.source, "r", require_anndata=False) as store:
            root = store.root
            obs = root["obs"]
            obs_index, _ = resolve_index(obs, "obs")
            n = element_len(obs_index)
            cache: Dict[str, np.ndarray] = {}
            with create_binary(folder / f"barcodes.tsv{ext}", gzipped=gzipped, level=GZIP_LEVEL) as fh:
                for lo in range(0, n, STRING_STEP):
                    block = read_str_chunk(obs_index, lo, min(lo + STRING_STEP, n), cache, obs)
                    _check_text(block, "barcode")
                    fh.write(("\n".join(block) + "\n").encode("utf-8"))

            matrix = describe(resolve(root, source.path))
            check_sparse(matrix)
            with console.status(f"[cyan]Counting the entries of {source.path}...[/]"):
                nnz = count_entries(matrix)
            with create_binary(folder / f"matrix.mtx{ext}", gzipped=gzipped, level=GZIP_LEVEL) as fh:
                write_header(
                    fh,
                    field=mtx_field(matrix.dtype),
                    n_rows=source.n_var,
                    n_cols=source.n_obs,
                    nnz=nnz,
                    comments=_metadata_comment(root),
                )
                with console.status(f"[cyan]Writing {source.path}...[/]"):
                    write_entries(fh, iter_matrix_entries(matrix, transpose=True))


def _metadata_comment(root: Any) -> List[str]:
    """`%metadata_json`, as Cell Ranger writes it, from what an import kept."""
    meta = cellranger_meta(root)
    if meta is None or "attrs" not in meta:
        return []
    values: Dict[str, Any] = {}
    for key in meta["attrs"].keys():
        item = meta["attrs"][key]
        if is_dataset(item) and getattr(item, "shape", None) == ():
            value = item[()]
            if isinstance(value, bytes):
                value = value.decode("utf-8", "replace")
            elif isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, (str, int, float)):
                values[key] = value
        elif is_dataset(item):
            strings = read_str_all(item)
            if len(strings) == 1:
                values[key] = strings[0]
    if not values:
        return []
    return [f"{_METADATA} {json.dumps(values, sort_keys=True)}"]
