"""The pieces every converter shares: its options, its plan, its interface."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, FrozenSet, List, Optional, Sequence

from rich.console import Console


@dataclass(frozen=True)
class Options:
    """What `adata convert` was asked for, beyond the input and output."""

    #: The flags the user actually passed, e.g. `{"--zarr-format"}`. A flag
    #: left at its default is not here, which is how one that does not apply
    #: to the conversion can be refused rather than silently ignored.
    given: FrozenSet[str] = frozenset()
    zarr_format: Optional[int] = None
    compression: Optional[str] = None
    force: bool = False
    overwrite: bool = False

    # Reading a 10x matrix.
    genome: Optional[str] = None
    var_names: str = "gene_ids"
    make_unique: bool = False
    feature_type: Optional[str] = None
    split_feature_types: bool = False

    # Writing a 10x matrix.
    layer: Optional[str] = None
    gene_ids_column: Optional[str] = None
    gene_names_column: Optional[str] = None
    feature_types_column: Optional[str] = None

    # Reading CellBender output.
    raw: Optional[Path] = None
    min_cell_probability: Optional[float] = None

    # Reading Parse split-pipe output (and, --unfiltered, Space Ranger's).
    sample: Optional[str] = None
    all_samples: bool = False
    unfiltered: bool = False

    # Reading Space Ranger output.
    library_id: Optional[str] = None
    no_images: bool = False
    bin_size: Optional[int] = None

    # Reading a loom.
    dense: bool = False
    obs_names_attr: Optional[str] = None
    var_names_attr: Optional[str] = None

    # Reading a text matrix (and, --transpose, a plain HDF5 one).
    delimiter: Optional[str] = None
    transpose: bool = False
    first_column_names: Optional[bool] = None

    # Reading a plain HDF5 matrix.
    key: Optional[str] = None
    obs_names_key: Optional[str] = None
    var_names_key: Optional[str] = None

    # Writing a SCope loom.
    scope: bool = False
    clusterings: Optional[str] = None
    default_embedding: Optional[str] = None

    # Reading or writing a MuData: the modality to read, or the one name to
    # write the whole AnnData under.
    mod: Optional[str] = None


@dataclass
class ConversionPlan:
    """Everything worked out about a conversion before any output exists.

    Building one reads metadata only -- shapes, names, encodings -- so every
    refusal happens while there is still nothing to clean up.
    """

    source: Path
    src_format: str
    dst_format: str
    outputs: List[Path]
    #: One line saying what will be written, printed before writing starts.
    summary: str
    warnings: List[str] = field(default_factory=list)
    #: Whatever the converter worked out that `write` needs.
    detail: Any = None


class Converter:
    """Turns a store in one format into one or more in another."""

    #: Format-specific flags this converter understands, e.g. `--genome`.
    options: FrozenSet[str] = frozenset()

    def plan(
        self,
        source: Path,
        output: Path,
        src_format: str,
        dst_format: str,
        options: Options,
    ) -> ConversionPlan:
        raise NotImplementedError

    def write(
        self, plan: ConversionPlan, staged: Sequence[Path], console: Console
    ) -> None:
        """Write `plan.outputs` to `staged`, the temporary paths standing in
        for them, in the same order."""
        raise NotImplementedError


@dataclass(frozen=True)
class Format:
    """A format `adata convert` reads, writes, or both."""

    name: str
    description: str
    #: Output extensions that select this format, lowercase with the dot.
    extensions: Sequence[str]
    #: Whether this is an AnnData store; AnnData is the hub every conversion
    #: goes through.
    anndata: bool = False
    #: Does an existing file hold this format? Content only, not extension.
    sniff: Optional[Callable[[Path], bool]] = None
    #: Reads this format into AnnData.
    importer: Optional[Converter] = None
    #: Writes this format from AnnData.
    exporter: Optional[Converter] = None
