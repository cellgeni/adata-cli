"""The `convert` command: rewrite a store in another format.

Every refusal happens while building the plan, before an output exists, and
the outputs are written beside their destination and moved into place only
once complete. A conversion that fails therefore leaves nothing behind, and
never a half-written store that the next command would read as a whole one.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.markup import escape

from adata.conversion import (
    Options,
    check_options,
    converter_for,
    detect_format,
    output_format,
)
from adata.util.path import check_output, is_zarr_path, same_path
from adata.util.staging import check_free_space, disk_size, staged_outputs

#: What a Matrix Market folder written by `--to 10x-mtx` holds. A folder
#: holding anything else is not replaced, even with --overwrite.
MTX_FILES = frozenset(
    f"{name}{ext}"
    for name in ("matrix.mtx", "barcodes.tsv", "features.tsv", "genes.tsv")
    for ext in ("", ".gz")
)


def convert_store(
    file: Path,
    output: Path,
    console: Console,
    *,
    src_format: Optional[str] = None,
    dst_format: Optional[str] = None,
    options: Options = Options(),
    space_check: bool = True,
) -> List[Path]:
    """Convert `file` into `output`; return the paths written."""
    file, output = Path(file), Path(output)
    if str(output) == "-":
        raise ValueError(
            "convert writes a store, which cannot go to stdout; give a path "
            "with -o."
        )
    if options.zarr_format not in (None, 2, 3):
        raise ValueError("--zarr-format must be 2 or 3.")

    src_format = src_format or detect_format(file)
    dst_format = output_format(output, dst_format)
    converter = converter_for(src_format, dst_format)
    check_options(converter, options, src_format, dst_format)

    plan = converter.plan(file, output, src_format, dst_format, options)
    for out in plan.outputs:
        # Writing a Zarr store over its own source empties it before a
        # single element has been read; and --overwrite replaces a store,
        # never a folder of other files that -o named by mistake.
        if dst_format != "10x-mtx":
            check_output(out, [file])
        elif same_path(file, out):
            raise ValueError(f"Output path is the input: {out}")
        elif out.is_dir() and any(p.name not in MTX_FILES for p in out.iterdir()):
            raise ValueError(
                f"'{out}' holds files other than a Matrix Market folder's: "
                "refusing to replace it. Name a new folder to write."
            )
        if out.exists() and not options.overwrite:
            raise FileExistsError(
                f"'{out}' already exists. Pass --overwrite to replace it."
            )
    # A full disk part way through an HDF5 copy can abort inside the
    # library, past any cleanup. Guessed from a store or file source; a
    # folder (Space Ranger's outs/, a Xenium bundle) holds images and BAMs
    # the output never carries, so it is not measured.
    if plan.outputs and (file.is_file() or is_zarr_path(file)):
        check_free_space(plan.outputs[0], disk_size(file), check=space_check)

    console.print(f"[cyan]Converting[/] {file}: {plan.summary}")
    for warning in plan.warnings:
        console.print(f"[yellow]Warning:[/] {escape(warning)}")

    with staged_outputs(plan.outputs) as staged:
        converter.write(plan, staged, console)

    for out in plan.outputs:
        console.print(f"[green]Wrote[/] {out}")
    return plan.outputs
