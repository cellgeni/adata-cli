"""Parquet tables, read a batch of rows at a time.

Visium HD writes its bin positions only as Parquet. pyarrow reads it, and
comes with the `spatial` extra, so it is imported here alone and only when
a Parquet file is read.

Parquet stores each column apart, so a pass reads only the columns it asks
for, and the file is read as a buffered stream: by default pyarrow reads a
whole column chunk of a row group at once, and Space Ranger writes its
positions as one row group, some 11 million rows for 2 µm bins.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Sequence, Tuple

import numpy as np

#: Rows handed out per block.
BATCH_ROWS = 1 << 16

#: Bytes pyarrow reads from the file at a time. Measured on Space Ranger's
#: 2 µm positions, this halves the peak of reading their numeric columns.
READ_BUFFER = 1 << 20


def require_pyarrow(what: str) -> None:
    try:
        import pyarrow.parquet  # noqa: F401
    except ImportError:
        raise ValueError(
            f"Reading {what} needs pyarrow, which reads its Parquet tables: "
            "pip install 'pyadata-cli[spatial]'."
        ) from None


@dataclass
class ParquetTable:
    path: Path
    n_rows: int
    #: Each column's Arrow type, by name, in the file's order.
    types: Dict[str, str]

    def blocks(
        self, columns: Sequence[str], batch_rows: int = BATCH_ROWS
    ) -> Iterator[Tuple[int, Dict[str, np.ndarray]]]:
        """(rows, {column: values}) for `columns`, in file order. Strings
        come as bytes, which is how a barcode join compares them."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        handle = pq.ParquetFile(self.path, buffer_size=READ_BUFFER, pre_buffer=False)
        for batch in handle.iter_batches(batch_size=batch_rows, columns=list(columns)):
            values: Dict[str, np.ndarray] = {}
            for name in columns:
                column = batch.column(name)
                if pa.types.is_string(column.type) or pa.types.is_large_string(column.type):
                    column = column.cast(pa.binary())
                values[name] = column.to_numpy(zero_copy_only=False)
            yield batch.num_rows, values


def open_table(path: Path) -> ParquetTable:
    """A Parquet file's row count and column types, from its footer alone."""
    import pyarrow.parquet as pq

    handle = pq.ParquetFile(path)
    schema = handle.schema_arrow
    return ParquetTable(
        path=Path(path),
        n_rows=handle.metadata.num_rows,
        types={field.name: str(field.type) for field in schema},
    )
