"""AnnData on-disk element handling: the spec, and reading/writing each element.

`spec` holds the encoding constants, `read` turns any on-disk layout into
usable values, `write` emits the current spec, and `strings` reconciles the two
backends' incompatible spellings of text.
"""

from adata.elements import read, spec, strings, write

__all__ = ["read", "spec", "strings", "write"]
