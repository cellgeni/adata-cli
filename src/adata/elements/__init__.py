"""AnnData on-disk element handling: the spec, and reading/writing each element.

`spec` holds the encoding constants, `read` turns any on-disk layout into
usable values, `write` emits the current spec, and `strings` reconciles the two
backends' incompatible spellings of text.

The submodules load on first use rather than here. `spec` imports nothing
else, so `adata.storage` can take its constants from it, while `read`,
`strings` and `write` import `adata.storage`: loading them eagerly with the
package would be an import cycle.
"""

from importlib import import_module
from typing import Any

__all__ = ["read", "spec", "strings", "write"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
