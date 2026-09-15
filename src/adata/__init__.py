"""adata-cli: streaming CLI for large AnnData .h5ad and .zarr stores."""

from importlib.metadata import PackageNotFoundError, version as _version

try:
    __version__ = _version("adata-cli")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
