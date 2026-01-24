"""DataFrame module for nirvana."""

from nirvana.dataframe.frame import DataFrame

# Try to import RustDataFrame if Rust backend is available
try:
    from nirvana.dataframe.rust_frame import DataFrame, rust_available
except ImportError:
    def rust_available():
        return False

__all__ = [
    "DataFrame",
    "rust_available",
]
