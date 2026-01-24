//! Error types for nirvana-core

use pyo3::exceptions::PyRuntimeError;
use pyo3::PyErr;
use thiserror::Error;

/// Custom error type for nirvana-core operations
#[derive(Error, Debug)]
pub enum NirvanaError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),

    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    #[error("Excel error: {0}")]
    Excel(String),

    #[error("Column not found: {0}")]
    ColumnNotFound(String),

    #[error("Index out of bounds: {index} (length: {length})")]
    IndexOutOfBounds { index: usize, length: usize },

    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Serialization error: {0}")]
    Serialization(String),
}

impl From<calamine::XlsxError> for NirvanaError {
    fn from(err: calamine::XlsxError) -> Self {
        NirvanaError::Excel(err.to_string())
    }
}

impl From<NirvanaError> for PyErr {
    fn from(err: NirvanaError) -> PyErr {
        PyRuntimeError::new_err(err.to_string())
    }
}

pub type Result<T> = std::result::Result<T, NirvanaError>;
