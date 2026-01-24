//! PyArrow interoperability utilities
//!
//! This module provides additional utilities for working with PyArrow
//! beyond the basic conversions in the dataframe module.

use arrow::pyarrow::ToPyArrow;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::dataframe::PyDataFrame;

/// Create a PyArrow RecordBatch from internal format
pub fn to_pyarrow_record_batch(df: &PyDataFrame, py: Python<'_>) -> PyResult<PyObject> {
    let pyarrow = py.import("pyarrow")?;

    let mut arrays: Vec<PyObject> = Vec::new();
    for col_name in df.get_column_order() {
        if let Some(col) = df.get_column(col_name) {
            arrays.push(col.to_pyarrow(py)?);
        }
    }

    let schema_py = df.get_schema().as_ref().to_pyarrow(py)?;
    let arrays_list = PyList::new(py, arrays)?;

    let batch = pyarrow.call_method1("record_batch", (arrays_list, schema_py))?;

    Ok(batch.into())
}

/// Utility to check if PyArrow is available
pub fn pyarrow_available(py: Python<'_>) -> bool {
    py.import("pyarrow").is_ok()
}
