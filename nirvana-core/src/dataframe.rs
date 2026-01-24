//! Core DataFrame implementation using Apache Arrow as the backend.
//!
//! This module provides a columnar DataFrame structure optimized for
//! batch access patterns common in LLM prompting scenarios.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BooleanArray, Float64Array, Int64Array, RecordBatch, RecordBatchReader,
    StringArray,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ffi_stream::ArrowArrayStreamReader;
use arrow::pyarrow::{FromPyArrow, ToPyArrow};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use rayon::prelude::*;

use crate::dtype::{infer::infer_dtype_from_strings, NirvanaDataType};
use crate::error::{NirvanaError, Result};
use crate::io::{csv::read_csv, excel::read_excel};

/// A single column with Arrow array backend
#[pyclass]
#[derive(Clone)]
pub struct PyColumn {
    name: String,
    data: ArrayRef,
    /// Data type for this column (extending Arrow types)
    dtype: NirvanaDataType,
}

#[pymethods]
impl PyColumn {
    /// Get the column name
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    /// Get the data type as a string
    /// Returns "image", "audio", or the Arrow type name
    #[getter]
    fn dtype(&self) -> String {
        match &self.dtype {
            NirvanaDataType::Image(_) => "image".to_string(),
            NirvanaDataType::Audio(_) => "audio".to_string(),
            NirvanaDataType::Arrow(dt) => format!("{:?}", dt),
        }
    }

    /// Get the number of elements
    fn __len__(&self) -> usize {
        self.data.len()
    }

    // ... [skipped __getitem__ implementation as it uses get_value] ...

    fn __getitem__(&self, idx: isize, py: Python<'_>) -> PyResult<PyObject> {
        let len = self.data.len() as isize;
        let actual_idx = if idx < 0 { len + idx } else { idx };

        if actual_idx < 0 || actual_idx >= len {
            return Err(NirvanaError::IndexOutOfBounds {
                index: idx as usize,
                length: self.data.len(),
            }
            .into());
        }

        self.get_value(actual_idx as usize, py)
    }

    /// Convert to a Python list
    fn to_list(&self, py: Python<'_>) -> PyResult<PyObject> {
        let list = PyList::empty(py);
        for i in 0..self.data.len() {
            list.append(self.get_value(i, py)?)?;
        }
        Ok(list.into())
    }

    /// Get values as a batch (for efficient LLM prompting)
    fn get_batch(&self, start: usize, end: usize, py: Python<'_>) -> PyResult<PyObject> {
        if end > self.data.len() {
            return Err(NirvanaError::IndexOutOfBounds {
                index: end,
                length: self.data.len(),
            }
            .into());
        }

        let list = PyList::empty(py);
        for i in start..end {
            list.append(self.get_value(i, py)?)?;
        }
        Ok(list.into())
    }

    fn __repr__(&self) -> String {
        format!(
            "PyColumn(name='{}', dtype={}, len={})",
            self.name,
            self.dtype(),
            self.data.len()
        )
    }
}

impl PyColumn {
    pub fn get_value(&self, idx: usize, py: Python<'_>) -> PyResult<PyObject> {
        if self.data.is_null(idx) {
            return Ok(py.None());
        }

        match self.data.data_type() {
            DataType::Int64 => {
                let arr = self.data.as_any().downcast_ref::<Int64Array>().unwrap();
                Ok(arr.value(idx).into_pyobject(py)?.into_any().unbind())
            }
            DataType::Float64 => {
                let arr = self.data.as_any().downcast_ref::<Float64Array>().unwrap();
                Ok(arr.value(idx).into_pyobject(py)?.into_any().unbind())
            }
            DataType::Utf8 | DataType::LargeUtf8 => {
                let arr = self.data.as_any().downcast_ref::<StringArray>().unwrap();
                Ok(arr.value(idx).into_pyobject(py)?.into_any().unbind())
            }
            DataType::Boolean => {
                let arr = self.data.as_any().downcast_ref::<BooleanArray>().unwrap();
                let value: bool = arr.value(idx);
                Ok(pyo3::types::PyBool::new(py, value)
                    .to_owned()
                    .into_any()
                    .unbind())
            }
            _ => Ok(format!("<unsupported type: {:?}>", self.data.data_type())
                .into_pyobject(py)?
                .into_any()
                .unbind()),
        }
    }

    pub fn array_ref(&self) -> &ArrayRef {
        &self.data
    }

    /// Convert column data to PyArrow array
    pub fn to_pyarrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        self.data.to_data().to_pyarrow(py)
    }
}

impl PyColumn {
    /// Create a new column with Arrow data type inferred from data
    pub fn new(name: String, data: ArrayRef) -> Self {
        let dtype = NirvanaDataType::Arrow(data.data_type().clone());
        Self { name, data, dtype }
    }

    /// Create a new column with specified Nirvana data type
    pub fn new_with_dtype(name: String, data: ArrayRef, dtype: NirvanaDataType) -> Self {
        Self { name, data, dtype }
    }
}

/// Columnar DataFrame backed by Apache Arrow
#[pyclass]
pub struct PyDataFrame {
    pub schema: SchemaRef,
    pub columns: HashMap<String, PyColumn>,
    pub column_order: Vec<String>,
    pub num_rows: usize,
}

#[pymethods]
impl PyDataFrame {
    /// Create an empty DataFrame
    #[new]
    pub fn new() -> Self {
        Self {
            schema: Arc::new(Schema::empty()),
            columns: HashMap::new(),
            column_order: Vec::new(),
            num_rows: 0,
        }
    }

    /// Create DataFrame from a Python dictionary
    #[staticmethod]
    fn from_dict(data: &Bound<'_, PyDict>, py: Python<'_>) -> PyResult<Self> {
        let mut columns = HashMap::new();
        let mut column_order = Vec::new();
        let mut fields = Vec::new();
        let mut num_rows = 0;

        for (key, value) in data.iter() {
            let col_name: String = key.extract()?;
            let values: Vec<PyObject> = value.extract()?;

            if num_rows == 0 {
                num_rows = values.len();
            } else if values.len() != num_rows {
                return Err(NirvanaError::InvalidOperation(format!(
                    "Column '{}' has {} rows, expected {}",
                    col_name,
                    values.len(),
                    num_rows
                ))
                .into());
            }

            // Infer type from first non-null value
            let (array, dtype) = Self::infer_and_create_array(&values, py)?;
            fields.push(Field::new(&col_name, dtype.to_arrow(), true));
            columns.insert(
                col_name.clone(),
                PyColumn::new_with_dtype(col_name.clone(), array, dtype),
            );
            column_order.push(col_name);
        }

        Ok(Self {
            schema: Arc::new(Schema::new(fields)),
            columns,
            column_order,
            num_rows,
        })
    }

    /// Load DataFrame from a CSV file
    #[staticmethod]
    #[pyo3(signature = (path, delimiter=None, has_header=None))]
    fn from_csv(path: &str, delimiter: Option<u8>, has_header: Option<bool>) -> PyResult<Self> {
        let delimiter = delimiter.unwrap_or(b',');
        let has_header = has_header.unwrap_or(true);
        let mut df = read_csv(path, delimiter, has_header).map_err(PyErr::from)?;
        df.infer_and_convert_media_columns().map_err(PyErr::from)?;
        Ok(df)
    }

    /// Load DataFrame from an Excel file
    #[staticmethod]
    #[pyo3(signature = (path, sheet=None))]
    fn from_excel(path: &str, sheet: Option<&str>) -> PyResult<Self> {
        let mut df = read_excel(path, sheet).map_err(PyErr::from)?;
        df.infer_and_convert_media_columns().map_err(PyErr::from)?;
        Ok(df)
    }

    /// Load DataFrame from a PyArrow Table
    #[staticmethod]
    fn from_pyarrow(table: &Bound<'_, PyAny>, _py: Python<'_>) -> PyResult<Self> {
        // Convert PyArrow Table to RecordBatchReader
        let reader = ArrowArrayStreamReader::from_pyarrow_bound(table)?;
        let schema = reader.schema();

        // Collect all batches
        let mut all_columns: HashMap<String, Vec<ArrayRef>> = HashMap::new();
        let mut total_rows = 0;

        for batch_result in reader {
            let batch = batch_result.map_err(|e| NirvanaError::Arrow(e))?;
            total_rows += batch.num_rows();

            for (i, field) in schema.fields().iter().enumerate() {
                let name = field.name().clone();
                all_columns
                    .entry(name)
                    .or_insert_with(Vec::new)
                    .push(batch.column(i).clone());
            }
        }

        // Concatenate arrays for each column
        let mut columns = HashMap::new();
        let mut column_order = Vec::new();

        for field in schema.fields() {
            let name = field.name().clone();
            let arrays = all_columns.get(&name).unwrap();

            // Concatenate if multiple batches
            let combined = if arrays.len() == 1 {
                arrays[0].clone()
            } else {
                let array_refs: Vec<&dyn Array> = arrays.iter().map(|a| a.as_ref()).collect();
                arrow::compute::concat(&array_refs).map_err(|e| NirvanaError::Arrow(e))?
            };

            columns.insert(name.clone(), PyColumn::new(name.clone(), combined));
            column_order.push(name);
        }

        let mut df = Self {
            schema,
            columns,
            column_order,
            num_rows: total_rows,
        };
        df.infer_and_convert_media_columns()
            .map_err(|e| NirvanaError::InvalidOperation(e.to_string()))?;
        Ok(df)
    }

    /// Convert to a PyArrow Table
    fn to_pyarrow(&self, py: Python<'_>) -> PyResult<PyObject> {
        let mut arrays: Vec<PyObject> = Vec::with_capacity(self.column_order.len());

        for name in &self.column_order {
            if let Some(col) = self.columns.get(name) {
                arrays.push(col.to_pyarrow(py)?);
            }
        }

        // Import pyarrow and create table
        let pyarrow = py.import("pyarrow")?;
        let schema_py = self.schema.as_ref().to_pyarrow(py)?;
        let arrays_list = PyList::new(py, arrays)?;
        let table = pyarrow.call_method1("table", (arrays_list, schema_py))?;

        Ok(table.into())
    }

    /// Get column names
    #[getter]
    fn columns(&self) -> Vec<String> {
        self.column_order.clone()
    }

    /// Get number of rows
    #[getter]
    fn nrows(&self) -> usize {
        self.num_rows
    }

    /// Get number of columns
    #[getter]
    fn ncols(&self) -> usize {
        self.column_order.len()
    }

    /// Get the schema as a dictionary of {column_name: dtype}
    /// Returns media dtypes (image/audio) when applicable
    fn schema_dict(&self) -> HashMap<String, String> {
        self.columns
            .iter()
            .map(|(name, col)| (name.clone(), col.dtype()))
            .collect()
    }

    /// Get a column by name
    fn __getitem__(&self, column: &str) -> PyResult<PyColumn> {
        self.columns
            .get(column)
            .cloned()
            .ok_or_else(|| NirvanaError::ColumnNotFound(column.to_string()).into())
    }

    /// Check if a column exists
    fn __contains__(&self, column: &str) -> bool {
        self.columns.contains_key(column)
    }

    fn __len__(&self) -> usize {
        self.num_rows
    }

    /// Set the media dtype for a column
    ///
    /// Args:
    ///     column: Column name
    ///     media_dtype: "image", "audio", or "text"
    fn set_column_media_dtype(&mut self, column: &str, media_dtype: &str) -> PyResult<()> {
        let dtype = match media_dtype {
            "image" => NirvanaDataType::Image(crate::dtype::image::ImageDtype::new()),
            "audio" => NirvanaDataType::Audio(crate::dtype::audio::AudioDtype::new()),
            "text" => NirvanaDataType::Arrow(DataType::Utf8),
            _ => {
                return Err(NirvanaError::InvalidOperation(format!(
                    "Invalid media dtype '{}'. Must be 'image', 'audio', or 'text'",
                    media_dtype
                ))
                .into())
            }
        };

        if let Some(col) = self.columns.get_mut(column) {
            col.dtype = dtype;
            Ok(())
        } else {
            Err(NirvanaError::ColumnNotFound(column.to_string()).into())
        }
    }

    /// Get a batch of data across specified columns
    ///
    /// This is optimized for LLM prompting where you need to pack
    /// multiple column values together for each row in a batch.
    fn get_batch(
        &self,
        columns: Vec<String>,
        start: usize,
        end: usize,
        py: Python<'_>,
    ) -> PyResult<PyObject> {
        if end > self.num_rows {
            return Err(NirvanaError::IndexOutOfBounds {
                index: end,
                length: self.num_rows,
            }
            .into());
        }

        let result = PyDict::new(py);
        for col_name in columns {
            if let Some(col) = self.columns.get(&col_name) {
                let batch = col.get_batch(start, end, py)?;
                result.set_item(&col_name, batch)?;
            } else {
                return Err(NirvanaError::ColumnNotFound(col_name).into());
            }
        }

        Ok(result.into())
    }

    /// Get rows as a list of dictionaries (for iteration)
    fn to_dicts(&self, py: Python<'_>) -> PyResult<PyObject> {
        let result = PyList::empty(py);

        for i in 0..self.num_rows {
            let row = PyDict::new(py);
            for col_name in &self.column_order {
                if let Some(col) = self.columns.get(col_name) {
                    row.set_item(col_name, col.get_value(i, py)?)?;
                }
            }
            result.append(row)?;
        }

        Ok(result.into())
    }

    /// Get first n rows as dicts
    #[pyo3(signature = (n=5))]
    fn head(&self, n: usize, py: Python<'_>) -> PyResult<PyObject> {
        let n = n.min(self.num_rows);
        let result = PyList::empty(py);

        for i in 0..n {
            let row = PyDict::new(py);
            for col_name in &self.column_order {
                if let Some(col) = self.columns.get(col_name) {
                    row.set_item(col_name, col.get_value(i, py)?)?;
                }
            }
            result.append(row)?;
        }

        Ok(result.into())
    }

    /// Get last n rows as dicts
    #[pyo3(signature = (n=5))]
    fn tail(&self, n: usize, py: Python<'_>) -> PyResult<PyObject> {
        let n = n.min(self.num_rows);
        let start = self.num_rows.saturating_sub(n);
        let result = PyList::empty(py);

        for i in start..self.num_rows {
            let row = PyDict::new(py);
            for col_name in &self.column_order {
                if let Some(col) = self.columns.get(col_name) {
                    row.set_item(col_name, col.get_value(i, py)?)?;
                }
            }
            result.append(row)?;
        }

        Ok(result.into())
    }

    /// Select specific columns, returning a new DataFrame
    fn select(&self, columns: Vec<String>) -> PyResult<Self> {
        let mut new_columns = HashMap::new();
        let mut new_fields = Vec::new();

        for col_name in &columns {
            if let Some(col) = self.columns.get(col_name) {
                new_columns.insert(col_name.clone(), col.clone());
                if let Ok(field) = self.schema.field_with_name(col_name) {
                    new_fields.push(field.clone());
                }
            } else {
                return Err(NirvanaError::ColumnNotFound(col_name.clone()).into());
            }
        }

        Ok(Self {
            schema: Arc::new(Schema::new(new_fields)),
            columns: new_columns,
            column_order: columns,
            num_rows: self.num_rows,
        })
    }

    /// Slice rows by range
    fn slice(&self, start: usize, length: usize) -> PyResult<Self> {
        if start >= self.num_rows {
            return Err(NirvanaError::IndexOutOfBounds {
                index: start,
                length: self.num_rows,
            }
            .into());
        }

        let actual_length = length.min(self.num_rows - start);
        let mut new_columns = HashMap::new();

        for (name, col) in &self.columns {
            let sliced_data = col.data.slice(start, actual_length);
            new_columns.insert(name.clone(), PyColumn::new(name.clone(), sliced_data));
        }

        Ok(Self {
            schema: self.schema.clone(),
            columns: new_columns,
            column_order: self.column_order.clone(),
            num_rows: actual_length,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "DataFrame(nrows={}, ncols={}, columns={:?})",
            self.num_rows,
            self.column_order.len(),
            self.column_order
        )
    }
}

impl PyDataFrame {
    /// Create from Arrow RecordBatch
    pub fn from_record_batch(batch: RecordBatch) -> Result<Self> {
        let schema = batch.schema();
        let num_rows = batch.num_rows();
        let mut columns = HashMap::new();
        let mut column_order = Vec::new();

        for (i, field) in schema.fields().iter().enumerate() {
            let name = field.name().clone();
            let array = batch.column(i).clone();
            columns.insert(name.clone(), PyColumn::new(name.clone(), array));
            column_order.push(name);
        }

        Ok(Self {
            schema,
            columns,
            column_order,
            num_rows,
        })
    }

    /// Get internal schema
    pub fn get_schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    /// Get column by name (internal)
    pub fn get_column(&self, name: &str) -> Option<&PyColumn> {
        self.columns.get(name)
    }

    /// Get column order
    pub fn get_column_order(&self) -> &[String] {
        &self.column_order
    }

    fn infer_and_create_array(
        values: &[PyObject],
        py: Python<'_>,
    ) -> PyResult<(ArrayRef, NirvanaDataType)> {
        // Find first non-None value to infer type
        let first_non_none = values.iter().find(|v| !v.bind(py).is_none());

        match first_non_none {
            Some(val) => {
                let bound_val = val.bind(py);

                if bound_val.extract::<i64>().is_ok() {
                    // Integer type
                    let int_values: Vec<Option<i64>> = values
                        .iter()
                        .map(|v| {
                            let bound = v.bind(py);
                            if bound.is_none() {
                                None
                            } else {
                                bound.extract::<i64>().ok()
                            }
                        })
                        .collect();
                    let array: ArrayRef = Arc::new(Int64Array::from(int_values));
                    Ok((array, NirvanaDataType::Arrow(DataType::Int64)))
                } else if bound_val.extract::<f64>().is_ok() {
                    // Float type
                    let float_values: Vec<Option<f64>> = values
                        .iter()
                        .map(|v| {
                            let bound = v.bind(py);
                            if bound.is_none() {
                                None
                            } else {
                                bound.extract::<f64>().ok()
                            }
                        })
                        .collect();
                    let array: ArrayRef = Arc::new(Float64Array::from(float_values));
                    Ok((array, NirvanaDataType::Arrow(DataType::Float64)))
                } else if bound_val.extract::<bool>().is_ok() {
                    // Boolean type
                    let bool_values: Vec<Option<bool>> = values
                        .iter()
                        .map(|v| {
                            let bound = v.bind(py);
                            if bound.is_none() {
                                None
                            } else {
                                bound.extract::<bool>().ok()
                            }
                        })
                        .collect();
                    let array: ArrayRef = Arc::new(BooleanArray::from(bool_values));
                    Ok((array, NirvanaDataType::Arrow(DataType::Boolean)))
                } else {
                    // Default to string, then infer specific media type
                    let mut non_null_strings = Vec::new();
                    let str_values: Vec<Option<String>> = values
                        .iter()
                        .map(|v| {
                            let bound = v.bind(py);
                            if bound.is_none() {
                                None
                            } else {
                                let s = bound.str().ok().map(|s| s.to_string());
                                if let Some(ref val) = s {
                                    non_null_strings.push(val.clone());
                                }
                                s
                            }
                        })
                        .collect();

                    // Infer media type from the strings
                    let nirvana_dtype = infer_dtype_from_strings(&non_null_strings);

                    // Perform conversion if media type is detected
                    let final_values: Vec<Option<String>> = match nirvana_dtype {
                        NirvanaDataType::Image(_) => str_values
                            .into_par_iter()
                            .map(|opt_s| {
                                opt_s.map(|s| {
                                    crate::dtype::infer::load_image_as_base64(&s).unwrap_or(s)
                                })
                            })
                            .collect(),
                        NirvanaDataType::Audio(_) => str_values
                            .into_par_iter()
                            .map(|opt_s| {
                                opt_s.map(|s| {
                                    crate::dtype::infer::load_audio_as_base64(&s).unwrap_or(s)
                                })
                            })
                            .collect(),
                        _ => str_values,
                    };

                    let array: ArrayRef = Arc::new(StringArray::from(final_values));

                    Ok((array, nirvana_dtype))
                }
            }
            None => {
                // All values are None, create null string array
                let null_values: Vec<Option<String>> = vec![None; values.len()];
                let array: ArrayRef = Arc::new(StringArray::from(null_values));
                Ok((array, NirvanaDataType::Arrow(DataType::Utf8)))
            }
        }
    }

    /// Helper to infer and convert media columns in place
    fn infer_and_convert_media_columns(&mut self) -> Result<()> {
        let mut new_columns = HashMap::new();
        let mut fields = Vec::new();

        for col_name in &self.column_order {
            if let Some(col) = self.columns.get(col_name) {
                // Only check String (Utf8) columns
                if matches!(col.dtype, NirvanaDataType::Arrow(DataType::Utf8)) {
                    // Extract strings to check
                    let arr = col
                        .data
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .ok_or_else(|| {
                            NirvanaError::InvalidOperation(format!(
                                "Column {} has Utf8 type but cannot cast to StringArray",
                                col_name
                            ))
                        })?;

                    let num_rows = arr.len();
                    // Sample first 100 non-null values to find valid strings
                    let mut sample_strings = Vec::new();
                    for i in 0..num_rows.min(100) {
                        if !arr.is_null(i) {
                            let s = arr.value(i);
                            if !s.is_empty() {
                                sample_strings.push(s.to_string());
                            }
                            if sample_strings.len() >= 10 {
                                break;
                            }
                        }
                    }

                    if sample_strings.is_empty() {
                        fields.push(Field::new(col_name, col.dtype.to_arrow(), true));
                        new_columns.insert(col_name.clone(), col.clone());
                        continue;
                    }

                    // Infer dtype
                    let nirvana_dtype = infer_dtype_from_strings(&sample_strings);

                    // If media type, convert values
                    if nirvana_dtype.is_image() || nirvana_dtype.is_audio() {
                        // Collect all values (handling nulls)
                        let str_values: Vec<Option<String>> = (0..num_rows)
                            .map(|i| {
                                if arr.is_null(i) {
                                    None
                                } else {
                                    Some(arr.value(i).to_string())
                                }
                            })
                            .collect();

                        // Convert in parallel
                        let final_values: Vec<Option<String>> = match nirvana_dtype {
                            NirvanaDataType::Image(_) => str_values
                                .into_par_iter()
                                .map(|opt_s| {
                                    opt_s.map(|s| {
                                        crate::dtype::infer::load_image_as_base64(&s).unwrap_or(s)
                                    })
                                })
                                .collect(),
                            NirvanaDataType::Audio(_) => str_values
                                .into_par_iter()
                                .map(|opt_s| {
                                    opt_s.map(|s| {
                                        crate::dtype::infer::load_audio_as_base64(&s).unwrap_or(s)
                                    })
                                })
                                .collect(),
                            _ => str_values, // Should not happen given outer check
                        };

                        let new_array = Arc::new(StringArray::from(final_values));
                        fields.push(Field::new(col_name, nirvana_dtype.to_arrow(), true));
                        new_columns.insert(
                            col_name.clone(),
                            PyColumn::new_with_dtype(col_name.clone(), new_array, nirvana_dtype),
                        );
                    } else {
                        // Keep original
                        fields.push(Field::new(col_name, col.dtype.to_arrow(), true));
                        new_columns.insert(col_name.clone(), col.clone());
                    }
                } else {
                    // Not a string column, keep as is
                    fields.push(Field::new(col_name, col.dtype.to_arrow(), true));
                    new_columns.insert(col_name.clone(), col.clone());
                }
            }
        }

        // Update schema and columns
        self.schema = Arc::new(Schema::new(fields));
        self.columns = new_columns;
        Ok(())
    }
}

impl Default for PyDataFrame {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_csv_media_conversion() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "id,image_path").unwrap();
        writeln!(file, "1,/path/to/image.jpg").unwrap();
        writeln!(file, "2,/path/to/image.png").unwrap();

        // The file needs to be closed or flushed to be readable? NamedTempFile handles it?
        // NamedTempFile destructor deletes the file, so we must keep it alive.
        // We pass path to from_csv.

        let df = PyDataFrame::from_csv(file.path().to_str().unwrap(), None, None).unwrap();

        let schema = df.schema_dict();
        assert_eq!(schema.get("image_path").unwrap(), "image");
        assert_eq!(schema.get("id").unwrap(), "Int64");
    }
}
