//! CSV file reading with automatic type inference

use std::collections::HashMap;
use std::fs::File;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use csv::ReaderBuilder;

use crate::dataframe::{PyColumn, PyDataFrame};
use crate::error::{NirvanaError, Result};

/// Read a CSV file into a PyDataFrame
pub fn read_csv(path: &str, delimiter: u8, has_header: bool) -> Result<PyDataFrame> {
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_header)
        .from_reader(file);

    // Get headers
    let headers: Vec<String> = if has_header {
        reader.headers()?.iter().map(|s| s.to_string()).collect()
    } else {
        // Generate column names col_0, col_1, etc based on first record
        let first_record = reader.records().next();
        match first_record {
            Some(Ok(record)) => (0..record.len()).map(|i| format!("col_{}", i)).collect(),
            _ => {
                return Err(NirvanaError::InvalidOperation("Empty CSV file".to_string()));
            }
        }
    };

    // Read all records into a vector of string columns
    let num_cols = headers.len();
    let mut string_columns: Vec<Vec<String>> = vec![Vec::new(); num_cols];

    // If no header, we need to re-read from the beginning
    let file = File::open(path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(delimiter)
        .has_headers(has_header)
        .from_reader(file);

    for result in reader.records() {
        let record = result?;
        for (i, field) in record.iter().enumerate() {
            if i < num_cols {
                string_columns[i].push(field.to_string());
            }
        }
    }

    let num_rows = string_columns.first().map(|c| c.len()).unwrap_or(0);

    // Infer types and create Arrow arrays
    let mut columns = HashMap::new();
    let mut column_order = Vec::new();
    let mut fields = Vec::new();

    for (i, col_name) in headers.iter().enumerate() {
        let string_col = &string_columns[i];
        let (array, dtype) = infer_column_type(string_col);

        fields.push(Field::new(col_name, dtype, true));
        columns.insert(col_name.clone(), PyColumn::new(col_name.clone(), array));
        column_order.push(col_name.clone());
    }

    Ok(PyDataFrame {
        schema: Arc::new(Schema::new(fields)),
        columns,
        column_order,
        num_rows,
    })
}

/// Infer column type from string values
fn infer_column_type(values: &[String]) -> (ArrayRef, DataType) {
    // Sample values to determine type
    let sample_size = values.len().min(100);
    let samples: Vec<&str> = values
        .iter()
        .take(sample_size)
        .map(|s| s.as_str())
        .collect();

    // Try to parse as integers
    let all_ints = samples
        .iter()
        .all(|s| s.is_empty() || s.parse::<i64>().is_ok());
    if all_ints && samples.iter().any(|s| !s.is_empty()) {
        let int_values: Vec<Option<i64>> = values
            .iter()
            .map(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse::<i64>().ok()
                }
            })
            .collect();
        return (Arc::new(Int64Array::from(int_values)), DataType::Int64);
    }

    // Try to parse as floats
    let all_floats = samples
        .iter()
        .all(|s| s.is_empty() || s.parse::<f64>().is_ok());
    if all_floats && samples.iter().any(|s| !s.is_empty()) {
        let float_values: Vec<Option<f64>> = values
            .iter()
            .map(|s| {
                if s.is_empty() {
                    None
                } else {
                    s.parse::<f64>().ok()
                }
            })
            .collect();
        return (
            Arc::new(Float64Array::from(float_values)),
            DataType::Float64,
        );
    }

    // Default to string
    let str_values: Vec<Option<&str>> = values
        .iter()
        .map(|s| if s.is_empty() { None } else { Some(s.as_str()) })
        .collect();
    (Arc::new(StringArray::from(str_values)), DataType::Utf8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::NamedTempFile;

    #[test]
    fn test_read_csv_with_header() {
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "name,age,score").unwrap();
        writeln!(file, "Alice,30,95.5").unwrap();
        writeln!(file, "Bob,25,88.0").unwrap();

        let df = read_csv(file.path().to_str().unwrap(), b',', true).unwrap();
        assert_eq!(df.num_rows, 2);
        assert_eq!(df.column_order, vec!["name", "age", "score"]);
    }

    #[test]
    fn test_type_inference() {
        let int_values = vec!["1".to_string(), "2".to_string(), "3".to_string()];
        let (_, dtype) = infer_column_type(&int_values);
        assert_eq!(dtype, DataType::Int64);

        let float_values = vec!["1.5".to_string(), "2.5".to_string(), "3.5".to_string()];
        let (_, dtype) = infer_column_type(&float_values);
        assert_eq!(dtype, DataType::Float64);

        let str_values = vec!["hello".to_string(), "world".to_string()];
        let (_, dtype) = infer_column_type(&str_values);
        assert_eq!(dtype, DataType::Utf8);
    }
}
