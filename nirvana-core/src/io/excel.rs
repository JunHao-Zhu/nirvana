//! Excel file reading using calamine

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use calamine::{open_workbook, Data, Reader, Xlsx};

use crate::dataframe::{PyColumn, PyDataFrame};
use crate::error::{NirvanaError, Result};

/// Read an Excel file into a PyDataFrame
pub fn read_excel(path: &str, sheet_name: Option<&str>) -> Result<PyDataFrame> {
    let mut workbook: Xlsx<_> = open_workbook(path)?;

    // Get sheet names
    let sheet_names = workbook.sheet_names().to_vec();
    if sheet_names.is_empty() {
        return Err(NirvanaError::InvalidOperation(
            "Excel file has no sheets".to_string(),
        ));
    }

    // Select sheet
    let target_sheet = match sheet_name {
        Some(name) => name.to_string(),
        None => sheet_names[0].clone(),
    };

    // Get range - worksheet_range returns Result in newer calamine versions
    let range = workbook
        .worksheet_range(&target_sheet)
        .map_err(|e| NirvanaError::Excel(e.to_string()))?;

    if range.is_empty() {
        return Ok(PyDataFrame::new());
    }

    // Get dimensions
    let (height, width) = range.get_size();
    if height == 0 || width == 0 {
        return Ok(PyDataFrame::new());
    }

    // First row is headers
    let headers: Vec<String> = (0..width)
        .map(|col| {
            range
                .get((0, col))
                .map(|cell| cell_to_string(cell))
                .unwrap_or_else(|| format!("col_{}", col))
        })
        .collect();

    // Read data rows
    let num_rows = height - 1;
    let mut column_data: Vec<Vec<Data>> = vec![Vec::with_capacity(num_rows); width];

    for row in 1..height {
        for col in 0..width {
            let cell = range.get((row, col)).cloned().unwrap_or(Data::Empty);
            column_data[col].push(cell);
        }
    }

    // Convert to Arrow arrays
    let mut columns = HashMap::new();
    let mut column_order = Vec::new();
    let mut fields = Vec::new();

    for (i, col_name) in headers.iter().enumerate() {
        let cells = &column_data[i];
        let (array, dtype) = cells_to_arrow_array(cells);

        fields.push(Field::new(col_name, dtype, true));
        columns.insert(
            col_name.clone(),
            PyColumn::new(col_name.clone(), array),
        );
        column_order.push(col_name.clone());
    }

    Ok(PyDataFrame {
        schema: Arc::new(Schema::new(fields)),
        columns,
        column_order,
        num_rows,
    })
}

fn cell_to_string(cell: &Data) -> String {
    match cell {
        Data::Int(i) => i.to_string(),
        Data::Float(f) => f.to_string(),
        Data::String(s) => s.clone(),
        Data::Bool(b) => b.to_string(),
        Data::DateTime(dt) => dt.to_string(),
        Data::DateTimeIso(s) => s.clone(),
        Data::DurationIso(s) => s.clone(),
        Data::Error(e) => format!("Error: {:?}", e),
        Data::Empty => String::new(),
    }
}

fn cells_to_arrow_array(cells: &[Data]) -> (ArrayRef, DataType) {
    // Infer type from first non-empty cell
    let first_non_empty = cells.iter().find(|c| !matches!(c, Data::Empty));

    match first_non_empty {
        Some(Data::Int(_)) => {
            let values: Vec<Option<i64>> = cells
                .iter()
                .map(|c| match c {
                    Data::Int(i) => Some(*i),
                    Data::Float(f) => Some(*f as i64),
                    Data::Empty => None,
                    _ => None,
                })
                .collect();
            (Arc::new(Int64Array::from(values)), DataType::Int64)
        }
        Some(Data::Float(_)) => {
            let values: Vec<Option<f64>> = cells
                .iter()
                .map(|c| match c {
                    Data::Float(f) => Some(*f),
                    Data::Int(i) => Some(*i as f64),
                    Data::Empty => None,
                    _ => None,
                })
                .collect();
            (Arc::new(Float64Array::from(values)), DataType::Float64)
        }
        _ => {
            // Default to string
            let values: Vec<Option<String>> = cells
                .iter()
                .map(|c| match c {
                    Data::Empty => None,
                    _ => Some(cell_to_string(c)),
                })
                .collect();
            (Arc::new(StringArray::from(values)), DataType::Utf8)
        }
    }
}
