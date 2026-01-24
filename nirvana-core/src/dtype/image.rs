use arrow::datatypes::DataType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ImageDtype;

impl ImageDtype {
    pub fn new() -> Self {
        Self
    }

    pub fn arrow_dtype(&self) -> DataType {
        DataType::Utf8
    }
}
