pub mod audio;
pub mod image;
pub mod infer;

use self::audio::AudioDtype;
use self::image::ImageDtype;
use arrow::datatypes::DataType;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NirvanaDataType {
    Arrow(DataType),
    Image(ImageDtype),
    Audio(AudioDtype),
}

impl NirvanaDataType {
    pub fn is_image(&self) -> bool {
        matches!(self, NirvanaDataType::Image(_))
    }

    pub fn is_audio(&self) -> bool {
        matches!(self, NirvanaDataType::Audio(_))
    }

    pub fn to_arrow(&self) -> DataType {
        match self {
            NirvanaDataType::Arrow(dt) => dt.clone(),
            NirvanaDataType::Image(dt) => dt.arrow_dtype(),
            NirvanaDataType::Audio(dt) => dt.arrow_dtype(),
        }
    }
}

impl From<DataType> for NirvanaDataType {
    fn from(dt: DataType) -> Self {
        NirvanaDataType::Arrow(dt)
    }
}

#[cfg(test)]
mod tests {
    use self::infer::infer_dtype_from_strings;
    use super::*;

    #[test]
    fn test_infer_column_media_type() {
        let image_col = vec![
            "/path/image1.jpg".to_string(),
            "/path/image2.png".to_string(),
        ];
        assert_eq!(
            infer_dtype_from_strings(&image_col),
            NirvanaDataType::Image(ImageDtype::new())
        );

        let text_col = vec!["hello".to_string(), "world".to_string()];
        assert_eq!(
            infer_dtype_from_strings(&text_col),
            NirvanaDataType::Arrow(DataType::Utf8)
        );
    }
}
