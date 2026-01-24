use std::fs;
use std::io::Read;
use std::path::Path;

use crate::dtype::{audio::AudioDtype, image::ImageDtype, NirvanaDataType};
use crate::error::{NirvanaError, Result};
use arrow::datatypes::DataType;

pub const IMAGE_EXTENSIONS: &[&str] = &[
    ".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp", ".svg", ".ico", ".jfif",
];

pub const AUDIO_EXTENSIONS: &[&str] = &[
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".opus",
];

fn is_image(value: &str) -> bool {
    let lower = value.to_lowercase();
    IMAGE_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

fn is_audio(value: &str) -> bool {
    let lower = value.to_lowercase();
    AUDIO_EXTENSIONS.iter().any(|ext| lower.ends_with(ext))
}

/// Infer media type from a file path or URL
pub fn infer_media_type(value: &str) -> NirvanaDataType {
    let lower = value.to_lowercase();

    // Check for image extensions
    for ext in IMAGE_EXTENSIONS {
        if lower.ends_with(ext) {
            return NirvanaDataType::Image(ImageDtype::new());
        }
    }

    // Check for audio extensions
    for ext in AUDIO_EXTENSIONS {
        if lower.ends_with(ext) {
            return NirvanaDataType::Audio(AudioDtype::new());
        }
    }

    NirvanaDataType::Arrow(DataType::Utf8)
}

/// Infer data type from a list of strings
pub fn infer_dtype_from_strings(values: &[String]) -> NirvanaDataType {
    for value in values.iter() {
        if value.is_empty() {
            continue;
        }

        if is_image(value) {
            return NirvanaDataType::Image(ImageDtype::new());
        }

        if is_audio(value) {
            return NirvanaDataType::Audio(AudioDtype::new());
        }
    }

    NirvanaDataType::Arrow(DataType::Utf8)
}

/// Convert an image file path or URL to a base64 format for OpenAI
///
/// Returns:
/// - For local files: "data:image/png;base64,{base64_content}"
/// - For HTTPS URLs: returns URL as-is (OpenAI supports direct URLs)
/// - For data URLs: returns as-is
/// - For S3 URLs: currently returns as-is (could be extended)
pub fn load_image_as_base64(path: &str) -> Result<String> {
    if path.is_empty() {
        return Ok(String::new());
    }

    // Already a data URL
    if path.starts_with("data:image") {
        return Ok(path.to_string());
    }

    // HTTPS URL - OpenAI accepts these directly
    if path.starts_with("https://") {
        return Ok(path.to_string());
    }

    // S3 URL - return as-is for now (could download)
    if path.starts_with("s3://") {
        // For S3, we'd need to download. For now, return as-is
        return Err(NirvanaError::InvalidOperation(
            "S3 URLs require additional handling. Please use local files or HTTPS URLs."
                .to_string(),
        ));
    }

    // Local file - read and encode
    let file_path = Path::new(path);
    if !file_path.exists() {
        return Err(NirvanaError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Image file not found: {}", path),
        )));
    }

    let mut file = fs::File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    // Detect MIME type from extension
    let extension = file_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("png")
        .to_lowercase();

    let mime_type = match extension.as_str() {
        "jpg" | "jpeg" | "jfif" => "image/jpeg",
        "png" => "image/png",
        "gif" => "image/gif",
        "webp" => "image/webp",
        _ => "image/png",
    };

    use base64::Engine;
    let encoded = base64::engine::general_purpose::STANDARD.encode(&buffer);
    Ok(format!("data:{};base64,{}", mime_type, encoded))
}

/// Convert an audio file path or URL to base64 format for OpenAI
///
/// Returns the base64-encoded audio content
pub fn load_audio_as_base64(path: &str) -> Result<String> {
    if path.is_empty() {
        return Ok(String::new());
    }

    // HTTPS URL - fetch and encode
    if path.starts_with("https://") || path.starts_with("http://") {
        let mut response = reqwest::blocking::get(path).map_err(|e| {
            NirvanaError::InvalidOperation(format!("Failed to fetch audio from URL: {}", e))
        })?;

        let mut buffer = Vec::new();
        response.read_to_end(&mut buffer)?;

        use base64::Engine;
        return Ok(base64::engine::general_purpose::STANDARD.encode(&buffer));
    }

    // Local file - read and encode
    let file_path = Path::new(path);
    if !file_path.exists() {
        return Err(NirvanaError::Io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            format!("Audio file not found: {}", path),
        )));
    }

    let mut file = fs::File::open(file_path)?;
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer)?;

    use base64::Engine;
    Ok(base64::engine::general_purpose::STANDARD.encode(&buffer))
}
