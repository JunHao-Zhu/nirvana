//! Nirvana Core - Rust-based columnar storage manager
//!
//! This crate provides high-performance in-memory columnar storage for the nirvana
//! DataFrame library, using Apache Arrow as the underlying data format.

mod dataframe;
mod dtype;
mod error;
mod io;

use crate::dtype::{infer, NirvanaDataType};
use pyo3::prelude::*;

/// Load an image file and convert to data URL format for OpenAI
#[pyfunction]
fn load_image(path: &str) -> PyResult<String> {
    infer::load_image_as_base64(path).map_err(|e| e.into())
}

/// Load an audio file and convert to base64 for OpenAI
#[pyfunction]
fn load_audio(path: &str) -> PyResult<String> {
    infer::load_audio_as_base64(path).map_err(|e| e.into())
}

/// Infer media type from a file path or URL
/// Returns: "image", "audio", or "text"
#[pyfunction]
fn infer_media_type(value: &str) -> String {
    match infer::infer_media_type(value) {
        NirvanaDataType::Image(_) => "image".to_string(),
        NirvanaDataType::Audio(_) => "audio".to_string(),
        NirvanaDataType::Arrow(_) => "text".to_string(),
    }
}

/// Python module for nirvana_core
#[pymodule]
fn nirvana_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<dataframe::PyDataFrame>()?;
    m.add_class::<dataframe::PyColumn>()?;
    m.add_function(wrap_pyfunction!(load_image, m)?)?;
    m.add_function(wrap_pyfunction!(load_audio, m)?)?;
    m.add_function(wrap_pyfunction!(infer_media_type, m)?)?;
    Ok(())
}
