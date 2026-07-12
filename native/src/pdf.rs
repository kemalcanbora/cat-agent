use pyo3::exceptions::{PyIOError, PyRuntimeError};
use pyo3::prelude::*;

pub fn parse_pdf_text(path: &str) -> PyResult<Vec<(u32, String)>> {
    let document = lopdf::Document::load(path)
        .map_err(|error| PyIOError::new_err(format!("failed to open PDF: {error}")))?;
    let mut pages = Vec::with_capacity(document.get_pages().len());
    for page_number in document.get_pages().keys().copied() {
        let text = document.extract_text(&[page_number]).map_err(|error| {
            PyRuntimeError::new_err(format!(
                "failed to extract text from PDF page {page_number}: {error}"
            ))
        })?;
        pages.push((page_number, text));
    }
    Ok(pages)
}

#[pyfunction(name = "parse_pdf_text")]
pub fn parse_pdf_text_py(path: &str) -> PyResult<Vec<(u32, String)>> {
    parse_pdf_text(path)
}
