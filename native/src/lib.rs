use std::collections::HashMap;
use std::fs;

use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use serde::{Deserialize, Serialize};

#[pyclass(module = "cat_agent._native", skip_from_py_object)]
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct RagIndex {
    term_frequencies: Vec<HashMap<String, usize>>,
    document_lengths: Vec<usize>,
    idf: HashMap<String, f64>,
    average_document_length: f64,
    k1: f64,
    b: f64,
    epsilon: f64,
}

impl RagIndex {
    fn build(corpus: Vec<Vec<String>>) -> PyResult<Self> {
        if corpus.is_empty() {
            return Err(PyValueError::new_err(
                "corpus must contain at least one document",
            ));
        }

        let document_lengths: Vec<usize> = corpus.iter().map(Vec::len).collect();
        let average_document_length =
            document_lengths.iter().sum::<usize>() as f64 / corpus.len() as f64;
        let mut term_frequencies = Vec::with_capacity(corpus.len());
        let mut document_frequencies: HashMap<String, usize> = HashMap::new();

        for document in corpus {
            let mut frequencies = HashMap::new();
            for term in document {
                *frequencies.entry(term).or_insert(0) += 1;
            }
            for term in frequencies.keys() {
                *document_frequencies.entry(term.clone()).or_insert(0) += 1;
            }
            term_frequencies.push(frequencies);
        }

        let document_count = term_frequencies.len() as f64;
        let mut idf = HashMap::with_capacity(document_frequencies.len());
        let mut idf_sum = 0.0;
        let mut negative_terms = Vec::new();
        for (term, frequency) in document_frequencies {
            let frequency = frequency as f64;
            let value = ((document_count - frequency + 0.5) / (frequency + 0.5)).ln();
            if value < 0.0 {
                negative_terms.push(term.clone());
            }
            idf_sum += value;
            idf.insert(term, value);
        }

        let epsilon = 0.25;
        if !idf.is_empty() {
            let average_idf = idf_sum / idf.len() as f64;
            let floor = epsilon * average_idf;
            for term in negative_terms {
                idf.insert(term, floor);
            }
        }

        Ok(Self {
            term_frequencies,
            document_lengths,
            idf,
            average_document_length,
            k1: 1.5,
            b: 0.75,
            epsilon,
        })
    }

    fn score_query(&self, query: &[String]) -> Vec<f64> {
        let mut scores = vec![0.0; self.term_frequencies.len()];
        if self.average_document_length == 0.0 {
            return scores;
        }

        for term in query {
            let Some(idf) = self.idf.get(term) else {
                continue;
            };
            for (index, frequencies) in self.term_frequencies.iter().enumerate() {
                let frequency = frequencies.get(term).copied().unwrap_or(0) as f64;
                let length_ratio =
                    self.document_lengths[index] as f64 / self.average_document_length;
                let denominator = frequency + self.k1 * (1.0 - self.b + self.b * length_ratio);
                if denominator > 0.0 {
                    scores[index] += idf * (frequency * (self.k1 + 1.0) / denominator);
                }
            }
        }
        scores
    }
}

#[pymethods]
impl RagIndex {
    #[new]
    fn new(corpus: Vec<Vec<String>>) -> PyResult<Self> {
        Self::build(corpus)
    }

    fn scores(&self, query: Vec<String>) -> Vec<f64> {
        self.score_query(&query)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        let bytes = serde_json::to_vec(self).map_err(|error| {
            PyIOError::new_err(format!("failed to serialize RAG index: {error}"))
        })?;
        fs::write(path, bytes)
            .map_err(|error| PyIOError::new_err(format!("failed to save RAG index: {error}")))
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let bytes = fs::read(path)
            .map_err(|error| PyIOError::new_err(format!("failed to read RAG index: {error}")))?;
        serde_json::from_slice(&bytes).map_err(|error| {
            PyIOError::new_err(format!("failed to deserialize RAG index: {error}"))
        })
    }

    fn __len__(&self) -> usize {
        self.term_frequencies.len()
    }
}

#[pyfunction]
fn parse_pdf_text(path: &str) -> PyResult<Vec<(u32, String)>> {
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

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RagIndex>()?;
    module.add_function(wrap_pyfunction!(parse_pdf_text, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranks_matching_document_first() {
        let index = RagIndex::build(vec![
            vec!["machine".into(), "learning".into()],
            vec!["python".into(), "programming".into()],
            vec!["machine".into(), "vision".into()],
        ])
        .unwrap();
        let scores = index.score_query(&["python".into()]);
        assert!(scores[1] > scores[0]);
        assert!(scores[1] > scores[2]);
    }

    #[test]
    fn serialized_index_preserves_scores() {
        let index =
            RagIndex::build(vec![vec!["one".into(), "two".into()], vec!["three".into()]]).unwrap();
        let path = std::env::temp_dir().join(format!(
            "cat-agent-rag-index-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let bytes = serde_json::to_vec(&index).unwrap();
        fs::write(&path, bytes).unwrap();
        let restored: RagIndex = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        fs::remove_file(path).unwrap();
        assert_eq!(
            index.score_query(&["three".into()]),
            restored.score_query(&["three".into()])
        );
    }
}
