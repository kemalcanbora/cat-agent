use num_traits::ToPrimitive;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use sha2::{Digest, Sha256};

use crate::keyword_tokenizer::split_text_into_keywords;

const MAX_TEXT_CHARS: usize = 2000;

fn token_bucket(token: &str, dimensions: usize) -> usize {
    let digest = Sha256::digest(token.as_bytes());
    let value = num_bigint::BigUint::from_bytes_be(&digest);
    let rem = value % dimensions;
    rem.to_usize().unwrap_or(0)
}

fn truncate_text(text: &str) -> &str {
    if text.chars().count() <= MAX_TEXT_CHARS {
        return text;
    }
    let end = text
        .char_indices()
        .nth(MAX_TEXT_CHARS)
        .map(|(index, _)| index)
        .unwrap_or(text.len());
    &text[..end]
}

fn hash_embed_text(text: &str, dimensions: usize) -> Vec<f32> {
    let mut vector = vec![0.0f32; dimensions];
    let truncated = truncate_text(text);

    for token in split_text_into_keywords(truncated) {
        let index = token_bucket(&token, dimensions);
        vector[index] += 1.0;
    }

    let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
    if norm > 0.0 {
        for value in &mut vector {
            *value /= norm;
        }
    }
    vector
}

pub fn hash_embed_texts(texts: &[String], dimensions: usize) -> PyResult<Vec<Vec<f32>>> {
    if dimensions == 0 {
        return Err(PyValueError::new_err(
            "dimensions must be greater than zero",
        ));
    }
    Ok(texts
        .iter()
        .map(|text| hash_embed_text(text, dimensions))
        .collect())
}

#[pyfunction(name = "hash_embed")]
#[pyo3(signature = (texts, dimensions = 384))]
pub fn hash_embed_py(texts: Vec<String>, dimensions: usize) -> PyResult<Vec<Vec<f32>>> {
    hash_embed_texts(&texts, dimensions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_text_yields_zero_vector() {
        let vector = hash_embed_text("", 8);
        assert_eq!(vector, vec![0.0; 8]);
    }

    #[test]
    fn normalized_nonzero_vector() {
        let vector = hash_embed_text("machine learning retrieval", 64);
        let norm: f32 = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn truncates_long_input() {
        let long = format!("alpha beta gamma{}", " x".repeat(3000));
        let truncated: String = long.chars().take(MAX_TEXT_CHARS).collect();
        let embedded = hash_embed_text(&long, 128);
        let expected = hash_embed_text(&truncated, 128);
        assert_eq!(embedded, expected);
    }
}
