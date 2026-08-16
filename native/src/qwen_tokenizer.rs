//! Heuristic token counting via OpenAI ``o200k_base`` (tiktoken-rs).
//!
//! Public Python names keep the historical ``qwen_*`` prefix for ABI stability;
//! the encoding is no longer the bundled Qwen BPE table.

use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use tiktoken_rs::o200k_base_singleton;
use unicode_normalization::UnicodeNormalization;

fn encode_normalized(text: &str) -> Vec<u32> {
    let normalized: String = text.nfc().collect();
    o200k_base_singleton().encode_with_special_tokens(&normalized)
}

pub fn count_qwen_tokens(text: &str) -> PyResult<usize> {
    Ok(encode_normalized(text).len())
}

pub fn encode_qwen_tokens(text: &str) -> PyResult<Vec<u32>> {
    Ok(encode_normalized(text))
}

pub fn decode_qwen_tokens(token_ids: &[u32]) -> PyResult<String> {
    o200k_base_singleton()
        .decode(token_ids.to_vec())
        .map_err(|error| PyIOError::new_err(format!("failed to decode tokens: {error}")))
}

pub fn truncate_qwen_text(text: &str, max_token: usize, keep_both_sides: bool) -> PyResult<String> {
    let mut token_ids = encode_normalized(text);
    if token_ids.len() <= max_token {
        return decode_qwen_tokens(&token_ids);
    }

    if keep_both_sides {
        let ellipsis = encode_normalized("...");
        let ellipsis_len = ellipsis.len();
        let available = max_token.saturating_sub(ellipsis_len);
        if available == 0 {
            token_ids.truncate(max_token);
            return decode_qwen_tokens(&token_ids);
        }
        let left_len = available / 2;
        let right_len = available - left_len;
        let tail_start = token_ids.len().saturating_sub(right_len);
        let mut merged = Vec::with_capacity(left_len + ellipsis_len + right_len);
        merged.extend_from_slice(&token_ids[..left_len]);
        merged.extend_from_slice(&ellipsis);
        merged.extend_from_slice(&token_ids[tail_start..]);
        return decode_qwen_tokens(&merged);
    }

    token_ids.truncate(max_token);
    decode_qwen_tokens(&token_ids)
}

/// No-op kept for callers that still invoke init before count/truncate.
/// ``vocab_path`` is ignored; encoding is the built-in ``o200k_base``.
#[pyfunction]
pub fn init_qwen_tokenizer(_vocab_path: &str) -> PyResult<()> {
    let _ = o200k_base_singleton();
    Ok(())
}

#[pyfunction(name = "count_qwen_tokens")]
pub fn count_qwen_tokens_py(text: &str) -> PyResult<usize> {
    count_qwen_tokens(text)
}

#[pyfunction(name = "encode_qwen_tokens")]
pub fn encode_qwen_tokens_py(text: &str) -> PyResult<Vec<u32>> {
    encode_qwen_tokens(text)
}

#[pyfunction(name = "decode_qwen_tokens")]
pub fn decode_qwen_tokens_py(token_ids: Vec<u32>) -> PyResult<String> {
    decode_qwen_tokens(&token_ids)
}

#[pyfunction(name = "truncate_qwen_text")]
pub fn truncate_qwen_text_py(
    text: &str,
    max_token: usize,
    keep_both_sides: bool,
) -> PyResult<String> {
    truncate_qwen_text(text, max_token, keep_both_sides)
}

#[pyfunction(name = "batch_count_qwen_tokens")]
pub fn batch_count_qwen_tokens_py(texts: Vec<String>) -> PyResult<Vec<usize>> {
    texts.iter().map(|text| count_qwen_tokens(text)).collect()
}
