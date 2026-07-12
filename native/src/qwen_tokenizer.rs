use std::collections::HashSet;
use std::fs;
use std::sync::Mutex;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyIOError;
use pyo3::prelude::*;
use rustc_hash::FxHashMap as HashMap;
use tiktoken_rs::CoreBPE;
use unicode_normalization::UnicodeNormalization;

const PAT_STR: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

static TOKENIZER: OnceCell<Mutex<CoreBPE>> = OnceCell::new();
static VOCAB_PATH: OnceCell<String> = OnceCell::new();
static ALLOWED_SPECIAL: OnceCell<HashSet<String>> = OnceCell::new();

fn special_tokens() -> HashMap<String, u32> {
    let mut tokens = HashMap::default();
    tokens.insert("<|endoftext|>".to_string(), 151_643);
    tokens.insert("<|im_start|>".to_string(), 151_644);
    tokens.insert("<|im_end|>".to_string(), 151_645);
    for index in 0..205 {
        tokens.insert(format!("<|extra_{index}|>"), 151_646 + index);
    }
    tokens
}

fn allowed_special_tokens() -> &'static HashSet<String> {
    ALLOWED_SPECIAL.get_or_init(|| special_tokens().into_keys().collect())
}

fn load_bpe(vocab_path: &str) -> PyResult<CoreBPE> {
    let contents = fs::read(vocab_path)
        .map_err(|error| PyIOError::new_err(format!("failed to read vocab file: {error}")))?;
    let mut mergeable_ranks = HashMap::default();
    for line in contents.split(|byte| *byte == b'\n') {
        if line.is_empty() {
            continue;
        }
        let mut parts = line.splitn(2, |byte| *byte == b' ');
        let Some(token_bytes) = parts.next() else {
            continue;
        };
        let Some(rank_bytes) = parts.next() else {
            continue;
        };
        let token = STANDARD
            .decode(token_bytes)
            .map_err(|error| PyIOError::new_err(format!("invalid vocab token: {error}")))?;
        let rank = std::str::from_utf8(rank_bytes)
            .map_err(|error| PyIOError::new_err(format!("invalid vocab rank: {error}")))?
            .parse::<u32>()
            .map_err(|error| PyIOError::new_err(format!("invalid vocab rank: {error}")))?;
        mergeable_ranks.insert(token, rank);
    }
    CoreBPE::new(mergeable_ranks, special_tokens(), PAT_STR)
        .map_err(|error| PyIOError::new_err(format!("failed to build Qwen tokenizer: {error}")))
}

fn tokenizer() -> PyResult<&'static Mutex<CoreBPE>> {
    if TOKENIZER.get().is_none() {
        let path = VOCAB_PATH.get().ok_or_else(|| {
            PyIOError::new_err("Qwen tokenizer is not initialized; call init_qwen_tokenizer first")
        })?;
        let bpe = load_bpe(path)?;
        let _ = TOKENIZER.set(Mutex::new(bpe));
    }
    Ok(TOKENIZER.get().expect("tokenizer initialized"))
}

pub fn count_qwen_tokens(text: &str) -> PyResult<usize> {
    let normalized: String = text.nfc().collect();
    let bpe = tokenizer()?.lock().expect("tokenizer lock");
    let allowed: HashSet<&str> = allowed_special_tokens()
        .iter()
        .map(String::as_str)
        .collect();
    let (tokens, _) = bpe.encode(&normalized, &allowed);
    Ok(tokens.len())
}

#[pyfunction]
pub fn init_qwen_tokenizer(vocab_path: &str) -> PyResult<()> {
    let _ = VOCAB_PATH.set(vocab_path.to_string());
    let bpe = load_bpe(vocab_path)?;
    if TOKENIZER.get().is_none() {
        let _ = TOKENIZER.set(Mutex::new(bpe));
    } else {
        *tokenizer()?.lock().expect("tokenizer lock") = bpe;
    }
    Ok(())
}

#[pyfunction(name = "count_qwen_tokens")]
pub fn count_qwen_tokens_py(text: &str) -> PyResult<usize> {
    count_qwen_tokens(text)
}
