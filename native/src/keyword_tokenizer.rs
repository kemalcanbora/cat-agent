use std::sync::OnceLock;

use jieba_rs::Jieba;
use pyo3::prelude::*;
use regex::Regex;
use rust_stemmers::{Algorithm, Stemmer};

use crate::stop_words::{punctuation_chars, stop_words};

static EN_TOKEN_RE: OnceLock<Regex> = OnceLock::new();
static SPECIAL_CASE_RE: OnceLock<Regex> = OnceLock::new();
static CHINESE_RE: OnceLock<Regex> = OnceLock::new();
static JIEBA: OnceLock<Jieba> = OnceLock::new();
static EN_STEMMER: OnceLock<Stemmer> = OnceLock::new();

fn en_token_re() -> &'static Regex {
    EN_TOKEN_RE.get_or_init(|| {
        Regex::new(
            r"(?x)
            (?:[A-Za-z]\.)+
            |\d+(?:\.\d+)?%?
            |\w+(?:[-']\w+)*
            |(?:[\w\-']@)+\w+",
        )
        .expect("valid english token regex")
    })
}

fn special_case_re() -> &'static Regex {
    SPECIAL_CASE_RE.get_or_init(|| {
        Regex::new(r"^(?:[A-Za-z]\.)+|\w+[@]\w+\.\w+|\d+%$|^(?:[\u4e00-\u9fff]+)$")
            .expect("valid special-case regex")
    })
}

fn chinese_re() -> &'static Regex {
    CHINESE_RE.get_or_init(|| {
        Regex::new(r"[\u4e00-\u9fff]").expect("valid chinese regex")
    })
}

fn jieba() -> &'static Jieba {
    JIEBA.get_or_init(Jieba::new)
}

fn english_stemmer() -> &'static Stemmer {
    EN_STEMMER.get_or_init(|| Stemmer::create(Algorithm::English))
}

fn has_chinese_chars(text: &str) -> bool {
    chinese_re().is_match(text)
}

fn is_punctuation_only(word: &str) -> bool {
    let punct = punctuation_chars();
    word.chars().all(|ch| punct.contains(ch))
}

pub fn clean_en_token(token: &str) -> String {
    if special_case_re().is_match(token) {
        return token.to_string();
    }
    token
        .trim_matches(|ch| punctuation_chars().contains(ch))
        .to_string()
}

pub fn tokenize_and_filter(input_text: &str) -> Vec<String> {
    let stop = stop_words();
    en_token_re()
        .find_iter(input_text)
        .map(|m| m.as_str())
        .map(|token| clean_en_token(token).to_lowercase())
        .filter(|token| !stop.contains(token.as_str()))
        .filter(|token| !is_punctuation_only(token))
        .collect()
}

fn stem_words(mut words: Vec<String>) -> Vec<String> {
    let stemmer = english_stemmer();
    for word in &mut words {
        *word = stemmer.stem(word).to_string();
    }
    words
}

pub fn string_tokenizer(text: &str) -> Vec<String> {
    let stop = stop_words();
    let text = text.to_lowercase();
    let text = text.trim();

    let mut wordlist: Vec<String> = if has_chinese_chars(text) {
        jieba()
            .cut(text, false)
            .into_iter()
            .filter(|word| !is_punctuation_only(word))
            .map(str::to_string)
            .collect()
    } else {
        tokenize_and_filter(text)
    };

    wordlist.retain(|word| !stop.contains(word.as_str()));
    stem_words(wordlist)
}

pub fn split_text_into_keywords(text: &str) -> Vec<String> {
    let stop = stop_words();
    string_tokenizer(text)
        .into_iter()
        .filter(|word| !stop.contains(word.as_str()))
        .collect()
}

#[pyfunction(name = "clean_en_token")]
pub fn clean_en_token_py(token: &str) -> String {
    clean_en_token(token)
}

#[pyfunction(name = "tokenize_and_filter")]
pub fn tokenize_and_filter_py(input_text: &str) -> Vec<String> {
    tokenize_and_filter(input_text)
}

#[pyfunction(name = "split_text_into_keywords")]
pub fn split_text_into_keywords_py(text: &str) -> Vec<String> {
    split_text_into_keywords(text)
}

#[pyfunction(name = "string_tokenizer")]
pub fn string_tokenizer_py(text: &str) -> Vec<String> {
    string_tokenizer(text)
}

#[pyfunction(name = "stem_words")]
pub fn stem_words_py(words: Vec<String>) -> Vec<String> {
    stem_words(words)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_stop_words() {
        let out = split_text_into_keywords("the quick brown");
        assert!(!out.contains(&"the".to_string()));
        assert!(out.iter().any(|w| w == "quick"));
        assert!(out.iter().any(|w| w == "brown"));
    }

    #[test]
    fn stems_english_words() {
        let out = split_text_into_keywords("running machines");
        assert!(out.iter().any(|w| w == "run" || w == "machin"));
    }
}
