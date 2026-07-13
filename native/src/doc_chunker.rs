use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use regex::Regex;
use std::sync::OnceLock;

use crate::qwen_tokenizer::{count_qwen_tokens, decode_qwen_tokens, encode_qwen_tokens};

#[derive(Clone, Debug)]
struct Paragraph {
    text: String,
    token: usize,
}

#[derive(Clone, Debug)]
pub(crate) struct Page {
    page_num: i64,
    content: Vec<Paragraph>,
}

#[derive(Clone, Debug)]
enum ChunkPart {
    PageMarker(String),
    Paragraph { text: String, page_num: i64 },
}

#[derive(Clone, Debug)]
pub(crate) struct ChunkOutput {
    content: String,
    source: String,
    title: String,
    chunk_id: usize,
    token: usize,
}

fn page_marker_re() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| Regex::new(r"^\[page: \d+\]$").expect("valid page marker regex"))
}

fn parse_doc(doc: &Bound<'_, PyAny>) -> PyResult<Vec<Page>> {
    let list = doc.cast::<PyList>()?;
    let mut pages = Vec::with_capacity(list.len());
    for item in list.iter() {
        let page_dict = item.cast::<PyDict>()?;
        let page_num: i64 = page_dict
            .get_item("page_num")?
            .ok_or_else(|| PyValueError::new_err("page_num missing"))?
            .extract()?;
        let content_item = page_dict
            .get_item("content")?
            .ok_or_else(|| PyValueError::new_err("content missing"))?;
        let content_list = content_item.cast::<PyList>()?;
        let mut content = Vec::with_capacity(content_list.len());
        for para_item in content_list.iter() {
            let para_dict = para_item.cast::<PyDict>()?;
            let text: String = if let Some(value) = para_dict.get_item("text")? {
                value.extract().unwrap_or_default()
            } else if let Some(value) = para_dict.get_item("table")? {
                value.extract().unwrap_or_default()
            } else {
                String::new()
            };
            let token: usize = para_dict
                .get_item("token")?
                .map(|value| value.extract().unwrap_or(0))
                .unwrap_or_else(|| count_qwen_tokens(&text).unwrap_or(0));
            content.push(Paragraph { text, token });
        }
        pages.push(Page { page_num, content });
    }
    Ok(pages)
}

fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();
    let mut index = 0;
    while index < chars.len() {
        current.push(chars[index]);
        let boundary = if chars[index] == '。' {
            true
        } else if chars[index] == '.' && index + 1 < chars.len() && chars[index + 1] == ' ' {
            current.push(chars[index + 1]);
            index += 1;
            true
        } else {
            false
        };
        if boundary {
            let trimmed = current.trim().to_string();
            if !trimmed.is_empty() {
                sentences.push(trimmed);
            }
            current.clear();
        }
        index += 1;
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        sentences.push(trimmed);
    }
    sentences
}

fn get_last_part(chunk: &[ChunkPart]) -> PyResult<String> {
    let need_page = match chunk.last() {
        Some(ChunkPart::Paragraph { page_num, .. }) => *page_num,
        _ => return Ok(String::new()),
    };
    let mut overlap = String::new();
    let mut available_len = 150usize;
    for part in chunk.iter().rev() {
        let ChunkPart::Paragraph {
            text: para,
            page_num,
        } = part
        else {
            continue;
        };
        if *page_num != need_page {
            return Ok(overlap);
        }
        if para.len() <= available_len {
            if overlap.is_empty() {
                overlap.clone_from(para);
            } else {
                overlap = format!("{para}\n{overlap}");
            }
            available_len = available_len.saturating_sub(para.len());
            continue;
        }
        let sentence_split_symbol = if para.contains('。') { '。' } else { '.' };
        let sentences = split_sentences(para);
        for sentence in sentences.into_iter().rev() {
            let sent = sentence.trim();
            if sent.is_empty() {
                continue;
            }
            if sent.len() <= available_len {
                if overlap.is_empty() {
                    overlap = sent.to_string();
                } else if sentence_split_symbol == '。' {
                    overlap = format!("{sent}{sentence_split_symbol}{overlap}");
                } else {
                    overlap = format!("{sent}. {overlap}");
                }
                available_len = available_len.saturating_sub(sent.len());
            } else {
                return Ok(overlap);
            }
        }
    }
    Ok(overlap)
}

struct FinalizeChunkParams<'a> {
    path: &'a str,
    title: &'a str,
    chunk_id: usize,
    parser_page_size: usize,
    available_token: usize,
    paragraph_split_symbol: &'a str,
}

fn finalize_chunk(
    chunk: &[ChunkPart],
    params: FinalizeChunkParams<'_>,
    results: &mut Vec<ChunkOutput>,
) -> PyResult<()> {
    let mut parts = chunk.to_vec();
    if matches!(parts.last(), Some(ChunkPart::PageMarker(marker)) if page_marker_re().is_match(marker))
    {
        parts.pop();
    }
    let content = parts
        .iter()
        .map(|part| match part {
            ChunkPart::PageMarker(marker) => marker.clone(),
            ChunkPart::Paragraph { text, .. } => text.clone(),
        })
        .collect::<Vec<_>>()
        .join(params.paragraph_split_symbol);
    results.push(ChunkOutput {
        content,
        source: params.path.to_string(),
        title: params.title.to_string(),
        chunk_id: params.chunk_id,
        token: params
            .parser_page_size
            .saturating_sub(params.available_token),
    });
    Ok(())
}

fn split_long_paragraph(text: &str, available_token: usize) -> PyResult<Vec<(String, usize)>> {
    let mut sentences = Vec::new();
    for sentence in split_sentences(text) {
        let token = count_qwen_tokens(&sentence)?;
        if sentence.trim().is_empty() || token == 0 {
            continue;
        }
        if token <= available_token {
            sentences.push((sentence, token));
        } else {
            let token_ids = encode_qwen_tokens(&sentence)?;
            let mut start = 0usize;
            while start < token_ids.len() {
                let end = (start + available_token).min(token_ids.len());
                let piece = decode_qwen_tokens(&token_ids[start..end])?;
                let piece_tokens = end - start;
                sentences.push((piece, piece_tokens));
                start = end;
            }
        }
    }
    Ok(sentences)
}

pub fn split_doc_to_chunks(
    doc: &[Page],
    path: &str,
    title: &str,
    parser_page_size: usize,
    paragraph_split_symbol: &str,
) -> PyResult<Vec<ChunkOutput>> {
    let mut results = Vec::new();
    let mut chunk: Vec<ChunkPart> = Vec::new();
    let mut available_token = parser_page_size;
    let mut has_para = false;

    for page in doc {
        let page_num = page.page_num;
        let marker = format!("[page: {page_num}]");
        if chunk.is_empty()
            || !matches!(chunk.first(), Some(ChunkPart::PageMarker(existing)) if existing == &marker)
        {
            chunk.push(ChunkPart::PageMarker(marker));
        }

        let mut idx = 0usize;
        while idx < page.content.len() {
            if chunk.is_empty() {
                chunk.push(ChunkPart::PageMarker(format!("[page: {page_num}]")));
            }
            let para = &page.content[idx];
            let token = para.token;
            if token <= available_token {
                available_token -= token;
                chunk.push(ChunkPart::Paragraph {
                    text: para.text.clone(),
                    page_num,
                });
                has_para = true;
                idx += 1;
            } else if has_para {
                finalize_chunk(
                    &chunk,
                    FinalizeChunkParams {
                        path,
                        title,
                        chunk_id: results.len(),
                        parser_page_size,
                        available_token,
                        paragraph_split_symbol,
                    },
                    &mut results,
                )?;
                let overlap_txt = get_last_part(&chunk)?;
                if !overlap_txt.trim().is_empty() {
                    let overlap_page = match chunk.last() {
                        Some(ChunkPart::Paragraph { page_num, .. }) => *page_num,
                        _ => page_num,
                    };
                    chunk = vec![
                        ChunkPart::PageMarker(format!("[page: {overlap_page}]")),
                        ChunkPart::Paragraph {
                            text: overlap_txt.clone(),
                            page_num: overlap_page,
                        },
                    ];
                    has_para = false;
                    available_token =
                        parser_page_size.saturating_sub(count_qwen_tokens(&overlap_txt)?);
                } else {
                    chunk.clear();
                    has_para = false;
                    available_token = parser_page_size;
                }
            } else {
                let sentences = split_long_paragraph(&para.text, available_token)?;
                let mut sent_index = 0usize;
                while sent_index < sentences.len() {
                    let (sentence, token) = sentences[sent_index].clone();
                    if chunk.is_empty() {
                        chunk.push(ChunkPart::PageMarker(format!("[page: {page_num}]")));
                    }
                    if token <= available_token || !has_para {
                        available_token = available_token.saturating_sub(token);
                        chunk.push(ChunkPart::Paragraph {
                            text: sentence,
                            page_num,
                        });
                        has_para = true;
                        sent_index += 1;
                    } else {
                        finalize_chunk(
                            &chunk,
                            FinalizeChunkParams {
                                path,
                                title,
                                chunk_id: results.len(),
                                parser_page_size,
                                available_token,
                                paragraph_split_symbol,
                            },
                            &mut results,
                        )?;
                        let overlap_txt = get_last_part(&chunk)?;
                        if !overlap_txt.trim().is_empty() {
                            let overlap_page = match chunk.last() {
                                Some(ChunkPart::Paragraph { page_num, .. }) => *page_num,
                                _ => page_num,
                            };
                            chunk = vec![
                                ChunkPart::PageMarker(format!("[page: {overlap_page}]")),
                                ChunkPart::Paragraph {
                                    text: overlap_txt.clone(),
                                    page_num: overlap_page,
                                },
                            ];
                            has_para = false;
                            available_token =
                                parser_page_size.saturating_sub(count_qwen_tokens(&overlap_txt)?);
                        } else {
                            chunk.clear();
                            has_para = false;
                            available_token = parser_page_size;
                        }
                    }
                }
                idx += 1;
            }
        }
    }

    if has_para {
        finalize_chunk(
            &chunk,
            FinalizeChunkParams {
                path,
                title,
                chunk_id: results.len(),
                parser_page_size,
                available_token,
                paragraph_split_symbol,
            },
            &mut results,
        )?;
    }
    Ok(results)
}

#[pyfunction(name = "split_doc_to_chunks")]
pub fn split_doc_to_chunks_py<'py>(
    py: Python<'py>,
    doc: &Bound<'py, PyAny>,
    path: &str,
    title: &str,
    parser_page_size: usize,
    paragraph_split_symbol: &str,
) -> PyResult<Bound<'py, PyList>> {
    let pages = parse_doc(doc)?;
    let outputs = split_doc_to_chunks(
        &pages,
        path,
        title,
        parser_page_size,
        paragraph_split_symbol,
    )?;
    let list = PyList::empty(py);
    for output in outputs {
        let dict = PyDict::new(py);
        dict.set_item("content", output.content)?;
        let metadata = PyDict::new(py);
        metadata.set_item("source", output.source)?;
        metadata.set_item("title", output.title)?;
        metadata.set_item("chunk_id", output.chunk_id)?;
        dict.set_item("metadata", metadata)?;
        dict.set_item("token", output.token)?;
        list.append(dict)?;
    }
    Ok(list)
}
