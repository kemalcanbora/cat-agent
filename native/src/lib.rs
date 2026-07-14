mod doc_chunker;
mod hash_embed;
mod keyword_tokenizer;
mod message_truncation;
mod pdf;
mod qwen_tokenizer;
mod rag_index;
mod stop_words;
mod vector_index;

use pyo3::prelude::*;

use doc_chunker::split_doc_to_chunks_py;
use hash_embed::hash_embed_py;
use keyword_tokenizer::{
    clean_en_token_py, split_text_into_keywords_py, stem_words_py, string_tokenizer_py,
    tokenize_and_filter_py,
};
use message_truncation::truncate_messages_py;
use pdf::parse_pdf_text_py;
use qwen_tokenizer::{
    batch_count_qwen_tokens_py, count_qwen_tokens_py, decode_qwen_tokens_py, encode_qwen_tokens_py,
    init_qwen_tokenizer, truncate_qwen_text_py,
};
use rag_index::RagIndex;
use stop_words::stop_words_list;
use vector_index::VectorIndex;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RagIndex>()?;
    module.add_class::<VectorIndex>()?;
    module.add_function(wrap_pyfunction!(parse_pdf_text_py, module)?)?;
    module.add_function(wrap_pyfunction!(string_tokenizer_py, module)?)?;
    module.add_function(wrap_pyfunction!(split_text_into_keywords_py, module)?)?;
    module.add_function(wrap_pyfunction!(tokenize_and_filter_py, module)?)?;
    module.add_function(wrap_pyfunction!(clean_en_token_py, module)?)?;
    module.add_function(wrap_pyfunction!(stem_words_py, module)?)?;
    module.add_function(wrap_pyfunction!(init_qwen_tokenizer, module)?)?;
    module.add_function(wrap_pyfunction!(count_qwen_tokens_py, module)?)?;
    module.add_function(wrap_pyfunction!(encode_qwen_tokens_py, module)?)?;
    module.add_function(wrap_pyfunction!(decode_qwen_tokens_py, module)?)?;
    module.add_function(wrap_pyfunction!(truncate_qwen_text_py, module)?)?;
    module.add_function(wrap_pyfunction!(batch_count_qwen_tokens_py, module)?)?;
    module.add_function(wrap_pyfunction!(hash_embed_py, module)?)?;
    module.add_function(wrap_pyfunction!(split_doc_to_chunks_py, module)?)?;
    module.add_function(wrap_pyfunction!(truncate_messages_py, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("WORDS_TO_IGNORE", stop_words_list())?;
    Ok(())
}
