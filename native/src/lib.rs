mod keyword_tokenizer;
mod pdf;
mod qwen_tokenizer;
mod rag_index;
mod stop_words;

use pyo3::prelude::*;

use keyword_tokenizer::{
    clean_en_token_py, split_text_into_keywords_py, stem_words_py, string_tokenizer_py,
    tokenize_and_filter_py,
};
use pdf::parse_pdf_text_py;
use qwen_tokenizer::{count_qwen_tokens_py, init_qwen_tokenizer};
use rag_index::RagIndex;
use stop_words::stop_words_list;

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<RagIndex>()?;
    module.add_function(wrap_pyfunction!(parse_pdf_text_py, module)?)?;
    module.add_function(wrap_pyfunction!(string_tokenizer_py, module)?)?;
    module.add_function(wrap_pyfunction!(split_text_into_keywords_py, module)?)?;
    module.add_function(wrap_pyfunction!(tokenize_and_filter_py, module)?)?;
    module.add_function(wrap_pyfunction!(clean_en_token_py, module)?)?;
    module.add_function(wrap_pyfunction!(stem_words_py, module)?)?;
    module.add_function(wrap_pyfunction!(init_qwen_tokenizer, module)?)?;
    module.add_function(wrap_pyfunction!(count_qwen_tokens_py, module)?)?;
    module.add("__version__", env!("CARGO_PKG_VERSION"))?;
    module.add("WORDS_TO_IGNORE", stop_words_list())?;
    Ok(())
}
