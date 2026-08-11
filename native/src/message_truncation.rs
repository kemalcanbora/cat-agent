use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;

use crate::qwen_tokenizer::{count_qwen_tokens, truncate_qwen_text};

/// Truncation carries structured tool-call fields alongside `text`.
/// `text` is used only for token estimation and may be omitted/truncated;
/// structured JSON fields survive the round-trip.
#[derive(Clone, Debug)]
pub(crate) struct NativeMessage {
    role: String,
    text: String,
    name: Option<String>,
    text_baseline: Option<String>,
    content_json: Option<String>,
    function_call_json: Option<String>,
    tool_calls_json: Option<String>,
    tool_call_id: Option<String>,
    extra_json: Option<String>,
    reasoning_content_json: Option<String>,
}

impl NativeMessage {
    fn with_text(&self, text: String) -> Self {
        Self {
            text,
            ..self.clone()
        }
    }
}

fn opt_json_field(dict: &Bound<'_, PyDict>, key: &str, py: Python<'_>) -> PyResult<Option<String>> {
    let Some(value) = dict.get_item(key)? else {
        return Ok(None);
    };
    if value.is_none() {
        return Ok(None);
    }
    // Always JSON-encode so dict/list/str round-trip uniformly through loads().
    let json = py.import("json")?;
    let dumped: String = json.call_method1("dumps", (value,))?.extract()?;
    Ok(Some(dumped))
}

fn set_json_field(
    dict: &Bound<'_, PyDict>,
    key: &str,
    raw: &Option<String>,
    py: Python<'_>,
    as_json: bool,
) -> PyResult<()> {
    let Some(value) = raw else {
        return Ok(());
    };
    if as_json {
        let json = py.import("json")?;
        let obj = json.call_method1("loads", (value,))?;
        dict.set_item(key, obj)?;
    } else {
        dict.set_item(key, value)?;
    }
    Ok(())
}

fn parse_messages(messages: &Bound<'_, PyAny>) -> PyResult<Vec<NativeMessage>> {
    let py = messages.py();
    let list = messages.cast::<PyList>()?;
    let mut parsed = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item.cast::<PyDict>()?;
        let role: String = dict
            .get_item("role")?
            .ok_or_else(|| PyValueError::new_err("role missing"))?
            .extract()?;
        let text: String = dict
            .get_item("text")?
            .map(|value| value.extract().unwrap_or_default())
            .unwrap_or_default();
        let name: Option<String> = dict
            .get_item("name")?
            .and_then(|value| value.extract().ok());
        let text_baseline: Option<String> = dict
            .get_item("text_baseline")?
            .and_then(|value| value.extract().ok());
        // content may be str or list/dict — always store as JSON when non-string
        let content_json = match dict.get_item("content")? {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => {
                if let Ok(s) = value.extract::<String>() {
                    Some(serde_json::to_string(&s).unwrap_or(s))
                } else {
                    let json = py.import("json")?;
                    Some(json.call_method1("dumps", (value,))?.extract()?)
                }
            }
        };
        let function_call_json = opt_json_field(dict, "function_call", py)?;
        let tool_calls_json = opt_json_field(dict, "tool_calls", py)?;
        let tool_call_id: Option<String> = dict
            .get_item("tool_call_id")?
            .and_then(|value| value.extract().ok());
        let extra_json = opt_json_field(dict, "extra", py)?;
        let reasoning_content_json = match dict.get_item("reasoning_content")? {
            None => None,
            Some(value) if value.is_none() => None,
            Some(value) => {
                if let Ok(s) = value.extract::<String>() {
                    Some(serde_json::to_string(&s).unwrap_or(s))
                } else {
                    let json = py.import("json")?;
                    Some(json.call_method1("dumps", (value,))?.extract()?)
                }
            }
        };
        parsed.push(NativeMessage {
            role,
            text,
            name,
            text_baseline,
            content_json,
            function_call_json,
            tool_calls_json,
            tool_call_id,
            extra_json,
            reasoning_content_json,
        });
    }
    Ok(parsed)
}

#[allow(clippy::collapsible_if)]
fn split_turn_into_steps(indexed_messages: &[IndexedMessage]) -> Vec<Vec<IndexedMessage>> {
    let mut steps: Vec<Vec<IndexedMessage>> = Vec::new();
    for indexed in indexed_messages {
        match indexed.message.role.as_str() {
            "user" => {
                if let Some(last_step) = steps.last_mut() {
                    if last_step.last().map(|item| item.message.role.as_str()) == Some("user") {
                        last_step.push(indexed.clone());
                        continue;
                    }
                }
                steps.push(vec![indexed.clone()]);
            }
            "assistant" => {
                if let Some(last_step) = steps.last_mut() {
                    if last_step.last().map(|item| item.message.role.as_str()) == Some("assistant")
                    {
                        last_step.push(indexed.clone());
                        continue;
                    }
                }
                steps.push(vec![indexed.clone()]);
            }
            "function" | "tool" => {
                if let Some(last_step) = steps.last_mut() {
                    last_step.push(indexed.clone());
                } else {
                    steps.push(vec![indexed.clone()]);
                }
            }
            _ => steps.push(vec![indexed.clone()]),
        }
    }
    steps
}

#[derive(Clone, Debug)]
struct IndexedMessage {
    index: usize,
    message: NativeMessage,
}

fn truncate_turn(
    indexed_messages: Vec<IndexedMessage>,
    message_tokens: &HashMap<usize, usize>,
    mut exceedance: usize,
    is_last_turn: bool,
) -> PyResult<(Vec<NativeMessage>, usize)> {
    let all_tokens: usize = indexed_messages
        .iter()
        .map(|item| message_tokens.get(&item.index).copied().unwrap_or(0))
        .sum();
    if all_tokens <= exceedance {
        return Ok((Vec::new(), exceedance.saturating_sub(all_tokens)));
    }
    if indexed_messages.len() == 1 {
        let item = &indexed_messages[0];
        let token_count = message_tokens.get(&item.index).copied().unwrap_or(0);
        let text = truncate_qwen_text(
            &item.message.text,
            token_count.saturating_sub(exceedance),
            true,
        )?;
        return Ok((vec![item.message.with_text(text)], 0));
    }

    let mut indexed_messages = indexed_messages;
    let mut message_tokens = message_tokens.clone();
    let messages_per_step = split_turn_into_steps(&indexed_messages);
    let last_step_idx = messages_per_step
        .last()
        .and_then(|step| step.first().map(|item| item.index))
        .unwrap_or(0);

    for item in &mut indexed_messages {
        if exceedance == 0 {
            break;
        }
        if (item.message.role != "function" && item.message.role != "tool")
            || (is_last_turn && item.index >= last_step_idx)
        {
            continue;
        }
        let token_count = message_tokens.get(&item.index).copied().unwrap_or(0);
        if token_count > exceedance {
            item.message.text =
                truncate_qwen_text(&item.message.text, token_count - exceedance, true)?;
            message_tokens.insert(item.index, token_count - exceedance);
            exceedance = 0;
            break;
        }
        // Omit body only — role, name, function_id (extra) stay on the message.
        item.message.text = "omit".to_string();
        message_tokens.insert(item.index, 0);
        exceedance -= token_count;
    }
    if exceedance == 0 {
        return Ok((
            indexed_messages
                .into_iter()
                .map(|item| item.message)
                .collect(),
            0,
        ));
    }

    let mut keep_idx = 0usize;
    for (step_index, step) in messages_per_step.iter().enumerate() {
        if step_index == 0 || step_index == messages_per_step.len() - 1 {
            continue;
        }
        let step_tokens: usize = step
            .iter()
            .map(|item| message_tokens.get(&item.index).copied().unwrap_or(0))
            .sum();
        if step_tokens >= exceedance {
            exceedance = 0;
            keep_idx = messages_per_step
                .get(step_index + 1)
                .and_then(|next| next.first().map(|item| item.index))
                .unwrap_or(0);
            break;
        }
        exceedance -= step_tokens;
        keep_idx = messages_per_step
            .get(step_index + 1)
            .and_then(|next| next.first().map(|item| item.index))
            .unwrap_or(0);
    }
    if exceedance == 0 {
        let first = messages_per_step
            .first()
            .cloned()
            .unwrap_or_default()
            .into_iter()
            .map(|item| item.message)
            .collect::<Vec<_>>();
        let tail = indexed_messages
            .iter()
            .filter(|item| item.index >= keep_idx)
            .map(|item| item.message.clone())
            .collect::<Vec<_>>();
        let mut combined = first;
        combined.extend(tail);
        return Ok((combined, 0));
    }

    let mut messages_to_keep = Vec::new();
    if let Some(last_step) = messages_per_step.last() {
        for item in last_step {
            let mut message = item.message.clone();
            if message.role == "function" || message.role == "tool" {
                let token_count = message_tokens.get(&item.index).copied().unwrap_or(0);
                if token_count > exceedance {
                    message.text =
                        truncate_qwen_text(&message.text, token_count - exceedance, true)?;
                    exceedance = 0;
                } else {
                    message.text = "omit".to_string();
                    exceedance -= token_count;
                }
            }
            messages_to_keep.push(message);
        }
    }
    if let Some(first_step) = messages_per_step.first() {
        let mut combined = first_step
            .iter()
            .map(|item| item.message.clone())
            .collect::<Vec<_>>();
        combined.append(&mut messages_to_keep);
        messages_to_keep = combined;
    }
    if exceedance == 0 {
        return Ok((messages_to_keep, 0));
    }

    for message in &mut messages_to_keep {
        let token_count = count_qwen_tokens(&message.text)?;
        if token_count > exceedance {
            message.text = truncate_qwen_text(&message.text, token_count - exceedance, true)?;
            break;
        }
        message.text = "omit".to_string();
        exceedance -= token_count;
    }
    Ok((messages_to_keep, 0))
}

pub fn truncate_messages_native(
    messages: &[NativeMessage],
    max_tokens: usize,
) -> PyResult<Vec<NativeMessage>> {
    if messages.is_empty() {
        return Ok(Vec::new());
    }
    let system_count = messages.iter().filter(|msg| msg.role == "system").count();
    if system_count >= 2 {
        return Err(PyValueError::new_err(
            "The input messages must contain no more than one system message.",
        ));
    }

    let mut available_token = max_tokens;
    let mut message_tokens = HashMap::new();
    let mut indexed_messages_per_user: HashMap<usize, Vec<IndexedMessage>> = HashMap::new();
    let mut last_user_idx = None;
    let mut new_messages = Vec::new();

    for (msg_idx, message) in messages.iter().enumerate() {
        if message.role == "system" {
            let token_count = count_qwen_tokens(&message.text)?;
            available_token = max_tokens.saturating_sub(token_count);
            new_messages.push(message.clone());
            continue;
        }
        let token_count = count_qwen_tokens(&message.text)?;
        message_tokens.insert(msg_idx, token_count);
        if message.role == "user" {
            last_user_idx = Some(msg_idx);
        }
        if let Some(user_idx) = last_user_idx {
            indexed_messages_per_user
                .entry(user_idx)
                .or_default()
                .push(IndexedMessage {
                    index: msg_idx,
                    message: message.clone(),
                });
        } else {
            return Err(PyValueError::new_err(
                "The input messages must start with a user message.",
            ));
        }
    }

    let all_tokens: usize = message_tokens.values().sum();
    if all_tokens <= available_token {
        return Ok(messages.to_vec());
    }
    if available_token == 0 {
        return Err(PyValueError::new_err(format!(
            "The input system has exceed the maximum input context length ({max_tokens} tokens)"
        )));
    }

    let mut exceedance = all_tokens.saturating_sub(available_token);
    let user_indices: Vec<usize> = indexed_messages_per_user.keys().copied().collect();
    for (turn_index, user_idx) in user_indices.iter().enumerate() {
        let indexed_messages = indexed_messages_per_user
            .remove(user_idx)
            .unwrap_or_default();
        if exceedance == 0 {
            new_messages.extend(indexed_messages.into_iter().map(|item| item.message));
        } else {
            let is_last_turn = turn_index == user_indices.len() - 1;
            let (turn_messages, remaining) =
                truncate_turn(indexed_messages, &message_tokens, exceedance, is_last_turn)?;
            exceedance = remaining;
            new_messages.extend(turn_messages);
        }
    }
    Ok(new_messages)
}

#[pyfunction(name = "truncate_messages")]
pub fn truncate_messages_py<'py>(
    py: Python<'py>,
    messages: &Bound<'py, PyAny>,
    max_tokens: usize,
) -> PyResult<Bound<'py, PyList>> {
    let parsed = parse_messages(messages)?;
    let truncated = truncate_messages_native(&parsed, max_tokens)?;
    let list = PyList::empty(py);
    for message in truncated {
        let dict = PyDict::new(py);
        dict.set_item("role", message.role)?;
        dict.set_item("text", message.text)?;
        if let Some(name) = message.name {
            dict.set_item("name", name)?;
        }
        if let Some(baseline) = message.text_baseline {
            dict.set_item("text_baseline", baseline)?;
        }
        // content / reasoning were stored via serde_json for strings, or json.dumps otherwise.
        set_json_field(&dict, "content", &message.content_json, py, true)?;
        set_json_field(
            &dict,
            "function_call",
            &message.function_call_json,
            py,
            true,
        )?;
        set_json_field(&dict, "tool_calls", &message.tool_calls_json, py, true)?;
        if let Some(tool_call_id) = message.tool_call_id {
            dict.set_item("tool_call_id", tool_call_id)?;
        }
        set_json_field(&dict, "extra", &message.extra_json, py, true)?;
        set_json_field(
            &dict,
            "reasoning_content",
            &message.reasoning_content_json,
            py,
            true,
        )?;
        list.append(dict)?;
    }
    Ok(list)
}
