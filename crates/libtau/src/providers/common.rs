use serde_json::{Value, json};

use crate::{
    context::{ContentPart, MediaData, ToolResult},
    providers::ProviderError,
};

pub fn json_as_text(value: &Value) -> Result<String, ProviderError> {
    Ok(serde_json::to_string(value)?)
}

pub fn assistant_content_as_text(part: &ContentPart) -> String {
    match part {
        ContentPart::Text { text } => text.clone(),
        ContentPart::Json { value } => value.to_string(),
        ContentPart::Image { media_type, data } | ContentPart::Binary { media_type, data } => {
            format!("[media content: {media_type}, {}]", media_data_label(data))
        }
    }
}

pub fn binary_content_as_text(media_type: &str, data: &MediaData) -> String {
    format!("[binary content: {media_type}, {}]", media_data_label(data))
}

pub fn media_data_label(data: &MediaData) -> String {
    match data {
        MediaData::Url(url) => format!("url={url}"),
        MediaData::Base64(data) => format!("base64_bytes={}", data.len()),
        MediaData::Path(path) => format!("path={path}"),
    }
}

pub fn media_to_url(media_type: &str, data: &MediaData) -> String {
    match data {
        MediaData::Url(url) => url.clone(),
        MediaData::Base64(data) => format!("data:{media_type};base64,{data}"),
        MediaData::Path(path) => path.clone(),
    }
}

pub fn tool_result_json(result: &ToolResult) -> Result<String, ProviderError> {
    Ok(serde_json::to_string(&json!({
        "name": result.name,
        "error": result.error,
        "content": result.content,
    }))?)
}
