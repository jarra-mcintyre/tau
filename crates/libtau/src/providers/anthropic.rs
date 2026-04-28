use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::{ContentPart, ConversationItem, MediaData, TauContext, ToolResult, ToolUse},
    providers::{Provider, ProviderError, ProviderResponse},
};

pub const PROVIDER_NAME: &str = "anthropic";
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 4096;

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    max_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            ProviderError::Configuration(
                "ANTHROPIC_API_KEY environment variable is not set".to_string(),
            )
        })?;
        Ok(Self::new(api_key))
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: base_url.into().trim_end_matches('/').to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    async fn respond(&self, context: &mut TauContext) -> Result<ProviderResponse, ProviderError> {
        let request = build_request(context, self.max_tokens)?;
        let response = self
            .client
            .post(format!("{}/messages", self.base_url))
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        let response: AnthropicResponse = serde_json::from_str(&body)?;
        parse_response(response)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct AnthropicRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<AnthropicContent>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "lowercase")]
enum AnthropicRole {
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
struct AnthropicMessage {
    role: AnthropicRole,
    content: Vec<AnthropicContent>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicContent {
    Text {
        text: String,
    },
    Image {
        source: AnthropicImageSource,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        is_error: Option<bool>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicImageSource {
    Base64 { media_type: String, data: String },
    Url { url: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct AnthropicTool {
    name: String,
    description: String,
    input_schema: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<AnthropicResponseContent>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicResponseContent {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    #[serde(other)]
    Other,
}

fn build_request(context: &TauContext, max_tokens: u32) -> Result<AnthropicRequest, ProviderError> {
    let model = context
        .model()
        .ok_or(ProviderError::MissingModel)?
        .to_string();

    let mut system = Vec::new();
    let mut messages = Vec::new();

    for item in &context.conversation().items {
        match item {
            ConversationItem::System { content } => system.extend(input_content_parts(content)?),
            ConversationItem::User { content } => push_message(
                &mut messages,
                AnthropicRole::User,
                input_content_parts(content)?,
            ),
            ConversationItem::Agent { content } => push_message(
                &mut messages,
                AnthropicRole::Assistant,
                output_content_parts(content),
            ),
            ConversationItem::ToolUse { calls } => push_message(
                &mut messages,
                AnthropicRole::Assistant,
                calls
                    .iter()
                    .map(|call| AnthropicContent::ToolUse {
                        id: call.id.clone(),
                        name: call.name.clone(),
                        input: call.input.clone(),
                    })
                    .collect(),
            ),
            ConversationItem::ToolResult { results } => push_message(
                &mut messages,
                AnthropicRole::User,
                results
                    .iter()
                    .map(tool_result_content)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
        }
    }

    let tools = context
        .tools()
        .map(|tool| AnthropicTool {
            name: tool.name.clone(),
            description: tool.description.clone(),
            input_schema: tool.input_schema.clone(),
        })
        .collect();

    Ok(AnthropicRequest {
        model,
        max_tokens,
        system,
        messages,
        tools,
    })
}

fn push_message(
    messages: &mut Vec<AnthropicMessage>,
    role: AnthropicRole,
    content: Vec<AnthropicContent>,
) {
    if content.is_empty() {
        return;
    }

    if let Some(last) = messages.last_mut()
        && last.role == role
    {
        last.content.extend(content);
        return;
    }

    messages.push(AnthropicMessage { role, content });
}

fn parse_response(response: AnthropicResponse) -> Result<ProviderResponse, ProviderError> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    for part in response.content {
        match part {
            AnthropicResponseContent::Text { text } => content.push(ContentPart::text(text)),
            AnthropicResponseContent::ToolUse { id, name, input } => {
                tool_calls.push(ToolUse { id, name, input });
            }
            AnthropicResponseContent::Other => {}
        }
    }

    Ok(ProviderResponse {
        content,
        tool_calls,
    })
}

fn input_content_parts(parts: &[ContentPart]) -> Result<Vec<AnthropicContent>, ProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => Ok(AnthropicContent::Text { text: text.clone() }),
            ContentPart::Json { value } => Ok(AnthropicContent::Text {
                text: serde_json::to_string(value)?,
            }),
            ContentPart::Image { media_type, data } => match data {
                MediaData::Base64(data) => Ok(AnthropicContent::Image {
                    source: AnthropicImageSource::Base64 {
                        media_type: media_type.clone(),
                        data: data.clone(),
                    },
                }),
                MediaData::Url(url) => Ok(AnthropicContent::Image {
                    source: AnthropicImageSource::Url { url: url.clone() },
                }),
                MediaData::Path(path) => Ok(AnthropicContent::Text {
                    text: format!("[image content: {media_type}, path={path}]"),
                }),
            },
            ContentPart::Binary { media_type, data } => Ok(AnthropicContent::Text {
                text: format!("[binary content: {media_type}, {}]", media_data_label(data)),
            }),
        })
        .collect()
}

fn output_content_parts(parts: &[ContentPart]) -> Vec<AnthropicContent> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => AnthropicContent::Text { text: text.clone() },
            ContentPart::Json { value } => AnthropicContent::Text {
                text: value.to_string(),
            },
            ContentPart::Image { media_type, data } | ContentPart::Binary { media_type, data } => {
                AnthropicContent::Text {
                    text: format!("[media content: {media_type}, {}]", media_data_label(data)),
                }
            }
        })
        .collect()
}

fn tool_result_content(result: &ToolResult) -> Result<AnthropicContent, ProviderError> {
    Ok(AnthropicContent::ToolResult {
        tool_use_id: result.call_id.clone(),
        content: serde_json::to_string(&serde_json::json!({
            "name": result.name,
            "error": result.error,
            "content": result.content,
        }))?,
        is_error: result.error.as_ref().map(|_| true),
    })
}

fn media_data_label(data: &MediaData) -> String {
    match data {
        MediaData::Url(url) => format!("url={url}"),
        MediaData::Base64(data) => format!("base64_bytes={}", data.len()),
        MediaData::Path(path) => format!("path={path}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{ToolDefinition, ToolOutput};
    use serde_json::json;

    fn callback(input: Value) -> Result<ToolOutput, crate::context::ToolCallError> {
        Ok(ToolOutput::json(input))
    }

    #[test]
    fn builds_messages_api_request_from_complete_history() {
        let mut context = TauContext::new();
        context.set_model("claude-sonnet-4-5");
        context.push_system_text("be helpful");
        context.push_user_text("hello");
        context.push_agent_text("I'll call a tool.");
        context.push_item(ConversationItem::ToolUse {
            calls: vec![ToolUse {
                id: "toolu_1".to_string(),
                name: "echo".to_string(),
                input: json!({"text":"hello"}),
            }],
        });
        context.push_item(ConversationItem::ToolResult {
            results: vec![ToolResult {
                call_id: "toolu_1".to_string(),
                name: "echo".to_string(),
                content: vec![ContentPart::json(json!({"text":"hello"}))],
                error: None,
            }],
        });
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();

        let request = build_request(&context, DEFAULT_MAX_TOKENS).unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["model"], "claude-sonnet-4-5");
        assert_eq!(value["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["tools"][0]["name"], "echo");
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][1]["role"], "assistant");
        assert_eq!(value["messages"][1]["content"][0]["type"], "text");
        assert_eq!(value["messages"][1]["content"][1]["type"], "tool_use");
        assert_eq!(value["messages"][2]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn parses_text_and_parallel_tool_calls() {
        let response = AnthropicResponse {
            content: vec![
                AnthropicResponseContent::Text {
                    text: "I'll check.".to_string(),
                },
                AnthropicResponseContent::ToolUse {
                    id: "toolu_a".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"Cargo.toml"}),
                },
                AnthropicResponseContent::ToolUse {
                    id: "toolu_b".to_string(),
                    name: "read_file".to_string(),
                    input: json!({"path":"README.md"}),
                },
            ],
        };

        let parsed = parse_response(response).unwrap();

        assert_eq!(parsed.content, vec![ContentPart::text("I'll check.")]);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].id, "toolu_a");
        assert_eq!(parsed.tool_calls[1].input, json!({"path":"README.md"}));
    }
}
