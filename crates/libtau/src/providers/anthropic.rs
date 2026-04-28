use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::{ContentPart, ConversationItem, MediaData, TauSession, ToolResult, ToolUse},
    providers::{
        Provider, ProviderApi, ProviderApiConfig, ProviderError, ProviderResponse, TokenUsage,
        common::{
            assistant_content_as_text, binary_content_as_text, json_as_text, tool_result_json,
        },
    },
};

pub const PROVIDER_NAME: &str = "anthropic";
pub const API_NAME: &str = "anthropic_messages";
pub const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const API: ProviderApi = ProviderApi {
    name: API_NAME,
    api_key_env: API_KEY_ENV,
    display_name: "Anthropic",
    build: build_provider,
};
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

fn build_provider(config: ProviderApiConfig) -> Arc<dyn Provider> {
    match config.base_url {
        Some(base_url) => Arc::new(AnthropicProvider::with_base_url(config.api_key, base_url)),
        None => Arc::new(AnthropicProvider::new(config.api_key)),
    }
}

fn normalize_base_url(base_url: impl Into<String>) -> String {
    let base_url = base_url.into();
    let trimmed = base_url.trim_end_matches('/');

    match reqwest::Url::parse(trimmed) {
        Ok(mut url) if url.path() == "/" || url.path().is_empty() => {
            url.set_path("/v1");
            url.to_string().trim_end_matches('/').to_string()
        }
        _ => trimmed.to_string(),
    }
}

impl AnthropicProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var(API_KEY_ENV).map_err(|_| {
            ProviderError::Configuration(format!("{API_KEY_ENV} environment variable is not set"))
        })?;
        Ok(Self::new(api_key))
    }

    pub fn with_base_url(api_key: impl Into<String>, base_url: impl Into<String>) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: normalize_base_url(base_url),
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

    async fn respond(&self, session: &mut TauSession) -> Result<ProviderResponse, ProviderError> {
        let request = build_request(session, self.max_tokens)?;
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
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
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

fn build_request(session: &TauSession, max_tokens: u32) -> Result<AnthropicRequest, ProviderError> {
    let model = session
        .model()
        .ok_or(ProviderError::MissingModel)?
        .to_string();

    let mut system = Vec::new();
    let mut messages = Vec::new();

    for item in &session.conversation().items {
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

    let tools = session
        .context()
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
    let usage = response.usage.map(|usage| {
        let total_tokens = match (usage.input_tokens, usage.output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        };
        TokenUsage {
            input_tokens: usage.input_tokens,
            output_tokens: usage.output_tokens,
            total_tokens,
        }
    });
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
        usage,
    })
}

fn input_content_parts(parts: &[ContentPart]) -> Result<Vec<AnthropicContent>, ProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => Ok(AnthropicContent::Text { text: text.clone() }),
            ContentPart::Json { value } => Ok(AnthropicContent::Text {
                text: json_as_text(value)?,
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
                text: binary_content_as_text(media_type, data),
            }),
        })
        .collect()
}

fn output_content_parts(parts: &[ContentPart]) -> Vec<AnthropicContent> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => AnthropicContent::Text { text: text.clone() },
            part => AnthropicContent::Text {
                text: assistant_content_as_text(part),
            },
        })
        .collect()
}

fn tool_result_content(result: &ToolResult) -> Result<AnthropicContent, ProviderError> {
    Ok(AnthropicContent::ToolResult {
        tool_use_id: result.call_id.clone(),
        content: tool_result_json(result)?,
        is_error: result.error.as_ref().map(|_| true),
    })
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
    fn derives_v1_base_path_for_host_only_urls() {
        assert_eq!(
            normalize_base_url("http://TP-MACMINI01.local:8080"),
            "http://tp-macmini01.local:8080/v1"
        );
        assert_eq!(
            normalize_base_url("http://TP-MACMINI01.local:8080/v1"),
            "http://TP-MACMINI01.local:8080/v1"
        );
    }

    #[test]
    fn builds_messages_api_request_from_complete_history() {
        let mut context = crate::context::TauContext::new();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();
        let mut session = context.session(AnthropicProvider::new("test-key"), "claude-sonnet-4-5");
        session.push_system_text("be helpful");
        session.push_user_text("hello");
        session.push_agent_text("I'll call a tool.");
        session.push_item(ConversationItem::ToolUse {
            calls: vec![ToolUse {
                id: "toolu_1".to_string(),
                name: "echo".to_string(),
                input: json!({"text":"hello"}),
            }],
        });
        session.push_item(ConversationItem::ToolResult {
            results: vec![ToolResult {
                call_id: "toolu_1".to_string(),
                name: "echo".to_string(),
                content: vec![ContentPart::json(json!({"text":"hello"}))],
                error: None,
            }],
        });
        let request = build_request(&session, DEFAULT_MAX_TOKENS).unwrap();
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
            usage: Some(AnthropicUsage {
                input_tokens: Some(50),
                output_tokens: Some(12),
            }),
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
        assert_eq!(parsed.usage.unwrap().total_tokens, Some(62));
    }
}
