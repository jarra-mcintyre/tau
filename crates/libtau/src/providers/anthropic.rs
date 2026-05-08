use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::{
        ContentPart, ConversationItem, MediaData, ResponsePart, ServerToolUse, TauSession,
        ToolResult, ToolUse,
    },
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
const WEB_SEARCH_TOOL_TYPE: &str = "web_search_20260209";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    max_tokens: u32,
    cache_ttl: Option<AnthropicCacheTtl>,
    web_search: Option<AnthropicWebSearchConfig>,
}

fn build_provider(config: ProviderApiConfig) -> Result<Arc<dyn Provider>, ProviderError> {
    let options = AnthropicOptions::from_value(config.options)?;
    let provider = match config.base_url {
        Some(base_url) => AnthropicProvider::with_base_url(config.api_key, base_url),
        None => AnthropicProvider::new(config.api_key),
    }
    .with_cache_ttl(options.cache_ttl)
    .with_web_search(options.web_search);

    Ok(Arc::new(provider))
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
        //Self::with_base_url(api_key, DEFAULT_BASE_URL)
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            max_tokens: DEFAULT_MAX_TOKENS,
            cache_ttl: Some(AnthropicCacheTtl::FiveMinutes),
            web_search: Some(AnthropicWebSearchConfig::enabled())
        }
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
            cache_ttl: Some(AnthropicCacheTtl::FiveMinutes),
            web_search: None,
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    pub fn with_cache_ttl(mut self, cache_ttl: Option<AnthropicCacheTtl>) -> Self {
        self.cache_ttl = cache_ttl;
        self
    }

    pub fn with_web_search(mut self, web_search: Option<AnthropicWebSearchConfig>) -> Self {
        self.web_search = web_search;
        self
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    async fn respond(&self, session: &mut TauSession) -> Result<ProviderResponse, ProviderError> {
        let request = build_request(
            session,
            self.max_tokens,
            self.cache_ttl,
            self.web_search.as_ref(),
        )?;
        let url = format!("{}/messages", self.base_url);
        let request_body = serde_json::to_string_pretty(&request)?;
        tracing::debug!(
            target: "tau::providers::anthropic",
            %url,
            body = %request_body,
            "request"
        );

        let response = self
            .client
            .post(&url)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        tracing::debug!(
            target: "tau::providers::anthropic",
            %status,
            body = %body,
            "response"
        );
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
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<AnthropicContent>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
struct AnthropicOptions {
    #[serde(default = "default_cache_ttl")]
    cache_ttl: Option<AnthropicCacheTtl>,
    #[serde(default)]
    web_search: Option<AnthropicWebSearchConfig>,
}

impl Default for AnthropicOptions {
    fn default() -> Self {
        Self {
            cache_ttl: default_cache_ttl(),
            web_search: None,
        }
    }
}

impl AnthropicOptions {
    fn from_value(value: Value) -> Result<Self, ProviderError> {
        if value.is_null() {
            return Ok(Self::default());
        }

        serde_json::from_value(value).map_err(|error| {
            ProviderError::Configuration(format!("invalid Anthropic options: {error}"))
        })
    }
}

fn default_cache_ttl() -> Option<AnthropicCacheTtl> {
    Some(AnthropicCacheTtl::FiveMinutes)
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicCacheTtl {
    #[serde(rename = "5m")]
    FiveMinutes,
    #[serde(rename = "1h")]
    OneHour,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub struct AnthropicWebSearchConfig {
    #[serde(default = "default_web_search_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub max_uses: Option<u32>,
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(default)]
    pub blocked_domains: Option<Vec<String>>,
}

impl AnthropicWebSearchConfig {
    fn enabled() -> Self {
        Self {
            enabled: true,
            max_uses: None,
            allowed_domains: None,
            blocked_domains: None
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
struct AnthropicCacheControl {
    #[serde(rename = "type")]
    kind: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    ttl: Option<AnthropicCacheTtl>,
}

impl AnthropicCacheControl {
    fn new(ttl: AnthropicCacheTtl) -> Self {
        Self {
            kind: "ephemeral",
            ttl: (ttl == AnthropicCacheTtl::OneHour).then_some(ttl),
        }
    }
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
    ServerToolUse {
        id: String,
        name: String,
        input: Value,
    },
    WebSearchToolResult {
        tool_use_id: String,
        content: Value,
    },
    Thinking {
        thinking: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
    },
    RedactedThinking {
        data: String,
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
#[serde(untagged)]
enum AnthropicTool {
    Custom {
        name: String,
        description: String,
        strict: bool,
        input_schema: Value,
    },
    Server {
        #[serde(rename = "type")]
        kind: &'static str,
        name: &'static str,
        #[serde(skip_serializing_if = "Option::is_none")]
        max_uses: Option<u32>,
        #[serde(skip_serializing_if = "Option::is_none")]
        allowed_domains: Option<Vec<String>>,
        #[serde(skip_serializing_if = "Option::is_none")]
        blocked_domains: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<Value>,
    usage: Option<AnthropicUsage>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicToolUseOutput {
    id: String,
    name: String,
    input: Value,
}

fn build_request(
    session: &TauSession,
    max_tokens: u32,
    cache_ttl: Option<AnthropicCacheTtl>,
    web_search: Option<&AnthropicWebSearchConfig>,
) -> Result<AnthropicRequest, ProviderError> {
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

    let tools = anthropic_tools(session, web_search);

    Ok(AnthropicRequest {
        model,
        max_tokens,
        cache_control: cache_ttl.map(AnthropicCacheControl::new),
        system,
        messages,
        tools,
    })
}

fn default_web_search_enabled() -> bool {
    true
}

fn anthropic_tools(
    session: &TauSession,
    web_search: Option<&AnthropicWebSearchConfig>,
) -> Vec<AnthropicTool> {
    let mut tools: Vec<_> = session
        .context()
        .tools()
        .map(|tool| AnthropicTool::Custom {
            name: tool.name.clone(),
            description: tool.description.clone(),
            strict: true,
            input_schema: tool.input_schema.clone(),
        })
        .collect();

    if let Some(web_search) = web_search.filter(|config| config.enabled) {
        tools.push(AnthropicTool::Server {
            kind: WEB_SEARCH_TOOL_TYPE,
            name: "web_search",
            max_uses: web_search.max_uses,
            allowed_domains: web_search.allowed_domains.clone(),
            blocked_domains: web_search.blocked_domains.clone(),
        });
    }

    tools
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
    let mut parts = Vec::new();
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    for part in response.content {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::Response("Anthropic text block missing text".to_string())
                })?;
                let content_part =
                    ContentPart::text_with_metadata(text, content_block_metadata(part.clone()));
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            Some("tool_use") => {
                let tool_use: AnthropicToolUseOutput = serde_json::from_value(part)?;
                let call = ToolUse {
                    id: tool_use.id,
                    name: tool_use.name,
                    input: tool_use.input,
                };
                parts.push(ResponsePart::ToolUse { call: call.clone() });
                tool_calls.push(call);
            }
            Some("server_tool_use") => {
                let call = ServerToolUse {
                    id: part
                        .get("id")
                        .and_then(Value::as_str)
                        .map(ToString::to_string),
                    name: part
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("web_search")
                        .to_string(),
                    input: part.get("input").cloned().unwrap_or(Value::Null),
                    metadata: Some(content_block_metadata(part.clone())),
                };
                parts.push(ResponsePart::ServerToolUse { call });
                let content_part =
                    ContentPart::json_with_metadata(part.clone(), content_block_metadata(part));
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            Some("thinking") => {
                let text = part
                    .get("thinking")
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ProviderError::Response(
                            "Anthropic thinking block missing thinking".to_string(),
                        )
                    })?;
                let signature = part
                    .get("signature")
                    .and_then(Value::as_str)
                    .map(ToString::to_string);
                let content_part = ContentPart::thinking_with_metadata(
                    text,
                    signature,
                    raw_metadata("anthropic.content_block", part.clone()),
                );
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            Some("redacted_thinking") => {
                let content_part = ContentPart::json_with_metadata(
                    part.clone(),
                    raw_metadata("anthropic.content_block", part),
                );
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            _ => {
                let content_part = ContentPart::json_with_metadata(
                    part.clone(),
                    raw_metadata("anthropic.content_block", part),
                );
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
        }
    }

    Ok(ProviderResponse {
        parts,
        content,
        tool_calls,
        usage,
    })
}

fn content_block_metadata(raw: Value) -> Value {
    let mut metadata = serde_json::json!({
        "provider": PROVIDER_NAME,
        "kind": "anthropic.content_block",
        "raw": raw,
    });

    if let Some(citations) = metadata["raw"].get("citations").cloned() {
        metadata["citations"] = citations;
    }

    metadata
}

fn raw_metadata(kind: &str, raw: Value) -> Value {
    serde_json::json!({
        "provider": PROVIDER_NAME,
        "kind": kind,
        "raw": raw,
    })
}

fn input_content_parts(parts: &[ContentPart]) -> Result<Vec<AnthropicContent>, ProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => Ok(AnthropicContent::Text { text: text.clone() }),
            ContentPart::Json { value, .. } => Ok(AnthropicContent::Text {
                text: json_as_text(value)?,
            }),
            ContentPart::Image {
                media_type, data, ..
            } => match data {
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
            ContentPart::Binary {
                media_type, data, ..
            } => Ok(AnthropicContent::Text {
                text: binary_content_as_text(media_type, data),
            }),
            ContentPart::Thinking { text, .. } => Ok(AnthropicContent::Text {
                text: format!("[thinking: {text}]"),
            }),
        })
        .collect()
}

fn output_content_parts(parts: &[ContentPart]) -> Vec<AnthropicContent> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => AnthropicContent::Text { text: text.clone() },
            ContentPart::Thinking {
                text, signature, ..
            } => AnthropicContent::Thinking {
                thinking: text.clone(),
                signature: signature.clone(),
            },
            ContentPart::Json { value, metadata }
                if is_anthropic_redacted_thinking(value, metadata) =>
            {
                AnthropicContent::RedactedThinking {
                    data: value
                        .get("data")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                }
            }
            ContentPart::Json { value, metadata }
                if is_anthropic_server_tool_use(value, metadata) =>
            {
                AnthropicContent::ServerToolUse {
                    id: value
                        .get("id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    name: value
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or("web_search")
                        .to_string(),
                    input: value.get("input").cloned().unwrap_or(Value::Null),
                }
            }
            ContentPart::Json { value, metadata }
                if is_anthropic_web_search_tool_result(value, metadata) =>
            {
                AnthropicContent::WebSearchToolResult {
                    tool_use_id: value
                        .get("tool_use_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_string(),
                    content: value.get("content").cloned().unwrap_or(Value::Null),
                }
            }
            part => AnthropicContent::Text {
                text: assistant_content_as_text(part),
            },
        })
        .collect()
}

fn is_anthropic_redacted_thinking(value: &Value, metadata: &Option<Value>) -> bool {
    is_anthropic_content_block(value, metadata, "redacted_thinking")
}

fn is_anthropic_server_tool_use(value: &Value, metadata: &Option<Value>) -> bool {
    is_anthropic_content_block(value, metadata, "server_tool_use")
}

fn is_anthropic_web_search_tool_result(value: &Value, metadata: &Option<Value>) -> bool {
    is_anthropic_content_block(value, metadata, "web_search_tool_result")
}

fn is_anthropic_content_block(value: &Value, metadata: &Option<Value>, block_type: &str) -> bool {
    value.get("type").and_then(Value::as_str) == Some(block_type)
        && metadata
            .as_ref()
            .and_then(|metadata| metadata.get("provider"))
            .and_then(Value::as_str)
            == Some(PROVIDER_NAME)
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
    use crate::tools::{ToolDefinition, ToolOutput};
    use serde_json::json;

    fn callback(input: Value) -> Result<ToolOutput, crate::tools::ToolCallError> {
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
        session.push_item(ConversationItem::Agent {
            content: vec![
                ContentPart::text("I'll call a tool."),
                ContentPart::Thinking {
                    text: "Need to echo.".to_string(),
                    signature: Some("sig_1".to_string()),
                    metadata: None,
                },
            ],
        });
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
        let web_search = AnthropicWebSearchConfig {
            enabled: true,
            max_uses: Some(3),
            allowed_domains: Some(vec!["example.com".to_string()]),
            blocked_domains: Some(vec!["bad.example".to_string()]),
        };
        let request = build_request(
            &session,
            DEFAULT_MAX_TOKENS,
            Some(AnthropicCacheTtl::FiveMinutes),
            Some(&web_search),
        )
        .unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["model"], "claude-sonnet-4-5");
        assert_eq!(value["max_tokens"], DEFAULT_MAX_TOKENS);
        assert_eq!(value["cache_control"]["type"], "ephemeral");
        assert!(value["cache_control"].get("ttl").is_none());
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["tools"][0]["name"], "echo");
        assert_eq!(value["tools"][0]["strict"], true);
        assert_eq!(value["tools"][1]["type"], WEB_SEARCH_TOOL_TYPE);
        assert_eq!(value["tools"][1]["name"], "web_search");
        assert_eq!(value["tools"][1]["max_uses"], 3);
        assert_eq!(value["tools"][1]["allowed_domains"], json!(["example.com"]));
        assert_eq!(value["tools"][1]["blocked_domains"], json!(["bad.example"]));
        assert_eq!(value["tools"].as_array().unwrap().len(), 2);
        assert_eq!(value["messages"][0]["role"], "user");
        assert_eq!(value["messages"][1]["role"], "assistant");
        assert_eq!(value["messages"][1]["content"][0]["type"], "text");
        assert_eq!(value["messages"][1]["content"][1]["type"], "thinking");
        assert_eq!(value["messages"][1]["content"][1]["signature"], "sig_1");
        assert_eq!(value["messages"][1]["content"][2]["type"], "tool_use");
        assert_eq!(value["messages"][2]["content"][0]["type"], "tool_result");
    }

    #[test]
    fn configures_one_hour_automatic_cache_ttl() {
        let context = crate::context::TauContext::new();
        let session = context.session(AnthropicProvider::new("test-key"), "claude-sonnet-4-5");
        let request = build_request(
            &session,
            DEFAULT_MAX_TOKENS,
            Some(AnthropicCacheTtl::OneHour),
            None,
        )
        .unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(
            value["cache_control"],
            json!({"type": "ephemeral", "ttl": "1h"})
        );
        assert!(value.get("tools").is_none());
    }

    #[test]
    fn parses_provider_options_with_web_search() {
        let options = AnthropicOptions::from_value(json!({
            "cache_ttl": "1h",
            "web_search": {
                "enabled": true,
                "max_uses": 2,
                "allowed_domains": ["example.com"],
                "blocked_domains": ["spam.example"]
            }
        }))
        .unwrap();

        assert_eq!(options.cache_ttl, Some(AnthropicCacheTtl::OneHour));
        let web_search = options.web_search.unwrap();
        assert!(web_search.enabled);
        assert_eq!(web_search.max_uses, Some(2));
        assert_eq!(web_search.allowed_domains, Some(vec!["example.com".to_string()]));
        assert_eq!(web_search.blocked_domains, Some(vec!["spam.example".to_string()]));
    }

    #[test]
    fn disables_web_search_when_configured_off() {
        let context = crate::context::TauContext::new();
        let session = context.session(AnthropicProvider::new("test-key"), "claude-sonnet-4-5");
        let web_search = AnthropicWebSearchConfig {
            enabled: false,
            max_uses: Some(1),
            allowed_domains: None,
            blocked_domains: None,
        };
        let request = build_request(
            &session,
            DEFAULT_MAX_TOKENS,
            Some(AnthropicCacheTtl::FiveMinutes),
            Some(&web_search),
        )
        .unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert!(value.get("tools").is_none());
    }

    #[test]
    fn parses_text_and_parallel_tool_calls() {
        let response = AnthropicResponse {
            usage: Some(AnthropicUsage {
                input_tokens: Some(50),
                output_tokens: Some(12),
            }),
            content: vec![
                json!({"type": "text", "text": "I'll check.", "citations": [{"type":"web_search_result_location","url":"https://example.com","title":"Example"}]}),
                json!({"type": "thinking", "thinking": "I should inspect the files.", "signature": "sig_1"}),
                json!({
                    "type": "tool_use",
                    "id": "toolu_a",
                    "name": "read_file",
                    "input": {"path":"Cargo.toml"}
                }),
                json!({
                    "type": "tool_use",
                    "id": "toolu_b",
                    "name": "read_file",
                    "input": {"path":"README.md"}
                }),
                json!({"type": "server_tool_use", "id": "srv_1", "name": "web_search"}),
            ],
        };

        let parsed = parse_response(response).unwrap();

        assert_eq!(parsed.content.len(), 3);
        match &parsed.content[0] {
            ContentPart::Text { text, metadata } => {
                assert_eq!(text, "I'll check.");
                assert_eq!(
                    metadata.as_ref().unwrap()["kind"],
                    "anthropic.content_block"
                );
                assert_eq!(
                    metadata.as_ref().unwrap()["citations"][0]["url"],
                    "https://example.com"
                );
            }
            other => panic!("expected text content, got {other:?}"),
        }
        match &parsed.content[1] {
            ContentPart::Thinking {
                text,
                signature,
                metadata,
            } => {
                assert_eq!(text, "I should inspect the files.");
                assert_eq!(signature.as_deref(), Some("sig_1"));
                assert_eq!(
                    metadata.as_ref().unwrap()["kind"],
                    "anthropic.content_block"
                );
            }
            other => panic!("expected thinking content, got {other:?}"),
        }
        match &parsed.content[2] {
            ContentPart::Json { value, metadata } => {
                assert_eq!(value["type"], "server_tool_use");
                assert_eq!(
                    metadata.as_ref().unwrap()["kind"],
                    "anthropic.content_block"
                );
            }
            other => panic!("expected json content, got {other:?}"),
        }
        assert!(matches!(
            &parsed.parts[4],
            ResponsePart::ServerToolUse { call }
                if call.id.as_deref() == Some("srv_1") && call.name == "web_search"
        ));
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].id, "toolu_a");
        assert_eq!(parsed.tool_calls[1].input, json!({"path":"README.md"}));
        assert_eq!(parsed.usage.unwrap().total_tokens, Some(62));
    }
}
