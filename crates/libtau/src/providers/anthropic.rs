use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::{
        ContentPart, ConversationItem, MediaData, ResponsePart, ResponseStop, ResponseStopReason,
        ServerToolUse, TauSession, ToolResult, ToolUse,
    },
    providers::{
        ModelCosts, ModelMetadata, Provider, ProviderApi, ProviderApiConfig, ProviderError,
        ProviderModelDetails, ProviderResponse, TokenUsage,
        common::{binary_content_as_text, json_as_text, tool_result_json},
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
    default_models,
};
const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";
const DEFAULT_MAX_TOKENS: u64 = 64_000;

#[derive(Debug, Clone, Copy, Serialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum AnthropicAdaptiveThinkingEffort {
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnthropicThinkingConfig {
    /// Anthropic adaptive thinking: `thinking: { type: "adaptive" }`.
    /// The optional effort is sent separately as `output_config.effort`.
    Adaptive {
        effort: Option<AnthropicAdaptiveThinkingEffort>,
    },
    /// Anthropic extended thinking: `thinking: { type: "enabled", budget_tokens: N }`.
    Enabled {
        budget_tokens: u32,
    },
    // FIXME: Handle this
    Disabled,
}

#[derive(Debug)]
pub struct AnthropicModelDetails {
    pub thinking: AnthropicThinkingConfig,
}

impl ProviderModelDetails for AnthropicModelDetails {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug)]
pub struct AnthropicModelCosts {
    pub input_token: f64,
    pub output_token: f64,
    pub cache_hit_token: Option<f64>,
    pub cache_write_5m_token: Option<f64>,
    pub cache_write_1h_token: Option<f64>,
    pub web_search: Option<f64>,
}

impl ModelCosts for AnthropicModelCosts {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn anthropic_model_details(thinking: AnthropicThinkingConfig) -> Arc<dyn ProviderModelDetails> {
    Arc::new(AnthropicModelDetails { thinking })
}

const THINKING_MAX: AnthropicThinkingConfig = AnthropicThinkingConfig::Adaptive {
    effort: Some(AnthropicAdaptiveThinkingEffort::Max),
};
const THINKING_XHIGH: AnthropicThinkingConfig = AnthropicThinkingConfig::Adaptive {
    effort: Some(AnthropicAdaptiveThinkingEffort::XHigh),
};
const THINKING_HIGH: AnthropicThinkingConfig = AnthropicThinkingConfig::Adaptive {
    effort: Some(AnthropicAdaptiveThinkingEffort::High),
};

fn anthropic_model_metadata(
    name: impl Into<String>,
    id: impl Into<String>,
    context_length: u64,
    max_tokens: u64,
    thinking: AnthropicThinkingConfig,
) -> ModelMetadata {
    ModelMetadata {
        name: name.into(),
        id: id.into(),
        context_length,
        max_tokens,
        provider_details: Some(anthropic_model_details(thinking)),
        costs: None,
    }
}

fn default_models() -> Vec<ModelMetadata> {
    vec![
        anthropic_model_metadata(
            "opus4-7-max",
            "claude-opus-4-7",
            1_000_000,
            128_000,
            THINKING_MAX,
        ),
        anthropic_model_metadata(
            "opus4-7-xhigh",
            "claude-opus-4-7",
            1_000_000,
            128_000,
            THINKING_XHIGH,
        ),
        anthropic_model_metadata(
            "opus-4-7-high",
            "claude-opus-4-7",
            1_000_000,
            128_000,
            THINKING_HIGH,
        ),
        anthropic_model_metadata(
            "opus-4-6-max",
            "claude-opus-4-6",
            200_000,
            128_000,
            THINKING_MAX,
        ),
        anthropic_model_metadata(
            "sonnet-4-6-xhigh",
            "claude-sonnet-4-6",
            1_000_000,
            64_000,
            THINKING_XHIGH,
        ),
        anthropic_model_metadata(
            "haiku-4-5-no-thinking",
            "claude-haiku-4-5-20251001",
            200_000,
            64_000,
            AnthropicThinkingConfig::Disabled,
        ),
        anthropic_model_metadata(
            "haiku-4-5-medium",
            "claude-haiku-4-5-20251001",
            200_000,
            64_000,
            AnthropicThinkingConfig::Enabled {
                budget_tokens: 4000,
            },
        ),
    ]
}
const ANTHROPIC_VERSION: &str = "2023-06-01";

#[derive(Debug, Clone)]
pub struct AnthropicProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
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
        Self {
            client: reqwest::Client::new(),
            api_key: api_key.into(),
            base_url: DEFAULT_BASE_URL.to_string(),
            cache_ttl: Some(AnthropicCacheTtl::FiveMinutes),
            web_search: Some(AnthropicWebSearchConfig::enabled()),
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
            cache_ttl: Some(AnthropicCacheTtl::FiveMinutes),
            web_search: None,
        }
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
        let (request, conversation) =
            build_request(session, self.cache_ttl, self.web_search.as_ref())?;
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
        let native_content = response.content.clone();
        let parsed = parse_response(response)?;
        record_anthropic_response(session, conversation, native_content)?;
        Ok(parsed)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct AnthropicRequest {
    model: String,
    max_tokens: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    cache_control: Option<AnthropicCacheControl>,
    #[serde(skip_serializing_if = "Option::is_none")]
    thinking: Option<AnthropicThinking>,
    #[serde(skip_serializing_if = "Option::is_none")]
    output_config: Option<AnthropicOutputConfig>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    system: Vec<AnthropicContent>,
    messages: Vec<AnthropicMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<AnthropicTool>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum AnthropicThinking {
    Adaptive,
    Enabled {
        budget_tokens: u32,
        #[serde(skip_serializing_if = "Option::is_none")]
        display: Option<&'static str>,
    },
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
struct AnthropicOutputConfig {
    effort: AnthropicAdaptiveThinkingEffort,
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

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "lowercase", deny_unknown_fields)]
pub struct AnthropicWebSearchConfig {
    // is this tool enabled (defaults to true)
    #[serde(default)]
    pub enabled: Option<bool>,
    #[serde(default)]
    pub max_uses: Option<u32>,
    #[serde(default)]
    pub allowed_domains: Option<Vec<String>>,
    #[serde(default)]
    pub blocked_domains: Option<Vec<String>>,
}

impl AnthropicWebSearchConfig {
    fn enabled() -> Self {
        Self::default()
    }

    fn options(&self) -> serde_json::Map<String, Value> {
        let mut options = serde_json::Map::new();

        if let Some(max_uses) = self.max_uses {
            options.insert("max_uses".to_string(), Value::from(max_uses));
        }
        if let Some(allowed_domains) = &self.allowed_domains {
            options.insert(
                "allowed_domains".to_string(),
                Value::Array(allowed_domains.iter().cloned().map(Value::String).collect()),
            );
        }
        if let Some(blocked_domains) = &self.blocked_domains {
            options.insert(
                "blocked_domains".to_string(),
                Value::Array(blocked_domains.iter().cloned().map(Value::String).collect()),
            );
        }

        options
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
        type_id: &'static str,
        name: &'static str,
        #[serde(flatten)]
        options: serde_json::Map<String, Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq)]
struct AnthropicConversationData {
    system: Vec<AnthropicContent>,
    messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Clone, Deserialize)]
struct AnthropicResponse {
    #[serde(default)]
    content: Vec<Value>,
    usage: Option<AnthropicUsage>,
    stop_reason: Option<String>,
    stop_sequence: Option<String>,
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
    session: &mut TauSession,
    cache_ttl: Option<AnthropicCacheTtl>,
    web_search: Option<&AnthropicWebSearchConfig>,
) -> Result<(AnthropicRequest, AnthropicConversationData), ProviderError> {
    let model = session
        .model_metadata()
        .ok_or(ProviderError::MissingModel)?
        .clone();
    let mut conversation = session
        .provider_conversation_data::<AnthropicConversationData>()
        .unwrap_or_default();
    append_anthropic_input_items(&mut conversation, pending_input_items(session))?;

    let tools = anthropic_tools(session, web_search);
    let (thinking, output_config) = anthropic_thinking_config(&model);

    Ok((
        AnthropicRequest {
            model: model.id.clone(),
            max_tokens: anthropic_max_tokens(&model),
            cache_control: cache_ttl.map(AnthropicCacheControl::new),
            thinking,
            output_config,
            system: conversation.system.clone(),
            messages: conversation.messages.clone(),
            tools,
        },
        conversation,
    ))
}

fn anthropic_thinking_config(
    model: &ModelMetadata,
) -> (Option<AnthropicThinking>, Option<AnthropicOutputConfig>) {
    let Some(details) = model
        .provider_details
        .as_deref()
        .and_then(|details| details.as_any().downcast_ref::<AnthropicModelDetails>())
    else {
        return (None, None);
    };

    match details.thinking {
        AnthropicThinkingConfig::Adaptive { effort } => (
            Some(AnthropicThinking::Adaptive),
            effort.map(|effort| AnthropicOutputConfig { effort }),
        ),
        AnthropicThinkingConfig::Enabled { budget_tokens } => (
            Some(AnthropicThinking::Enabled {
                budget_tokens,
                display: Some("summarized"),
            }),
            None,
        ),
        AnthropicThinkingConfig::Disabled => (None, None),
    }
}

fn anthropic_max_tokens(model: &ModelMetadata) -> u64 {
    if model.max_tokens == 0 {
        DEFAULT_MAX_TOKENS
    } else {
        model.max_tokens
    }
}

fn append_anthropic_input_items(
    data: &mut AnthropicConversationData,
    items: &[ConversationItem],
) -> Result<(), ProviderError> {
    for item in items {
        match item {
            ConversationItem::System { content } => {
                data.system.extend(input_content_parts(content)?)
            }
            ConversationItem::User { content } => push_message(
                &mut data.messages,
                AnthropicRole::User,
                input_content_parts(content)?,
            ),
            ConversationItem::ToolResult { results } => push_message(
                &mut data.messages,
                AnthropicRole::User,
                results
                    .iter()
                    .map(tool_result_content)
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            ConversationItem::Agent { .. }
            | ConversationItem::ToolUse { .. }
            | ConversationItem::ResponseStop { .. } => {}
        }
    }

    Ok(())
}

fn record_anthropic_response(
    session: &mut TauSession,
    mut conversation: AnthropicConversationData,
    native_content: Vec<Value>,
) -> Result<(), ProviderError> {
    if native_content.is_empty() {
        session.set_provider_conversation_data(&conversation)?;
        return Ok(());
    }

    let content = native_content
        .into_iter()
        .map(serde_json::from_value)
        .collect::<Result<Vec<AnthropicContent>, _>>()?;
    push_message(
        &mut conversation.messages,
        AnthropicRole::Assistant,
        content,
    );
    session.set_provider_conversation_data(&conversation)
}

fn pending_input_items(session: &TauSession) -> &[ConversationItem] {
    let items = &session.conversation().items;
    let start = items
        .iter()
        .rposition(|item| {
            matches!(
                item,
                ConversationItem::Agent { .. }
                    | ConversationItem::ToolUse { .. }
                    | ConversationItem::ResponseStop { .. }
            )
        })
        .map_or(0, |index| index + 1);
    &items[start..]
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

    if let Some(web_search) = web_search.filter(|config| config.enabled.unwrap_or(true)) {
        tools.push(AnthropicTool::Server {
            type_id: "web_search_20260209",
            name: "web_search",
            options: web_search.options(),
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

    let is_refusal = response.stop_reason.as_deref() == Some("refusal");
    let stop = anthropic_stop(
        response.stop_reason.as_deref(),
        response.stop_sequence.clone(),
    );

    for part in response.content {
        match part.get("type").and_then(Value::as_str) {
            Some("text") => {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::Response("Anthropic text block missing text".to_string())
                })?;
                let content_part = if is_refusal {
                    ContentPart::Refusal {
                        text: text.to_string(),
                        metadata: content_block_metadata(&part),
                    }
                } else {
                    ContentPart::Text {
                        text: text.to_string(),
                        metadata: content_block_metadata(&part),
                    }
                };
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
                    metadata: Some(provider_metadata("anthropic.server_tool_use")),
                };
                parts.push(ResponsePart::ServerToolUse { call });
                let content_part = ContentPart::json(part.clone());
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
                let content_part = ContentPart::Thinking {
                    text: text.to_string(),
                    signature,
                    metadata: None,
                };
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            Some("redacted_thinking") => {
                let content_part = ContentPart::json(part.clone());
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            _ => {
                let content_part = ContentPart::json(part.clone());
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
        }
    }

    if let Some(stop) = stop {
        parts.push(ResponsePart::Stop { stop });
    }

    Ok(ProviderResponse {
        parts,
        content,
        tool_calls,
        usage,
    })
}

fn anthropic_stop(reason: Option<&str>, sequence: Option<String>) -> Option<ResponseStop> {
    let reason = match reason? {
        "end_turn" => ResponseStopReason::EndTurn,
        "max_tokens" => ResponseStopReason::MaxTokens,
        "stop_sequence" => ResponseStopReason::StopSequence { sequence },
        "tool_use" => ResponseStopReason::ToolUse,
        "pause_turn" => ResponseStopReason::PauseTurn,
        "refusal" => ResponseStopReason::Refusal,
        other => ResponseStopReason::Other {
            value: other.to_string(),
        },
    };

    Some(ResponseStop {
        reason,
        metadata: Some(serde_json::json!({
            "provider": PROVIDER_NAME,
            "kind": "anthropic.stop",
        })),
    })
}

fn content_block_metadata(block: &Value) -> Option<Value> {
    block.get("citations").cloned().map(|citations| {
        serde_json::json!({
            "provider": PROVIDER_NAME,
            "kind": "citations",
            "citations": citations,
        })
    })
}

fn provider_metadata(kind: &str) -> Value {
    serde_json::json!({
        "provider": PROVIDER_NAME,
        "kind": kind,
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
            ContentPart::Refusal { text, .. } => Ok(AnthropicContent::Text { text: text.clone() }),
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
    use crate::tools::{ToolDefinition, ToolOutput};
    use serde_json::json;

    fn callback(input: Value) -> Result<ToolOutput, crate::tools::ToolCallError> {
        Ok(ToolOutput::json(input))
    }

    fn test_model() -> ModelMetadata {
        ModelMetadata {
            name: "claude-sonnet-4-5".to_string(),
            id: "claude-sonnet-4-5".to_string(),
            context_length: 200_000,
            max_tokens: 64_000,
            provider_details: None,
            costs: None,
        }
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
        let mut session = context.session(AnthropicProvider::new("test-key"), test_model());
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
        session
            .set_provider_conversation_data(&AnthropicConversationData {
                system: vec![AnthropicContent::Text {
                    text: "be helpful".to_string(),
                }],
                messages: vec![
                    AnthropicMessage {
                        role: AnthropicRole::User,
                        content: vec![AnthropicContent::Text {
                            text: "hello".to_string(),
                        }],
                    },
                    AnthropicMessage {
                        role: AnthropicRole::Assistant,
                        content: vec![
                            AnthropicContent::Text {
                                text: "I'll call a tool.".to_string(),
                            },
                            AnthropicContent::Thinking {
                                thinking: "Need to echo.".to_string(),
                                signature: Some("sig_1".to_string()),
                            },
                            AnthropicContent::ToolUse {
                                id: "toolu_1".to_string(),
                                name: "echo".to_string(),
                                input: json!({"text":"hello"}),
                            },
                        ],
                    },
                ],
            })
            .unwrap();
        let web_search = AnthropicWebSearchConfig {
            enabled: Some(true),
            max_uses: Some(3),
            allowed_domains: Some(vec!["example.com".to_string()]),
            blocked_domains: Some(vec!["bad.example".to_string()]),
        };
        let request = build_request(
            &mut session,
            Some(AnthropicCacheTtl::FiveMinutes),
            Some(&web_search),
        )
        .unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["model"], "claude-sonnet-4-5");
        assert_eq!(value["max_tokens"], 64_000);
        assert_eq!(value["cache_control"]["type"], "ephemeral");
        assert!(value["cache_control"].get("ttl").is_none());
        assert_eq!(value["system"][0]["type"], "text");
        assert_eq!(value["tools"][0]["name"], "echo");
        assert_eq!(value["tools"][0]["strict"], true);
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
    fn sends_adaptive_thinking_and_effort_for_default_anthropic_models() {
        let context = crate::context::TauContext::new();
        let model = default_models()
            .into_iter()
            .find(|model| model.name == "sonnet-4-6-xhigh")
            .unwrap();
        let mut session = context.session(AnthropicProvider::new("test-key"), model);

        let request = build_request(&mut session, None, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["thinking"], json!({"type": "adaptive"}));
        assert_eq!(value["output_config"], json!({"effort": "xhigh"}));
    }

    #[test]
    fn sends_enabled_thinking_budget_for_legacy_thinking_models() {
        let context = crate::context::TauContext::new();
        let model = default_models()
            .into_iter()
            .find(|model| model.name == "haiku-4-5-medium")
            .unwrap();
        let mut session = context.session(AnthropicProvider::new("test-key"), model);

        let request = build_request(&mut session, None, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(
            value["thinking"],
            json!({"type": "enabled", "budget_tokens": 4000, "display": "summarized"})
        );
        assert!(value.get("output_config").is_none());
    }

    #[test]
    fn omits_thinking_for_disabled_or_unknown_model_metadata() {
        let context = crate::context::TauContext::new();
        let model = default_models()
            .into_iter()
            .find(|model| model.name == "haiku-4-5-no-thinking")
            .unwrap();
        let mut session = context.session(AnthropicProvider::new("test-key"), model);
        let request = build_request(&mut session, None, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert!(value.get("thinking").is_none());
        assert!(value.get("output_config").is_none());

        let mut custom_session = context.session(AnthropicProvider::new("test-key"), test_model());
        let custom_request = build_request(&mut custom_session, None, None).unwrap();
        let custom_value = serde_json::to_value(custom_request.0).unwrap();

        assert!(custom_value.get("thinking").is_none());
        assert!(custom_value.get("output_config").is_none());
    }

    #[test]
    fn configured_model_without_metadata_uses_anthropic_max_tokens_default() {
        let context = crate::context::TauContext::new();
        let mut session = context.session(AnthropicProvider::new("test-key"), "llama.cpp-model");

        let request = build_request(&mut session, None, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["model"], "llama.cpp-model");
        assert_eq!(value["max_tokens"], DEFAULT_MAX_TOKENS);
    }

    #[test]
    fn configures_one_hour_automatic_cache_ttl() {
        let context = crate::context::TauContext::new();
        let mut session = context.session(AnthropicProvider::new("test-key"), test_model());
        let request = build_request(&mut session, Some(AnthropicCacheTtl::OneHour), None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

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
        assert_eq!(web_search.enabled, Some(true));
        assert_eq!(web_search.max_uses, Some(2));
        assert_eq!(
            web_search.allowed_domains,
            Some(vec!["example.com".to_string()])
        );
        assert_eq!(
            web_search.blocked_domains,
            Some(vec!["spam.example".to_string()])
        );
    }

    #[test]
    fn disables_web_search_when_configured_off() {
        let context = crate::context::TauContext::new();
        let mut session = context.session(AnthropicProvider::new("test-key"), test_model());
        let web_search = AnthropicWebSearchConfig {
            enabled: Some(false),
            max_uses: Some(1),
            allowed_domains: None,
            blocked_domains: None,
        };
        let request = build_request(
            &mut session,
            Some(AnthropicCacheTtl::FiveMinutes),
            Some(&web_search),
        )
        .unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert!(value.get("tools").is_none());
    }

    #[test]
    fn parses_text_and_parallel_tool_calls() {
        let response = AnthropicResponse {
            usage: Some(AnthropicUsage {
                input_tokens: Some(50),
                output_tokens: Some(12),
            }),
            stop_reason: Some("tool_use".to_string()),
            stop_sequence: None,
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
                assert_eq!(metadata.as_ref().unwrap()["kind"], "citations");
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
                assert!(metadata.is_none());
            }
            other => panic!("expected thinking content, got {other:?}"),
        }
        match &parsed.content[2] {
            ContentPart::Json { value, metadata } => {
                assert_eq!(value["type"], "server_tool_use");
                assert!(metadata.is_none());
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
