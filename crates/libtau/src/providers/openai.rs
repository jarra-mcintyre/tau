use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::{
        ContentPart, ConversationItem, ResponsePart, ResponseStop, ResponseStopReason, TauSession,
        ToolResult, ToolUse,
    },
    providers::{
        Provider, ProviderApi, ProviderApiConfig, ProviderError, ProviderResponse, TokenUsage,
        common::{
            assistant_content_as_text, binary_content_as_text, json_as_text, media_to_url,
            tool_result_json,
        },
    },
};

pub const PROVIDER_NAME: &str = "openai";
pub const API_NAME: &str = "openai_responses";
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";
pub const API: ProviderApi = ProviderApi {
    name: API_NAME,
    api_key_env: API_KEY_ENV,
    display_name: "OpenAI",
    build: build_provider,
};
const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug, Clone)]
pub struct OpenAiProvider {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiState {
    pub previous_response_id: Option<String>,
}

fn build_provider(config: ProviderApiConfig) -> Result<Arc<dyn Provider>, ProviderError> {
    Ok(match config.base_url {
        Some(base_url) => Arc::new(OpenAiProvider::with_base_url(config.api_key, base_url)),
        None => Arc::new(OpenAiProvider::new(config.api_key)),
    })
}

impl OpenAiProvider {
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
            base_url: base_url.into().trim_end_matches('/').to_string(),
        }
    }
}

#[async_trait]
impl Provider for OpenAiProvider {
    fn name(&self) -> &'static str {
        PROVIDER_NAME
    }

    async fn respond(&self, session: &mut TauSession) -> Result<ProviderResponse, ProviderError> {
        let request = build_request(session)?;
        let url = format!("{}/responses", self.base_url);
        let request_body = serde_json::to_string_pretty(&request)?;
        tracing::debug!(
            target: "tau::providers::openai",
            %url,
            body = %request_body,
            "request"
        );

        let response = self
            .client
            .post(&url)
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        tracing::debug!(
            target: "tau::providers::openai",
            %status,
            body = %body,
            "response"
        );
        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        let response: OpenAiResponse = serde_json::from_str(&body)?;
        session.set_provider_state(
            PROVIDER_NAME,
            OpenAiState {
                previous_response_id: Some(response.id.clone()),
            },
        );

        parse_response(response)
    }
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiRequest {
    model: String,
    input: Vec<OpenAiInputItem>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAiTool>,
    parallel_tool_calls: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    previous_response_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
enum OpenAiInputItem {
    Message(OpenAiMessage),
    FunctionCall(OpenAiFunctionCallItem),
    FunctionCallOutput(OpenAiFunctionCallOutputItem),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiMessage {
    role: OpenAiRole,
    content: Vec<OpenAiContent>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(rename_all = "snake_case")]
enum OpenAiRole {
    System,
    User,
    Assistant,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiContent {
    InputText { text: String },
    OutputText { text: String },
    InputImage { image_url: String },
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiFunctionCallItem {
    #[serde(rename = "type")]
    kind: &'static str,
    call_id: String,
    name: String,
    arguments: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiFunctionCallOutputItem {
    #[serde(rename = "type")]
    kind: &'static str,
    call_id: String,
    output: String,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
enum OpenAiTool {
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
        description: String,
        parameters: Value,
    },
    Server {
        #[serde(rename = "type")]
        kind: &'static str,
    },
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiResponse {
    id: String,
    #[serde(default)]
    output: Vec<Value>,
    usage: Option<OpenAiUsage>,
    status: Option<String>,
    incomplete_details: Option<OpenAiIncompleteDetails>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiIncompleteDetails {
    reason: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiUsage {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    total_tokens: Option<u64>,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiFunctionCallOutput {
    call_id: String,
    name: String,
    arguments: String,
}

fn build_request(session: &TauSession) -> Result<OpenAiRequest, ProviderError> {
    let model = session
        .model()
        .ok_or(ProviderError::MissingModel)?
        .to_string();

    let previous_response_id = session
        .provider_state::<OpenAiState>(PROVIDER_NAME)
        .and_then(|state| state.previous_response_id.clone());
    let items = incremental_input_items(session, previous_response_id.as_deref());

    let mut input = Vec::new();
    for item in items {
        match item {
            ConversationItem::System { content } => {
                input.push(OpenAiInputItem::Message(OpenAiMessage {
                    role: OpenAiRole::System,
                    content: input_content_parts(content)?,
                }))
            }
            ConversationItem::User { content } => {
                input.push(OpenAiInputItem::Message(OpenAiMessage {
                    role: OpenAiRole::User,
                    content: input_content_parts(content)?,
                }))
            }
            ConversationItem::Agent { content } => {
                input.push(OpenAiInputItem::Message(OpenAiMessage {
                    role: OpenAiRole::Assistant,
                    content: output_content_parts(content),
                }))
            }
            ConversationItem::ToolUse { calls } => {
                for call in calls {
                    input.push(OpenAiInputItem::FunctionCall(OpenAiFunctionCallItem {
                        kind: "function_call",
                        call_id: call.id.clone(),
                        name: call.name.clone(),
                        arguments: serde_json::to_string(&call.input)?,
                    }));
                }
            }
            ConversationItem::ToolResult { results } => {
                for result in results {
                    input.push(OpenAiInputItem::FunctionCallOutput(
                        OpenAiFunctionCallOutputItem {
                            kind: "function_call_output",
                            call_id: result.call_id.clone(),
                            output: tool_result_output(result)?,
                        },
                    ));
                }
            }
            ConversationItem::ResponseStop { .. } => {}
        }
    }

    let tools = openai_tools(session);

    Ok(OpenAiRequest {
        model,
        input,
        tools,
        parallel_tool_calls: true,
        previous_response_id,
    })
}

fn openai_tools(session: &TauSession) -> Vec<OpenAiTool> {
    let mut tools: Vec<_> = session
        .context()
        .tools()
        .map(|tool| OpenAiTool::Function {
            kind: "function",
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        })
        .collect();

    tools.push(OpenAiTool::Server { kind: "web_search" });

    tools
}

fn incremental_input_items<'a>(
    session: &'a TauSession,
    previous_response_id: Option<&str>,
) -> &'a [ConversationItem] {
    if previous_response_id.is_none() {
        return &session.conversation().items;
    }

    let items = &session.conversation().items;
    let start = items
        .iter()
        .rposition(|item| matches!(item, ConversationItem::ToolUse { .. }))
        .unwrap_or(items.len());
    &items[start..]
}

fn parse_response(response: OpenAiResponse) -> Result<ProviderResponse, ProviderError> {
    let usage = response.usage.map(|usage| TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
    });
    let mut parts = Vec::new();
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    let mut saw_refusal = false;

    for item in response.output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                saw_refusal |= parse_message_output_item(item, &mut parts, &mut content)?;
            }
            Some("function_call") => {
                let function_call: OpenAiFunctionCallOutput = serde_json::from_value(item.clone())?;
                let input = serde_json::from_str(&function_call.arguments).map_err(|error| {
                    ProviderError::Response(format!(
                        "function call {} arguments were not JSON: {error}",
                        function_call.call_id
                    ))
                })?;
                let call = ToolUse {
                    id: function_call.call_id,
                    name: function_call.name,
                    input,
                };
                parts.push(ResponsePart::ToolUse { call: call.clone() });
                tool_calls.push(call);
            }
            Some("reasoning") => {
                let content_part = openai_reasoning_content(item.clone());
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            _ => {
                let content_part = ContentPart::json_with_metadata(
                    item.clone(),
                    raw_metadata("openai.output_item", item),
                );
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
        }
    }

    if let Some(stop) = openai_stop(
        response.status.as_deref(),
        response.incomplete_details.as_ref(),
        saw_refusal,
        !tool_calls.is_empty(),
    ) {
        parts.push(ResponsePart::Stop { stop });
    }

    Ok(ProviderResponse {
        parts,
        content,
        tool_calls,
        usage,
    })
}

fn openai_stop(
    status: Option<&str>,
    incomplete_details: Option<&OpenAiIncompleteDetails>,
    saw_refusal: bool,
    has_tool_calls: bool,
) -> Option<ResponseStop> {
    let reason = if saw_refusal {
        ResponseStopReason::Refusal
    } else if has_tool_calls {
        ResponseStopReason::ToolUse
    } else {
        match status? {
            "completed" => ResponseStopReason::EndTurn,
            "incomplete" => {
                match incomplete_details.and_then(|details| details.reason.as_deref()) {
                    Some("max_output_tokens") => ResponseStopReason::MaxTokens,
                    Some("content_filter") => ResponseStopReason::Refusal,
                    Some(other) => ResponseStopReason::Other {
                        value: other.to_string(),
                    },
                    None => ResponseStopReason::Other {
                        value: "incomplete".to_string(),
                    },
                }
            }
            other => ResponseStopReason::Other {
                value: other.to_string(),
            },
        }
    };

    Some(ResponseStop {
        reason,
        metadata: Some(serde_json::json!({
            "provider": PROVIDER_NAME,
            "kind": "openai.stop",
            "status": status,
            "incomplete_details": incomplete_details.map(|details| serde_json::json!({
                "reason": details.reason,
            })),
        })),
    })
}

fn parse_message_output_item(
    item: Value,
    parts_out: &mut Vec<ResponsePart>,
    content: &mut Vec<ContentPart>,
) -> Result<bool, ProviderError> {
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::Response("OpenAI message output missing content array".to_string())
        })?;

    let mut saw_refusal = false;

    for part in parts {
        let metadata = raw_metadata("openai.message_content", part.clone());
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::Response("OpenAI output_text missing text".to_string())
                })?;
                let content_part = ContentPart::text_with_metadata(text, metadata);
                parts_out.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            Some("refusal") => {
                let refusal = part.get("refusal").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::Response("OpenAI refusal missing refusal text".to_string())
                })?;
                saw_refusal = true;
                let content_part = ContentPart::refusal_with_metadata(refusal, metadata);
                parts_out.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
            _ => {
                let content_part = ContentPart::json_with_metadata(part.clone(), metadata);
                parts_out.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
                content.push(content_part);
            }
        }
    }

    Ok(saw_refusal)
}

fn openai_reasoning_content(item: Value) -> ContentPart {
    let text = item
        .get("summary")
        .and_then(Value::as_array)
        .map(|summary| {
            summary
                .iter()
                .filter_map(|part| {
                    part.get("text")
                        .or_else(|| part.get("summary"))
                        .and_then(Value::as_str)
                })
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();

    ContentPart::thinking_with_metadata(text, None, raw_metadata("openai.reasoning", item))
}

fn raw_metadata(kind: &str, raw: Value) -> Value {
    serde_json::json!({
        "provider": PROVIDER_NAME,
        "kind": kind,
        "raw": raw,
    })
}

fn input_content_parts(parts: &[ContentPart]) -> Result<Vec<OpenAiContent>, ProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => Ok(OpenAiContent::InputText { text: text.clone() }),
            ContentPart::Json { value, .. } => Ok(OpenAiContent::InputText {
                text: json_as_text(value)?,
            }),
            ContentPart::Image {
                media_type, data, ..
            } => Ok(OpenAiContent::InputImage {
                image_url: media_to_url(media_type, data),
            }),
            ContentPart::Binary {
                media_type, data, ..
            } => Ok(OpenAiContent::InputText {
                text: binary_content_as_text(media_type, data),
            }),
            ContentPart::Thinking { text, .. } => Ok(OpenAiContent::InputText {
                text: format!("[thinking: {text}]"),
            }),
            ContentPart::Refusal { text, .. } => {
                Ok(OpenAiContent::InputText { text: text.clone() })
            }
        })
        .collect()
}

fn output_content_parts(parts: &[ContentPart]) -> Vec<OpenAiContent> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => OpenAiContent::OutputText { text: text.clone() },
            ContentPart::Thinking { text, .. } => OpenAiContent::OutputText {
                text: format!("[thinking: {text}]"),
            },
            ContentPart::Refusal { text, .. } => OpenAiContent::OutputText { text: text.clone() },
            part => OpenAiContent::OutputText {
                text: assistant_content_as_text(part),
            },
        })
        .collect()
}

fn tool_result_output(result: &ToolResult) -> Result<String, ProviderError> {
    tool_result_json(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::ToolResult,
        tools::{ToolDefinition, ToolOutput},
    };
    use serde_json::json;

    fn callback(input: Value) -> Result<ToolOutput, crate::tools::ToolCallError> {
        Ok(ToolOutput::json(input))
    }

    #[test]
    fn builds_responses_api_request_from_complete_history() {
        let mut context = crate::context::TauContext::new();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();
        let mut session = context.session(OpenAiProvider::new("test-key"), "gpt-4.1-mini");
        session.push_system_text("be helpful");
        session.push_user_text("hello");
        session.push_item(ConversationItem::ToolUse {
            calls: vec![ToolUse {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: json!({"text":"hello"}),
            }],
        });
        session.push_item(ConversationItem::ToolResult {
            results: vec![ToolResult {
                call_id: "call_1".to_string(),
                name: "echo".to_string(),
                content: vec![ContentPart::json(json!({"text":"hello"}))],
                error: None,
            }],
        });
        let request = build_request(&session).unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["model"], "gpt-4.1-mini");
        assert_eq!(value["parallel_tool_calls"], true);
        assert!(value.get("previous_response_id").is_none());
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][1]["type"], "web_search");
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][2]["type"], "function_call");
        assert_eq!(value["input"][3]["type"], "function_call_output");
    }

    #[test]
    fn uses_previous_response_id_for_incremental_requests() {
        let mut context = crate::context::TauContext::new();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();
        let mut session = context.session(OpenAiProvider::new("test-key"), "gpt-4.1-mini");
        session.push_system_text("be helpful");
        session.push_user_text("hello");
        session.push_agent_text("I'll call a tool.");
        session.push_item(ConversationItem::ToolUse {
            calls: vec![ToolUse {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: json!({"text":"hello"}),
            }],
        });
        session.push_item(ConversationItem::ToolResult {
            results: vec![ToolResult {
                call_id: "call_1".to_string(),
                name: "echo".to_string(),
                content: vec![ContentPart::json(json!({"text":"hello"}))],
                error: None,
            }],
        });
        session.set_provider_state(
            PROVIDER_NAME,
            OpenAiState {
                previous_response_id: Some("resp_previous".to_string()),
            },
        );

        let request = build_request(&session).unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["previous_response_id"], "resp_previous");
        assert_eq!(value["input"].as_array().unwrap().len(), 2);
        assert_eq!(value["input"][0]["type"], "function_call");
        assert_eq!(value["input"][1]["type"], "function_call_output");
    }

    #[test]
    fn parses_text_and_parallel_function_calls() {
        let response = OpenAiResponse {
            id: "resp_1".to_string(),
            usage: Some(OpenAiUsage {
                input_tokens: Some(100),
                output_tokens: Some(20),
                total_tokens: Some(120),
            }),
            status: Some("completed".to_string()),
            incomplete_details: None,
            output: vec![
                json!({
                    "type": "message",
                    "content": [{
                        "type": "output_text",
                        "text": "I'll check.",
                        "annotations": [{
                            "type": "url_citation",
                            "start_index": 0,
                            "end_index": 5,
                            "url": "https://example.com",
                            "title": "Example"
                        }]
                    }]
                }),
                json!({
                    "type": "function_call",
                    "call_id": "call_a",
                    "name": "read_file",
                    "arguments": "{\"path\":\"Cargo.toml\"}"
                }),
                json!({
                    "type": "function_call",
                    "call_id": "call_b",
                    "name": "read_file",
                    "arguments": "{\"path\":\"README.md\"}"
                }),
                json!({
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": "I should inspect the files."}]
                }),
                json!({
                    "type": "web_search_call",
                    "id": "ws_1",
                    "status": "completed",
                    "action": {"type": "search", "query": "latest news about AI"}
                }),
            ],
        };

        let parsed = parse_response(response).unwrap();

        assert_eq!(parsed.content.len(), 3);
        match &parsed.content[0] {
            ContentPart::Text { text, metadata } => {
                assert_eq!(text, "I'll check.");
                let metadata = metadata.as_ref().unwrap();
                assert_eq!(metadata["kind"], "openai.message_content");
                assert_eq!(metadata["raw"]["annotations"][0]["type"], "url_citation");
            }
            other => panic!("expected text content, got {other:?}"),
        }
        match &parsed.content[1] {
            ContentPart::Thinking { text, metadata, .. } => {
                assert_eq!(text, "I should inspect the files.");
                assert_eq!(metadata.as_ref().unwrap()["kind"], "openai.reasoning");
            }
            other => panic!("expected thinking content, got {other:?}"),
        }
        match &parsed.content[2] {
            ContentPart::Json { value, metadata } => {
                assert_eq!(value["type"], "web_search_call");
                assert_eq!(metadata.as_ref().unwrap()["kind"], "openai.output_item");
            }
            other => panic!("expected json content, got {other:?}"),
        }
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].id, "call_a");
        assert_eq!(parsed.tool_calls[1].input, json!({"path":"README.md"}));
        assert_eq!(parsed.usage.unwrap().total_tokens, Some(120));
    }
}
