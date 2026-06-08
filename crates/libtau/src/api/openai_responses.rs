use std::{
    sync::Arc,
    time::{SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    api::{
        ModelApi, ModelApiFactory, ProviderError, TokenUsage,
        common::{
            assistant_content_as_text, binary_content_as_text, media_to_url, tool_result_json,
        },
    },
    context::{
        Annotation, Citation, ContentPart, ConversationItem, ResponsePart, ResponseStop,
        ResponseStopReason, ServerToolUse, TauResponse, TauSession, ToolResult, ToolUse,
    },
    providers::{OAuthCredentials, ProviderConfig, ProviderCredentials, ThinkingEffort},
};

pub const API: ModelApiFactory = ModelApiFactory {
    name: "openai_responses",
    build: build_api,
};

#[derive(Debug, Clone)]
pub struct OpenAiResponsesApi {
    client: reqwest::Client,
    auth: ProviderCredentials,
    base_url: String,
    web_search: Option<OpenAiWebSearchConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
struct OpenAiOptions {
    #[serde(default)]
    web_search: Option<OpenAiWebSearchConfig>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq, Default)]
pub struct OpenAiWebSearchConfig {
    #[serde(default)]
    pub enabled: Option<bool>,
}

const OAUTH_EXPIRY_SKEW_MILLIS: i64 = 60_000;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OpenAiResponsesState {
    pub previous_response_id: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
struct OpenAiConversationData {
    previous_response_id: Option<String>,
    output: Vec<Value>,
}

fn build_api(config: ProviderConfig) -> Result<Arc<dyn ModelApi>, ProviderError> {
    let options = OpenAiOptions::from_value(config.options)?;
    Ok(Arc::new(OpenAiResponsesApi {
        client: reqwest::Client::new(),
        auth: config.auth,
        base_url: config.base_url.trim_end_matches('/').to_string(),
        web_search: options.web_search,
    }))
}

impl OpenAiResponsesApi {
    fn reauthentication_required(credentials: &OAuthCredentials) -> ProviderError {
        ProviderError::ReauthenticationRequired {
            access: credentials.access.clone(),
            refresh: credentials.refresh.clone(),
            expires: credentials.expires,
        }
    }

    fn authenticated_request(&self, url: &str) -> Result<reqwest::RequestBuilder, ProviderError> {
        match &self.auth {
            ProviderCredentials::API(api_key) => Ok(self.client.post(url).bearer_auth(api_key)),
            ProviderCredentials::OAuth(credentials) => {
                if credentials.expires <= unix_timestamp_millis()? + OAUTH_EXPIRY_SKEW_MILLIS {
                    return Err(Self::reauthentication_required(credentials));
                }

                Ok(self
                    .client
                    .post(url)
                    .bearer_auth(&credentials.access)
                    .header("chatgpt-account-id", &credentials.account_id)
                    .header("originator", "tau")
                    .header("User-Agent", "tau")
                    .header("OpenAI-Beta", "responses=experimental"))
            }
        }
    }
}

impl OpenAiOptions {
    fn from_value(value: Value) -> Result<Self, ProviderError> {
        if value.is_null() {
            return Ok(Self::default());
        }

        serde_json::from_value(value).map_err(|error| {
            ProviderError::Configuration(format!("invalid OpenAI options: {error}"))
        })
    }
}

fn unix_timestamp_millis() -> Result<i64, ProviderError> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            ProviderError::Configuration(format!("system clock is before UNIX_EPOCH: {error}"))
        })?
        .as_millis()
        .try_into()
        .map_err(|_| ProviderError::Configuration("system timestamp overflowed i64".to_string()))?)
}

#[async_trait]
impl ModelApi for OpenAiResponsesApi {
    fn name(&self) -> &'static str {
        crate::providers::openai::PROVIDER_NAME
    }

    async fn respond(&self, session: &mut TauSession) -> Result<TauResponse, ProviderError> {
        let (request, conversation) = build_request(session, self.web_search.as_ref())?;
        let url = format!("{}/responses", self.base_url);
        let request_body = serde_json::to_string_pretty(&request)?;
        tracing::debug!(
            target: "tau::providers::openai",
            %url,
            body = %request_body,
            "request"
        );

        let response = self
            .authenticated_request(&url)?
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
            if status == reqwest::StatusCode::UNAUTHORIZED {
                if let ProviderCredentials::OAuth(credentials) = &self.auth {
                    return Err(Self::reauthentication_required(credentials));
                }
            }
            return Err(ProviderError::Api { status, body });
        }

        let response: OpenAiResponse = serde_json::from_str(&body)?;
        let native_output = response.output.clone();
        let response_id = response.id.clone();
        let parsed = parse_response(response)?;
        record_openai_response(session, conversation, response_id, native_output)?;

        Ok(parsed)
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
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning: Option<OpenAiReasoning>,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
enum OpenAiInputItem {
    Message(OpenAiMessage),
    FunctionCall(OpenAiFunctionCallItem),
    FunctionCallOutput(OpenAiFunctionCallOutputItem),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
struct OpenAiReasoning {
    effort: String,
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
    output: OpenAiFunctionCallOutputContent,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
enum OpenAiFunctionCallOutputContent {
    Text(String),
    Blocks(Vec<OpenAiContent>),
}

#[derive(Debug, Clone, Serialize, PartialEq)]
#[serde(untagged)]
enum OpenAiTool {
    Function {
        #[serde(rename = "type")]
        kind: &'static str,
        name: String,
        description: String,
        strict: bool,
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

fn build_request(
    session: &mut TauSession,
    web_search: Option<&OpenAiWebSearchConfig>,
) -> Result<(OpenAiRequest, OpenAiConversationData), ProviderError> {
    let model = session
        .model()
        .ok_or(ProviderError::MissingModel)?
        .to_string();

    let conversation = session
        .provider_conversation_data::<OpenAiConversationData>()
        .or_else(|| {
            session
                .provider_state::<OpenAiResponsesState>(crate::providers::openai::PROVIDER_NAME)
                .map(|state| OpenAiConversationData {
                    previous_response_id: state.previous_response_id.clone(),
                    output: Vec::new(),
                })
        })
        .unwrap_or_default();
    let previous_response_id = conversation.previous_response_id.clone();

    let input_items = if previous_response_id.is_some() {
        pending_input_items(session)
    } else {
        &session.conversation().items
    };
    let mut input = Vec::new();
    for item in input_items {
        input.extend(openai_input_items_for_conversation_item(item)?);
    }

    let tools = openai_tools(session, web_search);

    Ok((
        OpenAiRequest {
            model,
            input,
            tools,
            parallel_tool_calls: true,
            previous_response_id,
            reasoning: openai_reasoning(session.thinking_effort()),
        },
        conversation,
    ))
}

fn openai_reasoning(effort: Option<ThinkingEffort>) -> Option<OpenAiReasoning> {
    let effort = match effort? {
        ThinkingEffort::Disabled => return None,
        ThinkingEffort::Low => "low",
        ThinkingEffort::Medium => "medium",
        ThinkingEffort::High => "high",
        ThinkingEffort::XHigh | ThinkingEffort::Max => "xhigh",
    };

    Some(OpenAiReasoning {
        effort: effort.to_string(),
    })
}

fn openai_input_items_for_conversation_item(
    item: &ConversationItem,
) -> Result<Vec<OpenAiInputItem>, ProviderError> {
    let mut input = Vec::new();
    match item {
        ConversationItem::System { content } => {
            input.push(OpenAiInputItem::Message(OpenAiMessage {
                role: OpenAiRole::System,
                content: input_content_parts(content)?,
            }))
        }
        ConversationItem::User { content } => input.push(OpenAiInputItem::Message(OpenAiMessage {
            role: OpenAiRole::User,
            content: input_content_parts(content)?,
        })),
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
    Ok(input)
}

fn record_openai_response(
    session: &mut TauSession,
    mut conversation: OpenAiConversationData,
    response_id: String,
    native_output: Vec<Value>,
) -> Result<(), ProviderError> {
    session.set_provider_state(
        crate::providers::openai::PROVIDER_NAME,
        OpenAiResponsesState {
            previous_response_id: Some(response_id.clone()),
        },
    );

    conversation.previous_response_id = Some(response_id);
    conversation.output.extend(native_output);
    session.set_provider_conversation_data(&conversation)
}

fn openai_tools(
    session: &TauSession,
    web_search: Option<&OpenAiWebSearchConfig>,
) -> Vec<OpenAiTool> {
    let mut tools: Vec<_> = session
        .context()
        .tools()
        .map(|tool| OpenAiTool::Function {
            kind: "function",
            name: tool.name.clone(),
            description: tool.description.clone(),
            strict: true,
            parameters: tool.input_schema.clone(),
        })
        .collect();

    if web_search.is_some_and(|config| config.enabled.unwrap_or(true)) {
        tools.push(OpenAiTool::Server { kind: "web_search" });
    }

    tools
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

fn parse_response(response: OpenAiResponse) -> Result<TauResponse, ProviderError> {
    let usage = response.usage.map(|usage| TokenUsage {
        input_tokens: usage.input_tokens,
        output_tokens: usage.output_tokens,
        total_tokens: usage.total_tokens,
    });
    let mut parts = Vec::new();
    let mut saw_refusal = false;
    let mut has_tool_calls = false;

    for item in response.output {
        match item.get("type").and_then(Value::as_str) {
            Some("message") => {
                saw_refusal |= parse_message_output_item(item, &mut parts)?;
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
                has_tool_calls = true;
            }
            Some("reasoning") => {
                let content_part = openai_reasoning_content(item.clone());
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
            }
            Some("web_search_call") => {
                parts.push(ResponsePart::ServerToolUse {
                    call: ServerToolUse {
                        id: item
                            .get("id")
                            .and_then(Value::as_str)
                            .map(ToString::to_string),
                        name: "web_search".to_string(),
                        input: item
                            .get("action")
                            .map_or_else(|| json!("[unspecified]"), Value::clone),
                    },
                });
            }
            _ => {
                let content_part = ContentPart::text(item.to_string());
                parts.push(ResponsePart::Content {
                    content: content_part.clone(),
                });
            }
        }
    }

    if let Some(stop) = openai_stop(
        response.status.as_deref(),
        response.incomplete_details.as_ref(),
        saw_refusal,
        has_tool_calls,
    ) {
        parts.push(ResponsePart::Stop { stop });
    }

    Ok(TauResponse { parts, usage })
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
            "provider": crate::providers::openai::PROVIDER_NAME,
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
) -> Result<bool, ProviderError> {
    let parts = item
        .get("content")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            ProviderError::Response("OpenAI message output missing content array".to_string())
        })?;

    let mut saw_refusal = false;

    for part in parts {
        match part.get("type").and_then(Value::as_str) {
            Some("output_text") => {
                let text = part.get("text").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::Response("OpenAI output_text missing text".to_string())
                })?;
                let content_part = ContentPart::Text {
                    text: text.to_string(),
                    annotations: openai_text_annotations(part)?,
                };
                parts_out.push(ResponsePart::Content {
                    content: content_part,
                });
            }
            Some("refusal") => {
                let refusal = part.get("refusal").and_then(Value::as_str).ok_or_else(|| {
                    ProviderError::Response("OpenAI refusal missing refusal text".to_string())
                })?;
                saw_refusal = true;
                let content_part = ContentPart::Refusal {
                    text: refusal.to_string(),
                };
                parts_out.push(ResponsePart::Content {
                    content: content_part,
                });
            }
            _ => {
                parts_out.push(ResponsePart::Content {
                    content: ContentPart::unknown(&part),
                });
            }
        }
    }

    Ok(saw_refusal)
}

fn openai_text_annotations(part: &Value) -> Result<Option<Vec<Annotation>>, ProviderError> {
    let Some(annotations) = part.get("annotations").and_then(Value::as_array) else {
        return Ok(None);
    };

    let citations = annotations
        .iter()
        .filter(|annotation| annotation.get("type").and_then(Value::as_str) == Some("url_citation"))
        .map(|annotation| {
            serde_json::from_value::<Citation>(annotation.clone())
                .map(Annotation::Citation)
                .map_err(ProviderError::from)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok((!citations.is_empty()).then_some(citations))
}

fn openai_reasoning_content(item: Value) -> ContentPart {
    ContentPart::Thinking {
        summary: match item.get("content").or(item.get("summary")) {
            Some(Value::Array(content)) => content
                .iter()
                .filter_map(|v| {
                    v.get("text").and_then(|v| match v {
                        Value::String(text) => Some(text.clone()),
                        _ => None,
                    })
                })
                .collect(),
            _ => Vec::new(),
        },
        signature: None,
    }
}

fn input_content_parts(parts: &[ContentPart]) -> Result<Vec<OpenAiContent>, ProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text, .. } => Ok(OpenAiContent::InputText { text: text.clone() }),
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
            ContentPart::Thinking { summary: text, .. } => Ok(OpenAiContent::InputText {
                text: format!("[thinking: {}]", text.join("\n")),
            }),
            ContentPart::Refusal { text, .. } => {
                Ok(OpenAiContent::InputText { text: text.clone() })
            }
            ContentPart::FailedToolCall { text, .. } => {
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
            ContentPart::Thinking { summary: text, .. } => OpenAiContent::OutputText {
                text: format!("[thinking: {}]", text.join("\n")),
            },
            ContentPart::Refusal { text, .. } => OpenAiContent::OutputText { text: text.clone() },
            ContentPart::FailedToolCall { text, .. } => {
                OpenAiContent::OutputText { text: text.clone() }
            }
            part => OpenAiContent::OutputText {
                text: assistant_content_as_text(part),
            },
        })
        .collect()
}

fn tool_result_output(
    result: &ToolResult,
) -> Result<OpenAiFunctionCallOutputContent, ProviderError> {
    if result
        .content
        .iter()
        .any(|part| matches!(part, ContentPart::Image { .. }))
    {
        return Ok(OpenAiFunctionCallOutputContent::Blocks(
            input_content_parts(&result.content)?,
        ));
    }

    if result
        .content
        .iter()
        .any(|part| matches!(part, ContentPart::FailedToolCall { .. }))
    {
        return Ok(OpenAiFunctionCallOutputContent::Text(tool_result_text(
            &result.content,
        )));
    }

    Ok(OpenAiFunctionCallOutputContent::Text(tool_result_json(
        result,
    )?))
}

fn tool_result_text(parts: &[ContentPart]) -> String {
    parts
        .iter()
        .map(assistant_content_as_text)
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::{MediaData, ToolResult},
        tools::{ToolDefinition, ToolOutput},
    };
    use serde_json::json;

    fn callback(input: Value) -> Result<ToolOutput, crate::tools::ToolCallError> {
        Ok(ToolOutput::json(input))
    }

    fn test_api(web_search: Option<OpenAiWebSearchConfig>) -> OpenAiResponsesApi {
        OpenAiResponsesApi {
            client: reqwest::Client::new(),
            auth: ProviderCredentials::API("test-key".to_string()),
            base_url: crate::providers::openai::DEFAULT_BASE_URL.to_string(),
            web_search,
        }
    }

    fn oauth_api(expires: i64) -> OpenAiResponsesApi {
        OpenAiResponsesApi {
            client: reqwest::Client::new(),
            auth: ProviderCredentials::OAuth(OAuthCredentials {
                access: "access-token".to_string(),
                refresh: "refresh-token".to_string(),
                expires,
                account_id: "account-id".to_string(),
            }),
            base_url: crate::providers::codex::DEFAULT_BASE_URL.to_string(),
            web_search: None,
        }
    }

    #[test]
    fn applies_api_key_auth_headers() {
        let request = test_api(None)
            .authenticated_request("https://example.com/responses")
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(request.headers()["authorization"], "Bearer test-key");
        assert!(!request.headers().contains_key("chatgpt-account-id"));
    }

    #[test]
    fn applies_oauth_auth_headers() {
        let request = oauth_api(i64::MAX)
            .authenticated_request("https://example.com/responses")
            .unwrap()
            .build()
            .unwrap();

        assert_eq!(request.headers()["authorization"], "Bearer access-token");
        assert_eq!(request.headers()["chatgpt-account-id"], "account-id");
        assert_eq!(request.headers()["originator"], "tau");
        assert_eq!(request.headers()["user-agent"], "tau");
        assert_eq!(request.headers()["openai-beta"], "responses=experimental");
    }

    #[test]
    fn expired_oauth_auth_requires_reauthentication() {
        let error = oauth_api(0)
            .authenticated_request("https://example.com/responses")
            .unwrap_err();

        assert!(matches!(
            error,
            ProviderError::ReauthenticationRequired {
                access,
                refresh,
                expires: 0,
            } if access == "access-token" && refresh == "refresh-token"
        ));
    }

    #[test]
    fn builds_responses_api_request_from_complete_history() {
        let mut context = crate::context::TauContext::default();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                readonly: true,
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();
        let mut session = context.session(
            test_api(Some(OpenAiWebSearchConfig::default())),
            "gpt-4.1-mini",
        );
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
                content: vec![ContentPart::text("hello")],
                error: None,
            }],
        });
        let request = build_request(&mut session, Some(&OpenAiWebSearchConfig::default())).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["model"], "gpt-4.1-mini");
        assert_eq!(value["parallel_tool_calls"], true);
        assert!(value.get("previous_response_id").is_none());
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["tools"][0]["strict"], true);
        assert_eq!(value["tools"][1]["type"], "web_search");
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][2]["type"], "function_call");
        assert_eq!(value["input"][3]["type"], "function_call_output");
    }

    #[test]
    fn disables_web_search_when_not_configured() {
        let mut context = crate::context::TauContext::default();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                readonly: true,
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();
        let mut session = context.session(test_api(None), "gpt-4.1-mini");

        let request = build_request(&mut session, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["tools"].as_array().unwrap().len(), 1);
        assert_eq!(value["tools"][0]["type"], "function");
    }

    #[test]
    fn uses_previous_response_id_for_incremental_requests() {
        let mut context = crate::context::TauContext::default();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                readonly: true,
                description: "echo input".to_string(),
                input_schema: json!({"type":"object"}),
                callback,
            })
            .unwrap();
        let mut session = context.session(test_api(None), "gpt-4.1-mini");
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
                content: vec![ContentPart::text("hello")],
                error: None,
            }],
        });
        session
            .set_provider_conversation_data(&OpenAiConversationData {
                previous_response_id: Some("resp_previous".to_string()),
                output: Vec::new(),
            })
            .unwrap();

        let request = build_request(&mut session, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["previous_response_id"], "resp_previous");
        assert_eq!(value["input"].as_array().unwrap().len(), 1);
        assert_eq!(value["input"][0]["type"], "function_call_output");
    }

    #[test]
    fn sends_tool_result_images_in_function_call_output() {
        let context = crate::context::TauContext::default();
        let mut session = context.session(test_api(None), "gpt-4.1-mini");
        session.push_item(ConversationItem::ToolResult {
            results: vec![ToolResult {
                call_id: "call_1".to_string(),
                name: "read_file".to_string(),
                content: vec![
                    ContentPart::text("image summary"),
                    ContentPart::Image {
                        media_type: "image/png".to_string(),
                        data: MediaData::Base64("abc123".to_string()),
                    },
                ],
                error: None,
            }],
        });

        let request = build_request(&mut session, None).unwrap();
        let value = serde_json::to_value(request.0).unwrap();

        assert_eq!(value["input"].as_array().unwrap().len(), 1);
        assert_eq!(value["input"][0]["type"], "function_call_output");
        assert_eq!(value["input"][0]["output"][0]["type"], "input_text");
        assert_eq!(value["input"][0]["output"][0]["text"], "image summary");
        assert_eq!(value["input"][0]["output"][1]["type"], "input_image");
        assert_eq!(
            value["input"][0]["output"][1]["image_url"],
            "data:image/png;base64,abc123"
        );
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

        let content = parsed.content();
        assert_eq!(content.len(), 2);
        match &content[0] {
            ContentPart::Text { text, annotations } => {
                assert_eq!(text, "I'll check.");
                let annotations = annotations.as_ref().unwrap();
                assert_eq!(annotations.len(), 1);
                match &annotations[0] {
                    Annotation::Citation(citation) => assert_eq!(
                        serde_json::to_value(citation).unwrap(),
                        json!({
                            "url": "https://example.com",
                            "title": "Example",
                            "citation": null,
                            "start_index": 0,
                            "end_index": 5
                        })
                    ),
                }
            }
            other => panic!("expected text content, got {other:?}"),
        }
        match &content[1] {
            ContentPart::Thinking { summary: text, .. } => {
                assert_eq!(text, &vec!["I should inspect the files."]);
            }
            other => panic!("expected thinking content, got {other:?}"),
        }
        assert!(matches!(
            &parsed.parts[4],
            ResponsePart::ServerToolUse { call }
                if call.id.as_deref() == Some("ws_1")
                    && call.name == "web_search"
                    && call.input == json!({"type": "search", "query": "latest news about AI"})
        ));
        let tool_calls = parsed.tool_calls();
        assert_eq!(tool_calls.len(), 2);
        assert_eq!(tool_calls[0].id, "call_a");
        assert_eq!(tool_calls[1].input, json!({"path":"README.md"}));
        assert_eq!(parsed.usage.unwrap().total_tokens, Some(120));
    }
}
