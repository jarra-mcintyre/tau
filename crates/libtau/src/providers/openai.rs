use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use crate::{
    context::{ContentPart, ConversationItem, MediaData, TauContext, ToolResult, ToolUse},
    providers::{Provider, ProviderError, ProviderResponse},
};

pub const PROVIDER_NAME: &str = "openai";
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

impl OpenAiProvider {
    pub fn new(api_key: impl Into<String>) -> Self {
        Self::with_base_url(api_key, DEFAULT_BASE_URL)
    }

    pub fn from_env() -> Result<Self, ProviderError> {
        let api_key = std::env::var("OPENAI_API_KEY").map_err(|_| {
            ProviderError::Configuration(
                "OPENAI_API_KEY environment variable is not set".to_string(),
            )
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

    async fn respond(&self, context: &mut TauContext) -> Result<ProviderResponse, ProviderError> {
        let request = build_request(context)?;
        let response = self
            .client
            .post(format!("{}/responses", self.base_url))
            .bearer_auth(&self.api_key)
            .json(&request)
            .send()
            .await?;

        let status = response.status();
        let body = response.text().await?;
        if !status.is_success() {
            return Err(ProviderError::Api { status, body });
        }

        let response: OpenAiResponse = serde_json::from_str(&body)?;
        context.set_provider_state(
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
struct OpenAiTool {
    #[serde(rename = "type")]
    kind: &'static str,
    name: String,
    description: String,
    parameters: Value,
}

#[derive(Debug, Clone, Deserialize)]
struct OpenAiResponse {
    id: String,
    #[serde(default)]
    output: Vec<OpenAiOutputItem>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiOutputItem {
    Message {
        content: Vec<OpenAiOutputContent>,
    },
    FunctionCall {
        call_id: String,
        name: String,
        arguments: String,
    },
    #[serde(other)]
    Other,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum OpenAiOutputContent {
    OutputText {
        text: String,
    },
    Refusal {
        refusal: String,
    },
    #[serde(other)]
    Other,
}

fn build_request(context: &TauContext) -> Result<OpenAiRequest, ProviderError> {
    let model = context
        .model()
        .ok_or(ProviderError::MissingModel)?
        .to_string();

    let mut input = Vec::new();
    for item in &context.conversation().items {
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
        }
    }

    let tools = context
        .tools()
        .map(|tool| OpenAiTool {
            kind: "function",
            name: tool.name.clone(),
            description: tool.description.clone(),
            parameters: tool.input_schema.clone(),
        })
        .collect();

    Ok(OpenAiRequest {
        model,
        input,
        tools,
        parallel_tool_calls: true,
    })
}

fn parse_response(response: OpenAiResponse) -> Result<ProviderResponse, ProviderError> {
    let mut content = Vec::new();
    let mut tool_calls = Vec::new();

    for item in response.output {
        match item {
            OpenAiOutputItem::Message { content: parts } => {
                for part in parts {
                    match part {
                        OpenAiOutputContent::OutputText { text } => {
                            content.push(ContentPart::text(text))
                        }
                        OpenAiOutputContent::Refusal { refusal } => {
                            content.push(ContentPart::text(refusal))
                        }
                        OpenAiOutputContent::Other => {}
                    }
                }
            }
            OpenAiOutputItem::FunctionCall {
                call_id,
                name,
                arguments,
            } => {
                let input = serde_json::from_str(&arguments).map_err(|error| {
                    ProviderError::Response(format!(
                        "function call {call_id} arguments were not JSON: {error}"
                    ))
                })?;
                tool_calls.push(ToolUse {
                    id: call_id,
                    name,
                    input,
                });
            }
            OpenAiOutputItem::Other => {}
        }
    }

    Ok(ProviderResponse {
        content,
        tool_calls,
    })
}

fn input_content_parts(parts: &[ContentPart]) -> Result<Vec<OpenAiContent>, ProviderError> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => Ok(OpenAiContent::InputText { text: text.clone() }),
            ContentPart::Json { value } => Ok(OpenAiContent::InputText {
                text: serde_json::to_string(value)?,
            }),
            ContentPart::Image { media_type, data } => Ok(OpenAiContent::InputImage {
                image_url: media_to_url(media_type, data),
            }),
            ContentPart::Binary { media_type, data } => Ok(OpenAiContent::InputText {
                text: format!("[binary content: {media_type}, {}]", media_data_label(data)),
            }),
        })
        .collect()
}

fn output_content_parts(parts: &[ContentPart]) -> Vec<OpenAiContent> {
    parts
        .iter()
        .map(|part| match part {
            ContentPart::Text { text } => OpenAiContent::OutputText { text: text.clone() },
            ContentPart::Json { value } => OpenAiContent::OutputText {
                text: value.to_string(),
            },
            ContentPart::Image { media_type, data } | ContentPart::Binary { media_type, data } => {
                OpenAiContent::OutputText {
                    text: format!("[media content: {media_type}, {}]", media_data_label(data)),
                }
            }
        })
        .collect()
}

fn tool_result_output(result: &ToolResult) -> Result<String, ProviderError> {
    Ok(serde_json::to_string(&json!({
        "name": result.name,
        "error": result.error,
        "content": result.content,
    }))?)
}

fn media_to_url(media_type: &str, data: &MediaData) -> String {
    match data {
        MediaData::Url(url) => url.clone(),
        MediaData::Base64(data) => format!("data:{media_type};base64,{data}"),
        MediaData::Path(path) => path.clone(),
    }
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

    fn callback(input: Value) -> Result<ToolOutput, crate::context::ToolCallError> {
        Ok(ToolOutput::json(input))
    }

    #[test]
    fn builds_responses_api_request_from_complete_history() {
        let mut context = TauContext::new();
        context.set_model("gpt-4.1-mini");
        context.push_system_text("be helpful");
        context.push_user_text("hello");
        context.push_item(ConversationItem::ToolUse {
            calls: vec![ToolUse {
                id: "call_1".to_string(),
                name: "echo".to_string(),
                input: json!({"text":"hello"}),
            }],
        });
        context.push_item(ConversationItem::ToolResult {
            results: vec![ToolResult {
                call_id: "call_1".to_string(),
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

        let request = build_request(&context).unwrap();
        let value = serde_json::to_value(request).unwrap();

        assert_eq!(value["model"], "gpt-4.1-mini");
        assert_eq!(value["parallel_tool_calls"], true);
        assert_eq!(value["tools"][0]["type"], "function");
        assert_eq!(value["input"][0]["role"], "system");
        assert_eq!(value["input"][2]["type"], "function_call");
        assert_eq!(value["input"][3]["type"], "function_call_output");
    }

    #[test]
    fn parses_text_and_parallel_function_calls() {
        let response = OpenAiResponse {
            id: "resp_1".to_string(),
            output: vec![
                OpenAiOutputItem::Message {
                    content: vec![OpenAiOutputContent::OutputText {
                        text: "I'll check.".to_string(),
                    }],
                },
                OpenAiOutputItem::FunctionCall {
                    call_id: "call_a".to_string(),
                    name: "read_file".to_string(),
                    arguments: "{\"path\":\"Cargo.toml\"}".to_string(),
                },
                OpenAiOutputItem::FunctionCall {
                    call_id: "call_b".to_string(),
                    name: "read_file".to_string(),
                    arguments: "{\"path\":\"README.md\"}".to_string(),
                },
            ],
        };

        let parsed = parse_response(response).unwrap();

        assert_eq!(parsed.content, vec![ContentPart::text("I'll check.")]);
        assert_eq!(parsed.tool_calls.len(), 2);
        assert_eq!(parsed.tool_calls[0].id, "call_a");
        assert_eq!(parsed.tool_calls[1].input, json!({"path":"README.md"}));
    }
}
