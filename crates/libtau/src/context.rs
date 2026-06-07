use std::{any::Any, collections::BTreeMap, sync::Arc};

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    providers::{ModelApi, ModelMetadata, ProviderError, ThinkingEffort, TokenUsage},
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub type ProviderState = Arc<dyn Any + Send + Sync>;

#[derive(Clone, Default)]
pub struct TauContext {
    tools: BTreeMap<String, ToolDefinition>,
}

#[derive(Clone)]
pub struct TauSession {
    context: Arc<TauContext>,
    conversation: Conversation,
    model: Option<ModelMetadata>,
    provider: Arc<dyn ModelApi>,
    config: TauSessionConfig,
    provider_state: BTreeMap<String, ProviderState>,
    last_token_usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TauSessionConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<ThinkingEffort>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub model: Option<String>,
    pub items: Vec<ConversationItem>,
    /// Provider-specific conversation state for the session's current provider.
    ///
    /// Generic Tau content is stored in `items`; providers use this field to
    /// persist their native replay format (for example Anthropic content blocks
    /// with cache-control-sensitive structure, or OpenAI response ids). Switching
    /// provider invalidates this state.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_data: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationItem {
    System { content: Vec<ContentPart> },
    User { content: Vec<ContentPart> },
    Agent { content: Vec<ContentPart> },
    ToolUse { calls: Vec<ToolUse> },
    ToolResult { results: Vec<ToolResult> },
    ResponseStop { stop: ResponseStop },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    Image {
        media_type: String,
        data: MediaData,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    Binary {
        media_type: String,
        data: MediaData,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    Thinking {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        signature: Option<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    Refusal {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
    // This is a text type
    FailedToolCall {
        text: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        metadata: Option<Value>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum MediaData {
    Base64(String),
    Url(String),
    Path(String),
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolUse {
    pub id: String,
    pub name: String,
    pub input: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolResult {
    pub call_id: String,
    pub name: String,
    pub content: Vec<ContentPart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponsePart {
    Content { content: ContentPart },
    ToolUse { call: ToolUse },
    ServerToolUse { call: ServerToolUse },
    ServerToolResult { result: ServerToolResult },
    Stop { stop: ResponseStop },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ResponseStopReason {
    EndTurn,
    MaxTokens,
    StopSequence {
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sequence: Option<String>,
    },
    ToolUse,
    PauseTurn,
    Refusal,
    Other {
        value: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ResponseStop {
    pub reason: ResponseStopReason,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerToolUse {
    pub id: Option<String>,
    pub name: String,
    pub input: Value,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ServerToolResult {
    pub tool_use_id: Option<String>,
    pub name: String,
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct TauResponse {
    pub parts: Vec<ResponsePart>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub usage: Option<TokenUsage>,
}

impl TauContext {
    pub fn session(
        &self,
        provider: impl ModelApi + 'static,
        model: impl Into<ModelMetadata>,
    ) -> TauSession {
        TauSession::new(self.clone(), provider, model)
    }

    pub fn session_with_provider_arc(
        &self,
        provider: Arc<dyn ModelApi>,
        model: impl Into<ModelMetadata>,
    ) -> TauSession {
        TauSession::new_with_provider_arc(self.clone(), provider, model)
    }

    pub fn register_tool(
        &mut self,
        definition: ToolDefinition,
    ) -> Result<(), ToolRegistrationError> {
        if self.tools.contains_key(&definition.name) {
            return Err(ToolRegistrationError::DuplicateName(definition.name));
        }

        self.tools.insert(definition.name.clone(), definition);
        Ok(())
    }

    pub fn tools(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools.values()
    }

    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    pub fn call_tool(&self, name: &str, input: Value) -> Result<ToolOutput, ToolCallError> {
        let tool = self
            .get_tool(name)
            .ok_or_else(|| ToolCallError::UnknownTool(name.to_string()))?;

        (tool.callback)(input)
    }

    pub fn call_tools_parallel(&self, calls: &[ToolUse]) -> Vec<ToolResult> {
        std::thread::scope(|scope| {
            let handles: Vec<_> = calls
                .iter()
                .map(|call| {
                    scope.spawn(
                        move || match self.call_tool(&call.name, call.input.clone()) {
                            Ok(output) => ToolResult {
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                content: output.content,
                                error: None,
                            },
                            Err(error) => ToolResult {
                                call_id: call.id.clone(),
                                name: call.name.clone(),
                                content: vec![ContentPart::text(format!("{error:?}"))],
                                error: Some(format!("{error:?}")),
                            },
                        },
                    )
                })
                .collect();

            handles
                .into_iter()
                .map(|handle| handle.join().expect("tool call thread panicked"))
                .collect()
        })
    }
}

impl TauSession {
    pub fn new(
        context: TauContext,
        provider: impl ModelApi + 'static,
        model: impl Into<ModelMetadata>,
    ) -> Self {
        Self::new_with_provider_arc(context, Arc::new(provider), model)
    }

    pub fn new_with_provider_arc(
        context: TauContext,
        provider: Arc<dyn ModelApi>,
        model: impl Into<ModelMetadata>,
    ) -> Self {
        let model = model.into();
        let mut conversation = Conversation::default();
        conversation.model = Some(model.id.clone());
        Self {
            context: Arc::new(context),
            conversation,
            model: Some(model),
            provider,
            config: TauSessionConfig::default(),
            provider_state: BTreeMap::new(),
            last_token_usage: None,
        }
    }

    pub fn context(&self) -> &TauContext {
        &self.context
    }

    pub fn provider(&self) -> &dyn ModelApi {
        self.provider.as_ref()
    }

    pub fn config(&self) -> &TauSessionConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut TauSessionConfig {
        &mut self.config
    }

    pub fn thinking_effort(&self) -> Option<ThinkingEffort> {
        self.config.thinking_effort
    }

    pub fn set_thinking_effort(&mut self, thinking_effort: Option<ThinkingEffort>) {
        self.config.thinking_effort = thinking_effort;
    }

    pub fn set_config(&mut self, config: TauSessionConfig) {
        self.config = config;
    }

    /// Set the provider. This clears provider state.
    pub fn set_provider(&mut self, provider: Arc<dyn ModelApi>) {
        self.provider = provider;
        self.provider_state.clear();
        self.conversation.provider_data = None;
        self.last_token_usage = None;
    }

    /// Refresh the provider. This preserves provider state and will panic if changing provider.
    /// The principle intended use-case is refreshing provider auth information
    pub fn refresh_provider(&mut self, provider: Arc<dyn ModelApi>) {
        assert_eq!(self.provider.name(), provider.name());
        self.provider = provider;
        self.last_token_usage = None;
    }

    /// Set both the provider and model, clearing provider conversation state
    pub fn set_provider_and_model(
        &mut self,
        provider: Arc<dyn ModelApi>,
        model: impl Into<ModelMetadata>,
    ) {
        self.set_provider(provider);
        self.set_model(model);
    }

    pub fn last_token_usage(&self) -> Option<&TokenUsage> {
        self.last_token_usage.as_ref()
    }

    pub fn set_model(&mut self, model: impl Into<ModelMetadata>) {
        let model = model.into();
        self.conversation.model = Some(model.id.clone());
        self.model = Some(model);
    }

    pub fn model(&self) -> Option<&str> {
        self.model.as_ref().map(|model| model.id.as_str())
    }

    pub fn model_metadata(&self) -> Option<&ModelMetadata> {
        self.model.as_ref()
    }

    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    pub fn conversation_mut(&mut self) -> &mut Conversation {
        &mut self.conversation
    }

    pub fn provider_conversation_data<T>(&self) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.conversation
            .provider_data
            .clone()
            .and_then(|value| serde_json::from_value(value).ok())
    }

    pub fn set_provider_conversation_data<T>(&mut self, data: &T) -> Result<(), ProviderError>
    where
        T: Serialize,
    {
        self.conversation.provider_data = Some(serde_json::to_value(data)?);
        Ok(())
    }

    pub fn clear_provider_conversation_data(&mut self) {
        self.conversation.provider_data = None;
    }

    pub fn push_item(&mut self, item: ConversationItem) {
        self.conversation.items.push(item);
    }

    pub fn set_system_message(&mut self, text: impl Into<String>) {
        self.set_system_content(vec![ContentPart::text(text)]);
    }

    pub fn set_system_content(&mut self, content: Vec<ContentPart>) {
        if let Some(ConversationItem::System { content: existing }) = self
            .conversation
            .items
            .iter_mut()
            .find(|item| matches!(item, ConversationItem::System { .. }))
        {
            *existing = content;
        } else {
            self.conversation
                .items
                .insert(0, ConversationItem::System { content });
        }
    }

    pub fn push_system_text(&mut self, text: impl Into<String>) {
        self.push_item(ConversationItem::System {
            content: vec![ContentPart::text(text)],
        });
    }

    pub fn push_user_text(&mut self, text: impl Into<String>) {
        self.push_user_content(vec![ContentPart::text(text)]);
    }

    pub fn push_user_content(&mut self, content: Vec<ContentPart>) {
        self.push_item(ConversationItem::User { content });
    }

    pub fn push_agent_text(&mut self, text: impl Into<String>) {
        self.push_item(ConversationItem::Agent {
            content: vec![ContentPart::text(text)],
        });
    }

    pub fn set_provider_state<T>(&mut self, provider: impl Into<String>, state: T)
    where
        T: Any + Send + Sync,
    {
        self.provider_state.insert(provider.into(), Arc::new(state));
    }

    pub fn provider_state<T>(&self, provider: &str) -> Option<Arc<T>>
    where
        T: Any + Send + Sync,
    {
        self.provider_state
            .get(provider)
            .cloned()
            .and_then(|state| state.downcast::<T>().ok())
    }

    pub fn call_tools_parallel(&self, calls: &[ToolUse]) -> Vec<ToolResult> {
        self.context.call_tools_parallel(calls)
    }

    pub fn call_tools_parallel_and_record(&mut self, calls: &[ToolUse]) -> Vec<ToolResult> {
        let results = self.call_tools_parallel(calls);
        self.push_item(ConversationItem::ToolResult {
            results: results.clone(),
        });
        results
    }

    pub async fn send_message(
        &mut self,
        text: impl Into<String>,
    ) -> Result<TauResponse, ProviderError> {
        self.send_content(vec![ContentPart::text(text)]).await
    }

    pub async fn send_content(
        &mut self,
        content: Vec<ContentPart>,
    ) -> Result<TauResponse, ProviderError> {
        self.push_user_content(content);
        self.request_response().await
    }

    pub async fn request_response(&mut self) -> Result<TauResponse, ProviderError> {
        let provider = self.provider.clone();
        let response = provider.respond(self).await?;
        self.last_token_usage = response.usage.clone();
        self.record_provider_response(&response);

        Ok(response)
    }

    fn record_provider_response(&mut self, response: &TauResponse) {
        let mut content = Vec::new();
        let mut tool_calls = Vec::new();
        let mut stops = Vec::new();

        for part in &response.parts {
            match part {
                ResponsePart::Content { content: part } => content.push(part.clone()),
                ResponsePart::ToolUse { call } => tool_calls.push(call.clone()),
                ResponsePart::ServerToolUse { .. } | ResponsePart::ServerToolResult { .. } => {}
                ResponsePart::Stop { stop } => stops.push(stop.clone()),
            }
        }

        if !content.is_empty() {
            self.push_item(ConversationItem::Agent { content });
        }
        if !tool_calls.is_empty() {
            self.push_item(ConversationItem::ToolUse { calls: tool_calls });
        }
        for stop in stops {
            self.push_item(ConversationItem::ResponseStop { stop });
        }
    }
}

impl TauResponse {
    pub fn content(&self) -> Vec<ContentPart> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                ResponsePart::Content { content } => Some(content.clone()),
                _ => None,
            })
            .collect()
    }

    pub fn tool_calls(&self) -> Vec<ToolUse> {
        self.parts
            .iter()
            .filter_map(|part| match part {
                ResponsePart::ToolUse { call } => Some(call.clone()),
                _ => None,
            })
            .collect()
    }
}

impl ContentPart {
    /// Used to call out unknown/unsupported content types in fallback branches
    pub fn unknown(value: &Value) -> Self {
        Self::text(format!("[Unknown content]: {}", value))
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self::Text {
            text: text.into(),
            metadata: None,
        }
    }

    pub fn thinking(text: impl Into<String>) -> Self {
        Self::Thinking {
            text: text.into(),
            signature: None,
            metadata: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{providers::ModelApi, tools};

    #[derive(Debug)]
    struct StubProvider;

    #[async_trait::async_trait]
    impl ModelApi for StubProvider {
        fn name(&self) -> &'static str {
            "stub"
        }

        async fn respond(&self, _session: &mut TauSession) -> Result<TauResponse, ProviderError> {
            let call = ToolUse {
                id: "call_1".to_string(),
                name: "read_file".to_string(),
                input: serde_json::json!({"path":"README.md"}),
            };
            Ok(TauResponse {
                parts: vec![ResponsePart::ToolUse { call: call.clone() }],
                usage: Some(TokenUsage {
                    input_tokens: Some(10),
                    output_tokens: Some(5),
                    total_tokens: Some(15),
                }),
            })
        }
    }

    #[test]
    fn registers_builtin_tools() {
        let mut context = TauContext::default();
        tools::register_builtin_tools(&mut context).unwrap();

        let names: Vec<_> = context
            .tools()
            .map(|definition| definition.name.as_str())
            .collect();

        assert_eq!(
            names,
            vec![
                "bash",
                "edit_file",
                "find_files",
                "grep",
                "read_file",
                "write_file"
            ]
        );
        assert!(context.get_tool("bash").is_some());
        assert!(context.get_tool("read_file").is_some());
        assert!(context.get_tool("edit_file").is_some());
        assert!(context.get_tool("find_files").is_some());
        assert!(context.get_tool("grep").is_some());
        assert!(context.get_tool("write_file").is_some());
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
            Ok(ToolOutput::json(input))
        }

        let mut context = TauContext::default();
        let definition = ToolDefinition {
            name: "duplicate".to_string(),
            description: "first".to_string(),
            readonly: true,
            input_schema: serde_json::json!({ "type": "object" }),
            callback,
        };
        context.register_tool(definition.clone()).unwrap();

        assert_eq!(
            context.register_tool(definition),
            Err(ToolRegistrationError::DuplicateName(
                "duplicate".to_string()
            ))
        );
    }

    #[test]
    fn sends_message_through_configured_provider() {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let mut session = TauContext::default().session(StubProvider, "gpt-test");
                session.set_system_message("be helpful");

                let response = session.send_message("read the readme").await.unwrap();

                assert_eq!(
                    response,
                    TauResponse {
                        parts: vec![ResponsePart::ToolUse {
                            call: ToolUse {
                                id: "call_1".to_string(),
                                name: "read_file".to_string(),
                                input: serde_json::json!({"path":"README.md"}),
                            }
                        }],
                        usage: Some(TokenUsage {
                            input_tokens: Some(10),
                            output_tokens: Some(5),
                            total_tokens: Some(15),
                        }),
                    }
                );
                assert_eq!(session.model(), Some("gpt-test"));
                assert_eq!(session.conversation().items.len(), 3);
                assert_eq!(session.last_token_usage().unwrap().total_tokens, Some(15));
            });
    }

    #[test]
    fn session_stores_conversation_history_and_parallel_tool_results() {
        fn echo(input: Value) -> Result<ToolOutput, ToolCallError> {
            Ok(ToolOutput::json(input))
        }

        let mut context = TauContext::default();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                description: "echo".to_string(),
                readonly: true,
                input_schema: serde_json::json!({ "type": "object" }),
                callback: echo,
            })
            .unwrap();
        let mut session = context.session(StubProvider, "gpt-4.1");
        session.push_system_text("system");
        session.push_user_text("hello");

        let calls = vec![ToolUse {
            id: "call_1".to_string(),
            name: "echo".to_string(),
            input: serde_json::json!({ "ok": true }),
        }];
        let results = session.call_tools_parallel_and_record(&calls);

        assert_eq!(session.model(), Some("gpt-4.1"));
        assert_eq!(session.conversation().items.len(), 3);
        assert_eq!(
            results[0].content,
            vec![ContentPart::text(
                serde_json::json!({ "ok": true }).to_string()
            )]
        );
    }
}
