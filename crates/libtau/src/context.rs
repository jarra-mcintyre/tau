use std::{any::Any, collections::BTreeMap, fmt, sync::Arc};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::providers::{Provider, ProviderError, ProviderResponse, TokenUsage};

pub type ToolCallback = fn(Value) -> Result<ToolOutput, ToolCallError>;
pub type ProviderState = Arc<dyn Any + Send + Sync>;

#[derive(Clone, Default)]
pub struct TauContext {
    tools: BTreeMap<String, ToolDefinition>,
}

#[derive(Clone)]
pub struct TauSession {
    context: Arc<TauContext>,
    conversation: Conversation,
    provider: Arc<dyn Provider>,
    provider_state: BTreeMap<String, ProviderState>,
    last_token_usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct Conversation {
    pub model: Option<String>,
    pub items: Vec<ConversationItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConversationItem {
    System { content: Vec<ContentPart> },
    User { content: Vec<ContentPart> },
    Agent { content: Vec<ContentPart> },
    ToolUse { calls: Vec<ToolUse> },
    ToolResult { results: Vec<ToolResult> },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentPart {
    Text { text: String },
    Json { value: Value },
    Image { media_type: String, data: MediaData },
    Binary { media_type: String, data: MediaData },
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
pub struct ToolOutput {
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TauResponse {
    Message(Vec<ContentPart>),
    ToolUse(Vec<ToolUse>),
    MessageAndToolUse {
        content: Vec<ContentPart>,
        tool_calls: Vec<ToolUse>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub enum TauEvent {
    UserMessage(Vec<ContentPart>),
    AgentMessage(Vec<ContentPart>),
    ToolUseRequested(Vec<ToolUse>),
    ToolResult(Vec<ToolResult>),
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing)]
    pub callback: ToolCallback,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolRegistrationError {
    #[error("duplicate tool name: {0}")]
    DuplicateName(String),
    #[error("failed to serialize tool schema: {0}")]
    SchemaSerializationFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolCallError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("invalid tool input: {0}")]
    InvalidInput(String),
    #[error("failed to serialize tool output: {0}")]
    OutputSerializationFailed(String),
}

impl fmt::Debug for TauContext {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TauContext")
            .field("tools", &self.tools)
            .finish()
    }
}

impl fmt::Debug for TauSession {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("TauSession")
            .field("context", &self.context)
            .field("conversation", &self.conversation)
            .field("provider", &self.provider.name())
            .field(
                "provider_state_keys",
                &self.provider_state.keys().collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl TauContext {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn session(
        &self,
        provider: impl Provider + 'static,
        model: impl Into<String>,
    ) -> TauSession {
        TauSession::new(self.clone(), provider, model)
    }

    pub fn session_with_provider_arc(
        &self,
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
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

    pub fn call_tool_json(&self, name: &str, input: Value) -> Result<Value, ToolCallError> {
        match self.call_tool(name, input)?.content.as_slice() {
            [ContentPart::Json { value }] => Ok(value.clone()),
            other => serde_json::to_value(other)
                .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string())),
        }
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
        provider: impl Provider + 'static,
        model: impl Into<String>,
    ) -> Self {
        Self::new_with_provider_arc(context, Arc::new(provider), model)
    }

    pub fn new_with_provider_arc(
        context: TauContext,
        provider: Arc<dyn Provider>,
        model: impl Into<String>,
    ) -> Self {
        let mut conversation = Conversation::default();
        conversation.model = Some(model.into());
        Self {
            context: Arc::new(context),
            conversation,
            provider,
            provider_state: BTreeMap::new(),
            last_token_usage: None,
        }
    }

    pub fn context(&self) -> &TauContext {
        &self.context
    }

    pub fn provider(&self) -> &dyn Provider {
        self.provider.as_ref()
    }

    pub fn last_token_usage(&self) -> Option<&TokenUsage> {
        self.last_token_usage.as_ref()
    }

    pub fn set_model(&mut self, model: impl Into<String>) {
        self.conversation.model = Some(model.into());
    }

    pub fn model(&self) -> Option<&str> {
        self.conversation.model.as_deref()
    }

    pub fn conversation(&self) -> &Conversation {
        &self.conversation
    }

    pub fn conversation_mut(&mut self) -> &mut Conversation {
        &mut self.conversation
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
        self.call_tools_parallel_and_record_with_events(calls, |_| {})
    }

    pub fn call_tools_parallel_and_record_with_events(
        &mut self,
        calls: &[ToolUse],
        mut on_event: impl FnMut(TauEvent),
    ) -> Vec<ToolResult> {
        let results = self.call_tools_parallel(calls);
        self.push_item(ConversationItem::ToolResult {
            results: results.clone(),
        });
        on_event(TauEvent::ToolResult(results.clone()));
        results
    }

    pub async fn send_message(
        &mut self,
        text: impl Into<String>,
    ) -> Result<TauResponse, ProviderError> {
        self.send_message_with_events(text, |_| {}).await
    }

    pub async fn send_message_with_events(
        &mut self,
        text: impl Into<String>,
        on_event: impl FnMut(TauEvent),
    ) -> Result<TauResponse, ProviderError> {
        self.send_content_with_events(vec![ContentPart::text(text)], on_event)
            .await
    }

    pub async fn send_content(
        &mut self,
        content: Vec<ContentPart>,
    ) -> Result<TauResponse, ProviderError> {
        self.send_content_with_events(content, |_| {}).await
    }

    pub async fn send_content_with_events(
        &mut self,
        content: Vec<ContentPart>,
        mut on_event: impl FnMut(TauEvent),
    ) -> Result<TauResponse, ProviderError> {
        self.push_user_content(content.clone());
        on_event(TauEvent::UserMessage(content));

        let response = self.request_response().await?;
        emit_tau_response(&response, &mut on_event);

        Ok(response)
    }

    pub async fn request_response(&mut self) -> Result<TauResponse, ProviderError> {
        let provider = self.provider.clone();
        let response = provider.respond(self).await?;
        self.last_token_usage = response.usage.clone();
        self.record_provider_response(&response);

        Ok(response.into())
    }

    fn record_provider_response(&mut self, response: &ProviderResponse) {
        if !response.content.is_empty() {
            self.push_item(ConversationItem::Agent {
                content: response.content.clone(),
            });
        }
        if !response.tool_calls.is_empty() {
            self.push_item(ConversationItem::ToolUse {
                calls: response.tool_calls.clone(),
            });
        }
    }
}

fn emit_tau_response(response: &TauResponse, on_event: &mut impl FnMut(TauEvent)) {
    match response {
        TauResponse::Message(content) => on_event(TauEvent::AgentMessage(content.clone())),
        TauResponse::ToolUse(tool_calls) => {
            on_event(TauEvent::ToolUseRequested(tool_calls.clone()));
        }
        TauResponse::MessageAndToolUse {
            content,
            tool_calls,
        } => {
            on_event(TauEvent::AgentMessage(content.clone()));
            on_event(TauEvent::ToolUseRequested(tool_calls.clone()));
        }
    }
}

impl From<ProviderResponse> for TauResponse {
    fn from(response: ProviderResponse) -> Self {
        match (response.content.is_empty(), response.tool_calls.is_empty()) {
            (false, true) => Self::Message(response.content),
            (true, false) => Self::ToolUse(response.tool_calls),
            _ => Self::MessageAndToolUse {
                content: response.content,
                tool_calls: response.tool_calls,
            },
        }
    }
}

impl ContentPart {
    pub fn text(text: impl Into<String>) -> Self {
        Self::Text { text: text.into() }
    }

    pub fn json(value: Value) -> Self {
        Self::Json { value }
    }
}

impl ToolOutput {
    pub fn json(value: Value) -> Self {
        Self {
            content: vec![ContentPart::json(value)],
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::text(text)],
        }
    }
}

impl ToolDefinition {
    pub fn new<Input>(
        name: &str,
        description: &str,
        callback: ToolCallback,
    ) -> Result<Self, ToolRegistrationError>
    where
        Input: JsonSchema,
    {
        let input_schema = serde_json::to_value(schemars::schema_for!(Input))
            .map_err(|error| ToolRegistrationError::SchemaSerializationFailed(error.to_string()))?;

        Ok(Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
            callback,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{providers::Provider, tools};

    #[derive(Debug)]
    struct StubProvider;

    #[async_trait::async_trait]
    impl Provider for StubProvider {
        fn name(&self) -> &'static str {
            "stub"
        }

        async fn respond(
            &self,
            _session: &mut TauSession,
        ) -> Result<ProviderResponse, ProviderError> {
            Ok(ProviderResponse {
                content: vec![],
                tool_calls: vec![ToolUse {
                    id: "call_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path":"README.md"}),
                }],
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
        let mut context = TauContext::new();
        tools::register_builtin_tools(&mut context).unwrap();

        let names: Vec<_> = context
            .tools()
            .map(|definition| definition.name.as_str())
            .collect();

        assert_eq!(names, vec!["bash", "edit_file", "read_file", "write_file"]);
        assert!(context.get_tool("bash").is_some());
        assert!(context.get_tool("read_file").is_some());
        assert!(context.get_tool("edit_file").is_some());
        assert!(context.get_tool("write_file").is_some());
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
            Ok(ToolOutput::json(input))
        }

        let mut context = TauContext::new();
        let definition = ToolDefinition {
            name: "duplicate".to_string(),
            description: "first".to_string(),
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
    fn sends_message_through_configured_provider_and_emits_updates() {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let mut session = TauContext::new().session(StubProvider, "gpt-test");
                session.set_system_message("be helpful");

                let mut events = Vec::new();
                let response = session
                    .send_message_with_events("read the readme", |event| events.push(event))
                    .await
                    .unwrap();

                assert_eq!(
                    response,
                    TauResponse::ToolUse(vec![ToolUse {
                        id: "call_1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path":"README.md"}),
                    }])
                );
                assert_eq!(
                    events,
                    vec![
                        TauEvent::UserMessage(vec![ContentPart::text("read the readme")]),
                        TauEvent::ToolUseRequested(vec![ToolUse {
                            id: "call_1".to_string(),
                            name: "read_file".to_string(),
                            input: serde_json::json!({"path":"README.md"}),
                        }])
                    ]
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

        let mut context = TauContext::new();
        context
            .register_tool(ToolDefinition {
                name: "echo".to_string(),
                description: "echo".to_string(),
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
            vec![ContentPart::json(serde_json::json!({ "ok": true }))]
        );
    }
}
