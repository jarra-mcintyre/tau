use std::sync::Arc;

use async_trait::async_trait;

use crate::{
    context::{ContentPart, ResponsePart, TauSession, ToolUse},
    providers::ProviderConfig,
};

pub mod anthropic_messages;
pub mod common;
pub mod openai_responses;

#[derive(Debug, Clone, PartialEq)]
pub struct ApiResponse {
    /// Ordered response blocks as returned by the provider.
    pub parts: Vec<ResponsePart>,
    /// Convenience view of all agent content blocks in `parts`.
    pub content: Vec<ContentPart>,
    /// Convenience view of all client-executable tool calls in `parts`.
    pub tool_calls: Vec<ToolUse>,
    pub usage: Option<TokenUsage>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TokenUsage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
}

impl ApiResponse {
    pub fn is_tool_call_only(&self) -> bool {
        self.content.is_empty() && !self.tool_calls.is_empty()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ProviderError {
    #[error("context does not have a provider")]
    MissingProvider,
    #[error("context does not have a model")]
    MissingModel,
    #[error("provider configuration error: {0}")]
    Configuration(String),
    #[error("http request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("provider returned an error ({status}): {body}")]
    Api {
        status: reqwest::StatusCode,
        body: String,
    },
    #[error("failed to serialize provider request: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("provider response was not understood: {0}")]
    Response(String),
}

#[async_trait]
pub trait ModelApi: Send + Sync {
    fn name(&self) -> &'static str;

    async fn respond(&self, session: &mut TauSession) -> Result<ApiResponse, ProviderError>;
}

#[derive(Debug, Clone, Copy)]
pub struct ModelApiFactory {
    pub name: &'static str,
    pub build: fn(ProviderConfig) -> Result<Arc<dyn ModelApi>, ProviderError>,
}

impl ModelApiFactory {
    pub fn build_api(&self, config: ProviderConfig) -> Result<Arc<dyn ModelApi>, ProviderError> {
        (self.build)(config)
    }
}

pub fn available_model_apis() -> &'static [ModelApiFactory] {
    &[openai_responses::API, anthropic_messages::API]
}

pub fn find_model_api(name: &str) -> Option<&'static ModelApiFactory> {
    available_model_apis()
        .iter()
        .find(|api| api.name == name)
}
