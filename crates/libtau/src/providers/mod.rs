use std::sync::Arc;

use async_trait::async_trait;

use crate::context::{ContentPart, ResponsePart, TauSession, ToolUse};

pub mod anthropic;
pub mod common;
pub mod openai;

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderResponse {
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

impl ProviderResponse {
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
pub trait Provider: Send + Sync {
    fn name(&self) -> &'static str;

    async fn respond(&self, session: &mut TauSession) -> Result<ProviderResponse, ProviderError>;
}

#[derive(Debug, Clone)]
pub struct ProviderApiConfig {
    pub api_key: String,
    pub base_url: Option<String>,
    pub options: serde_json::Value,
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderApi {
    pub name: &'static str,
    pub api_key_env: &'static str,
    pub display_name: &'static str,
    pub build: fn(ProviderApiConfig) -> Result<Arc<dyn Provider>, ProviderError>,
}

impl ProviderApi {
    pub fn build_provider(
        &self,
        config: ProviderApiConfig,
    ) -> Result<Arc<dyn Provider>, ProviderError> {
        (self.build)(config)
    }
}

pub fn available_provider_apis() -> &'static [ProviderApi] {
    &[openai::API, anthropic::API]
}

pub fn find_provider_api(name: &str) -> Option<&'static ProviderApi> {
    available_provider_apis()
        .iter()
        .find(|api| api.name == name)
}
