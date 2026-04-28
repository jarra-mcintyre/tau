use std::sync::Arc;

use async_trait::async_trait;

use crate::context::{ContentPart, TauSession, ToolUse};

pub mod anthropic;
pub mod common;
pub mod openai;

#[derive(Debug, Clone, PartialEq)]
pub struct ProviderResponse {
    pub content: Vec<ContentPart>,
    pub tool_calls: Vec<ToolUse>,
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
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderApi {
    pub name: &'static str,
    pub api_key_env: &'static str,
    pub display_name: &'static str,
    pub build: fn(ProviderApiConfig) -> Arc<dyn Provider>,
}

impl ProviderApi {
    pub fn build_provider(&self, config: ProviderApiConfig) -> Arc<dyn Provider> {
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
