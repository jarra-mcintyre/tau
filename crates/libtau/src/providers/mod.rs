use std::{any::Any, fmt, sync::Arc};

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

pub trait ModelCosts: fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

pub trait ProviderModelDetails: fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// User-facing model name used in Tau model selections.
    pub name: String,
    /// Provider API model identifier sent in requests.
    pub id: String,
    /// maximum context length of the model
    pub context_length: u64,
    /// maximum number of tokens to generate in a single response
    pub max_tokens: u64,
    /// Provider-specific model capabilities/configuration, such as thinking effort.
    pub provider_details: Option<Arc<dyn ProviderModelDetails>>,
    /// Provider-specific prices. USD per one million units unless noted otherwise.
    pub costs: Option<Arc<dyn ModelCosts>>,
}

impl ModelMetadata {
    pub fn custom(model: impl Into<String>) -> Self {
        let model = model.into();
        Self {
            name: model.clone(),
            id: model,
            context_length: 0,
            max_tokens: 0,
            provider_details: None,
            costs: None,
        }
    }
}

impl From<String> for ModelMetadata {
    fn from(model: String) -> Self {
        Self::custom(model)
    }
}

impl From<&str> for ModelMetadata {
    fn from(model: &str) -> Self {
        Self::custom(model)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ProviderApi {
    pub name: &'static str,
    pub api_key_env: &'static str,
    pub display_name: &'static str,
    pub build: fn(ProviderApiConfig) -> Result<Arc<dyn Provider>, ProviderError>,
    pub default_models: fn() -> Vec<ModelMetadata>,
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
