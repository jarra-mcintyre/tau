use serde::{Deserialize, Serialize};
use std::{any::Any, fmt, sync::Arc};

use crate::api;

pub mod anthropic;
pub mod openai;

pub trait ModelCosts: fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

pub trait ProviderModelConfig: fmt::Debug + Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum ThinkingEffort {
    Disabled,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

#[derive(Debug, Clone)]
pub struct ModelMetadata {
    /// User-facing model name used in Tau model selections.
    pub name: String,
    /// Provider API model identifier sent in requests.
    pub id: String,
    /// maximum context length of the model
    pub context_length: u64,
    /// maximum number of tokens to generate in a single response (0 to leave unspecified)
    pub max_tokens: u64,
    /// Default thinking effort for this model, if configured.
    pub thinking_effort: Option<ThinkingEffort>,
    /// Provider-specific model configuration.
    pub provider_config: Option<Arc<dyn ProviderModelConfig>>,
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
            thinking_effort: None,
            provider_config: None,
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
pub struct ProviderMetadata {
    /// User-facing provider name used in Tau model selections.
    pub name: &'static str,
    /// API integration used to communicate with this provider.
    pub api: &'static api::ModelApiFactory,
    /// Environment variable used for this provider's API key.
    pub api_key_env: &'static str,
    /// Human readable provider name used in errors.
    pub display_name: &'static str,
    /// Default API base URL for this provider.
    pub base_url: &'static str,
    /// Predefined model list for this provider.
    pub models: fn() -> Vec<ModelMetadata>,
}

#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
    pub base_url: String,
    pub options: serde_json::Value,
}

impl ProviderMetadata {
    pub fn default_models(&self) -> Vec<ModelMetadata> {
        (self.models)()
    }
}

pub const OPENAI: ProviderMetadata = ProviderMetadata {
    name: openai::PROVIDER_NAME,
    api: &api::openai_responses::API,
    api_key_env: openai::API_KEY_ENV,
    display_name: "OpenAI",
    base_url: openai::DEFAULT_BASE_URL,
    models: openai::default_models,
};

pub const ANTHROPIC: ProviderMetadata = ProviderMetadata {
    name: anthropic::PROVIDER_NAME,
    api: &api::anthropic_messages::API,
    api_key_env: anthropic::API_KEY_ENV,
    display_name: "Anthropic",
    base_url: anthropic::DEFAULT_BASE_URL,
    models: anthropic::default_models,
};

pub fn predefined_providers() -> &'static [ProviderMetadata] {
    &[OPENAI, ANTHROPIC]
}

pub fn find_predefined_provider(name: &str) -> Option<&'static ProviderMetadata> {
    predefined_providers()
        .iter()
        .find(|provider| provider.name == name)
}

// Backwards-compatible re-exports for callers that still import API items from `providers`.
pub use crate::api::{
    ModelApi, ModelApiFactory, ProviderError, TokenUsage, available_model_apis, find_model_api,
};
