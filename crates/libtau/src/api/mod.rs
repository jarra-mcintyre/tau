use std::sync::Arc;

use async_trait::async_trait;

use serde::{Deserialize, Serialize};

use crate::{
    context::{ResponsePart, TauResponse, TauSession},
    providers::ProviderConfig,
};

pub mod anthropic_messages;
pub mod common;
pub mod openai_responses;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenUsage {
    pub uncached_input_tokens: Option<u64>,
    pub cache_read_input_tokens: Option<u64>,
    pub cache_creation_input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub total_tokens: Option<u64>,
    //pub expected_cost: Option<u64>
}

impl TokenUsage {
    pub(crate) fn sum(left: Option<&Self>, right: Option<&Self>) -> Option<Self> {
        match (left, right) {
            (Some(l), Some(r)) => Some(TokenUsage {
                uncached_input_tokens: add_optional(
                    l.uncached_input_tokens,
                    r.uncached_input_tokens,
                ),
                cache_read_input_tokens: add_optional(
                    l.cache_read_input_tokens,
                    r.cache_read_input_tokens,
                ),
                cache_creation_input_tokens: add_optional(
                    l.cache_creation_input_tokens,
                    r.cache_creation_input_tokens,
                ),
                output_tokens: add_optional(l.output_tokens, r.output_tokens),
                total_tokens: add_optional(l.total_tokens, r.total_tokens),
            }),
            _ => left.or(right).cloned(),
        }
    }
}

fn add_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        _ => left.or(right),
    }
}

impl TauResponse {
    pub fn is_tool_call_only(&self) -> bool {
        !self.parts.is_empty()
            && self
                .parts
                .iter()
                .all(|part| matches!(part, ResponsePart::ToolUse { .. }))
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
    #[error("provider requires OAuth re-authentication")]
    ReauthenticationRequired {
        access: String,
        refresh: String,
        expires: i64,
    },
    #[error("failed to serialize provider request: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("provider response was not understood: {0}")]
    Response(String),
}

#[async_trait]
pub trait ModelApi: Send + Sync {
    fn name(&self) -> &'static str;

    async fn respond(&self, session: &mut TauSession) -> Result<TauResponse, ProviderError>;
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
    available_model_apis().iter().find(|api| api.name == name)
}
