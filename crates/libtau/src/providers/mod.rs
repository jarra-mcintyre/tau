use async_trait::async_trait;

use crate::context::{ContentPart, TauContext, ToolUse};

pub mod anthropic;
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

    async fn respond(&self, context: &mut TauContext) -> Result<ProviderResponse, ProviderError>;
}
