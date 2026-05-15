use std::sync::Arc;

use serde::Deserialize;

use crate::providers::{ModelCosts, ModelMetadata, ProviderModelConfig};

pub const PROVIDER_NAME: &str = "anthropic";
pub const API_KEY_ENV: &str = "ANTHROPIC_API_KEY";
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com/v1";

#[derive(Debug)]
pub struct AnthropicModelCosts {
    pub input_token: f64,
    pub output_token: f64,
    pub cache_hit_token: Option<f64>,
    pub cache_write_5m_token: Option<f64>,
    pub cache_write_1h_token: Option<f64>,
    pub web_search: Option<f64>,
}

impl ModelCosts for AnthropicModelCosts {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnthropicModelConfig {
    pub legacy_thinking_budget: bool,
}

impl ProviderModelConfig for AnthropicModelConfig {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn anthropic_model_metadata(
    name: impl Into<String>,
    id: impl Into<String>,
    context_length: u64,
    max_tokens: u64,
    legacy_thinking_budget: bool,
) -> ModelMetadata {
    ModelMetadata {
        name: name.into(),
        id: id.into(),
        context_length,
        max_tokens,
        thinking_effort: None,
        provider_config: Some(Arc::new(AnthropicModelConfig {
            legacy_thinking_budget,
        })),
        costs: None,
    }
}

pub fn default_models() -> Vec<ModelMetadata> {
    vec![
        anthropic_model_metadata("opus4-7", "claude-opus-4-7", 1_000_000, 128_000, false),
        anthropic_model_metadata("opus-4-6", "claude-opus-4-6", 200_000, 128_000, false),
        anthropic_model_metadata("sonnet-4-6", "claude-sonnet-4-6", 1_000_000, 64_000, false),
        anthropic_model_metadata(
            "haiku-4-5",
            "claude-haiku-4-5-20251001",
            200_000,
            64_000,
            true,
        ),
    ]
}
