use std::sync::Arc;

use crate::providers::{ModelCosts, ModelMetadata};

pub const PROVIDER_NAME: &str = "openai";
pub const API_KEY_ENV: &str = "OPENAI_API_KEY";
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

#[derive(Debug)]
pub struct OpenAiModelCosts {
    pub input_token: f64,
    pub output_token: f64,
    pub cached_input_token: Option<f64>,
}

impl ModelCosts for OpenAiModelCosts {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

fn openai_model_costs(
    input_token: f64,
    output_token: f64,
    cached_input_token: Option<f64>,
) -> Arc<dyn ModelCosts> {
    Arc::new(OpenAiModelCosts {
        input_token,
        output_token,
        cached_input_token,
    })
}

pub fn default_models() -> Vec<ModelMetadata> {
    vec![
        ModelMetadata {
            name: "gpt-5.5".to_string(),
            id: "gpt-5.5".to_string(),
            context_length: 1_000_000,
            max_tokens: 0,
            thinking_effort: None,
            provider_config: None,
            costs: Some(openai_model_costs(5.0, 30.0, Some(0.50))),
        },
        ModelMetadata {
            name: "gpt-5.4".to_string(),
            id: "gpt-5.4".to_string(),
            context_length: 1_000_000,
            max_tokens: 0,
            thinking_effort: None,
            provider_config: None,
            costs: Some(openai_model_costs(2.50, 15.0, Some(0.25))),
        },
        ModelMetadata {
            name: "gpt-5.4-mini".to_string(),
            id: "gpt-5.4-mini".to_string(),
            context_length: 400_000,
            max_tokens: 0,
            thinking_effort: None,
            provider_config: None,
            costs: Some(openai_model_costs(0.75, 4.50, Some(0.075))),
        },
    ]
}
