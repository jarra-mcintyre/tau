use crate::providers::{ModelMetadata, openai};

pub const PROVIDER_NAME: &str = "openai-codex";
pub const API_KEY_ENV: &str = "OPENAI_CODEX_ACCESS_TOKEN";
pub const DEFAULT_BASE_URL: &str = "https://chatgpt.com/backend-api/codex";

pub fn default_models() -> Vec<ModelMetadata> {
    openai::default_models()
}
