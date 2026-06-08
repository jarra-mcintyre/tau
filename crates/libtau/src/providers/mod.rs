use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{any::Any, fmt, sync::Arc};

use crate::api;

pub mod anthropic;
pub mod codex;
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

// short-cut for unit tests
impl From<&str> for ModelMetadata {
    fn from(model: &str) -> Self {
        Self::custom(model)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthCredentials {
    pub access: String,
    pub refresh: String,
    pub expires: i64,
    pub account_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderCredentials {
    API(String),
    OAuth(OAuthCredentials),
}

/// Runtime API configuration consumed by a concrete model API implementation.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub auth: ProviderCredentials,
    pub base_url: String,
    pub options: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub struct ApiKeyProviderConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CodexProviderConfig {
    pub access: String,
    pub refresh: String,
    pub expires: i64,
    #[serde(rename = "accountId")]
    pub account_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OAuthProviderCredentials {
    pub provider: String,
    pub access: String,
    pub refresh: String,
    pub expires: i64,
    pub account_id: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type", content = "config")]
pub enum ProviderConfigEntry {
    #[serde(rename = "openai-api")]
    OpenAiApi(ApiKeyProviderConfig),
    #[serde(rename = "openai-codex")]
    OpenAiCodex(CodexProviderConfig),
    #[serde(rename = "anthropic-api")]
    AnthropicApi(ApiKeyProviderConfig),
    #[serde(rename = "custom")]
    Custom(CustomProviderConfig),
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct CustomProviderConfig {
    pub name: String,
    pub api: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub api_key_env: Option<String>,
    pub base_url: String,
    #[serde(default, skip_serializing_if = "Value::is_null")]
    pub options: Value,
    #[serde(default)]
    pub models: Vec<ModelConfigEntry>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ModelConfigEntry {
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub thinking_effort: Option<ThinkingEffort>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_config: Option<Value>,
}

#[derive(Debug)]
pub struct ConfiguredProvider {
    pub name: String,
    pub api: &'static api::ModelApiFactory,
    pub display_name: String,
    pub auth: ProviderAuthConfig,
    pub base_url: String,
    pub options: Value,
    pub models: Vec<ModelMetadata>,
}

#[derive(Debug, Clone)]
pub enum ProviderAuthConfig {
    ApiKey {
        api_key: Option<String>,
        api_key_env: Option<String>,
    },
    OAuth(OAuthCredentials),
}

impl ProviderConfigEntry {
    pub fn provider_name(&self) -> &str {
        match self {
            ProviderConfigEntry::OpenAiApi(_) => openai::PROVIDER_NAME,
            ProviderConfigEntry::OpenAiCodex(_) => codex::PROVIDER_NAME,
            ProviderConfigEntry::AnthropicApi(_) => anthropic::PROVIDER_NAME,
            ProviderConfigEntry::Custom(config) => &config.name,
        }
    }

    pub fn oauth_credentials(&self) -> Option<OAuthProviderCredentials> {
        match self {
            ProviderConfigEntry::OpenAiCodex(config) => Some(OAuthProviderCredentials {
                provider: codex::PROVIDER_NAME.to_string(),
                access: config.access.clone(),
                refresh: config.refresh.clone(),
                expires: config.expires,
                account_id: Some(config.account_id.clone()),
            }),
            _ => None,
        }
    }
}

impl ConfiguredProvider {
    pub fn build_api(&self) -> Result<Arc<dyn api::ModelApi>, Box<dyn std::error::Error>> {
        let auth: ProviderCredentials = match &self.auth {
            ProviderAuthConfig::ApiKey {
                api_key,
                api_key_env,
            } => api_key
                .as_ref()
                .map(|k| k.clone())
                .or_else(|| api_key_env.as_ref().and_then(|v| std::env::var(v).ok()))
                .map(|k| ProviderCredentials::API(k))
                .ok_or_else(|| {
                    match api_key_env {
                        Some(k) => format!("missing {} API key. Set the {} environment variable or directly configure the API key", self.display_name, k),
                        None => format!("missing {} API key. Either configure an API key environment variable or directly configure the API key", self.display_name)
                    }
                })?,
            ProviderAuthConfig::OAuth(config) => ProviderCredentials::OAuth(config.clone())
        };

        Ok(self.api.build_api(ProviderConfig {
            auth,
            base_url: self.base_url.clone(),
            options: self.options.clone(),
        })?)
    }
}

pub async fn refresh_oauth_provider(
    provider: &str,
    refresh_token: &str,
) -> Result<ProviderConfigEntry, Box<dyn std::error::Error>> {
    match provider {
        codex::PROVIDER_NAME => Ok(ProviderConfigEntry::OpenAiCodex(
            codex::refresh_credentials(refresh_token).await?,
        )),
        _ => Err(format!("provider '{provider}' does not support OAuth refresh").into()),
    }
}

pub fn configured_provider_from_entry(
    entry: &ProviderConfigEntry,
) -> Result<ConfiguredProvider, Box<dyn std::error::Error>> {
    match entry {
        ProviderConfigEntry::OpenAiApi(config) => Ok(configured_openai_provider(config)),
        ProviderConfigEntry::OpenAiCodex(config) => Ok(configured_codex_provider(config)),
        ProviderConfigEntry::AnthropicApi(config) => Ok(configured_anthropic_provider(config)),
        ProviderConfigEntry::Custom(config) => configured_custom_provider(config),
    }
}

fn configured_openai_provider(config: &ApiKeyProviderConfig) -> ConfiguredProvider {
    ConfiguredProvider {
        name: openai::PROVIDER_NAME.to_string(),
        api: &api::openai_responses::API,
        display_name: "OpenAI".to_string(),
        auth: api_key_auth(config, openai::API_KEY_ENV),
        base_url: config
            .base_url
            .clone()
            .unwrap_or_else(|| openai::DEFAULT_BASE_URL.to_string()),
        options: Value::Null,
        models: openai::default_models(),
    }
}

fn configured_anthropic_provider(config: &ApiKeyProviderConfig) -> ConfiguredProvider {
    ConfiguredProvider {
        name: anthropic::PROVIDER_NAME.to_string(),
        api: &api::anthropic_messages::API,
        display_name: "Anthropic".to_string(),
        auth: api_key_auth(config, anthropic::API_KEY_ENV),
        base_url: config
            .base_url
            .clone()
            .unwrap_or_else(|| anthropic::DEFAULT_BASE_URL.to_string()),
        options: Value::Null,
        models: anthropic::default_models(),
    }
}

fn api_key_auth(config: &ApiKeyProviderConfig, default_env: &str) -> ProviderAuthConfig {
    ProviderAuthConfig::ApiKey {
        api_key: config.api_key.clone(),
        api_key_env: config
            .api_key_env
            .clone()
            .or_else(|| Option::Some(default_env.to_string())),
    }
}

fn configured_codex_provider(config: &CodexProviderConfig) -> ConfiguredProvider {
    ConfiguredProvider {
        name: codex::PROVIDER_NAME.to_string(),
        api: &api::openai_responses::API,
        display_name: "OpenAI Codex".to_string(),
        auth: ProviderAuthConfig::OAuth(OAuthCredentials {
            access: config.access.clone(),
            refresh: config.refresh.clone(),
            expires: config.expires,
            account_id: config.account_id.clone(),
        }),
        base_url: codex::DEFAULT_BASE_URL.to_string(),
        options: Value::Null,
        models: codex::default_models(),
    }
}

fn configured_custom_provider(
    config: &CustomProviderConfig,
) -> Result<ConfiguredProvider, Box<dyn std::error::Error>> {
    let api = api::find_model_api(&config.api)
        .ok_or_else(|| format!("unsupported provider API: {}", config.api))?;
    // FIXME: should be able to automatically discover models
    if config.models.is_empty() {
        return Err(format!(
            "custom provider '{}' must specify at least one model",
            config.name
        )
        .into());
    }

    Ok(ConfiguredProvider {
        name: config.name.clone(),
        api,
        display_name: config.api.clone(),
        auth: ProviderAuthConfig::ApiKey {
            api_key: config.api_key.clone(),
            api_key_env: config.api_key_env.clone(),
        },
        base_url: config.base_url.clone(),
        options: config.options.clone(),
        models: configured_models_for_api(api, &config.models)?,
    })
}

fn configured_models_for_api(
    api: &'static api::ModelApiFactory,
    models: &[ModelConfigEntry],
) -> Result<Vec<ModelMetadata>, Box<dyn std::error::Error>> {
    models
        .iter()
        .map(|model| configured_model(api, model))
        .collect()
}

fn configured_model(
    api: &'static api::ModelApiFactory,
    model: &ModelConfigEntry,
) -> Result<ModelMetadata, Box<dyn std::error::Error>> {
    Ok(ModelMetadata {
        name: model.name.clone(),
        id: model.id.clone().unwrap_or_else(|| model.name.clone()),
        context_length: model.context_length.unwrap_or_default(),
        max_tokens: model.max_tokens.unwrap_or_default(),
        thinking_effort: model.thinking_effort.or(Some(ThinkingEffort::Max)),
        provider_config: parse_provider_model_config(api, model.provider_config.as_ref())?,
        costs: None,
    })
}

fn parse_provider_model_config(
    api: &'static api::ModelApiFactory,
    value: Option<&Value>,
) -> Result<Option<Arc<dyn ProviderModelConfig>>, Box<dyn std::error::Error>> {
    match value {
        Some(value) if !value.is_null() => match api.name {
            "anthropic_messages" => Ok(Some(Arc::new(serde_json::from_value::<
                anthropic::AnthropicModelConfig,
            >(value.clone())?))),
            _ => Err(format!(
                "provider_config is not supported for provider API {}",
                api.name
            )
            .into()),
        },
        _ => Ok(None),
    }
}

// Backwards-compatible re-exports for callers that still import API items from `providers`.
pub use crate::api::{
    ModelApi, ModelApiFactory, ProviderError, TokenUsage, available_model_apis, find_model_api,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deserializes_typed_provider_config_entries() {
        let entry: ProviderConfigEntry = serde_json::from_value(serde_json::json!({
            "type": "custom",
            "config": {
                "name": "local",
                "api": "openai_responses",
                "api_key": "none",
                "base_url": "http://localhost:8080",
                "models": [{"name": "model-a"}]
            }
        }))
        .unwrap();

        match entry {
            ProviderConfigEntry::Custom(config) => {
                assert_eq!(config.name, "local");
                assert_eq!(config.models.len(), 1);
            }
            _ => panic!("expected custom provider config"),
        }
    }

    #[test]
    fn builds_builtin_provider_from_api_key_entry() {
        let provider =
            configured_provider_from_entry(&ProviderConfigEntry::OpenAiApi(ApiKeyProviderConfig {
                api_key: Some("sk-test".to_string()),
                api_key_env: None,
                base_url: None,
            }))
            .unwrap();

        assert_eq!(provider.name, openai::PROVIDER_NAME);
        match provider.auth {
            ProviderAuthConfig::ApiKey { api_key, .. } => {
                assert_eq!(api_key.as_deref(), Some("sk-test"))
            }
            ProviderAuthConfig::OAuth { .. } => panic!("expected API key auth"),
        }
        assert!(!provider.models.is_empty());
    }
}
