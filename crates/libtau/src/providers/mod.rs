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

/// Runtime API configuration consumed by a concrete model API implementation.
#[derive(Debug, Clone)]
pub struct ProviderConfig {
    pub api_key: String,
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
#[serde(untagged)]
pub enum ModelConfigEntry {
    Name(String),
    Detailed {
        name: String,
        #[serde(default)]
        id: Option<String>,
        #[serde(default)]
        context_length: u64,
        #[serde(default)]
        max_tokens: u64,
        #[serde(default)]
        thinking_effort: Option<ThinkingEffort>,
        #[serde(default)]
        provider_config: Value,
    },
}

#[derive(Debug)]
pub struct ConfiguredProvider {
    pub name: String,
    pub api: &'static api::ModelApiFactory,
    pub display_name: String,
    pub auth: ProviderAuth,
    pub base_url: String,
    pub options: Value,
    pub models: Vec<ConfiguredModel>,
}

#[derive(Debug)]
pub enum ProviderAuth {
    ApiKey {
        api_key: Option<String>,
        api_key_env: String,
    },
    OAuth {
        access: String,
        refresh: String,
        expires: i64,
        account_id: String,
    },
}

#[derive(Debug)]
pub struct ConfiguredModel {
    pub name: String,
    pub id: String,
    pub metadata: Option<ModelMetadata>,
}

impl ConfiguredProvider {
    pub fn build_api(&self) -> Result<Arc<dyn api::ModelApi>, Box<dyn std::error::Error>> {
        let api_key = match &self.auth {
            ProviderAuth::ApiKey {
                api_key,
                api_key_env,
            } => std::env::var(api_key_env)
                .ok()
                .or_else(|| api_key.clone())
                .ok_or_else(|| {
                    format!(
                        "missing {} API key; set {} or configure provider '{}' in ~/.tau/providers.json",
                        self.display_name, api_key_env, self.name
                    )
                })?,
            ProviderAuth::OAuth { access, .. } => access.clone(),
        };

        Ok(self.api.build_api(ProviderConfig {
            api_key,
            base_url: self.base_url.clone(),
            options: self.runtime_options(),
        })?)
    }

    fn runtime_options(&self) -> Value {
        match &self.auth {
            ProviderAuth::ApiKey { .. } => self.options.clone(),
            ProviderAuth::OAuth {
                refresh,
                expires,
                account_id,
                ..
            } => {
                let mut options = self.options.clone();
                if !options.is_object() {
                    options = serde_json::json!({});
                }
                let map = options.as_object_mut().expect("options was object above");
                map.insert("refresh".to_string(), Value::String(refresh.clone()));
                map.insert("expires".to_string(), Value::Number((*expires).into()));
                map.insert("accountId".to_string(), Value::String(account_id.clone()));
                options
            }
        }
    }
}

pub fn configured_provider_from_entry(
    entry: &ProviderConfigEntry,
) -> Result<ConfiguredProvider, Box<dyn std::error::Error>> {
    match entry {
        ProviderConfigEntry::OpenAiApi(config) => configured_openai_provider(config),
        ProviderConfigEntry::OpenAiCodex(config) => Ok(configured_codex_provider(config)),
        ProviderConfigEntry::AnthropicApi(config) => configured_anthropic_provider(config),
        ProviderConfigEntry::Custom(config) => configured_custom_provider(config),
    }
}

fn configured_openai_provider(
    config: &ApiKeyProviderConfig,
) -> Result<ConfiguredProvider, Box<dyn std::error::Error>> {
    Ok(ConfiguredProvider {
        name: openai::PROVIDER_NAME.to_string(),
        api: &api::openai_responses::API,
        display_name: "OpenAI".to_string(),
        auth: api_key_auth(config, openai::API_KEY_ENV),
        base_url: config
            .base_url
            .clone()
            .unwrap_or_else(|| openai::DEFAULT_BASE_URL.to_string()),
        options: Value::Null,
        models: models_from_metadata(openai::default_models()),
    })
}

fn configured_anthropic_provider(
    config: &ApiKeyProviderConfig,
) -> Result<ConfiguredProvider, Box<dyn std::error::Error>> {
    Ok(ConfiguredProvider {
        name: anthropic::PROVIDER_NAME.to_string(),
        api: &api::anthropic_messages::API,
        display_name: "Anthropic".to_string(),
        auth: api_key_auth(config, anthropic::API_KEY_ENV),
        base_url: config
            .base_url
            .clone()
            .unwrap_or_else(|| anthropic::DEFAULT_BASE_URL.to_string()),
        options: Value::Null,
        models: models_from_metadata(anthropic::default_models()),
    })
}

fn api_key_auth(config: &ApiKeyProviderConfig, default_env: &str) -> ProviderAuth {
    ProviderAuth::ApiKey {
        api_key: config.api_key.clone(),
        api_key_env: config
            .api_key_env
            .clone()
            .unwrap_or_else(|| default_env.to_string()),
    }
}

fn configured_codex_provider(config: &CodexProviderConfig) -> ConfiguredProvider {
    ConfiguredProvider {
        name: codex::PROVIDER_NAME.to_string(),
        api: &api::openai_responses::API,
        display_name: "OpenAI Codex".to_string(),
        auth: ProviderAuth::OAuth {
            access: config.access.clone(),
            refresh: config.refresh.clone(),
            expires: config.expires,
            account_id: config.account_id.clone(),
        },
        base_url: codex::DEFAULT_BASE_URL.to_string(),
        options: Value::Null,
        models: models_from_metadata(codex::default_models()),
    }
}

fn configured_custom_provider(
    config: &CustomProviderConfig,
) -> Result<ConfiguredProvider, Box<dyn std::error::Error>> {
    let api = api::find_model_api(&config.api)
        .ok_or_else(|| format!("unsupported provider API: {}", config.api))?;
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
        auth: ProviderAuth::ApiKey {
            api_key: config.api_key.clone(),
            api_key_env: config
                .api_key_env
                .clone()
                .unwrap_or_else(|| "TAU_API_KEY".to_string()),
        },
        base_url: config.base_url.clone(),
        options: config.options.clone(),
        models: configured_models_for_api(api, &config.models)?,
    })
}

fn configured_models_for_api(
    api: &'static api::ModelApiFactory,
    models: &[ModelConfigEntry],
) -> Result<Vec<ConfiguredModel>, Box<dyn std::error::Error>> {
    models
        .iter()
        .map(|model| configured_model(api, model))
        .collect()
}

fn configured_model(
    api: &'static api::ModelApiFactory,
    model: &ModelConfigEntry,
) -> Result<ConfiguredModel, Box<dyn std::error::Error>> {
    match model {
        ModelConfigEntry::Name(name) => Ok(ConfiguredModel {
            name: name.clone(),
            id: name.clone(),
            metadata: Some(ModelMetadata::custom(name.clone())),
        }),
        ModelConfigEntry::Detailed {
            name,
            id,
            context_length,
            max_tokens,
            thinking_effort,
            provider_config,
        } => {
            let id = id.clone().unwrap_or_else(|| name.clone());
            Ok(ConfiguredModel {
                name: name.clone(),
                id: id.clone(),
                metadata: Some(ModelMetadata {
                    name: name.clone(),
                    id,
                    context_length: *context_length,
                    max_tokens: *max_tokens,
                    thinking_effort: *thinking_effort,
                    provider_config: parse_provider_model_config(api, provider_config)?,
                    costs: None,
                }),
            })
        }
    }
}

fn parse_provider_model_config(
    api: &'static api::ModelApiFactory,
    value: &Value,
) -> Result<Option<Arc<dyn ProviderModelConfig>>, Box<dyn std::error::Error>> {
    if value.is_null() {
        return Ok(None);
    }

    match api.name {
        "anthropic_messages" => Ok(Some(Arc::new(serde_json::from_value::<
            anthropic::AnthropicModelConfig,
        >(value.clone())?))),
        _ => Err(format!(
            "provider_config is not supported for provider API {}",
            api.name
        )
        .into()),
    }
}

fn models_from_metadata(models: Vec<ModelMetadata>) -> Vec<ConfiguredModel> {
    models
        .into_iter()
        .map(|metadata| ConfiguredModel {
            name: metadata.name.to_string(),
            id: metadata.id.to_string(),
            metadata: Some(metadata),
        })
        .collect()
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
                "models": ["model-a"]
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
            ProviderAuth::ApiKey { api_key, .. } => assert_eq!(api_key.as_deref(), Some("sk-test")),
            ProviderAuth::OAuth { .. } => panic!("expected API key auth"),
        }
        assert!(!provider.models.is_empty());
    }
}
