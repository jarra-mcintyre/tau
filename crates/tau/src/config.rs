use std::{fmt, fs, path::PathBuf, str::FromStr, sync::Arc};

use libtau::{
    context::{TauContext, TauSession},
    providers::{Provider, ProviderApi, ProviderApiConfig, find_provider_api, openai},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const CONFIG_PATH: &str = ".tau/providers.json";
const DEFAULT_PROVIDER: &str = "openai";
const DEFAULT_API: &str = openai::API_NAME;
const DEFAULT_MODEL: &str = "gpt-4.1-mini";

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
struct ProvidersConfig {
    #[serde(default)]
    current_model: Option<String>,
    #[serde(default)]
    providers: Vec<ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
struct ProviderConfig {
    /// The user-facing provider name, e.g. "personal-openai".
    #[serde(default)]
    name: Option<String>,
    /// The provider API implementation to use, e.g. "openai_responses" or "anthropic_messages".
    #[serde(default)]
    api: Option<String>,
    #[serde(default)]
    api_key: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    options: Value,
    #[serde(default)]
    models: Vec<String>,
}

#[derive(Debug, Clone)]
struct ConfiguredProvider {
    name: String,
    provider_api: &'static ProviderApi,
    api_key: Option<String>,
    base_url: Option<String>,
    options: Value,
    models: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelSelection {
    provider_name: String,
    model: String,
}

#[derive(Debug, Clone)]
pub(crate) struct CliConfig {
    current_model: ModelSelection,
    providers_config: ProvidersConfig,
    providers: Vec<ConfiguredProvider>,
    config_path: Option<PathBuf>,
}

impl CliConfig {
    pub(crate) fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let (providers_config, config_path) = load_providers_config()?;
        let providers = configured_providers(&providers_config, config_path.as_ref())?;
        let current_model_ref = std::env::var("TAU_MODEL")
            .ok()
            .or(providers_config.current_model.clone())
            .or_else(|| first_configured_model_ref(&providers))
            .unwrap_or_else(|| format!("{DEFAULT_PROVIDER}/{DEFAULT_MODEL}"));
        let current_model = current_model_ref.parse()?;
        validate_model_selection(&providers, &current_model, config_path.as_ref())?;

        Ok(Self {
            current_model,
            providers_config,
            providers,
            config_path,
        })
    }

    pub(crate) fn current_model(&self) -> &ModelSelection {
        &self.current_model
    }

    pub(crate) fn config_path(&self) -> Option<&PathBuf> {
        self.config_path.as_ref()
    }

    pub(crate) fn session_for_current_model(
        &self,
        context: &TauContext,
    ) -> Result<TauSession, Box<dyn std::error::Error>> {
        let provider = self.build_provider_for_selection(&self.current_model)?;
        Ok(context.session_with_provider_arc(provider, self.current_model.model.clone()))
    }

    pub(crate) fn restore_current_model(
        &mut self,
        selection: ModelSelection,
    ) -> Result<(), Box<dyn std::error::Error>> {
        validate_model_selection(&self.providers, &selection, self.config_path.as_ref())?;
        self.current_model = selection;
        Ok(())
    }

    pub(crate) fn switch_model(
        &mut self,
        session: &mut TauSession,
        model_ref: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let selection = model_ref.parse()?;
        validate_model_selection(&self.providers, &selection, self.config_path.as_ref())?;
        let provider = self.build_provider_for_selection(&selection)?;
        self.save_current_model(&selection)?;
        session.set_provider_and_model(provider, selection.model.clone());
        self.current_model = selection;
        Ok(())
    }

    pub(crate) fn print_models(&self) {
        let mut printed_current = false;
        for provider in &self.providers {
            for model in &provider.models {
                let selection = ModelSelection {
                    provider_name: provider.name.clone(),
                    model: model.clone(),
                };
                if selection == self.current_model {
                    printed_current = true;
                    println!("* {selection}");
                } else {
                    println!("  {selection}");
                }
            }
        }

        if !printed_current {
            println!("* {}", self.current_model);
        }
    }

    pub(crate) fn print_current_model(&self) {
        println!("model: {}", self.current_model);
    }

    fn build_provider_for_selection(
        &self,
        selection: &ModelSelection,
    ) -> Result<Arc<dyn Provider>, Box<dyn std::error::Error>> {
        let provider = self
            .providers
            .iter()
            .find(|provider| provider.name == selection.provider_name)
            .ok_or_else(|| format!("provider '{}' is not configured", selection.provider_name))?;

        let api_key = std::env::var(provider.provider_api.api_key_env)
            .ok()
            .or(provider.api_key.clone())
            .ok_or_else(|| {
                format!(
                    "missing {} API key; set {} or providers.{}.api_key in ~/.tau/providers.json",
                    provider.provider_api.display_name,
                    provider.provider_api.api_key_env,
                    provider.name
                )
            })?;

        Ok(provider.provider_api.build_provider(ProviderApiConfig {
            api_key,
            base_url: provider.base_url.clone(),
            options: provider.options.clone(),
        })?)
    }

    fn save_current_model(
        &mut self,
        selection: &ModelSelection,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let path = self
            .config_path
            .clone()
            .or_else(providers_config_path)
            .ok_or("cannot persist current model because HOME is not set")?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        self.providers_config.current_model = Some(selection.to_string());
        fs::write(
            &path,
            serde_json::to_string_pretty(&self.providers_config)? + "\n",
        )?;
        self.config_path = Some(path);
        Ok(())
    }
}

impl ModelSelection {
    pub(crate) fn model(&self) -> &str {
        &self.model
    }
}

impl FromStr for ModelSelection {
    type Err = String;

    fn from_str(model_ref: &str) -> Result<Self, Self::Err> {
        let Some((provider_name, model)) = model_ref.split_once('/') else {
            return Err(format!(
                "model must be specified as provider/model, got '{model_ref}'"
            ));
        };
        if provider_name.is_empty() || model.is_empty() {
            return Err(format!(
                "model must be specified as provider/model, got '{model_ref}'"
            ));
        }

        Ok(Self {
            provider_name: provider_name.to_string(),
            model: model.to_string(),
        })
    }
}

impl fmt::Display for ModelSelection {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}/{}", self.provider_name, self.model)
    }
}

fn configured_providers(
    config: &ProvidersConfig,
    config_path: Option<&PathBuf>,
) -> Result<Vec<ConfiguredProvider>, Box<dyn std::error::Error>> {
    let mut providers = Vec::new();

    for provider_config in &config.providers {
        let name = provider_config.name.clone().ok_or_else(|| {
            format!(
                "provider entry in {} is missing a name",
                config_path_label(config_path)
            )
        })?;
        let api_name = provider_config
            .api
            .clone()
            .or_else(|| default_provider_api_name(&name).map(str::to_string))
            .ok_or_else(|| {
                format!(
                    "provider '{name}' must specify an api in {}",
                    config_path_label(config_path)
                )
            })?;
        let provider_api = find_provider_api(&api_name)
            .ok_or_else(|| format!("unsupported provider API: {api_name}"))?;

        providers.push(ConfiguredProvider {
            name,
            provider_api,
            api_key: provider_config.api_key.clone(),
            base_url: provider_config.base_url.clone(),
            options: provider_config.options.clone(),
            models: provider_config.models.clone(),
        });
    }

    if providers.is_empty() {
        providers.push(ConfiguredProvider {
            name: DEFAULT_PROVIDER.to_string(),
            provider_api: find_provider_api(DEFAULT_API)
                .expect("default provider API is registered"),
            api_key: None,
            base_url: None,
            options: Value::Null,
            models: vec![DEFAULT_MODEL.to_string()],
        });
    }

    Ok(providers)
}

fn first_configured_model_ref(providers: &[ConfiguredProvider]) -> Option<String> {
    providers.iter().find_map(|provider| {
        provider
            .models
            .first()
            .map(|model| format!("{}/{model}", provider.name))
    })
}

fn validate_model_selection(
    providers: &[ConfiguredProvider],
    selection: &ModelSelection,
    config_path: Option<&PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let provider = providers
        .iter()
        .find(|provider| provider.name == selection.provider_name)
        .ok_or_else(|| {
            format!(
                "provider '{}' is not configured in {}",
                selection.provider_name,
                config_path_label(config_path)
            )
        })?;

    if !provider.models.is_empty() && !provider.models.contains(&selection.model) {
        return Err(format!(
            "model '{}' is not listed for provider '{}' in {}",
            selection.model,
            selection.provider_name,
            config_path_label(config_path)
        )
        .into());
    }

    Ok(())
}

fn config_path_label(config_path: Option<&PathBuf>) -> String {
    config_path
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| CONFIG_PATH.to_string())
}

fn load_providers_config() -> Result<(ProvidersConfig, Option<PathBuf>), Box<dyn std::error::Error>>
{
    let Some(path) = providers_config_path() else {
        return Ok((ProvidersConfig::default(), None));
    };

    if !path.exists() {
        return Ok((ProvidersConfig::default(), None));
    }

    let contents = fs::read_to_string(&path)?;
    let config = serde_json::from_str(&contents)
        .map_err(|error| format!("failed to parse {}: {error}", path.display()))?;

    Ok((config, Some(path)))
}

fn default_provider_api_name(provider_name: &str) -> Option<&'static str> {
    (provider_name == DEFAULT_PROVIDER).then_some(DEFAULT_API)
}

fn providers_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(CONFIG_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn provider(name: &str, models: &[&str]) -> ConfiguredProvider {
        ConfiguredProvider {
            name: name.to_string(),
            provider_api: find_provider_api(DEFAULT_API).unwrap(),
            api_key: None,
            base_url: None,
            options: Value::Null,
            models: models.iter().map(|model| model.to_string()).collect(),
        }
    }

    #[test]
    fn parses_and_displays_model_selection() {
        let selection: ModelSelection = "openai/gpt-4.1".parse().unwrap();

        assert_eq!(selection.provider_name, "openai");
        assert_eq!(selection.model, "gpt-4.1");
        assert_eq!(selection.to_string(), "openai/gpt-4.1");
    }

    #[test]
    fn rejects_malformed_model_selection() {
        assert!("gpt-4.1".parse::<ModelSelection>().is_err());
        assert!("/gpt-4.1".parse::<ModelSelection>().is_err());
        assert!("openai/".parse::<ModelSelection>().is_err());
    }

    #[test]
    fn validates_model_selection_across_providers() {
        let providers = vec![
            provider("openai", &["gpt-4.1", "gpt-4.1-mini"]),
            provider("anthropic", &["claude-sonnet-4"]),
        ];

        assert!(
            validate_model_selection(
                &providers,
                &"anthropic/claude-sonnet-4".parse().unwrap(),
                None
            )
            .is_ok()
        );
        assert!(
            validate_model_selection(&providers, &"openai/claude-sonnet-4".parse().unwrap(), None)
                .is_err()
        );
        assert!(
            validate_model_selection(&providers, &"google/gemini-pro".parse().unwrap(), None)
                .is_err()
        );
    }

    #[test]
    fn chooses_first_configured_model_ref() {
        let providers = vec![
            provider("empty", &[]),
            provider("anthropic", &["claude-sonnet-4"]),
            provider("openai", &["gpt-4.1"]),
        ];

        assert_eq!(
            first_configured_model_ref(&providers),
            Some("anthropic/claude-sonnet-4".to_string())
        );
    }
}
