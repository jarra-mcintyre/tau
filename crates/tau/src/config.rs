#![allow(dead_code)]

use std::{
    error::Error,
    fmt,
    fs::{self, OpenOptions},
    path::{Path, PathBuf},
    str::FromStr,
    sync::Arc,
    thread,
    time::{Duration, Instant},
};

use libtau::{
    api::ModelApi,
    context::{TauContext, TauSession},
    providers::{
        ApiKeyProviderConfig, ConfiguredProvider, ModelMetadata, ProviderConfigEntry,
        configured_provider_from_entry, refresh_oauth_provider,
    },
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const CONFIG_PATH: &str = ".tau/providers.json";
const DEFAULT_PROVIDER: &str = "openai";
const DEFAULT_MODEL: &str = "gpt-5.4-mini";
const PROVIDERS_CONFIG_LOCK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, Deserialize, Serialize, Default)]
#[serde(deny_unknown_fields)]
pub(crate) struct ProvidersConfig {
    #[serde(default)]
    current_model: Option<String>,
    #[serde(default)]
    providers: Vec<ProviderConfigEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ModelSelection {
    provider_name: String,
    model: String,
}

#[derive(Debug)]
pub(crate) struct CliConfig {
    current_model: ModelSelection,
    providers_config: ProvidersConfig,
    providers: Vec<ConfiguredProvider>,
    config_path: Option<PathBuf>,
}

impl CliConfig {
    pub(crate) fn load() -> Result<Self, Box<dyn Error>> {
        let (providers_config, config_path) = load_providers_config()?;
        let providers = configured_providers(&providers_config)?;
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

    pub(crate) fn current_model_metadata(&self) -> Result<ModelMetadata, Box<dyn Error>> {
        self.resolve_model_metadata(&self.current_model)
    }

    pub(crate) fn config_path(&self) -> Option<&PathBuf> {
        self.config_path.as_ref()
    }

    pub(crate) fn session_for_current_model(
        &self,
        context: &TauContext,
    ) -> Result<TauSession, Box<dyn Error>> {
        let provider = self.build_provider_for_selection(&self.current_model)?;
        let model = self.resolve_model_metadata(&self.current_model)?;
        let thinking_effort = model.thinking_effort;
        let mut session = context.session_with_provider_arc(provider, model);
        if thinking_effort.is_some() {
            session.set_thinking_effort(thinking_effort);
        }
        Ok(session)
    }

    pub(crate) fn build_provider_for_current_model(
        &self,
    ) -> Result<Arc<dyn libtau::api::ModelApi>, Box<dyn Error>> {
        self.build_provider_for_selection(&self.current_model)
    }

    pub(crate) async fn refresh_oauth_provider(
        &mut self,
        provider: &String,
        expected: OAuthRefreshRequest,
    ) -> Result<(), Box<dyn Error>> {
        let stored = if let Some(current) = oauth_provider_entry_if_changed(&provider, &expected)? {
            current
        } else {
            let refreshed = match refresh_oauth_provider(&provider, &expected.refresh).await {
                Ok(refreshed) => refreshed,
                Err(error) => {
                    if let Some(current) = oauth_provider_entry_if_changed(&provider, &expected)? {
                        self.apply_provider_entry(current)?;
                        return Ok(());
                    }
                    return Err(error);
                }
            };
            update_oauth_credentials(&provider, &expected, refreshed)?
        };
        self.apply_provider_entry(stored)
    }

    fn apply_provider_entry(&mut self, entry: ProviderConfigEntry) -> Result<(), Box<dyn Error>> {
        upsert_provider_entry(&mut self.providers_config.providers, entry);
        self.providers = configured_providers(&self.providers_config)?;
        Ok(())
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
        let model = self.resolve_model_metadata(&selection)?;
        let thinking_effort = model.thinking_effort;
        session.set_provider_and_model(provider, model);
        if thinking_effort.is_some() {
            session.set_thinking_effort(thinking_effort);
        }
        self.current_model = selection;
        Ok(())
    }

    pub(crate) fn print_models(&self) {
        let mut printed_current = false;
        for provider in &self.providers {
            for model in &provider.models {
                let selection = ModelSelection {
                    provider_name: provider.name.clone(),
                    model: model.name.clone(),
                };
                let summary = model_metadata_summary(model);
                if selection == self.current_model {
                    printed_current = true;
                    println!("* {selection}{summary}");
                } else {
                    println!("  {selection}{summary}");
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

    fn resolve_model_metadata(
        &self,
        selection: &ModelSelection,
    ) -> Result<ModelMetadata, Box<dyn std::error::Error>> {
        let provider = self
            .providers
            .iter()
            .find(|provider| provider.name == selection.provider_name)
            .ok_or_else(|| format!("provider '{}' is not configured", selection.provider_name))?;
        let model = provider
            .models
            .iter()
            .find(|model| model.name == selection.model || model.id == selection.model)
            .ok_or_else(|| {
                format!(
                    "model '{}' is not listed for provider '{}' in {}",
                    selection.model,
                    selection.provider_name,
                    config_path_label(self.config_path.as_ref())
                )
            })?;

        Ok(model.clone())
    }

    fn build_provider_for_selection(
        &self,
        selection: &ModelSelection,
    ) -> Result<Arc<dyn ModelApi>, Box<dyn Error>> {
        let provider = self
            .providers
            .iter()
            .find(|provider| provider.name == selection.provider_name)
            .ok_or_else(|| format!("provider '{}' is not configured", selection.provider_name))?;
        provider.build_api()
    }

    fn save_current_model(&mut self, selection: &ModelSelection) -> Result<(), Box<dyn Error>> {
        self.providers_config.current_model = Some(selection.to_string());
        self.save_providers_config()
    }

    fn save_providers_config(&mut self) -> Result<(), Box<dyn Error>> {
        let path = self
            .config_path
            .clone()
            .or_else(providers_config_path)
            .ok_or("cannot persist provider configuration because HOME is not set")?;

        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        fs::write(
            &path,
            serde_json::to_string_pretty(&self.providers_config)? + "\n",
        )?;
        self.config_path = Some(path);
        Ok(())
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

pub(crate) fn configure_provider_entry(entry: ProviderConfigEntry) -> Result<(), Box<dyn Error>> {
    let (mut config, path) = load_providers_config()?;
    upsert_provider_entry(&mut config.providers, entry);
    save_providers_config_to(path, &config)
}

#[derive(Debug, Clone)]
pub(crate) struct OAuthRefreshRequest {
    pub(crate) access: String,
    pub(crate) refresh: String,
    pub(crate) expires: i64,
}

pub(crate) fn list_providers(json_output: bool) -> Result<(), Box<dyn Error>> {
    let (config, config_path) = load_providers_config()?;
    let configured = config.providers;

    if json_output {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "config_path": config_path.or_else(providers_config_path).map(|path| path.display().to_string()),
                "providers": provider_statuses_json(&configured),
            }))?
        );
        return Ok(());
    }

    for line in provider_status_lines(&configured) {
        println!("{line}");
    }
    Ok(())
}

fn oauth_provider_entry_if_changed(
    provider: &str,
    expected: &OAuthRefreshRequest,
) -> Result<Option<ProviderConfigEntry>, Box<dyn Error>> {
    let path = providers_config_path()
        .ok_or("cannot read provider configuration because HOME is not set")?;
    let _lock = ProvidersConfigLock::acquire(&path)?;
    let (config, _) = load_providers_config()?;
    let current = config
        .providers
        .iter()
        .find(|entry| entry.provider_name() == provider)
        .ok_or_else(|| format!("provider '{provider}' is no longer configured"))?;

    if !entry_matches_expected_oauth(provider, current, expected)? {
        return Ok(Some(current.clone()));
    }
    Ok(None)
}

fn entry_matches_expected_oauth(
    provider: &str,
    entry: &ProviderConfigEntry,
    expected: &OAuthRefreshRequest,
) -> Result<bool, Box<dyn Error>> {
    let credentials = entry
        .oauth_credentials()
        .ok_or_else(|| format!("provider '{provider}' is not configured with OAuth"))?;
    Ok(credentials.access == expected.access
        && credentials.refresh == expected.refresh
        && credentials.expires == expected.expires)
}

fn update_oauth_credentials(
    provider: &str,
    expected: &OAuthRefreshRequest,
    refreshed: ProviderConfigEntry,
) -> Result<ProviderConfigEntry, Box<dyn Error>> {
    let path = providers_config_path()
        .ok_or("cannot persist provider configuration because HOME is not set")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let _lock = ProvidersConfigLock::acquire(&path)?;
    let (mut config, _) = load_providers_config()?;
    let entry = config
        .providers
        .iter_mut()
        .find(|entry| entry.provider_name() == provider)
        .ok_or_else(|| format!("provider '{provider}' is no longer configured"))?;

    if !entry_matches_expected_oauth(provider, entry, expected)? {
        return Ok(entry.clone());
    }

    if refreshed.provider_name() != provider {
        return Err(format!(
            "OAuth refresh for provider '{}' returned credentials for '{}'",
            provider,
            refreshed.provider_name()
        )
        .into());
    }

    *entry = refreshed.clone();
    let temp_path = path.with_extension(format!("json.tmp-{}", uuid::Uuid::new_v4()));
    fs::write(&temp_path, serde_json::to_string_pretty(&config)? + "\n")?;
    fs::rename(&temp_path, &path)?;
    Ok(refreshed)
}

struct ProvidersConfigLock {
    path: PathBuf,
}

impl ProvidersConfigLock {
    fn acquire(config_path: &Path) -> Result<Self, Box<dyn Error>> {
        let lock_path = config_path.with_extension("json.lock");
        if let Some(parent) = lock_path.parent() {
            fs::create_dir_all(parent)?;
        }
        let start = Instant::now();
        loop {
            match OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(_) => return Ok(Self { path: lock_path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    if start.elapsed() >= PROVIDERS_CONFIG_LOCK_TIMEOUT {
                        return Err(format!(
                            "timed out waiting for provider configuration lock {}",
                            lock_path.display()
                        )
                        .into());
                    }
                    thread::sleep(Duration::from_millis(50));
                }
                Err(error) => return Err(error.into()),
            }
        }
    }
}

impl Drop for ProvidersConfigLock {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn configured_providers(
    config: &ProvidersConfig,
) -> Result<Vec<ConfiguredProvider>, Box<dyn Error>> {
    let entries = if config.providers.is_empty() {
        vec![ProviderConfigEntry::OpenAiApi(
            ApiKeyProviderConfig::default(),
        )]
    } else {
        config.providers.clone()
    };

    entries
        .iter()
        .map(configured_provider_from_entry)
        .collect::<Result<Vec<_>, _>>()
}

fn model_metadata_summary(model: &ModelMetadata) -> String {
    let costs = model
        .costs
        .as_ref()
        .map(|costs| format!(", costs: {costs:?}"))
        .unwrap_or_default();
    let provider_config = model
        .provider_config
        .as_ref()
        .map(|config| format!(", provider_config: {config:?}"))
        .unwrap_or_default();

    format!(
        " (id: {}, context: {}{provider_config}{costs})",
        model.id, model.context_length
    )
}

fn first_configured_model_ref(providers: &[ConfiguredProvider]) -> Option<String> {
    providers.iter().find_map(|provider| {
        provider
            .models
            .first()
            .map(|model| format!("{}/{}", provider.name, model.name))
    })
}

fn validate_model_selection(
    providers: &[ConfiguredProvider],
    selection: &ModelSelection,
    config_path: Option<&PathBuf>,
) -> Result<(), Box<dyn Error>> {
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

    if !provider.models.is_empty()
        && !provider
            .models
            .iter()
            .any(|model| model.name == selection.model || model.id == selection.model)
    {
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

fn upsert_provider_entry(entries: &mut Vec<ProviderConfigEntry>, entry: ProviderConfigEntry) {
    let key = provider_entry_key(&entry).to_string();
    if let Some(existing) = entries
        .iter_mut()
        .find(|existing| provider_entry_key(existing) == key)
    {
        *existing = entry;
    } else {
        entries.push(entry);
    }
}

fn provider_entry_key(entry: &ProviderConfigEntry) -> &str {
    match entry {
        ProviderConfigEntry::OpenAiApi(_) => "openai-api",
        ProviderConfigEntry::OpenAiCodex(_) => "openai-codex",
        ProviderConfigEntry::AnthropicApi(_) => "anthropic-api",
        ProviderConfigEntry::Custom(config) => &config.name,
    }
}

fn provider_status_lines(entries: &[ProviderConfigEntry]) -> Vec<String> {
    let mut lines = vec![
        builtin_provider_status_line("openai-api", entries),
        builtin_provider_status_line("openai-codex", entries),
        builtin_provider_status_line("anthropic-api", entries),
    ];
    lines.extend(entries.iter().filter_map(|entry| match entry {
        ProviderConfigEntry::Custom(config) => Some(format!("custom/{}: configured", config.name)),
        _ => None,
    }));
    lines
}

fn builtin_provider_status_line(provider_type: &str, entries: &[ProviderConfigEntry]) -> String {
    let configured = entries
        .iter()
        .any(|entry| provider_entry_key(entry) == provider_type);
    format!(
        "{provider_type}: {}",
        if configured {
            "configured"
        } else {
            "not configured"
        }
    )
}

fn provider_statuses_json(entries: &[ProviderConfigEntry]) -> Value {
    let mut statuses = vec![
        builtin_provider_status_json("openai-api", entries),
        builtin_provider_status_json("openai-codex", entries),
        builtin_provider_status_json("anthropic-api", entries),
    ];
    statuses.extend(entries.iter().filter_map(|entry| match entry {
        ProviderConfigEntry::Custom(config) => Some(json!({
            "type": "custom",
            "name": config.name,
            "configured": true,
        })),
        _ => None,
    }));
    Value::Array(statuses)
}

fn builtin_provider_status_json(provider_type: &str, entries: &[ProviderConfigEntry]) -> Value {
    json!({
        "type": provider_type,
        "configured": entries.iter().any(|entry| provider_entry_key(entry) == provider_type),
    })
}

fn config_path_label(config_path: Option<&PathBuf>) -> String {
    config_path
        .map(|path| path.display().to_string())
        .unwrap_or_else(|| CONFIG_PATH.to_string())
}

fn load_providers_config() -> Result<(ProvidersConfig, Option<PathBuf>), Box<dyn Error>> {
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

fn save_providers_config_to(
    path: Option<PathBuf>,
    config: &ProvidersConfig,
) -> Result<(), Box<dyn Error>> {
    let path = path
        .or_else(providers_config_path)
        .ok_or("cannot persist provider configuration because HOME is not set")?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, serde_json::to_string_pretty(config)? + "\n")?;
    Ok(())
}

fn providers_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(CONFIG_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;
    use libtau::{api::find_model_api, providers::ProviderAuthConfig};

    fn provider(name: &str, models: &[&str]) -> ConfiguredProvider {
        ConfiguredProvider {
            name: name.to_string(),
            api: find_model_api("openai_responses").unwrap(),
            display_name: name.to_string(),
            auth: ProviderAuthConfig::ApiKey {
                api_key: None,
                api_key_env: Some("OPENAI_API_KEY".to_string()),
            },
            base_url: "https://example.com".to_string(),
            options: Value::Null,
            models: models
                .iter()
                .map(|model| ModelMetadata::custom(*model))
                .collect(),
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

    #[test]
    fn upserts_builtin_provider_entries() {
        let mut entries = vec![ProviderConfigEntry::AnthropicApi(ApiKeyProviderConfig {
            api_key: Some("old".to_string()),
            api_key_env: None,
            base_url: None,
        })];

        upsert_provider_entry(
            &mut entries,
            ProviderConfigEntry::AnthropicApi(ApiKeyProviderConfig {
                api_key: Some("new".to_string()),
                api_key_env: None,
                base_url: None,
            }),
        );

        assert_eq!(entries.len(), 1);
        match &entries[0] {
            ProviderConfigEntry::AnthropicApi(config) => {
                assert_eq!(config.api_key.as_deref(), Some("new"));
            }
            _ => panic!("expected anthropic config"),
        }
    }
}
