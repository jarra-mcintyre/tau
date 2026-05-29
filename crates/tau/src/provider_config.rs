use clap::ValueEnum;
use libtau::providers::{ApiKeyProviderConfig, ProviderConfigEntry, anthropic, codex, openai};

use crate::config;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum ConfigurableProvider {
    OpenaiApi,
    OpenaiCodex,
    AnthropicApi,
}

impl ConfigurableProvider {
    fn available_message() -> String {
        let providers = Self::value_variants()
            .iter()
            .map(|provider| {
                provider
                    .to_possible_value()
                    .expect("provider has a name")
                    .get_name()
                    .to_string()
            })
            .collect::<Vec<_>>()
            .join(", ");
        format!("available providers: {providers}")
    }
}

pub(crate) async fn configure_provider(
    provider: Option<ConfigurableProvider>,
) -> Result<(), Box<dyn std::error::Error>> {
    let entry = match provider {
        Some(ConfigurableProvider::OpenaiApi) => {
            ProviderConfigEntry::OpenAiApi(configure_api_key("openai-api", openai::API_KEY_ENV)?)
        }
        Some(ConfigurableProvider::AnthropicApi) => ProviderConfigEntry::AnthropicApi(
            configure_api_key("anthropic-api", anthropic::API_KEY_ENV)?,
        ),
        Some(ConfigurableProvider::OpenaiCodex) => {
            println!("Configuring openai-codex.");
            ProviderConfigEntry::OpenAiCodex(codex::authenticate().await?)
        }
        None => {
            return Err(format!(
                "provider is required; {}",
                ConfigurableProvider::available_message()
            )
            .into());
        }
    };

    let provider_name = provider_entry_name(&entry).to_string();
    config::configure_provider_entry(entry)?;
    println!("Stored provider configuration for {provider_name} in ~/.tau/providers.json.");
    Ok(())
}

fn configure_api_key(
    provider_name: &str,
    env_var: &str,
) -> Result<ApiKeyProviderConfig, Box<dyn std::error::Error>> {
    println!("Configuring {provider_name}.");
    println!("Enter an API key to store, or press Enter to read {env_var} from the environment:");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let api_key = match input.trim() {
        "" => std::env::var(env_var).map_err(|_| format!("{env_var} is not set"))?,
        value => value.to_string(),
    };

    Ok(ApiKeyProviderConfig {
        api_key: Some(api_key),
        api_key_env: None,
        base_url: None,
    })
}

fn provider_entry_name(entry: &ProviderConfigEntry) -> &'static str {
    match entry {
        ProviderConfigEntry::OpenAiApi(_) => "openai-api",
        ProviderConfigEntry::OpenAiCodex(_) => "openai-codex",
        ProviderConfigEntry::AnthropicApi(_) => "anthropic-api",
        ProviderConfigEntry::Custom(_) => "custom",
    }
}
