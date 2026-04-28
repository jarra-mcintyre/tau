use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use libtau::{
    context::{ContentPart, TauContext, TauResponse, TauSession, ToolResult, ToolUse},
    providers::{ProviderApi, ProviderApiConfig, TokenUsage, find_provider_api, openai},
    tools,
};
use serde::Deserialize;

const CONFIG_PATH: &str = ".tau/providers.json";
const DEFAULT_PROVIDER: &str = "openai";
const DEFAULT_API: &str = openai::API_NAME;
const DEFAULT_MODEL: &str = "gpt-4.1-mini";
const SYSTEM_MESSAGE: &str = r#"You are Tau, a coding agent running in a terminal.

You can inspect and modify files using tools. When the user asks you to read, write, or edit files, use the available tools."#;

#[derive(Debug, Clone, Deserialize, Default)]
struct ProvidersConfig {
    #[serde(default)]
    default_provider: Option<String>,
    #[serde(default)]
    default_model: Option<String>,
    #[serde(default)]
    providers: Vec<ProviderConfig>,
}

#[derive(Debug, Clone, Deserialize, Default)]
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
    default_model: Option<String>,
    #[serde(default)]
    models: Vec<String>,
}

#[derive(Debug, Clone)]
struct CliConfig {
    provider_name: String,
    provider_api: &'static ProviderApi,
    model: String,
    available_models: Vec<String>,
    provider_config: ProviderApiConfig,
    config_path: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let cli_config = load_cli_config()?;
    let mut context = TauContext::new();
    tools::register_builtin_tools(&mut context)?;

    let provider = cli_config
        .provider_api
        .build_provider(cli_config.provider_config.clone());
    let mut session = context.session_with_provider_arc(provider, cli_config.model.clone());
    session.set_system_message(SYSTEM_MESSAGE);

    println!("Tau interactive shell");
    println!("provider: {}", cli_config.provider_name);
    println!("api: {}", cli_config.provider_api.name);
    println!("model: {}", cli_config.model);
    if let Some(path) = &cli_config.config_path {
        println!("config: {}", path.display());
    } else {
        println!("config: not found, using environment/defaults");
    }
    println!("type /models to list configured models");
    println!("type /exit or press Ctrl-D to quit\n");

    let stdin = io::stdin();
    loop {
        print!("tau> ");
        io::stdout().flush()?;

        let mut line = String::new();
        let bytes_read = stdin.read_line(&mut line)?;
        if bytes_read == 0 {
            println!();
            break;
        }

        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if matches!(line, "/exit" | "/quit") {
            break;
        }
        if line == "/models" {
            print_models(&cli_config);
            continue;
        }

        if let Err(error) = run_turn(&mut session, line).await {
            eprintln!("error: {error}");
        }
    }

    Ok(())
}

fn load_cli_config() -> Result<CliConfig, Box<dyn std::error::Error>> {
    let (providers_config, config_path) = load_providers_config()?;

    let provider_name = std::env::var("TAU_PROVIDER")
        .ok()
        .or(providers_config.default_provider.clone())
        .unwrap_or_else(|| DEFAULT_PROVIDER.to_string());

    let provider_config = providers_config
        .providers
        .iter()
        .find(|provider| provider.name.as_deref() == Some(provider_name.as_str()))
        .cloned()
        .unwrap_or_default();

    let provider_api_name = provider_config
        .api
        .clone()
        .or_else(|| default_provider_api_name(&provider_name).map(str::to_string))
        .ok_or_else(|| {
            format!(
                "provider '{provider_name}' must specify an api in {}",
                config_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| CONFIG_PATH.to_string())
            )
        })?;
    let provider_api = find_provider_api(&provider_api_name)
        .ok_or_else(|| format!("unsupported provider API: {provider_api_name}"))?;

    let model = std::env::var("TAU_MODEL")
        .ok()
        .or(providers_config.default_model.clone())
        .or(provider_config.default_model.clone())
        .or_else(|| provider_config.models.first().cloned())
        .unwrap_or_else(|| DEFAULT_MODEL.to_string());

    if !provider_config.models.is_empty() && !provider_config.models.contains(&model) {
        return Err(format!(
            "model '{model}' is not listed for provider '{provider_name}' in {}",
            config_path
                .as_ref()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| CONFIG_PATH.to_string())
        )
        .into());
    }

    let api_key = std::env::var(provider_api.api_key_env)
        .ok()
        .or(provider_config.api_key.clone())
        .ok_or_else(|| {
            format!(
                "missing {} API key; set {} or providers.{provider_name}.api_key in ~/.tau/providers.json",
                provider_api.display_name, provider_api.api_key_env
            )
        })?;

    Ok(CliConfig {
        provider_name,
        provider_api,
        model,
        available_models: provider_config.models,
        provider_config: ProviderApiConfig {
            api_key,
            base_url: provider_config.base_url,
        },
        config_path,
    })
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

fn print_models(config: &CliConfig) {
    if config.available_models.is_empty() {
        println!(
            "no models configured for provider '{}' ({})",
            config.provider_name, config.provider_api.name
        );
        println!("current model: {}", config.model);
        return;
    }

    for model in &config.available_models {
        if model == &config.model {
            println!("* {model}");
        } else {
            println!("  {model}");
        }
    }
}

async fn run_turn(
    context: &mut TauSession,
    user_message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut response = context.send_message(user_message).await?;
    print_token_usage(context.last_token_usage());

    loop {
        match response {
            TauResponse::Message(content) => {
                print_content(&content);
                return Ok(());
            }
            TauResponse::ToolUse(tool_calls) => {
                run_tools(context, &tool_calls);
                response = context.request_response().await?;
                print_token_usage(context.last_token_usage());
            }
            TauResponse::MessageAndToolUse {
                content,
                tool_calls,
            } => {
                print_content(&content);
                run_tools(context, &tool_calls);
                response = context.request_response().await?;
                print_token_usage(context.last_token_usage());
            }
        }
    }
}

fn run_tools(context: &mut TauSession, tool_calls: &[ToolUse]) -> Vec<ToolResult> {
    for call in tool_calls {
        println!("[tool] {}({})", call.name, compact_json(&call.input));
    }

    let results = context.call_tools_parallel_and_record(tool_calls);

    for result in &results {
        match &result.error {
            Some(error) => println!("[tool] {} failed: {error}", result.name),
            None => println!("[tool] {} completed", result.name),
        }
    }

    results
}

fn print_token_usage(usage: Option<&TokenUsage>) {
    let Some(usage) = usage else {
        return;
    };

    match (usage.input_tokens, usage.output_tokens, usage.total_tokens) {
        (Some(input), Some(output), Some(total)) => {
            println!("[tokens] input={input}, output={output}, total={total}");
        }
        (input, output, total) => {
            println!(
                "[tokens] input={}, output={}, total={}",
                format_optional_u64(input),
                format_optional_u64(output),
                format_optional_u64(total)
            );
        }
    }
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn print_content(content: &[ContentPart]) {
    for part in content {
        match part {
            ContentPart::Text { text } => println!("{text}"),
            ContentPart::Json { value } => println!("{}", pretty_json(value)),
            ContentPart::Image { media_type, data } => {
                println!("[image: {media_type}, {data:?}]");
            }
            ContentPart::Binary { media_type, data } => {
                println!("[binary: {media_type}, {data:?}]");
            }
        }
    }
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid json>".to_string())
}

fn pretty_json(value: &serde_json::Value) -> String {
    serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
}
