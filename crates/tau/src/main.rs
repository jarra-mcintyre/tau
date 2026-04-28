use std::{
    fs,
    io::{self, Write},
    path::PathBuf,
};

use libtau::{
    context::{ContentPart, TauContext, TauResponse, ToolResult, ToolUse},
    providers::{anthropic::AnthropicProvider, openai::OpenAiProvider},
    tools,
};
use serde::Deserialize;

const CONFIG_PATH: &str = ".tau/providers.json";
const DEFAULT_PROVIDER: &str = "openai";
const DEFAULT_API: &str = "openai_responses";
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
    provider_api: String,
    model: String,
    available_models: Vec<String>,
    api_key: String,
    base_url: Option<String>,
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
    match cli_config.provider_api.as_str() {
        "openai_responses" => match &cli_config.base_url {
            Some(base_url) => context.set_provider(OpenAiProvider::with_base_url(
                cli_config.api_key.clone(),
                base_url,
            )),
            None => context.set_provider(OpenAiProvider::new(cli_config.api_key.clone())),
        },
        "anthropic_messages" => match &cli_config.base_url {
            Some(base_url) => context.set_provider(AnthropicProvider::with_base_url(
                cli_config.api_key.clone(),
                base_url,
            )),
            None => context.set_provider(AnthropicProvider::new(cli_config.api_key.clone())),
        },
        api => return Err(format!("unsupported provider API: {api}").into()),
    }
    context.set_model(cli_config.model.clone());
    context.set_system_message(SYSTEM_MESSAGE);
    tools::register_builtin_tools(&mut context)?;

    println!("Tau interactive shell");
    println!("provider: {}", cli_config.provider_name);
    println!("api: {}", cli_config.provider_api);
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

        if let Err(error) = run_turn(&mut context, line).await {
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

    let provider_api = provider_config
        .api
        .clone()
        .or_else(|| {
            if provider_name == DEFAULT_PROVIDER {
                Some(DEFAULT_API.to_string())
            } else {
                None
            }
        })
        .ok_or_else(|| {
            format!(
                "provider '{provider_name}' must specify an api in {}",
                config_path
                    .as_ref()
                    .map(|path| path.display().to_string())
                    .unwrap_or_else(|| CONFIG_PATH.to_string())
            )
        })?;

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

    let api_key = match provider_api.as_str() {
        "openai_responses" => std::env::var("OPENAI_API_KEY")
            .ok()
            .or(provider_config.api_key.clone())
            .ok_or_else(|| {
                format!(
                    "missing OpenAI API key; set OPENAI_API_KEY or providers.{provider_name}.api_key in ~/.tau/providers.json"
                )
            })?,
        "anthropic_messages" => std::env::var("ANTHROPIC_API_KEY")
            .ok()
            .or(provider_config.api_key.clone())
            .ok_or_else(|| {
                format!(
                    "missing Anthropic API key; set ANTHROPIC_API_KEY or providers.{provider_name}.api_key in ~/.tau/providers.json"
                )
            })?,
        api => return Err(format!("unsupported provider API: {api}").into()),
    };

    Ok(CliConfig {
        provider_name,
        provider_api,
        model,
        available_models: provider_config.models,
        api_key,
        base_url: provider_config.base_url,
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

fn providers_config_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(CONFIG_PATH))
}

fn print_models(config: &CliConfig) {
    if config.available_models.is_empty() {
        println!(
            "no models configured for provider '{}' ({})",
            config.provider_name, config.provider_api
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
    context: &mut TauContext,
    user_message: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut response = context.send_message(user_message).await?;

    loop {
        match response {
            TauResponse::Message(content) => {
                print_content(&content);
                return Ok(());
            }
            TauResponse::ToolUse(tool_calls) => {
                run_tools(context, &tool_calls);
                response = context.request_response().await?;
            }
            TauResponse::MessageAndToolUse {
                content,
                tool_calls,
            } => {
                print_content(&content);
                run_tools(context, &tool_calls);
                response = context.request_response().await?;
            }
        }
    }
}

fn run_tools(context: &mut TauContext, tool_calls: &[ToolUse]) -> Vec<ToolResult> {
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
