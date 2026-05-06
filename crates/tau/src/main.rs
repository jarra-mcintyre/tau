use std::{
    io::{self, Write},
    path::PathBuf,
};

use clap::Parser;
use libtau::{
    context::{ContentPart, TauResponse, TauSession, ToolResult, ToolUse},
    providers::TokenUsage,
    tools,
};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod config;
mod session;

use config::CliConfig;
use session::{load_or_create_session, save_session};

const SYSTEM_MESSAGE: &str = r#"You are Tau, a coding agent running in a terminal.

You can inspect and modify files using tools. When the user asks you to read, write, or edit files, use the available tools."#;

#[derive(Debug, Parser)]
#[command(version, about = "Tau interactive coding agent")]
struct Args {
    /// Resume a previous session by id.
    #[arg(long, value_name = "SESSION_ID")]
    resume: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _log_guard = init_file_logging()?;

    tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?
        .block_on(run())
}

fn init_file_logging() -> Result<WorkerGuard, Box<dyn std::error::Error>> {
    let default_filter = if cfg!(debug_assertions) {
        "debug"
    } else {
        "warn"
    };
    let filter = EnvFilter::try_from_env("TAU_LOG_LEVEL")
        .or_else(|_| EnvFilter::try_from_env("RUST_LOG"))
        .unwrap_or_else(|_| EnvFilter::new(default_filter));
    let log_dir = std::env::var_os("TAU_LOG_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("logs"));
    std::fs::create_dir_all(&log_dir)?;

    let file_appender = tracing_appender::rolling::daily(log_dir, "tau.log");
    let (non_blocking, guard) = tracing_appender::non_blocking(file_appender);

    tracing_subscriber::registry()
        .with(filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_ansi(false),
        )
        .try_init()?;

    Ok(guard)
}

async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let mut cli_config = CliConfig::load()?;
    let mut context = libtau::context::TauContext::new();
    tools::register_builtin_tools(&mut context)?;

    let (mut session, persistence) =
        load_or_create_session(&context, &mut cli_config, SYSTEM_MESSAGE, args.resume)?;
    save_session(&persistence, cli_config.current_model(), &session)?;

    println!("Tau interactive shell");
    println!("session: {}", persistence.id);
    cli_config.print_current_model();
    if let Some(path) = cli_config.config_path() {
        println!("config: {}", path.display());
    } else {
        println!("config: not found, using environment/defaults");
    }
    println!("type /models to list configured models");
    println!("type /model provider/model to switch models");
    println!("resume later with: tau --resume {}", persistence.id);
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
            cli_config.print_models();
            continue;
        }
        if let Some(model_ref) = line.strip_prefix("/model ") {
            match cli_config.switch_model(&mut session, model_ref.trim()) {
                Ok(()) => {
                    save_session(&persistence, cli_config.current_model(), &session)?;
                    cli_config.print_current_model();
                }
                Err(error) => eprintln!("error: {error}"),
            }
            continue;
        }

        match run_turn(&mut session, line).await {
            Ok(()) => save_session(&persistence, cli_config.current_model(), &session)?,
            Err(error) => {
                let _ = save_session(&persistence, cli_config.current_model(), &session);
                eprintln!("error: {error}");
            }
        }
    }

    Ok(())
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
            ContentPart::Text { text, .. } => println!("{text}"),
            ContentPart::Json { value, .. } => println!("{}", pretty_json(value)),
            ContentPart::Thinking { text, .. } => println!("[thinking]\n{text}"),
            ContentPart::Image {
                media_type, data, ..
            } => {
                println!("[image: {media_type}, {data:?}]");
            }
            ContentPart::Binary {
                media_type, data, ..
            } => {
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
