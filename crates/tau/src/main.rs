use std::{
    fs,
    io::{self, IsTerminal},
    path::PathBuf,
    process::Command as ProcessCommand,
};

use libtau::{
    context::{ContentPart, ResponsePart, ServerToolResult, TauSession, ToolResult, ToolUse},
    providers::TokenUsage,
    tools,
};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod cli;
mod config;
mod provider_config;
mod session;
mod state;

use cli::{Command, parse_cli_from};
use config::{CliConfig, OAuthRefreshRequest};
use session::{SessionPersistence, load_session_from_path, save_session, session_path_for_id};
use state::{SessionRecord, StateDb};

const SYSTEM_MESSAGE: &str = r#"You are Tau, a coding agent running in a terminal.

You can inspect and modify files using tools. When the user asks you to read, write, or edit files, use the available tools."#;

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
    let invocation = parse_cli_from(std::env::args_os()).unwrap_or_else(|error| error.exit());
    let output = OutputStyle::detect();

    match invocation.command {
        Command::Message { contents } => {
            let message = match contents {
                Some(contents) => contents,
                None => edit_message()?,
            };
            if message.trim().is_empty() {
                return Err("message is empty".into());
            }

            let mut cli_config = CliConfig::load()?;
            if let Some(model_ref) = invocation.modifiers.model.as_deref() {
                cli_config.restore_current_model(model_ref.parse()?)?;
            }

            let state = StateDb::open(state_path()?)?;
            let mut context = libtau::context::TauContext::default();
            tools::register_builtin_tools(&mut context)?;

            let (record, mut session, persistence) = load_or_create_current_session(
                &state,
                &context,
                &mut cli_config,
                invocation.modifiers.conversation.as_deref(),
                invocation.modifiers.read_only,
            )?;

            match run_turn(&mut session, &mut cli_config, &message, &output).await {
                Ok(usage) => {
                    save_session(&persistence, cli_config.current_model(), &session)?;
                    state.touch_session(&record.id)?;
                    print_token_usage(usage.as_ref(), &output);
                    Ok(())
                }
                Err(error) => {
                    let _ = save_session(&persistence, cli_config.current_model(), &session);
                    Err(error)
                }
            }
        }
        Command::Conversation { alias } => {
            let mut cli_config = CliConfig::load()?;
            if let Some(model_ref) = invocation.modifiers.model.as_deref() {
                cli_config.restore_current_model(model_ref.parse()?)?;
            }

            let state = StateDb::open(state_path()?)?;
            let context = libtau::context::TauContext::default();
            let readonly = invocation.modifiers.read_only && !invocation.modifiers.writes_allowed;
            let (record, _session, _persistence) = create_new_session(
                &state,
                &context,
                &mut cli_config,
                alias.as_deref(),
                readonly,
            )?;

            println!("conversation: {}", record.alias);
            println!("session: {}", record.id);
            println!("model: {}", record.provider);
            if record.readonly {
                println!("mode: read-only");
            }
            Ok(())
        }
        Command::ProviderConfig { provider } => provider_config::configure_provider(provider).await,
        Command::ProviderList => config::list_providers(invocation.modifiers.json),
        Command::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(format!("command not implemented yet: {other:?}").into()),
    }
}

fn load_or_create_current_session(
    state: &StateDb,
    context: &libtau::context::TauContext,
    cli_config: &mut CliConfig,
    alias: Option<&str>,
    readonly: bool,
) -> Result<(SessionRecord, TauSession, SessionPersistence), Box<dyn std::error::Error>> {
    if let Some(alias) = alias {
        let record = state
            .get_session_by_alias(alias)?
            .ok_or_else(|| format!("conversation '{alias}' does not exist"))?;
        cli_config.restore_current_model(record.provider.parse()?)?;
        let (session, persistence) =
            load_session_from_path(context, cli_config, record.contents_path.clone())?;
        state.set_current_session(&record.id)?;
        return Ok((record, session, persistence));
    }

    if let Some(session_id) = state.current_session_id()? {
        if let Some(record) = state.get_session_by_id(&session_id)? {
            cli_config.restore_current_model(record.provider.parse()?)?;
            let (session, persistence) =
                load_session_from_path(context, cli_config, record.contents_path.clone())?;
            return Ok((record, session, persistence));
        }
        state.clear_current_session()?;
    }

    create_new_session(state, context, cli_config, None, readonly)
}

fn create_new_session(
    state: &StateDb,
    context: &libtau::context::TauContext,
    cli_config: &mut CliConfig,
    alias: Option<&str>,
    readonly: bool,
) -> Result<(SessionRecord, TauSession, SessionPersistence), Box<dyn std::error::Error>> {
    let alias_generated = alias.is_none();
    let alias = match alias {
        Some(alias) => alias.to_string(),
        None => generate_unique_session_alias(state)?,
    };
    let placeholder_id = format!("session-{}", uuid::Uuid::new_v4());
    let contents_path = session_path_for_id(&placeholder_id)?;
    let record = state.create_session(
        &alias,
        &cli_config.current_model().to_string(),
        readonly,
        alias_generated,
        &contents_path,
    )?;
    state.set_current_session(&record.id)?;

    let mut session = cli_config.session_for_current_model(context)?;
    session.set_system_message(SYSTEM_MESSAGE);
    let persistence = SessionPersistence {
        id: record.id.clone(),
        path: record.contents_path.clone(),
    };
    save_session(&persistence, cli_config.current_model(), &session)?;

    Ok((record, session, persistence))
}

fn generate_unique_session_alias(state: &StateDb) -> Result<String, Box<dyn std::error::Error>> {
    for _ in 0..100 {
        let alias = random_alias();
        if state.get_session_by_alias(&alias)?.is_none() {
            return Ok(alias);
        }
    }

    Err("failed to generate a unique conversation alias".into())
}

fn random_alias() -> String {
    const ALPHABET: &[u8] = b"23456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz_";
    let uuid = uuid::Uuid::new_v4();
    uuid.as_bytes()
        .iter()
        .take(8)
        .map(|byte| ALPHABET[*byte as usize % ALPHABET.len()] as char)
        .collect()
}

fn state_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err("cannot open Tau state because HOME is not set".into());
    };
    Ok(PathBuf::from(home).join(".tau/state.db"))
}

fn edit_message() -> Result<String, Box<dyn std::error::Error>> {
    let editor = std::env::var_os("EDITOR").ok_or("EDITOR is not set")?;
    let path = std::env::temp_dir().join(format!("tau-message-{}.md", uuid::Uuid::new_v4()));
    fs::write(&path, "")?;
    let status = ProcessCommand::new(editor).arg(&path).status()?;
    if !status.success() {
        return Err("editor exited unsuccessfully".into());
    }
    let message = fs::read_to_string(&path)?;
    let _ = fs::remove_file(path);
    Ok(message)
}

async fn run_turn(
    context: &mut TauSession,
    cli_config: &mut CliConfig,
    user_message: &str,
    output: &OutputStyle,
) -> Result<Option<TokenUsage>, Box<dyn std::error::Error>> {
    output.println_styled("muted", "[Sending message]");
    context.push_user_content(vec![ContentPart::text(user_message)]);
    output.println_styled("muted", "[Message sent. Waiting for model response]");
    let mut response = request_response_with_reauth(context, cli_config).await?;
    let mut total_usage = context.last_token_usage().cloned();

    loop {
        let mut tool_calls = Vec::new();
        for part in response.parts {
            match part {
                ResponsePart::Content { content } => print_content(&content, output),
                ResponsePart::ToolUse { call } => tool_calls.push(call),
                ResponsePart::ServerToolUse { call } => {
                    output.println_indented_styled("tool", &format_server_tool_use(&call));
                }
                ResponsePart::ServerToolResult { result } => {
                    print_server_tool_result(&result, output)
                }
                ResponsePart::Stop { .. } => {}
            }
        }

        if tool_calls.is_empty() {
            return Ok(total_usage);
        }

        run_tools(context, &tool_calls, output);
        output.println_styled(
            "muted",
            "[Sending tool results. Waiting for model response]",
        );
        response = request_response_with_reauth(context, cli_config).await?;
        add_usage(&mut total_usage, context.last_token_usage());
    }
}

async fn request_response_with_reauth(
    context: &mut TauSession,
    cli_config: &mut CliConfig,
) -> Result<libtau::context::TauResponse, Box<dyn std::error::Error>> {
    match context.request_response().await {
        Ok(response) => Ok(response),
        Err(libtau::api::ProviderError::ReauthenticationRequired {
            access,
            refresh,
            expires,
        }) => {
            cli_config
                .refresh_oauth_provider(
                    &context.provider().name().to_string(),
                    OAuthRefreshRequest {
                        access,
                        refresh,
                        expires,
                    },
                )
                .await?;
            let provider = cli_config.build_provider_for_current_model()?;
            context.refresh_provider(provider);
            context.request_response().await.map_err(Into::into)
        }
        Err(error) => Err(error.into()),
    }
}

fn run_tools(
    context: &mut TauSession,
    tool_calls: &[ToolUse],
    output: &OutputStyle,
) -> Vec<ToolResult> {
    for call in tool_calls {
        output.println_indented_styled(
            "tool",
            &format!("[tool] {}({})", call.name, compact_json(&call.input)),
        );
    }

    let results = context.call_tools_parallel_and_record(tool_calls);

    for result in &results {
        match &result.error {
            Some(error) => output.println_indented_styled(
                "tool",
                &format!("[tool] {} failed: {error}", result.name),
            ),
            None => {
                output.println_indented_styled("tool", &format!("[tool] {} completed", result.name))
            }
        }
    }

    results
}

fn add_usage(total: &mut Option<TokenUsage>, usage: Option<&TokenUsage>) {
    let Some(usage) = usage else {
        return;
    };
    match total {
        Some(total) => {
            total.input_tokens = add_optional(total.input_tokens, usage.input_tokens);
            total.output_tokens = add_optional(total.output_tokens, usage.output_tokens);
            total.total_tokens = add_optional(total.total_tokens, usage.total_tokens);
        }
        None => *total = Some(usage.clone()),
    }
}

fn add_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        _ => None,
    }
}

fn print_token_usage(usage: Option<&TokenUsage>, output: &OutputStyle) {
    let Some(usage) = usage else {
        return;
    };

    let line = match (usage.input_tokens, usage.output_tokens, usage.total_tokens) {
        (Some(input), Some(output_tokens), Some(total)) => {
            format!("[tokens] input={input}, output={output_tokens}, total={total}")
        }
        (input, output_tokens, total) => format!(
            "[tokens] input={}, output={}, total={}",
            format_optional_u64(input),
            format_optional_u64(output_tokens),
            format_optional_u64(total)
        ),
    };
    output.println_styled("muted", &line);
}

fn format_optional_u64(value: Option<u64>) -> String {
    value
        .map(|value| value.to_string())
        .unwrap_or_else(|| "unknown".to_string())
}

fn print_content(content: &ContentPart, output: &OutputStyle) {
    match content {
        ContentPart::Text { text, .. } => output.println_styled("agent", text),
        ContentPart::Thinking { summary, .. } => {
            if !summary.is_empty() {
                for text in summary {
                    output.println_indented_styled("muted", &format!("[thinking]\n{text}"))
                }
            } else {
                output.println_indented_styled("muted", "[redacted thinking]");
            }
        }
        ContentPart::Refusal { text, .. } => {
            output.println_indented_styled("muted", &format!("[refusal]\n{text}"))
        }
        ContentPart::FailedToolCall { text, .. } => {
            output.println_indented_styled("error", &format!("[failed tool call]\n{text}"))
        }
        ContentPart::Image {
            media_type, data, ..
        } => {
            output.println_indented_styled("muted", &format!("[image: {media_type}, {data:?}]"));
        }
        ContentPart::Binary {
            media_type, data, ..
        } => {
            output.println_indented_styled("muted", &format!("[binary: {media_type}, {data:?}]"));
        }
    }
}

fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid json>".to_string())
}

fn print_server_tool_result(result: &ServerToolResult, output: &OutputStyle) {
    for content in &result.content {
        match content {
            ContentPart::Text { text, .. } => output.println_indented_styled("tool", text),
            other => print_content(other, output),
        }
    }
}

fn format_server_tool_use(call: &libtau::context::ServerToolUse) -> String {
    if call.name == "web_search"
        && let Some(query) = call.input.get("query").and_then(serde_json::Value::as_str)
    {
        return format!("[server tool] web_search\nquery: {query}");
    }

    format!("[server tool] {}", call.name)
}

fn indent_display_block(text: &str) -> String {
    const INDENT: &str = "  ";
    text.lines()
        .map(|line| format!("{INDENT}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Debug, Clone, Copy)]
struct OutputStyle {
    color: bool,
}

impl OutputStyle {
    fn detect() -> Self {
        let color = io::stdout().is_terminal()
            && std::env::var_os("NO_COLOR").is_none()
            && std::env::var("TERM")
                .map(|term| term != "dumb")
                .unwrap_or(true);
        Self { color }
    }

    fn println_styled(&self, style: &str, text: &str) {
        if !self.color {
            println!("{text}");
            return;
        }

        let code = match style {
            "muted" => "90",
            "tool" => "36",
            "agent" => "97",
            _ => "0",
        };
        println!("\x1b[{code}m{text}\x1b[0m");
    }

    fn println_indented_styled(&self, style: &str, text: &str) {
        self.println_styled(style, &indent_display_block(text));
    }
}
