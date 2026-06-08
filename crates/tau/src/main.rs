use std::path::PathBuf;

use libtau::{context::TauSession, tools};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod agent;
mod cli;
mod config;
mod editor;
mod output;
mod provider_config;
mod session;
mod state;

use agent::run_turn;
use cli::{Command, parse_cli_from};
use config::CliConfig;
use editor::edit_message;
use output::{OutputStyle, print_token_usage};
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

fn state_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err("cannot open Tau state because HOME is not set".into());
    };
    Ok(PathBuf::from(home).join(".tau/state.db"))
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
