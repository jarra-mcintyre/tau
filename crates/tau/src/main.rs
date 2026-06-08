use std::path::PathBuf;

use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{EnvFilter, layer::SubscriberExt, util::SubscriberInitExt};

mod agent;
mod cli;
mod commands;
mod config;
mod editor;
mod output;
mod provider_config;
mod session;
mod session_manager;
mod state;

use cli::parse_cli_from;
use output::OutputStyle;

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

    commands::dispatch(invocation, output).await
}
