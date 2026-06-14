use std::{error::Error, path::PathBuf};

use libtau::context::ConversationItem;

use crate::{
    cli::{CliInvocation, Command},
    config::CliConfig,
    output::{OutputStyle, format_token_usage},
    session::load_session_from_path,
    state::StateDb,
};

mod conversation;
mod message;
mod provider;

pub(crate) async fn dispatch(
    invocation: CliInvocation,
    output: OutputStyle,
) -> Result<(), Box<dyn Error>> {
    let CliInvocation {
        modifiers,
        command,
        definition,
    } = invocation;
    match command {
        Command::Message(command) => message::dispatch(modifiers, command, output).await,
        Command::Conversation(command) => conversation::dispatch(modifiers, command),
        Command::Provider(command) => provider::dispatch(modifiers, command).await,
        Command::Status => status(),
        Command::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(format!(
            "command not implemented yet: {} ({other:?})",
            definition.name
        )
        .into()),
    }
}

fn status() -> Result<(), Box<dyn Error>> {
    let mut cli_config = CliConfig::load()?;
    let state = StateDb::open(state_path()?)?;
    let Some(session_id) = state.current_session_id()? else {
        print_model_status(&cli_config.current_model().to_string());
        println!("conversation: none");
        println!("message offset: 0");
        println!("token usage: none");
        return Ok(());
    };

    let Some(record) = state.get_session_by_id(&session_id)? else {
        state.clear_current_session()?;
        print_model_status(&cli_config.current_model().to_string());
        println!("conversation: none");
        println!("message offset: 0");
        println!("token usage: none");
        return Ok(());
    };

    let context = libtau::context::TauContext::default();
    let (session, _) = load_session_from_path(&context, &mut cli_config, record.contents_path)?;

    print_model_status(&cli_config.current_model().to_string());
    println!("conversation: {} ({})", record.id, record.alias);
    println!(
        "message offset: {}",
        message_offset(session.conversation().items.iter())
    );
    print_total_token_usage(session.total_token_usage());
    Ok(())
}

fn print_model_status(model_ref: &str) {
    let (provider, model) = model_ref.split_once('/').unwrap_or((model_ref, ""));
    println!("provider: {provider}");
    println!("model: {model}");
}

fn message_offset<'a>(items: impl Iterator<Item = &'a ConversationItem>) -> usize {
    items
        .filter(|item| !matches!(item, ConversationItem::System { .. }))
        .count()
}

fn print_total_token_usage(usage: Option<&libtau::providers::TokenUsage>) {
    let Some(usage) = usage else {
        println!("token usage: none");
        return;
    };

    let usage = format_token_usage(usage);
    if usage.is_empty() {
        println!("token usage: none");
    } else {
        println!("token usage: {usage}");
    }
}

pub(super) fn state_path() -> Result<PathBuf, Box<dyn Error>> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err("cannot open Tau state because HOME is not set".into());
    };
    Ok(PathBuf::from(home).join(".tau/state.db"))
}
