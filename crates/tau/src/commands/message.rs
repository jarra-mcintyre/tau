use std::error::Error;

use libtau::tools;

use crate::{
    agent::run_turn,
    cli::{MessageCommand, Modifiers},
    config::CliConfig,
    editor::edit_message,
    output::{OutputStyle, print_token_usage},
    state::StateDb,
};

use super::{conversation::load_or_create_current_session, state_path};

pub(super) async fn dispatch(
    modifiers: Modifiers,
    command: MessageCommand,
    output: OutputStyle,
) -> Result<(), Box<dyn Error>> {
    match command {
        MessageCommand::Message { contents } => send(modifiers, contents, output).await,
        other => Err(format!("command not implemented yet: {other:?}").into()),
    }
}

async fn send(
    modifiers: Modifiers,
    contents: Option<String>,
    output: OutputStyle,
) -> Result<(), Box<dyn Error>> {
    let message = match contents {
        Some(contents) => contents,
        None => edit_message()?,
    };
    if message.trim().is_empty() {
        return Err("message is empty".into());
    }

    let mut cli_config = CliConfig::load()?;
    if let Some(model_ref) = modifiers.model.as_deref() {
        cli_config.restore_current_model(model_ref.parse()?)?;
    }

    let state = StateDb::open(state_path()?)?;
    let mut context = libtau::context::TauContext::default();
    tools::register_builtin_tools(&mut context)?;

    let (record, mut session, persistence) = load_or_create_current_session(
        &state,
        &context,
        &mut cli_config,
        modifiers.conversation.as_deref(),
        modifiers.read_only,
    )?;

    match run_turn(&mut session, &mut cli_config, &message, &output).await {
        Ok(usage) => {
            crate::session::save_session(&persistence, cli_config.current_model(), &session)?;
            state.touch_session(&record.id)?;
            print_token_usage(usage.as_ref(), &output);
            Ok(())
        }
        Err(error) => {
            let _ =
                crate::session::save_session(&persistence, cli_config.current_model(), &session);
            Err(error)
        }
    }
}
