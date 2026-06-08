use std::error::Error;

use libtau::context::TauContext;

use crate::{
    cli::{ConversationCommand, Modifiers},
    config::CliConfig,
    session_manager::SessionManager,
    state::StateDb,
};

use super::state_path;

pub(super) fn dispatch(
    modifiers: Modifiers,
    command: ConversationCommand,
) -> Result<(), Box<dyn Error>> {
    match command {
        ConversationCommand::Create { alias } => create(modifiers, alias),
        other => Err(format!("command not implemented yet: {other:?}").into()),
    }
}

fn create(modifiers: Modifiers, alias: Option<String>) -> Result<(), Box<dyn Error>> {
    let mut cli_config = CliConfig::load()?;
    if let Some(model_ref) = modifiers.model.as_deref() {
        cli_config.restore_current_model(model_ref.parse()?)?;
    }

    let state = StateDb::open(state_path()?)?;
    let context = TauContext::default();
    let readonly = modifiers.read_only && !modifiers.writes_allowed;
    let mut session_manager = SessionManager::builder()
        .state(&state)
        .context(&context)
        .config(&mut cli_config)
        .build();
    let loaded_session = session_manager.create(alias.as_deref(), readonly)?;
    let record = loaded_session.record;

    println!("conversation: {}", record.alias);
    println!("session: {}", record.id);
    println!("model: {}", record.provider);
    if record.readonly {
        println!("mode: read-only");
    }
    Ok(())
}
