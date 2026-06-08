use std::error::Error;

use libtau::context::{TauContext, TauSession};

use crate::{
    cli::{ConversationCommand, Modifiers},
    config::CliConfig,
    session::{SessionPersistence, load_session_from_path, save_session, session_path_for_id},
    state::{SessionRecord, StateDb},
};

use super::state_path;

const SYSTEM_MESSAGE: &str = r#"You are Tau, a coding agent running in a terminal.

You can inspect and modify files using tools. When the user asks you to read, write, or edit files, use the available tools."#;

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

pub(super) fn load_or_create_current_session(
    state: &StateDb,
    context: &TauContext,
    cli_config: &mut CliConfig,
    alias: Option<&str>,
    readonly: bool,
) -> Result<(SessionRecord, TauSession, SessionPersistence), Box<dyn Error>> {
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
    context: &TauContext,
    cli_config: &mut CliConfig,
    alias: Option<&str>,
    readonly: bool,
) -> Result<(SessionRecord, TauSession, SessionPersistence), Box<dyn Error>> {
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

fn generate_unique_session_alias(state: &StateDb) -> Result<String, Box<dyn Error>> {
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
