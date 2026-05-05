use std::{fs, path::PathBuf};

use libtau::context::{Conversation, TauContext, TauSession};
use serde::{Deserialize, Serialize};

use crate::config::{CliConfig, ModelSelection};

const SESSIONS_DIR: &str = ".tau/sessions";

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PersistedSession {
    id: String,
    current_model: String,
    conversation: Conversation,
}

#[derive(Debug, Clone)]
pub(crate) struct SessionPersistence {
    pub(crate) id: String,
    path: PathBuf,
}

pub(crate) fn load_or_create_session(
    context: &TauContext,
    config: &mut CliConfig,
    system_message: &str,
    session_id: Option<String>,
) -> Result<(TauSession, SessionPersistence), Box<dyn std::error::Error>> {
    if let Some(session_id) = session_id {
        let path = session_path(&session_id)?;
        let persisted = read_session(&path)?;
        let current_model: ModelSelection = persisted.current_model.parse()?;
        config.restore_current_model(current_model)?;

        let mut session = config.session_for_current_model(context)?;
        *session.conversation_mut() = persisted.conversation;
        session.set_model(config.current_model().model().to_string());

        return Ok((
            session,
            SessionPersistence {
                id: persisted.id,
                path,
            },
        ));
    }

    let mut session = config.session_for_current_model(context)?;
    session.set_system_message(system_message);
    let id = format!("session-{}", uuid::Uuid::new_v4());
    let path = session_path(&id)?;

    Ok((session, SessionPersistence { id, path }))
}

pub(crate) fn save_session(
    persistence: &SessionPersistence,
    current_model: &ModelSelection,
    session: &TauSession,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = persistence.path.parent() {
        fs::create_dir_all(parent)?;
    }

    let persisted = PersistedSession {
        id: persistence.id.clone(),
        current_model: current_model.to_string(),
        conversation: session.conversation().clone(),
    };
    fs::write(
        &persistence.path,
        serde_json::to_string_pretty(&persisted)? + "\n",
    )?;
    Ok(())
}

fn read_session(path: &PathBuf) -> Result<PersistedSession, Box<dyn std::error::Error>> {
    let contents = fs::read_to_string(path)
        .map_err(|error| format!("failed to read session {}: {error}", path.display()))?;
    serde_json::from_str(&contents)
        .map_err(|error| format!("failed to parse session {}: {error}", path.display()).into())
}

fn session_path(session_id: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err("cannot persist session because HOME is not set".into());
    };
    Ok(PathBuf::from(home)
        .join(SESSIONS_DIR)
        .join(format!("{session_id}.json")))
}
