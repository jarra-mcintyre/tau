use std::error::Error;

use bon::Builder;
use libtau::context::{TauContext, TauSession};

use crate::{
    config::CliConfig,
    session::{SessionPersistence, load_session_from_path, save_session, session_path_for_id},
    state::{SessionRecord, StateDb},
};

const SYSTEM_MESSAGE: &str = r#"You are Tau, a coding agent running in a terminal.

You can inspect and modify files using tools. When the user asks you to read, write, or edit files, use the available tools."#;

#[derive(Builder)]
pub(crate) struct LoadedSession {
    pub(crate) record: SessionRecord,
    pub(crate) session: TauSession,
    pub(crate) persistence: SessionPersistence,
}

#[derive(Builder)]
pub(crate) struct SessionManager<'a> {
    state: &'a StateDb,
    context: &'a TauContext,
    config: &'a mut CliConfig,
}

impl SessionManager<'_> {
    pub(crate) fn load_or_create_current(
        &mut self,
        alias: Option<&str>,
        readonly: bool,
    ) -> Result<LoadedSession, Box<dyn Error>> {
        if let Some(alias) = alias {
            return self.load_named(alias);
        }

        if let Some(session_id) = self.state.current_session_id()? {
            if let Some(record) = self.state.get_session_by_id(&session_id)? {
                return self.load(record);
            }
            self.state.clear_current_session()?;
        }

        self.create(alias, readonly)
    }

    pub(crate) fn create(
        &mut self,
        alias: Option<&str>,
        readonly: bool,
    ) -> Result<LoadedSession, Box<dyn Error>> {
        let alias_generated = alias.is_none();
        let alias = match alias {
            Some(alias) => alias.to_string(),
            None => generate_unique_session_alias(self.state)?,
        };
        let session_id = format!("session-{}", uuid::Uuid::new_v4());
        let contents_path = session_path_for_id(&session_id)?;
        let record = self.state.create_session_with_id(
            &session_id,
            &alias,
            &self.config.current_model().to_string(),
            readonly,
            alias_generated,
            &contents_path,
        )?;
        self.state.set_current_session(&record.id)?;

        let mut session = self.config.session_for_current_model(self.context)?;
        session.set_system_message(SYSTEM_MESSAGE);
        let persistence = SessionPersistence {
            id: record.id.clone(),
            path: record.contents_path.clone(),
        };
        save_session(&persistence, self.config.current_model(), &session)?;

        Ok(LoadedSession::builder()
            .record(record)
            .session(session)
            .persistence(persistence)
            .build())
    }

    fn load_named(&mut self, alias: &str) -> Result<LoadedSession, Box<dyn Error>> {
        let record = self
            .state
            .get_session_by_alias(alias)?
            .ok_or_else(|| format!("conversation '{alias}' does not exist"))?;
        self.state.set_current_session(&record.id)?;
        self.load(record)
    }

    fn load(&mut self, record: SessionRecord) -> Result<LoadedSession, Box<dyn Error>> {
        self.config
            .restore_current_model(record.provider.parse()?)?;
        let (session, persistence) =
            load_session_from_path(self.context, self.config, record.contents_path.clone())?;
        Ok(LoadedSession::builder()
            .record(record)
            .session(session)
            .persistence(persistence)
            .build())
    }
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
