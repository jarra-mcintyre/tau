#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};
use serde_json::Value;

const SCHEMA_VERSION: i64 = 3;

pub(crate) type StateResult<T> = Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
pub(crate) struct StateDb {
    connection: Connection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SessionRecord {
    pub(crate) id: String,
    pub(crate) alias: String,
    /// Unix timestamp in milliseconds.
    pub(crate) created_at: i64,
    /// Unix timestamp in milliseconds.
    pub(crate) updated_at: i64,
    /// Provider/model reference used for the session, e.g. `openai/gpt-4.1-mini`.
    pub(crate) provider: String,
    pub(crate) readonly: bool,
    pub(crate) alias_generated: bool,
    pub(crate) contents_path: PathBuf,
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StagedMessageRecord {
    pub(crate) session_id: String,
    /// Unix timestamp in milliseconds.
    pub(crate) created_at: i64,
    /// Unix timestamp in milliseconds.
    pub(crate) updated_at: i64,
    /// JSON array/object describing message parts (text, image, path reference, etc.).
    pub(crate) parts: Value,
}

impl StateDb {
    pub(crate) fn open(path: impl AsRef<Path>) -> StateResult<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }

        let connection = Connection::open(path)?;
        let db = Self { connection };
        db.migrate()?;
        Ok(db)
    }

    pub(crate) fn create_session(
        &self,
        alias: &str,
        provider: &str,
        readonly: bool,
        alias_generated: bool,
        contents_path: impl AsRef<Path>,
    ) -> StateResult<SessionRecord> {
        let now = unix_timestamp_millis()?;
        let record = SessionRecord {
            id: format!("session-{}", uuid::Uuid::new_v4()),
            alias: alias.to_string(),
            created_at: now,
            updated_at: now,
            provider: provider.to_string(),
            readonly,
            alias_generated,
            contents_path: contents_path.as_ref().to_path_buf(),
        };
        self.insert_session(&record)?;
        Ok(record)
    }

    pub(crate) fn insert_session(&self, record: &SessionRecord) -> StateResult<()> {
        self.connection.execute(
            "INSERT INTO sessions (id, alias, created_at, updated_at, provider, readonly, alias_generated, contents_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            params![
                record.id,
                record.alias,
                record.created_at,
                record.updated_at,
                record.provider,
                if record.readonly { 1 } else { 0 },
                if record.alias_generated { 1 } else { 0 },
                record.contents_path.to_string_lossy(),
            ],
        )?;
        Ok(())
    }

    pub(crate) fn get_session_by_alias(&self, alias: &str) -> StateResult<Option<SessionRecord>> {
        self.connection
            .query_row(
                "SELECT id, alias, created_at, updated_at, provider, readonly, alias_generated, contents_path
                 FROM sessions WHERE alias = ?1",
                params![alias],
                session_record_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn get_session_by_id(&self, id: &str) -> StateResult<Option<SessionRecord>> {
        self.connection
            .query_row(
                "SELECT id, alias, created_at, updated_at, provider, readonly, alias_generated, contents_path
                 FROM sessions WHERE id = ?1",
                params![id],
                session_record_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn list_sessions(&self) -> StateResult<Vec<SessionRecord>> {
        let mut statement = self.connection.prepare(
            "SELECT id, alias, created_at, updated_at, provider, readonly, alias_generated, contents_path
             FROM sessions ORDER BY updated_at DESC, created_at DESC",
        )?;
        let rows = statement.query_map([], session_record_from_row)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub(crate) fn rename_session_alias(&self, id: &str, alias: &str) -> StateResult<bool> {
        let updated_at = unix_timestamp_millis()?;
        let changed = self.connection.execute(
            "UPDATE sessions SET alias = ?1, alias_generated = 0, updated_at = ?2 WHERE id = ?3",
            params![alias, updated_at, id],
        )?;
        Ok(changed > 0)
    }

    pub(crate) fn touch_session(&self, id: &str) -> StateResult<bool> {
        let changed = self.connection.execute(
            "UPDATE sessions SET updated_at = ?1 WHERE id = ?2",
            params![unix_timestamp_millis()?, id],
        )?;
        Ok(changed > 0)
    }

    pub(crate) fn delete_session_by_alias(&self, alias: &str) -> StateResult<bool> {
        let changed = self
            .connection
            .execute("DELETE FROM sessions WHERE alias = ?1", params![alias])?;
        Ok(changed > 0)
    }

    pub(crate) fn upsert_staged_message(&self, session_id: &str, parts: &Value) -> StateResult<()> {
        let now = unix_timestamp_millis()?;
        let parts_json = serde_json::to_string(parts)?;
        self.connection.execute(
            "INSERT INTO staged_messages (session_id, created_at, updated_at, parts_json)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(session_id) DO UPDATE SET
                 updated_at = excluded.updated_at,
                 parts_json = excluded.parts_json",
            params![session_id, now, now, parts_json],
        )?;
        Ok(())
    }

    pub(crate) fn get_staged_message(
        &self,
        session_id: &str,
    ) -> StateResult<Option<StagedMessageRecord>> {
        self.connection
            .query_row(
                "SELECT session_id, created_at, updated_at, parts_json
                 FROM staged_messages WHERE session_id = ?1",
                params![session_id],
                staged_message_record_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn delete_staged_message(&self, session_id: &str) -> StateResult<bool> {
        let changed = self.connection.execute(
            "DELETE FROM staged_messages WHERE session_id = ?1",
            params![session_id],
        )?;
        Ok(changed > 0)
    }

    pub(crate) fn set_current_session(&self, session_id: &str) -> StateResult<()> {
        self.connection.execute(
            "INSERT INTO current_session (singleton, session_id)
             VALUES (1, ?1)
             ON CONFLICT(singleton) DO UPDATE SET session_id = excluded.session_id",
            params![session_id],
        )?;
        Ok(())
    }

    pub(crate) fn current_session_id(&self) -> StateResult<Option<String>> {
        self.connection
            .query_row(
                "SELECT session_id FROM current_session WHERE singleton = 1",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn clear_current_session(&self) -> StateResult<()> {
        self.connection
            .execute("DELETE FROM current_session WHERE singleton = 1", [])?;
        Ok(())
    }

    fn migrate(&self) -> StateResult<()> {
        self.connection.execute_batch(
            "PRAGMA journal_mode = wal;
            PRAGMA synchronous = normal;
            PRAGMA foreign_keys = ON;",
        )?;

        let version: i64 = self
            .connection
            .query_row("PRAGMA user_version", [], |row| row.get(0))?;
        if version != 0 && version != SCHEMA_VERSION {
            return Err(format!(
                "unsupported Tau state database schema version {version}; expected {SCHEMA_VERSION}"
            )
            .into());
        }

        self.connection.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY NOT NULL,
                alias TEXT NOT NULL UNIQUE,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                provider TEXT NOT NULL,
                readonly INTEGER NOT NULL,
                alias_generated INTEGER NOT NULL DEFAULT 0,
                contents_path TEXT NOT NULL
            ) WITHOUT ROWID;
            CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
            CREATE TABLE IF NOT EXISTS staged_messages (
                session_id TEXT PRIMARY KEY NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                parts_json TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            ) WITHOUT ROWID;
            CREATE TABLE IF NOT EXISTS current_session (
                singleton INTEGER PRIMARY KEY NOT NULL CHECK (singleton = 1),
                session_id TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );",
        )?;

        self.connection
            .execute_batch(&format!("PRAGMA user_version = {SCHEMA_VERSION};"))?;

        Ok(())
    }
}

fn session_record_from_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<SessionRecord> {
    Ok(SessionRecord {
        id: row.get(0)?,
        alias: row.get(1)?,
        created_at: row.get(2)?,
        updated_at: row.get(3)?,
        provider: row.get(4)?,
        readonly: row.get::<_, i64>(5)? != 0,
        alias_generated: row.get::<_, i64>(6)? != 0,
        contents_path: PathBuf::from(row.get::<_, String>(7)?),
    })
}

fn staged_message_record_from_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<StagedMessageRecord> {
    let parts_json: String = row.get(3)?;
    let parts = serde_json::from_str(&parts_json).map_err(|error| {
        rusqlite::Error::FromSqlConversionFailure(3, rusqlite::types::Type::Text, Box::new(error))
    })?;

    Ok(StagedMessageRecord {
        session_id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        parts,
    })
}

fn unix_timestamp_millis() -> StateResult<i64> {
    Ok(SystemTime::now()
        .duration_since(UNIX_EPOCH)?
        .as_millis()
        .try_into()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn temp_db_path(test_name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("tau-state-{test_name}-{}.db", uuid::Uuid::new_v4()))
    }

    #[test]
    fn creates_and_loads_session_records() {
        let db = StateDb::open(temp_db_path("creates-and-loads")).unwrap();
        let record = db
            .create_session(
                "Feature work",
                "openai/gpt-4.1-mini",
                true,
                false,
                "/home/me/.tau/sessions/session-1.json",
            )
            .unwrap();

        assert_eq!(record.alias, "Feature work");
        assert!(!record.alias_generated);
        assert!(record.id.starts_with("session-"));

        assert_eq!(
            db.get_session_by_alias("Feature work").unwrap(),
            Some(record.clone())
        );
        assert_eq!(db.get_session_by_id(&record.id).unwrap(), Some(record));
    }

    #[test]
    fn enforces_unique_aliases() {
        let db = StateDb::open(temp_db_path("unique-aliases")).unwrap();
        db.create_session("same", "openai/gpt-4.1-mini", false, false, "one.json")
            .unwrap();

        assert!(
            db.create_session("same", "openai/gpt-4.1-mini", false, false, "two.json")
                .is_err()
        );
    }

    #[test]
    fn lists_renames_touches_and_deletes_sessions() {
        let db = StateDb::open(temp_db_path("updates")).unwrap();
        let first = db
            .create_session("first", "openai/gpt-4.1-mini", false, false, "first.json")
            .unwrap();
        let second = db
            .create_session(
                "second",
                "anthropic/claude-sonnet-4",
                true,
                false,
                "second.json",
            )
            .unwrap();

        assert_eq!(db.list_sessions().unwrap().len(), 2);
        assert!(db.rename_session_alias(&first.id, "renamed").unwrap());
        assert!(db.get_session_by_alias("first").unwrap().is_none());
        assert_eq!(
            db.get_session_by_alias("renamed").unwrap().unwrap().id,
            first.id
        );
        assert!(db.touch_session(&second.id).unwrap());
        assert!(db.delete_session_by_alias("renamed").unwrap());
        assert!(!db.delete_session_by_alias("missing").unwrap());
        assert_eq!(db.list_sessions().unwrap().len(), 1);
    }

    #[test]
    fn upserts_one_staged_message_per_session() {
        let db = StateDb::open(temp_db_path("staged-message")).unwrap();
        let session = db
            .create_session("work", "openai/gpt-4.1-mini", false, false, "work.json")
            .unwrap();

        let first_parts = json!([
            { "type": "text", "text": "hello" },
            { "type": "path_reference", "path": "src/main.rs" }
        ]);
        let first = db.upsert_staged_message(&session.id, &first_parts).unwrap();

        let second_parts =
            json!([{ "type": "image", "media_type": "image/png", "path": "shot.png" }]);
        let second = db
            .upsert_staged_message(&session.id, &second_parts)
            .unwrap();

        assert_eq!(db.list_staged_message_count(), 1);
        assert!(db.delete_staged_message(&session.id).unwrap());
        assert!(db.get_staged_message(&session.id).unwrap().is_none());
    }

    #[test]
    fn deleting_session_cascades_to_staged_message_and_current_session() {
        let db = StateDb::open(temp_db_path("cascade")).unwrap();
        let session = db
            .create_session("work", "openai/gpt-4.1-mini", false, false, "work.json")
            .unwrap();
        db.upsert_staged_message(&session.id, &json!([{ "type": "text", "text": "hello" }]))
            .unwrap();
        db.set_current_session(&session.id).unwrap();

        assert!(db.delete_session_by_alias("work").unwrap());

        assert!(db.get_staged_message(&session.id).unwrap().is_none());
        assert_eq!(db.current_session_id().unwrap(), None);
    }

    #[test]
    fn tracks_current_session() {
        let db = StateDb::open(temp_db_path("current-session")).unwrap();
        let first = db
            .create_session("first", "openai/gpt-4.1-mini", false, false, "first.json")
            .unwrap();
        let second = db
            .create_session("second", "openai/gpt-4.1-mini", false, false, "second.json")
            .unwrap();

        assert_eq!(db.current_session_id().unwrap(), None);
        db.set_current_session(&first.id).unwrap();
        assert_eq!(db.current_session_id().unwrap(), Some(first.id));
        db.set_current_session(&second.id).unwrap();
        assert_eq!(db.current_session_id().unwrap(), Some(second.id));
        db.clear_current_session().unwrap();
        assert_eq!(db.current_session_id().unwrap(), None);
    }

    trait TestStateDbExt {
        fn list_staged_message_count(&self) -> usize;
    }

    impl TestStateDbExt for StateDb {
        fn list_staged_message_count(&self) -> usize {
            self.connection
                .query_row("SELECT COUNT(*) FROM staged_messages", [], |row| {
                    row.get::<_, i64>(0)
                })
                .unwrap() as usize
        }
    }
}
