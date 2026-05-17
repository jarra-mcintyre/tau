#![allow(dead_code)]

use std::{
    fs,
    path::{Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};

use rusqlite::{Connection, OptionalExtension, params};

const SCHEMA_VERSION: i64 = 1;

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
    pub(crate) contents_path: PathBuf,
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
            contents_path: contents_path.as_ref().to_path_buf(),
        };
        self.insert_session(&record)?;
        Ok(record)
    }

    pub(crate) fn insert_session(&self, record: &SessionRecord) -> StateResult<()> {
        self.connection.execute(
            "INSERT INTO sessions (id, alias, created_at, updated_at, provider, readonly, contents_path)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                record.id,
                record.alias,
                record.created_at,
                record.updated_at,
                record.provider,
                if record.readonly { 1 } else { 0 },
                record.contents_path.to_string_lossy(),
            ],
        )?;
        Ok(())
    }

    pub(crate) fn get_session_by_alias(&self, alias: &str) -> StateResult<Option<SessionRecord>> {
        self.connection
            .query_row(
                "SELECT id, alias, created_at, updated_at, provider, readonly, contents_path
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
                "SELECT id, alias, created_at, updated_at, provider, readonly, contents_path
                 FROM sessions WHERE id = ?1",
                params![id],
                session_record_from_row,
            )
            .optional()
            .map_err(Into::into)
    }

    pub(crate) fn list_sessions(&self) -> StateResult<Vec<SessionRecord>> {
        let mut statement = self.connection.prepare(
            "SELECT id, alias, created_at, updated_at, provider, readonly, contents_path
             FROM sessions ORDER BY updated_at DESC, created_at DESC",
        )?;
        let rows = statement.query_map([], session_record_from_row)?;
        rows.collect::<Result<Vec<_>, _>>().map_err(Into::into)
    }

    pub(crate) fn rename_session_alias(&self, id: &str, alias: &str) -> StateResult<bool> {
        let updated_at = unix_timestamp_millis()?;
        let changed = self.connection.execute(
            "UPDATE sessions SET alias = ?1, updated_at = ?2 WHERE id = ?3",
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
                 contents_path TEXT NOT NULL
             ) WITHOUT ROWID;
             CREATE INDEX IF NOT EXISTS idx_sessions_updated_at ON sessions(updated_at DESC);
             PRAGMA user_version = 1;",
        )?;

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
        contents_path: PathBuf::from(row.get::<_, String>(6)?),
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
                "/home/me/.tau/sessions/session-1.json",
            )
            .unwrap();

        assert_eq!(record.alias, "Feature work");
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
        db.create_session("same", "openai/gpt-4.1-mini", false, "one.json")
            .unwrap();

        assert!(
            db.create_session("same", "openai/gpt-4.1-mini", false, "two.json")
                .is_err()
        );
    }

    #[test]
    fn lists_renames_touches_and_deletes_sessions() {
        let db = StateDb::open(temp_db_path("updates")).unwrap();
        let first = db
            .create_session("first", "openai/gpt-4.1-mini", false, "first.json")
            .unwrap();
        let second = db
            .create_session("second", "anthropic/claude-sonnet-4", true, "second.json")
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
}
