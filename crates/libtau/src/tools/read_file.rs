use std::{fs, io, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::{TauContext, ToolCallError, ToolDefinition, ToolRegistrationError};

pub const NAME: &str = "read_file";
pub const DESCRIPTION: &str = "Read a UTF-8 text file from disk.";

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileInput {
    pub path: PathBuf,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadFileStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileOutput {
    pub status: ReadFileStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ReadFileError>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileError {
    pub kind: ReadFileErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadFileErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(definition()?)
}

pub fn definition() -> Result<ToolDefinition, ToolRegistrationError> {
    ToolDefinition::new::<ReadFileInput>(NAME, DESCRIPTION, callback)
}

fn callback(input: Value) -> Result<Value, ToolCallError> {
    let input: ReadFileInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let output = read_file(input);
    serde_json::to_value(output)
        .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))
}

pub fn read_file(input: ReadFileInput) -> ReadFileOutput {
    match fs::read_to_string(&input.path) {
        Ok(contents) => ReadFileOutput {
            status: ReadFileStatus::Success,
            contents: Some(contents),
            error: None,
        },
        Err(error) => ReadFileOutput {
            status: ReadFileStatus::Error,
            contents: None,
            error: Some(ReadFileError::from_io_error(error)),
        },
    }
}

impl ReadFileError {
    fn from_io_error(error: io::Error) -> Self {
        Self {
            kind: ReadFileErrorKind::from_io_error_kind(error.kind()),
            message: error.to_string(),
        }
    }
}

impl ReadFileErrorKind {
    fn from_io_error_kind(kind: io::ErrorKind) -> Self {
        match kind {
            io::ErrorKind::NotFound => Self::NotFound,
            io::ErrorKind::PermissionDenied => Self::PermissionDenied,
            io::ErrorKind::InvalidInput => Self::InvalidInput,
            _ => Self::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_file_contents() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-success.txt",
            std::process::id()
        ));
        fs::write(&path, "hello tau").unwrap();

        let output = read_file(ReadFileInput { path: path.clone() });

        assert_eq!(output.status, ReadFileStatus::Success);
        assert_eq!(output.contents, Some("hello tau".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn returns_not_found_error() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-missing.txt",
            std::process::id()
        ));
        let output = read_file(ReadFileInput { path });

        assert_eq!(output.status, ReadFileStatus::Error);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::NotFound);
    }
}
