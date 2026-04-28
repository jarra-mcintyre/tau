use std::{fs, io, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::{
    TauContext, ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError,
};

pub const NAME: &str = "write_file";
pub const DESCRIPTION: &str = "Create or overwrite a text file";

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct WriteFileInput {
    pub path: PathBuf,
    pub contents: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WriteFileStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct WriteFileOutput {
    pub status: WriteFileStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes_written: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<WriteFileError>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct WriteFileError {
    pub kind: WriteFileErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum WriteFileErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(definition()?)
}

pub fn definition() -> Result<ToolDefinition, ToolRegistrationError> {
    ToolDefinition::new::<WriteFileInput>(NAME, DESCRIPTION, callback)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: WriteFileInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let output = write_file(input);
    let value = serde_json::to_value(output)
        .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))?;
    Ok(ToolOutput::json(value))
}

pub fn write_file(input: WriteFileInput) -> WriteFileOutput {
    let bytes_written = input.contents.len();

    match fs::write(&input.path, input.contents) {
        Ok(()) => WriteFileOutput {
            status: WriteFileStatus::Success,
            bytes_written: Some(bytes_written),
            error: None,
        },
        Err(error) => WriteFileOutput {
            status: WriteFileStatus::Error,
            bytes_written: None,
            error: Some(WriteFileError::from_io_error(error)),
        },
    }
}

impl WriteFileError {
    fn from_io_error(error: io::Error) -> Self {
        Self {
            kind: WriteFileErrorKind::from_io_error_kind(error.kind()),
            message: error.to_string(),
        }
    }
}

impl WriteFileErrorKind {
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
    fn creates_file() {
        let path = temp_path("create");
        let _ = fs::remove_file(&path);

        let output = write_file(WriteFileInput {
            path: path.clone(),
            contents: "hello tau".to_string(),
        });

        assert_eq!(output.status, WriteFileStatus::Success);
        assert_eq!(output.bytes_written, Some("hello tau".len()));
        assert_eq!(output.error, None);
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello tau");

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn overwrites_file() {
        let path = temp_path("overwrite");
        fs::write(&path, "old contents").unwrap();

        let output = write_file(WriteFileInput {
            path: path.clone(),
            contents: "new contents".to_string(),
        });

        assert_eq!(output.status, WriteFileStatus::Success);
        assert_eq!(output.bytes_written, Some("new contents".len()));
        assert_eq!(output.error, None);
        assert_eq!(fs::read_to_string(&path).unwrap(), "new contents");

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn returns_not_found_for_missing_parent_directory() {
        let path = std::env::temp_dir()
            .join(format!(
                "tau-write-file-test-{}-missing-parent",
                std::process::id()
            ))
            .join("file.txt");

        let output = write_file(WriteFileInput {
            path,
            contents: "hello tau".to_string(),
        });

        assert_eq!(output.status, WriteFileStatus::Error);
        assert_eq!(output.bytes_written, None);
        assert_eq!(output.error.unwrap().kind, WriteFileErrorKind::NotFound);
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tau-write-file-test-{}-{name}.txt",
            std::process::id()
        ))
    }
}
