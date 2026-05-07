use std::{fs, io, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::TauContext,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "read_file";
pub const DESCRIPTION: &str =
    "Read a text file from disk, optionally limited to a 1-based inclusive line range.";

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileInput {
    pub path: PathBuf,
    /// Optional 1-based first line to read. Defaults to the first line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_line: Option<i64>,
    /// Optional 1-based last line to read, inclusive. Defaults to the last line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_line: Option<i64>,
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

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: ReadFileInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let output = read_file(input);
    let value = serde_json::to_value(output)
        .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))?;
    Ok(ToolOutput::json(value))
}

pub fn read_file(input: ReadFileInput) -> ReadFileOutput {
    if let Err(error) = validate_line_range(input.first_line, input.last_line) {
        return ReadFileOutput {
            status: ReadFileStatus::Error,
            contents: None,
            error: Some(error),
        };
    }

    match fs::read_to_string(&input.path) {
        Ok(contents) => ReadFileOutput {
            status: ReadFileStatus::Success,
            contents: Some(apply_line_range(
                &contents,
                input.first_line,
                input.last_line,
            )),
            error: None,
        },
        Err(error) => ReadFileOutput {
            status: ReadFileStatus::Error,
            contents: None,
            error: Some(ReadFileError::from_io_error(error)),
        },
    }
}

fn validate_line_range(
    first_line: Option<i64>,
    last_line: Option<i64>,
) -> Result<(), ReadFileError> {
    if first_line.is_some_and(|line| line <= 0) || last_line.is_some_and(|line| line <= 0) {
        return Err(ReadFileError::invalid_input(
            "line numbers are 1-based and must be greater than zero".to_string(),
        ));
    }

    if let (Some(first_line), Some(last_line)) = (first_line, last_line) {
        if first_line > last_line {
            return Err(ReadFileError::invalid_input(format!(
                "first_line ({first_line}) must be less than or equal to last_line ({last_line})"
            )));
        }
    }

    Ok(())
}

fn apply_line_range(contents: &str, first_line: Option<i64>, last_line: Option<i64>) -> String {
    if first_line.is_none() && last_line.is_none() {
        return contents.to_string();
    }

    let first_line = first_line.unwrap_or(1) as usize;
    let last_line = last_line.map(|line| line as usize);
    contents
        .split_inclusive('\n')
        .enumerate()
        .filter_map(|(index, line)| {
            let line_number = index + 1;
            if line_number < first_line
                || last_line.is_some_and(|last_line| line_number > last_line)
            {
                None
            } else {
                Some(line)
            }
        })
        .collect()
}

impl ReadFileError {
    fn from_io_error(error: io::Error) -> Self {
        Self {
            kind: ReadFileErrorKind::from_io_error_kind(error.kind()),
            message: error.to_string(),
        }
    }

    fn invalid_input(message: String) -> Self {
        Self {
            kind: ReadFileErrorKind::InvalidInput,
            message,
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

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
        });

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
        let output = read_file(ReadFileInput {
            path,
            first_line: None,
            last_line: None,
        });

        assert_eq!(output.status, ReadFileStatus::Error);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::NotFound);
    }

    #[test]
    fn reads_inclusive_line_range() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-range.txt",
            std::process::id()
        ));
        fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: Some(2),
            last_line: Some(3),
        });

        assert_eq!(output.status, ReadFileStatus::Success);
        assert_eq!(output.contents, Some("two\nthree\n".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_from_start_when_only_last_line_is_set() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-end-line.txt",
            std::process::id()
        ));
        fs::write(&path, "one\ntwo\nthree").unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: Some(2),
        });

        assert_eq!(output.status, ReadFileStatus::Success);
        assert_eq!(output.contents, Some("one\ntwo\n".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn returns_invalid_input_for_invalid_line_range() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-invalid-range.txt",
            std::process::id()
        ));

        let output = read_file(ReadFileInput {
            path,
            first_line: Some(3),
            last_line: Some(2),
        });

        assert_eq!(output.status, ReadFileStatus::Error);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::InvalidInput);
    }

    #[test]
    fn returns_invalid_input_for_negative_line_number() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-negative-line.txt",
            std::process::id()
        ));

        let output = read_file(ReadFileInput {
            path,
            first_line: Some(-1),
            last_line: None,
        });

        assert_eq!(output.status, ReadFileStatus::Error);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::InvalidInput);
    }
}
