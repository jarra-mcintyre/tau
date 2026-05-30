use std::{fs, io, path::PathBuf};

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::TauContext,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "find_files";
pub const DESCRIPTION: &str = "Find files under a directory whose filenames match a regular expression. By default, version-control metadata directories such as .git, .hg, and .svn are not searched, and results are limited to 100 files.";

const BLACKLISTED_DIRS: &[&str] = &[".git", ".hg", ".svn"];
const DEFAULT_MAX_RESULTS: usize = 100;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct FindFilesInput {
    /// Directory to search recursively.
    pub directory: PathBuf,
    /// Regular expression matched against each file's filename, not its full path.
    pub filename_regex: String,
    /// Search normally blacklisted directories such as .git, .hg, and .svn when true.
    #[serde(default, skip_serializing_if = "is_false")]
    pub include_blacklisted_directories: bool,
    /// Maximum number of matching files to return. Defaults to 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FindFilesStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct FindFilesOutput {
    pub status: FindFilesStatus,
    pub matches: Vec<PathBuf>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<FindFilesError>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct FindFilesError {
    pub kind: FindFilesErrorKind,
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum FindFilesErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(ToolDefinition::new::<FindFilesInput>(
        "find_files",
        DESCRIPTION,
        callback,
    )?)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: FindFilesInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let output = find_files(input);
    let value = serde_json::to_value(output)
        .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))?;
    Ok(ToolOutput::json(value))
}

pub fn find_files(input: FindFilesInput) -> FindFilesOutput {
    let regex = match Regex::new(&input.filename_regex) {
        Ok(regex) => regex,
        Err(error) => {
            return FindFilesOutput {
                status: FindFilesStatus::Error,
                matches: Vec::new(),
                errors: vec![FindFilesError::new(
                    FindFilesErrorKind::InvalidInput,
                    input.directory,
                    error,
                )],
            };
        }
    };

    match fs::metadata(&input.directory) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => {
            return FindFilesOutput {
                status: FindFilesStatus::Error,
                matches: Vec::new(),
                errors: vec![FindFilesError::new(
                    FindFilesErrorKind::InvalidInput,
                    input.directory,
                    "directory is not a directory",
                )],
            };
        }
        Err(error) => {
            return FindFilesOutput {
                status: FindFilesStatus::Error,
                matches: Vec::new(),
                errors: vec![FindFilesError::from_io_error(input.directory, error)],
            };
        }
    }

    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
    let mut matches = Vec::new();
    let mut errors = Vec::new();
    walk_directory(
        input.directory,
        &regex,
        input.include_blacklisted_directories,
        &mut matches,
        &mut errors,
    );
    matches.sort();
    matches.truncate(max_results);

    let status = if errors.is_empty() {
        FindFilesStatus::Success
    } else {
        FindFilesStatus::Error
    };

    FindFilesOutput {
        status,
        matches,
        errors,
    }
}

fn walk_directory(
    directory: PathBuf,
    regex: &Regex,
    include_blacklisted_directories: bool,
    matches: &mut Vec<PathBuf>,
    errors: &mut Vec<FindFilesError>,
) {
    let entries = match fs::read_dir(&directory) {
        Ok(entries) => entries,
        Err(error) => {
            errors.push(FindFilesError::from_io_error(directory, error));
            return;
        }
    };

    for entry in entries {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                errors.push(FindFilesError::from_io_error(directory.clone(), error));
                continue;
            }
        };
        let path = entry.path();
        let file_type = match entry.file_type() {
            Ok(file_type) => file_type,
            Err(error) => {
                errors.push(FindFilesError::from_io_error(path, error));
                continue;
            }
        };

        if file_type.is_dir() {
            if include_blacklisted_directories || !is_blacklisted_directory(&entry.file_name()) {
                walk_directory(
                    path,
                    regex,
                    include_blacklisted_directories,
                    matches,
                    errors,
                );
            }
        } else if file_type.is_file()
            && regex.is_match(entry.file_name().to_string_lossy().as_ref())
        {
            matches.push(path);
        }
    }
}

fn is_blacklisted_directory(name: &std::ffi::OsStr) -> bool {
    BLACKLISTED_DIRS
        .iter()
        .any(|blacklisted| name == std::ffi::OsStr::new(blacklisted))
}

fn is_false(value: &bool) -> bool {
    !value
}

impl FindFilesError {
    fn from_io_error(path: PathBuf, error: io::Error) -> Self {
        Self {
            kind: FindFilesErrorKind::from_io_error_kind(error.kind()),
            path,
            message: error.to_string(),
        }
    }

    fn new(kind: FindFilesErrorKind, path: PathBuf, message: impl ToString) -> Self {
        Self {
            kind,
            path,
            message: message.to_string(),
        }
    }
}

impl FindFilesErrorKind {
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

    fn test_directory(name: &str) -> PathBuf {
        let path =
            std::env::temp_dir().join(format!("tau-find-files-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn finds_files_matching_filename_regex() {
        let root = test_directory("matches");
        fs::write(root.join("alpha.rs"), "").unwrap();
        fs::write(root.join("alpha.txt"), "").unwrap();
        fs::create_dir(root.join("nested")).unwrap();
        fs::write(root.join("nested").join("beta.rs"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: root.clone(),
            filename_regex: r".*\.rs$".to_string(),
            include_blacklisted_directories: false,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(
            output.matches,
            vec![root.join("alpha.rs"), root.join("nested").join("beta.rs")]
        );
        assert!(output.errors.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skips_blacklisted_directories_by_default() {
        let root = test_directory("blacklist");
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".git").join("config"), "").unwrap();
        fs::write(root.join("config"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: root.clone(),
            filename_regex: "config".to_string(),
            include_blacklisted_directories: false,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join("config")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn includes_blacklisted_directories_when_requested() {
        let root = test_directory("include-blacklist");
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".git").join("config"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: root.clone(),
            filename_regex: "config".to_string(),
            include_blacklisted_directories: true,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join(".git").join("config")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn limits_results_to_default_maximum() {
        let root = test_directory("default-limit");
        for index in 0..101 {
            fs::write(root.join(format!("file-{index:03}.txt")), "").unwrap();
        }

        let output = find_files(FindFilesInput {
            directory: root.clone(),
            filename_regex: r".*\.txt$".to_string(),
            include_blacklisted_directories: false,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches.len(), DEFAULT_MAX_RESULTS);
        assert_eq!(output.matches[0], root.join("file-000.txt"));
        assert_eq!(output.matches[99], root.join("file-099.txt"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn limits_results_to_requested_maximum() {
        let root = test_directory("requested-limit");
        for index in 0..3 {
            fs::write(root.join(format!("file-{index}.txt")), "").unwrap();
        }

        let output = find_files(FindFilesInput {
            directory: root.clone(),
            filename_regex: r".*\.txt$".to_string(),
            include_blacklisted_directories: false,
            max_results: Some(2),
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(
            output.matches,
            vec![root.join("file-0.txt"), root.join("file-1.txt")]
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn returns_invalid_input_for_invalid_regex() {
        let root = test_directory("invalid-regex");

        let output = find_files(FindFilesInput {
            directory: root.clone(),
            filename_regex: "(".to_string(),
            include_blacklisted_directories: false,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Error);
        assert_eq!(output.matches, Vec::<PathBuf>::new());
        assert_eq!(output.errors[0].kind, FindFilesErrorKind::InvalidInput);

        fs::remove_dir_all(root).unwrap();
    }
}
