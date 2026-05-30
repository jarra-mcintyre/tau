use std::{ffi::OsStr, fs, io, path::PathBuf};

use regex::Regex;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::TauContext,
    path_ignore::IgnoreRules,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "find_files";
pub const DESCRIPTION: &str = "Find files in directory matching pattern. By default ignores hidden folders and respects .gitignore and friends";

const DEFAULT_MAX_RESULTS: usize = 100;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct FindFilesInput {
    /// Directory to search recursively.
    pub directory: Option<PathBuf>,
    /// Regex matched against each file's filename, not its full path.
    pub pattern: String,
    /// Include files ignored by .gitignore etc
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_ignored: Option<bool>,
    /// Search hidden directories when true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_hidden: Option<bool>,
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
    let directory = input.directory.unwrap_or_else(|| PathBuf::from("."));

    let regex = match Regex::new(&input.pattern) {
        Ok(regex) => regex,
        Err(error) => {
            return FindFilesOutput {
                status: FindFilesStatus::Error,
                matches: Vec::new(),
                errors: vec![FindFilesError::new(
                    FindFilesErrorKind::InvalidInput,
                    directory,
                    error,
                )],
            };
        }
    };

    match fs::metadata(&directory) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => {
            return FindFilesOutput {
                status: FindFilesStatus::Error,
                matches: Vec::new(),
                errors: vec![FindFilesError::new(
                    FindFilesErrorKind::InvalidInput,
                    directory,
                    "directory is not a directory",
                )],
            };
        }
        Err(error) => {
            return FindFilesOutput {
                status: FindFilesStatus::Error,
                matches: Vec::new(),
                errors: vec![FindFilesError::from_io_error(directory, error)],
            };
        }
    }

    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
    let include_ignored = input.include_ignored.unwrap_or(false);
    let ignore_rules = if !include_ignored {
        match IgnoreRules::from_directory_chain(&directory) {
            Ok(rules) => Some(rules),
            Err(error) => {
                return FindFilesOutput {
                    status: FindFilesStatus::Error,
                    matches: Vec::new(),
                    errors: vec![FindFilesError::from_io_error(directory, error)],
                };
            }
        }
    } else {
        None
    };
    let mut matches = Vec::new();
    let mut errors = Vec::new();
    walk_directory(
        directory,
        &regex,
        input.include_hidden.unwrap_or(false),
        ignore_rules,
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
    include_hidden_directories: bool,
    ignore_rules: Option<IgnoreRules>,
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

        if ignore_rules
            .as_ref()
            .is_some_and(|rules| rules.is_ignored(&path, file_type.is_dir()))
        {
            continue;
        }

        if file_type.is_dir() {
            if include_hidden_directories || !is_hidden_directory(&entry.file_name()) {
                let ignore_rules = match &ignore_rules {
                    Some(rules) => match rules.with_rules_from_directory(&path) {
                        Ok(updated) => Some(updated),
                        Err(error) => {
                            errors.push(FindFilesError::from_io_error(directory, error));
                            return;
                        }
                    },
                    None => None
                };
                walk_directory(
                    path,
                    regex,
                    include_hidden_directories,
                    ignore_rules,
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

fn is_hidden_directory(name: &OsStr) -> bool {
    name.to_string_lossy().starts_with('.')
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
            directory: Some(root.clone()),
            pattern: r".*\.rs$".to_string(),
            include_ignored: None,
            include_hidden: None,
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
    fn skips_hidden_directories_by_default() {
        let root = test_directory("hidden");
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".git").join("config"), "").unwrap();
        fs::write(root.join("config"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: Some(root.clone()),
            pattern: "config".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join("config")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn includes_hidden_directories_when_requested() {
        let root = test_directory("include-hidden");
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".git").join("config"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: Some(root.clone()),
            pattern: "config".to_string(),
            include_ignored: None,
            include_hidden: Some(true),
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join(".git").join("config")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skips_all_hidden_directories_by_default() {
        let root = test_directory("all-hidden");
        fs::create_dir(root.join(".cache")).unwrap();
        fs::write(root.join(".cache").join("data.txt"), "").unwrap();
        fs::write(root.join("data.txt"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: Some(root.clone()),
            pattern: r".*\.txt$".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join("data.txt")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn obeys_ignore_files() {
        let root = test_directory("ignore-files");
        fs::write(root.join(".gitignore"), "*.log\nignored-dir/\n").unwrap();
        fs::write(root.join("debug.log"), "").unwrap();
        fs::write(root.join("keep.txt"), "").unwrap();
        fs::create_dir(root.join("ignored-dir")).unwrap();
        fs::write(root.join("ignored-dir").join("nested.txt"), "").unwrap();
        fs::create_dir(root.join("subdir")).unwrap();
        fs::write(root.join("subdir").join(".svnignore"), "ignored.txt\n").unwrap();
        fs::write(root.join("subdir").join("ignored.txt"), "").unwrap();
        fs::write(root.join("subdir").join("keep.txt"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: Some(root.clone()),
            pattern: r".*\.(log|txt)$".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(
            output.matches,
            vec![root.join("keep.txt"), root.join("subdir").join("keep.txt")]
        );

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn loads_ignore_files_from_repository_root_to_search_directory() {
        let root = test_directory("ignore-chain");
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".gitignore"), "*.log\n").unwrap();
        fs::create_dir(root.join("subdir")).unwrap();
        fs::write(root.join("subdir").join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(root.join("subdir").join("debug.log"), "").unwrap();
        fs::write(root.join("subdir").join("ignored.txt"), "").unwrap();
        fs::write(root.join("subdir").join("keep.txt"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: Some(root.join("subdir")),
            pattern: r".*\.(log|txt)$".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join("subdir").join("keep.txt")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn includes_ignored_files_when_requested() {
        let root = test_directory("include-ignored");
        fs::write(root.join(".gitignore"), "*.log\n").unwrap();
        fs::write(root.join("debug.log"), "").unwrap();

        let output = find_files(FindFilesInput {
            directory: Some(root.clone()),
            pattern: r".*\.log$".to_string(),
            include_ignored: Some(true),
            include_hidden: None,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Success);
        assert_eq!(output.matches, vec![root.join("debug.log")]);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn limits_results_to_default_maximum() {
        let root = test_directory("default-limit");
        for index in 0..101 {
            fs::write(root.join(format!("file-{index:03}.txt")), "").unwrap();
        }

        let output = find_files(FindFilesInput {
            directory: Some(root.clone()),
            pattern: r".*\.txt$".to_string(),
            include_ignored: None,
            include_hidden: None,
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
            directory: Some(root.clone()),
            pattern: r".*\.txt$".to_string(),
            include_ignored: None,
            include_hidden: None,
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
            directory: Some(root.clone()),
            pattern: "(".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert_eq!(output.status, FindFilesStatus::Error);
        assert_eq!(output.matches, Vec::<PathBuf>::new());
        assert_eq!(output.errors[0].kind, FindFilesErrorKind::InvalidInput);

        fs::remove_dir_all(root).unwrap();
    }
}
