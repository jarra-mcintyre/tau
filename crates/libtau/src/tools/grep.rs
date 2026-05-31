use std::{fs, io, path::PathBuf};

use grep_regex::RegexMatcher;
use grep_searcher::{BinaryDetection, SearcherBuilder, sinks::Lossy};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::TauContext,
    path_ignore,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "grep";
pub const DESCRIPTION: &str = "Recursively search file contents. By default ignores hidden files and respects .gitignore and friends";

const DEFAULT_MAX_RESULTS: usize = 100;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct GrepInput {
    /// Directory to search recursively.
    pub directory: Option<PathBuf>,
    /// Regex matched against file contents.
    pub pattern: String,
    /// Include files ignored by .gitignore etc
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_ignored: Option<bool>,
    /// Search hidden directories when true.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_hidden: Option<bool>,
    /// Maximum number of results to return. Applies to both errors and matches. Default 100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_results: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct GrepOutput {
    pub okay: bool,
    pub matches: Vec<GrepMatch>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub errors: Vec<GrepError>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq, PartialOrd, Ord)]
pub struct GrepMatch {
    pub path: PathBuf,
    pub line_number: u64,
    pub line: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct GrepError {
    pub kind: GrepErrorKind,
    pub path: PathBuf,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum GrepErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(ToolDefinition::new::<GrepInput>(
        NAME,
        DESCRIPTION,
        true,
        callback,
    )?)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: GrepInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    Ok(grep(input).into_tool_output())
}

pub fn grep(input: GrepInput) -> GrepOutput {
    let directory = input.directory.unwrap_or_else(|| PathBuf::from("."));

    let matcher = match RegexMatcher::new_line_matcher(&input.pattern) {
        Ok(matcher) => matcher,
        Err(error) => {
            return GrepOutput {
                okay: false,
                matches: Vec::new(),
                errors: vec![GrepError::new(
                    GrepErrorKind::InvalidInput,
                    directory,
                    error,
                )],
            };
        }
    };

    match fs::metadata(&directory) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => {
            return GrepOutput {
                okay: false,
                matches: Vec::new(),
                errors: vec![GrepError::new(
                    GrepErrorKind::InvalidInput,
                    directory,
                    "directory is not a directory",
                )],
            };
        }
        Err(error) => {
            return GrepOutput {
                okay: false,
                matches: Vec::new(),
                errors: vec![GrepError::from_io_error(directory, error)],
            };
        }
    }

    let max_results = input.max_results.unwrap_or(DEFAULT_MAX_RESULTS);
    let mut matches = Vec::new();
    let mut errors = Vec::new();
    search_directory(
        &directory,
        &matcher,
        input.include_hidden.unwrap_or(false),
        input.include_ignored.unwrap_or(false),
        &mut matches,
        &mut errors,
    );
    matches.sort();
    matches.truncate(max_results);
    errors.truncate(max_results);

    GrepOutput {
        okay: errors.is_empty(),
        matches,
        errors,
    }
}

fn search_directory(
    directory: &PathBuf,
    matcher: &RegexMatcher,
    include_hidden: bool,
    include_ignored: bool,
    matches: &mut Vec<GrepMatch>,
    errors: &mut Vec<GrepError>,
) {
    let mut searcher = SearcherBuilder::new()
        .binary_detection(BinaryDetection::quit(b'\x00'))
        .build();

    for entry in path_ignore::find_files(directory, include_hidden, include_ignored) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(error) => {
                errors.push(GrepError::new(
                    GrepErrorKind::Other,
                    path_ignore::error_path(&error),
                    error,
                ));
                continue;
            }
        };

        if !entry
            .file_type()
            .is_some_and(|file_type| file_type.is_file())
        {
            continue;
        }

        let path = entry.into_path();
        if let Err(error) = search_file(&mut searcher, matcher, &path, matches) {
            errors.push(GrepError::from_io_error(path, error));
        }
    }
}

fn search_file(
    searcher: &mut grep_searcher::Searcher,
    matcher: &RegexMatcher,
    path: &PathBuf,
    matches: &mut Vec<GrepMatch>,
) -> io::Result<()> {
    searcher.search_path(
        matcher,
        path,
        Lossy(|line_number, line| {
            matches.push(GrepMatch {
                path: path.clone(),
                line_number,
                line: line.trim_end_matches(['\r', '\n']).to_string(),
            });
            Ok(true)
        }),
    )
}

impl GrepOutput {
    fn into_tool_output(self) -> ToolOutput {
        let mut lines: Vec<String> = self
            .matches
            .iter()
            .map(|matched| {
                format!(
                    "{}:{}:{}",
                    matched.path.display(),
                    matched.line_number,
                    matched.line
                )
            })
            .collect();
        lines.extend(
            self.errors
                .iter()
                .map(|error| format!("{}: {}", error.path.display(), error.message)),
        );

        let text = if lines.is_empty() {
            "no matches".to_string()
        } else {
            lines.join("\n")
        };

        if self.okay {
            ToolOutput::text(text)
        } else {
            ToolOutput::error(text)
        }
    }
}

impl GrepError {
    fn from_io_error(path: PathBuf, error: io::Error) -> Self {
        Self {
            kind: GrepErrorKind::from_io_error_kind(error.kind()),
            path,
            message: error.to_string(),
        }
    }

    fn new(kind: GrepErrorKind, path: PathBuf, message: impl ToString) -> Self {
        Self {
            kind,
            path,
            message: message.to_string(),
        }
    }
}

impl GrepErrorKind {
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
            std::env::temp_dir().join(format!("tau-grep-test-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn finds_lines_matching_regex() {
        let root = test_directory("matches");
        fs::write(root.join("alpha.txt"), "one\ntwo needles\nthree\n").unwrap();
        fs::create_dir(root.join("nested")).unwrap();
        fs::write(root.join("nested").join("beta.txt"), "needle here\nnope\n").unwrap();

        let output = grep(GrepInput {
            directory: Some(root.clone()),
            pattern: "needle".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert!(output.okay);
        assert_eq!(
            output.matches,
            vec![
                GrepMatch {
                    path: root.join("alpha.txt"),
                    line_number: 2,
                    line: "two needles".to_string(),
                },
                GrepMatch {
                    path: root.join("nested").join("beta.txt"),
                    line_number: 1,
                    line: "needle here".to_string(),
                },
            ]
        );
        assert!(output.errors.is_empty());

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn skips_hidden_directories_by_default() {
        let root = test_directory("hidden");
        fs::create_dir(root.join(".git")).unwrap();
        fs::write(root.join(".git").join("config"), "needle\n").unwrap();
        fs::write(root.join("config"), "needle\n").unwrap();

        let output = grep(GrepInput {
            directory: Some(root.clone()),
            pattern: "needle".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert!(output.okay);
        assert_eq!(output.matches.len(), 1);
        assert_eq!(output.matches[0].path, root.join("config"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn obeys_ignore_files() {
        let root = test_directory("ignore-files");
        fs::write(root.join(".gitignore"), "ignored.txt\n").unwrap();
        fs::write(root.join("ignored.txt"), "needle\n").unwrap();
        fs::write(root.join("keep.txt"), "needle\n").unwrap();

        let output = grep(GrepInput {
            directory: Some(root.clone()),
            pattern: "needle".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert!(output.okay);
        assert_eq!(output.matches.len(), 1);
        assert_eq!(output.matches[0].path, root.join("keep.txt"));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn limits_results_to_requested_maximum() {
        let root = test_directory("requested-limit");
        fs::write(root.join("many.txt"), "needle 1\nneedle 2\nneedle 3\n").unwrap();

        let output = grep(GrepInput {
            directory: Some(root.clone()),
            pattern: "needle".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: Some(2),
        });

        assert!(output.okay);
        assert_eq!(output.matches.len(), 2);
        assert_eq!(output.matches[0].line_number, 1);
        assert_eq!(output.matches[1].line_number, 2);

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn returns_invalid_input_for_invalid_regex() {
        let root = test_directory("invalid-regex");

        let output = grep(GrepInput {
            directory: Some(root.clone()),
            pattern: "(".to_string(),
            include_ignored: None,
            include_hidden: None,
            max_results: None,
        });

        assert!(!output.okay);
        assert_eq!(output.matches, Vec::<GrepMatch>::new());
        assert_eq!(output.errors[0].kind, GrepErrorKind::InvalidInput);

        fs::remove_dir_all(root).unwrap();
    }
}
