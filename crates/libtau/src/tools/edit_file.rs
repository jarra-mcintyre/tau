use std::{fs, io, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::{TauContext, ToolCallError, ToolDefinition, ToolRegistrationError};

pub const NAME: &str = "edit_file";
pub const DESCRIPTION: &str =
    "Edit a UTF-8 text file using exact old contents to new contents substitutions.";

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct EditFileInput {
    pub path: PathBuf,
    pub substitutions: Vec<Substitution>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct Substitution {
    pub old_contents: String,
    pub new_contents: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EditFileStatus {
    Success,
    Error,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct EditFileOutput {
    pub status: EditFileStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub substitutions_applied: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<EditFileError>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct EditFileError {
    pub kind: EditFileErrorKind,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub substitution_index: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EditFileErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    NoMatch,
    NonUniqueMatch,
    OverlappingSubstitutions,
    Other,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ResolvedSubstitution {
    index: usize,
    start: usize,
    end: usize,
    new_contents: String,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(definition()?)
}

pub fn definition() -> Result<ToolDefinition, ToolRegistrationError> {
    ToolDefinition::new::<EditFileInput>(NAME, DESCRIPTION, callback)
}

fn callback(input: Value) -> Result<Value, ToolCallError> {
    let input: EditFileInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let output = edit_file(input);
    serde_json::to_value(output)
        .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))
}

pub fn edit_file(input: EditFileInput) -> EditFileOutput {
    let contents = match fs::read_to_string(&input.path) {
        Ok(contents) => contents,
        Err(error) => return error_output(EditFileError::from_io_error(error, None)),
    };

    let mut resolved = Vec::with_capacity(input.substitutions.len());

    for (index, substitution) in input.substitutions.iter().enumerate() {
        let matches = match find_exact_matches(&contents, &substitution.old_contents) {
            Ok(matches) => matches,
            Err(message) => {
                return error_output(EditFileError {
                    kind: EditFileErrorKind::InvalidInput,
                    message,
                    substitution_index: Some(index),
                });
            }
        };

        match matches.as_slice() {
            [] => {
                return error_output(EditFileError {
                    kind: EditFileErrorKind::NoMatch,
                    message: "substitution old contents did not match the file".to_string(),
                    substitution_index: Some(index),
                });
            }
            [(start, end)] => resolved.push(ResolvedSubstitution {
                index,
                start: *start,
                end: *end,
                new_contents: substitution.new_contents.clone(),
            }),
            _ => {
                return error_output(EditFileError {
                    kind: EditFileErrorKind::NonUniqueMatch,
                    message: format!(
                        "substitution old contents matched {} locations in the file",
                        matches.len()
                    ),
                    substitution_index: Some(index),
                });
            }
        }
    }

    if let Some((left, right)) = first_overlap(&resolved) {
        return error_output(EditFileError {
            kind: EditFileErrorKind::OverlappingSubstitutions,
            message: format!(
                "substitution {} overlaps substitution {}",
                left.index, right.index
            ),
            substitution_index: Some(right.index),
        });
    }

    resolved.sort_by_key(|substitution| substitution.start);

    let mut edited = contents;
    for substitution in resolved.iter().rev() {
        edited.replace_range(
            substitution.start..substitution.end,
            &substitution.new_contents,
        );
    }

    match fs::write(&input.path, edited) {
        Ok(()) => EditFileOutput {
            status: EditFileStatus::Success,
            substitutions_applied: Some(input.substitutions.len()),
            error: None,
        },
        Err(error) => error_output(EditFileError::from_io_error(error, None)),
    }
}

fn error_output(error: EditFileError) -> EditFileOutput {
    EditFileOutput {
        status: EditFileStatus::Error,
        substitutions_applied: None,
        error: Some(error),
    }
}

impl EditFileError {
    fn from_io_error(error: io::Error, substitution_index: Option<usize>) -> Self {
        Self {
            kind: EditFileErrorKind::from_io_error_kind(error.kind()),
            message: error.to_string(),
            substitution_index,
        }
    }
}

impl EditFileErrorKind {
    fn from_io_error_kind(kind: io::ErrorKind) -> Self {
        match kind {
            io::ErrorKind::NotFound => Self::NotFound,
            io::ErrorKind::PermissionDenied => Self::PermissionDenied,
            io::ErrorKind::InvalidInput => Self::InvalidInput,
            _ => Self::Other,
        }
    }
}

fn find_exact_matches(haystack: &str, needle: &str) -> Result<Vec<(usize, usize)>, String> {
    if needle.is_empty() {
        return Err("substitution old contents must not be empty".to_string());
    }

    Ok(haystack
        .match_indices(needle)
        .map(|(start, matched)| (start, start + matched.len()))
        .collect())
}

fn first_overlap(
    substitutions: &[ResolvedSubstitution],
) -> Option<(&ResolvedSubstitution, &ResolvedSubstitution)> {
    let mut sorted: Vec<_> = substitutions.iter().collect();
    sorted.sort_by_key(|substitution| substitution.start);

    for pair in sorted.windows(2) {
        let left = pair[0];
        let right = pair[1];
        if left.end > right.start {
            return Some((left, right));
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn applies_unique_substitution() {
        let path = temp_path("success");
        fs::write(&path, "hello world\n").unwrap();

        let output = edit_file(EditFileInput {
            path: path.clone(),
            substitutions: vec![Substitution {
                old_contents: "hello world".to_string(),
                new_contents: "hello tau".to_string(),
            }],
        });

        assert_eq!(output.status, EditFileStatus::Success);
        assert_eq!(output.substitutions_applied, Some(1));
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello tau\n");

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn uses_exact_matching_only() {
        let path = temp_path("exact-only");
        fs::write(&path, "fn main() {\n    println!(\"hi\");\n}\n").unwrap();

        let output = edit_file(EditFileInput {
            path: path.clone(),
            substitutions: vec![Substitution {
                old_contents: "fn main(){ println!(\"hi\"); }".to_string(),
                new_contents: "fn main() {\n    println!(\"hello\");\n}".to_string(),
            }],
        });

        assert_eq!(output.status, EditFileStatus::Error);
        assert_eq!(output.error.unwrap().kind, EditFileErrorKind::NoMatch);
        assert_eq!(
            fs::read_to_string(&path).unwrap(),
            "fn main() {\n    println!(\"hi\");\n}\n"
        );

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_non_unique_matches_without_writing() {
        let path = temp_path("non-unique");
        fs::write(&path, "same\nsame\n").unwrap();

        let output = edit_file(EditFileInput {
            path: path.clone(),
            substitutions: vec![Substitution {
                old_contents: "same".to_string(),
                new_contents: "different".to_string(),
            }],
        });

        assert_eq!(output.status, EditFileStatus::Error);
        assert_eq!(
            output.error.unwrap().kind,
            EditFileErrorKind::NonUniqueMatch
        );
        assert_eq!(fs::read_to_string(&path).unwrap(), "same\nsame\n");

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_missing_matches_without_writing() {
        let path = temp_path("missing");
        fs::write(&path, "hello world\n").unwrap();

        let output = edit_file(EditFileInput {
            path: path.clone(),
            substitutions: vec![Substitution {
                old_contents: "not here".to_string(),
                new_contents: "replacement".to_string(),
            }],
        });

        assert_eq!(output.status, EditFileStatus::Error);
        assert_eq!(output.error.unwrap().kind, EditFileErrorKind::NoMatch);
        assert_eq!(fs::read_to_string(&path).unwrap(), "hello world\n");

        fs::remove_file(path).unwrap();
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tau-edit-file-test-{}-{name}.txt",
            std::process::id()
        ))
    }
}
