use std::{fs, path::PathBuf};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::TauContext,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "write_file";
pub const DESCRIPTION: &str = "Create or overwrite a text file";

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct WriteFileInput {
    /// File path
    pub path: PathBuf,
    /// Text to write
    pub contents: String,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(ToolDefinition::new::<WriteFileInput>(
        NAME,
        DESCRIPTION,
        false,
        callback,
    )?)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: WriteFileInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    Result::Ok(write_file(input))
}

fn write_file(input: WriteFileInput) -> ToolOutput {
    match fs::write(&input.path, input.contents) {
        Ok(()) => ToolOutput::text(format!("Wrote {}", input.path.display())),
        Err(err) => ToolOutput::text(format!(
            "Error {} when writing file {}",
            err,
            input.path.display()
        )),
    }
}

#[cfg(test)]
mod tests {
    use crate::context::ContentPart;

    use super::*;
    use std::assert_matches;

    #[test]
    fn creates_file() {
        let path = temp_path("create");
        let _ = fs::remove_file(&path);

        let output = write_file(WriteFileInput {
            path: path.clone(),
            contents: "hello tau".to_string(),
        });

        assert_eq!(1, output.content.len());
        assert_matches!(&output.content[0], ContentPart::Text{text,..} if text.contains("Wrote"));
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

        assert_eq!(1, output.content.len());
        assert_matches!(&output.content[0], ContentPart::Text{text,..} if text.contains("Wrote"));
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

        assert_eq!(1, output.content.len());
        if let ContentPart::Text { text, .. } = &output.content[0] {
            println!("Got message: '{}'", text);
            assert!(text.contains("Error"));
        } else {
            panic!("Expected text content");
        }
    }

    fn temp_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "tau-write-file-test-{}-{name}.txt",
            std::process::id()
        ))
    }
}
