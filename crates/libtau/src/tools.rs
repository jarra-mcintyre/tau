use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::{ContentPart, TauContext};

pub mod bash;
pub mod edit_file;
pub mod find_files;
pub mod read_file;
pub mod write_file;

pub type ToolCallback = fn(Value) -> Result<ToolOutput, ToolCallError>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolOutput {
    pub content: Vec<ContentPart>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing)]
    pub callback: ToolCallback,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolRegistrationError {
    #[error("duplicate tool name: {0}")]
    DuplicateName(String),
    #[error("failed to serialize tool schema: {0}")]
    SchemaSerializationFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ToolCallError {
    #[error("unknown tool: {0}")]
    UnknownTool(String),
    #[error("invalid tool input: {0}")]
    InvalidInput(String),
    #[error("failed to serialize tool output: {0}")]
    OutputSerializationFailed(String),
}

pub fn register_builtin_tools(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    bash::register(context)?;
    read_file::register(context)?;
    edit_file::register(context)?;
    find_files::register(context)?;
    write_file::register(context)?;
    Ok(())
}

impl ToolOutput {
    pub fn json(value: Value) -> Self {
        Self {
            content: vec![ContentPart::json(value)],
        }
    }

    pub fn text(text: impl Into<String>) -> Self {
        Self {
            content: vec![ContentPart::text(text)],
        }
    }
}

impl ToolDefinition {
    pub fn new<Input>(
        name: &str,
        description: &str,
        callback: ToolCallback,
    ) -> Result<Self, ToolRegistrationError>
    where
        Input: JsonSchema,
    {
        let input_schema = serde_json::to_value(schemars::schema_for!(Input))
            .map_err(|error| ToolRegistrationError::SchemaSerializationFailed(error.to_string()))?;

        Ok(Self {
            name: name.to_string(),
            description: description.to_string(),
            input_schema,
            callback,
        })
    }
}
