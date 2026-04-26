use std::collections::BTreeMap;

use schemars::JsonSchema;
use serde::Serialize;
use serde_json::Value;

pub type ToolCallback = fn(Value) -> Result<Value, ToolCallError>;

#[derive(Debug, Clone)]
pub struct TauContext {
    tools: BTreeMap<String, ToolDefinition>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
    #[serde(skip_serializing)]
    pub callback: ToolCallback,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolRegistrationError {
    DuplicateName(String),
    SchemaSerializationFailed(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolCallError {
    UnknownTool(String),
    InvalidInput(String),
    OutputSerializationFailed(String),
}

impl TauContext {
    pub fn new() -> Self {
        Self {
            tools: BTreeMap::new(),
        }
    }

    pub fn register_tool(
        &mut self,
        definition: ToolDefinition,
    ) -> Result<(), ToolRegistrationError> {
        if self.tools.contains_key(&definition.name) {
            return Err(ToolRegistrationError::DuplicateName(definition.name));
        }

        self.tools.insert(definition.name.clone(), definition);

        Ok(())
    }

    pub fn tools(&self) -> impl Iterator<Item = &ToolDefinition> {
        self.tools.values()
    }

    pub fn get_tool(&self, name: &str) -> Option<&ToolDefinition> {
        self.tools.get(name)
    }

    pub fn call_tool(&self, name: &str, input: Value) -> Result<Value, ToolCallError> {
        let tool = self
            .get_tool(name)
            .ok_or_else(|| ToolCallError::UnknownTool(name.to_string()))?;

        (tool.callback)(input)
    }
}

impl Default for TauContext {
    fn default() -> Self {
        Self::new()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools;

    #[test]
    fn registers_builtin_tools() {
        let mut context = TauContext::new();
        tools::register_builtin_tools(&mut context).unwrap();

        let names: Vec<_> = context
            .tools()
            .map(|definition| definition.name.as_str())
            .collect();

        assert_eq!(names, vec!["edit_file", "read_file", "write_file"]);
        assert!(context.get_tool("read_file").is_some());
        assert!(context.get_tool("edit_file").is_some());
        assert!(context.get_tool("write_file").is_some());
    }

    #[test]
    fn rejects_duplicate_tool_names() {
        fn callback(input: Value) -> Result<Value, ToolCallError> {
            Ok(input)
        }

        let mut context = TauContext::new();
        let definition = ToolDefinition {
            name: "duplicate".to_string(),
            description: "first".to_string(),
            input_schema: serde_json::json!({ "type": "object" }),
            callback,
        };
        context.register_tool(definition.clone()).unwrap();

        assert_eq!(
            context.register_tool(definition),
            Err(ToolRegistrationError::DuplicateName(
                "duplicate".to_string()
            ))
        );
    }
}
