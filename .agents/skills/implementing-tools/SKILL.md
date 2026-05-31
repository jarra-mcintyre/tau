---
name: implementing-tools
description: guidlines for implementing tools in Tau
---

# General considerations

Tool inputs are converted into a JSON schema to pass to model APIs. This schema forms part of the context.
- Keep the name and description brief and to the point
- Keep input field names brief
- Always including a rust-doc comment on input fields. These are used as the field description in the resulting schema.

Tool outputs are just content blocks returned to the model.
- Content should be returned using appropriate content blocks
- Success can be as simple as a text content block saying "done", "written", "succedded" or similar
- Errors should be returned using a `FailedToolCall` block with text describing the error
- Unless instructed otherwise never encode the output as JSON. It's counter productive.
- Unless instructed otherwise never create complex structures to describe the output shape. It's counter productive. Just stick to plain text
- Errors should be informative and help the model adjust

Overall: output consumes context. Treat it as a precious resource. Don't cram it with extra information

# Implementation

Tools live under `crates/libtau/src/tools`.
- Do have a look at `write_file.rs` for a simple example.
- Do not read every single tool definition. It's an unnecessary waste of context

Each tool defines a callback. The callback returns either a result (a `ToolOuput` structure with a list of content blocks), or a `ToolCallError`.
- `ToolCallError` should only be used for invalid tool invocations (e.g. incorrect arguments etc)
- Normally errors occuring during execution should be returned as a `FailedToolCall` content in the tool output.

Input structures should be handled as follows
- Always use `Option` for optional parameter. Never omit booleans
- For complex input structures with multiple optional parameters it's handy to derive `Builder`. This keeps the unit tests cleaner

```rust
#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq, Builder)]
pub struct ReadFileInput {
    pub path: PathBuf,
    /// First line to read (optional; starts from 1)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_line: Option<i64>,
    /// Last line to read (optional; starts from 1)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_line: Option<i64>,
    /// Content type override (optional; e.g. image/png)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Allow reading large text files (optional; default false)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_soft_limit: Option<bool>,
}
```