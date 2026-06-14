use std::io::{self, IsTerminal};

use libtau::{
    context::{ContentPart, ServerToolResult},
    providers::TokenUsage,
};

#[derive(Debug, Clone, Copy)]
pub(crate) enum Style {
    Muted,
    Tool,
    Agent,
    Error,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct OutputStyle {
    color: bool,
}

impl OutputStyle {
    pub(crate) fn detect() -> Self {
        let color = io::stdout().is_terminal()
            && std::env::var_os("NO_COLOR").is_none()
            && std::env::var("TERM")
                .map(|term| term != "dumb")
                .unwrap_or(true);
        Self { color }
    }

    pub(crate) fn println_styled(&self, style: Style, text: &str) {
        if !self.color {
            println!("{text}");
            return;
        }

        let code = match style {
            Style::Muted => "90",
            Style::Tool => "36",
            Style::Agent => "97",
            Style::Error => "31",
        };
        println!("\x1b[{code}m{text}\x1b[0m");
    }

    pub(crate) fn println_indented_styled(&self, style: Style, text: &str) {
        self.println_styled(style, &indent_display_block(text));
    }
}

pub(crate) fn print_token_usage(usage: Option<&TokenUsage>, output: &OutputStyle) {
    let Some(usage) = usage else {
        return;
    };

    let line = format!("[tokens] {}", format_token_usage(usage));
    output.println_styled(Style::Muted, &line);
}

pub(crate) fn format_token_usage(usage: &TokenUsage) -> String {
    let mut fields = Vec::new();
    if let Some(uncached_input) = usage.uncached_input_tokens {
        fields.push(format!("uncached_input={uncached_input}"));
    }
    if let Some(cache_read_input) = usage.cache_read_input_tokens {
        fields.push(format!("cache_read_input={cache_read_input}"));
    }
    if let Some(cache_creation_input) = usage.cache_creation_input_tokens {
        fields.push(format!("cache_creation_input={cache_creation_input}"));
    }
    if let Some(output) = usage.output_tokens {
        fields.push(format!("output={output}"));
    }
    if let Some(total) = usage.total_tokens {
        fields.push(format!("total={total}"));
    }

    fields.join(", ")
}

pub(crate) fn print_content(content: &ContentPart, output: &OutputStyle) {
    match content {
        ContentPart::Text { text, .. } => output.println_styled(Style::Agent, text),
        ContentPart::Thinking { summary, .. } => {
            if !summary.is_empty() {
                for text in summary {
                    output.println_indented_styled(Style::Muted, &format!("[thinking]\n{text}"))
                }
            } else {
                output.println_indented_styled(Style::Muted, "[redacted thinking]");
            }
        }
        ContentPart::Refusal { text, .. } => {
            output.println_indented_styled(Style::Muted, &format!("[refusal]\n{text}"))
        }
        ContentPart::FailedToolCall { text, .. } => {
            output.println_indented_styled(Style::Error, &format!("[failed tool call]\n{text}"))
        }
        ContentPart::Image {
            media_type, data, ..
        } => {
            output
                .println_indented_styled(Style::Muted, &format!("[image: {media_type}, {data:?}]"));
        }
        ContentPart::Binary {
            media_type, data, ..
        } => {
            output.println_indented_styled(
                Style::Muted,
                &format!("[binary: {media_type}, {data:?}]"),
            );
        }
    }
}

pub(crate) fn compact_json(value: &serde_json::Value) -> String {
    serde_json::to_string(value).unwrap_or_else(|_| "<invalid json>".to_string())
}

pub(crate) fn print_server_tool_result(result: &ServerToolResult, output: &OutputStyle) {
    for content in &result.content {
        match content {
            ContentPart::Text { text, .. } => output.println_indented_styled(Style::Tool, text),
            other => print_content(other, output),
        }
    }
}

pub(crate) fn format_server_tool_use(call: &libtau::context::ServerToolUse) -> String {
    if call.name == "web_search"
        && let Some(query) = call.input.get("query").and_then(serde_json::Value::as_str)
    {
        return format!("[server tool] web_search\nquery: {query}");
    }

    format!("[server tool] {}", call.name)
}

fn indent_display_block(text: &str) -> String {
    const INDENT: &str = "  ";
    text.lines()
        .map(|line| format!("{INDENT}{line}"))
        .collect::<Vec<_>>()
        .join("\n")
}
