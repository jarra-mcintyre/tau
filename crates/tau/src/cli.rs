use std::{ffi::OsString, path::PathBuf};

use clap::{ArgAction, CommandFactory, Parser, Subcommand};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliInvocation {
    pub modifiers: Modifiers,
    pub command: Command,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Modifiers {
    pub yes: bool,
    pub conversation: Option<String>,
    pub model: Option<String>,
    pub read_only: bool,
    pub writes_allowed: bool,
    pub json: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Message { contents: Option<String> },
    MessageAdd { contents: Option<String> },
    MessageInclude { paths: Vec<PathBuf> },
    MessagePaste,
    MessageReference { paths: Vec<PathBuf> },
    MessageSend,
    MessageEcho,
    MessageClear,

    Conversation { alias: Option<String> },
    ConversationAlias { alias: String },
    ConversationHistory,
    ConversationCompact,
    ConversationFork { alias: Option<String> },
    ConversationSwitch { alias: String },
    ConversationList,
    ConversationDelete { alias: String },

    Provider { model: String },
    ProviderThinking { level: ThinkingLevel },
    ProviderList,

    Query { query: String },
    Prompt { prompt: String },
    File { path: PathBuf },
    Version,
    Help { question: Option<String> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingLevel {
    Disabled,
    Low,
    Medium,
    High,
    XHigh,
    Max,
}

impl std::str::FromStr for ThinkingLevel {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "disabled" => Ok(Self::Disabled),
            "low" => Ok(Self::Low),
            "medium" => Ok(Self::Medium),
            "high" => Ok(Self::High),
            "xhigh" => Ok(Self::XHigh),
            "max" => Ok(Self::Max),
            other => Err(format!(
                "invalid thinking level '{other}'; expected disabled, low, medium, high, xhigh, or max"
            )),
        }
    }
}

pub fn parse_cli_from<I, T>(args: I) -> Result<CliInvocation, clap::Error>
where
    I: IntoIterator<Item = T>,
    T: Into<OsString> + Clone,
{
    RawCli::try_parse_from(args).and_then(CliInvocation::try_from)
}

#[derive(Debug, Parser)]
#[command(
    name = "tau",
    about = "Tau command line coding agent",
    disable_help_flag = true,
    disable_help_subcommand = true,
    disable_version_flag = true
)]
struct RawCli {
    /// Skips any confirmation prompts.
    #[arg(long, short = 'y', global = true, action = ArgAction::SetTrue)]
    yes: bool,

    /// Specifies which conversation a command should act on.
    #[arg(long, short = 'c', global = true, value_name = "ALIAS")]
    conversation: Option<String>,

    /// Specifies which model to use.
    #[arg(
        long,
        short = 'm',
        global = true,
        value_name = "PROVIDER/MODEL",
        id = "model_modifier"
    )]
    model: Option<String>,

    /// Disables the editor and shell tools.
    #[arg(long, short = 'r', global = true, action = ArgAction::SetTrue)]
    read_only: bool,

    /// Enables the editor and shell tools.
    #[arg(long, short = 'w', global = true, action = ArgAction::SetTrue)]
    writes_allowed: bool,

    /// Formats Tau output as JSON.
    #[arg(long, short = 'j', global = true, action = ArgAction::SetTrue)]
    json: bool,

    /// Ask a general question with no project context.
    #[arg(long, short = 'q', value_name = "QUERY")]
    query: Option<String>,

    /// Execute a prompt with project context.
    #[arg(long, short = 'p', value_name = "PROMPT")]
    prompt: Option<String>,

    /// Execute a prompt read from a file.
    #[arg(long, short = 'f', value_name = "PATH")]
    file: Option<PathBuf>,

    /// Show the current Tau version.
    #[arg(long, short = 'v', action = ArgAction::SetTrue)]
    version: bool,

    /// Show help, optionally asking a question about Tau.
    #[arg(long = "help", short = 'h', value_name = "QUESTION", num_args = 0..=1, action = ArgAction::Set)]
    help: Option<Option<String>>,

    #[command(subcommand)]
    command: Option<RawCommand>,
}

#[derive(Debug, Subcommand)]
enum RawCommand {
    #[command(name = "message", alias = "m")]
    Message { contents: Vec<String> },
    #[command(name = "message-add", alias = "ma")]
    MessageAdd { contents: Vec<String> },
    #[command(name = "message-include", alias = "mi")]
    MessageInclude { paths: Vec<PathBuf> },
    #[command(name = "message-paste", alias = "mp")]
    MessagePaste,
    #[command(name = "message-reference", alias = "mr")]
    MessageReference { paths: Vec<PathBuf> },
    #[command(name = "message-send", alias = "ms")]
    MessageSend,
    #[command(name = "message-echo", alias = "me")]
    MessageEcho,
    #[command(name = "message-clear", alias = "mc")]
    MessageClear,

    #[command(name = "conversation", alias = "c")]
    Conversation { alias: Option<String> },
    #[command(name = "conversation-alias", alias = "ca")]
    ConversationAlias { alias: String },
    #[command(name = "conversation-history", alias = "ch")]
    ConversationHistory,
    #[command(name = "conversation-compact", alias = "cc")]
    ConversationCompact,
    #[command(name = "conversation-fork", alias = "cf")]
    ConversationFork { alias: Option<String> },
    #[command(name = "conversation-switch", alias = "cs")]
    ConversationSwitch { alias: String },
    #[command(name = "conversation-list", alias = "cl")]
    ConversationList,
    #[command(name = "conversation-delete", alias = "cd")]
    ConversationDelete { alias: String },

    #[command(name = "provider", alias = "p")]
    Provider {
        #[arg(value_name = "PROVIDER/MODEL")]
        model_ref: String,
    },
    #[command(name = "provider-thinking", alias = "pt")]
    ProviderThinking { level: ThinkingLevel },
    #[command(name = "provider-list", alias = "pl")]
    ProviderList,

    #[command(name = "query")]
    Query { query: String },
    #[command(name = "prompt")]
    Prompt { prompt: String },
    #[command(name = "file")]
    File { path: PathBuf },
    #[command(name = "version")]
    Version,
    #[command(name = "help", alias = "h")]
    Help { question: Vec<String> },
}

impl TryFrom<RawCli> for CliInvocation {
    type Error = clap::Error;

    fn try_from(raw: RawCli) -> Result<Self, Self::Error> {
        let modifiers = Modifiers {
            yes: raw.yes,
            conversation: raw.conversation,
            model: raw.model,
            read_only: raw.read_only,
            writes_allowed: raw.writes_allowed,
            json: raw.json,
        };

        let top_level_command_count = [
            raw.query.is_some(),
            raw.prompt.is_some(),
            raw.file.is_some(),
            raw.version,
            raw.help.is_some(),
            raw.command.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count();

        if top_level_command_count != 1 {
            return Err(RawCli::command().error(
                clap::error::ErrorKind::MissingSubcommand,
                "expected exactly one Tau command",
            ));
        }

        let command = if let Some(query) = raw.query {
            Command::Query { query }
        } else if let Some(prompt) = raw.prompt {
            Command::Prompt { prompt }
        } else if let Some(path) = raw.file {
            Command::File { path }
        } else if raw.version {
            Command::Version
        } else if let Some(question) = raw.help {
            Command::Help { question }
        } else {
            raw.command.expect("checked above").into()
        };

        Ok(Self { modifiers, command })
    }
}

impl From<RawCommand> for Command {
    fn from(command: RawCommand) -> Self {
        match command {
            RawCommand::Message { contents } => Command::Message {
                contents: join_optional(contents),
            },
            RawCommand::MessageAdd { contents } => Command::MessageAdd {
                contents: join_optional(contents),
            },
            RawCommand::MessageInclude { paths } => Command::MessageInclude { paths },
            RawCommand::MessagePaste => Command::MessagePaste,
            RawCommand::MessageReference { paths } => Command::MessageReference { paths },
            RawCommand::MessageSend => Command::MessageSend,
            RawCommand::MessageEcho => Command::MessageEcho,
            RawCommand::MessageClear => Command::MessageClear,
            RawCommand::Conversation { alias } => Command::Conversation { alias },
            RawCommand::ConversationAlias { alias } => Command::ConversationAlias { alias },
            RawCommand::ConversationHistory => Command::ConversationHistory,
            RawCommand::ConversationCompact => Command::ConversationCompact,
            RawCommand::ConversationFork { alias } => Command::ConversationFork { alias },
            RawCommand::ConversationSwitch { alias } => Command::ConversationSwitch { alias },
            RawCommand::ConversationList => Command::ConversationList,
            RawCommand::ConversationDelete { alias } => Command::ConversationDelete { alias },
            RawCommand::Provider { model_ref } => Command::Provider { model: model_ref },
            RawCommand::ProviderThinking { level } => Command::ProviderThinking { level },
            RawCommand::ProviderList => Command::ProviderList,
            RawCommand::Query { query } => Command::Query { query },
            RawCommand::Prompt { prompt } => Command::Prompt { prompt },
            RawCommand::File { path } => Command::File { path },
            RawCommand::Version => Command::Version,
            RawCommand::Help { question } => Command::Help {
                question: join_optional(question),
            },
        }
    }
}

fn join_optional(parts: Vec<String>) -> Option<String> {
    if parts.is_empty() {
        None
    } else {
        Some(parts.join(" "))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> CliInvocation {
        parse_cli_from(args).unwrap_or_else(|error| panic!("{error}"))
    }

    #[test]
    fn parses_message_command_with_alias_and_contents() {
        assert_eq!(
            parse(&["tau", "m", "Also", "check", "tests"]),
            CliInvocation {
                modifiers: Modifiers::default(),
                command: Command::Message {
                    contents: Some("Also check tests".to_string())
                },
            }
        );
    }

    #[test]
    fn parses_message_without_contents_for_editor() {
        assert_eq!(
            parse(&["tau", "message"]),
            CliInvocation {
                modifiers: Modifiers::default(),
                command: Command::Message { contents: None },
            }
        );
    }

    #[test]
    fn parses_message_building_commands() {
        assert_eq!(
            parse(&["tau", "-c", "bug-1234", "mi", "a.rs", "image.png"]),
            CliInvocation {
                modifiers: Modifiers {
                    conversation: Some("bug-1234".to_string()),
                    ..Modifiers::default()
                },
                command: Command::MessageInclude {
                    paths: vec![PathBuf::from("a.rs"), PathBuf::from("image.png")]
                },
            }
        );

        assert_eq!(parse(&["tau", "mp"]).command, Command::MessagePaste);
        assert_eq!(
            parse(&["tau", "mr", "src/main.rs"]).command,
            Command::MessageReference {
                paths: vec![PathBuf::from("src/main.rs")]
            }
        );
        assert_eq!(parse(&["tau", "ms"]).command, Command::MessageSend);
        assert_eq!(parse(&["tau", "me"]).command, Command::MessageEcho);
        assert_eq!(parse(&["tau", "-y", "mc"]).command, Command::MessageClear);
    }

    #[test]
    fn parses_conversation_commands() {
        assert_eq!(
            parse(&["tau", "c", "Feature work phase 2"]).command,
            Command::Conversation {
                alias: Some("Feature work phase 2".to_string())
            }
        );
        assert_eq!(
            parse(&["tau", "ca", "My awesome feature"]).command,
            Command::ConversationAlias {
                alias: "My awesome feature".to_string()
            }
        );
        assert_eq!(parse(&["tau", "ch"]).command, Command::ConversationHistory);
        assert_eq!(parse(&["tau", "cc"]).command, Command::ConversationCompact);
        assert_eq!(
            parse(&["tau", "cf", "forked"]).command,
            Command::ConversationFork {
                alias: Some("forked".to_string())
            }
        );
        assert_eq!(
            parse(&["tau", "cs", "bug-1234"]).command,
            Command::ConversationSwitch {
                alias: "bug-1234".to_string()
            }
        );
        assert_eq!(parse(&["tau", "cl"]).command, Command::ConversationList);
        assert_eq!(
            parse(&["tau", "cd", "old-feature"]).command,
            Command::ConversationDelete {
                alias: "old-feature".to_string()
            }
        );
    }

    #[test]
    fn parses_provider_commands() {
        assert_eq!(
            parse(&["tau", "-y", "p", "anthropic/opus-4.7"]),
            CliInvocation {
                modifiers: Modifiers {
                    yes: true,
                    ..Modifiers::default()
                },
                command: Command::Provider {
                    model: "anthropic/opus-4.7".to_string()
                },
            }
        );
        assert_eq!(
            parse(&["tau", "pt", "xhigh"]).command,
            Command::ProviderThinking {
                level: ThinkingLevel::XHigh
            }
        );
        assert_eq!(parse(&["tau", "pl"]).command, Command::ProviderList);
    }

    #[test]
    fn parses_non_interactive_short_flags() {
        assert_eq!(
            parse(&[
                "tau",
                "-r",
                "-m",
                "anthropic/haiku-4.5",
                "-p",
                "Document the API"
            ]),
            CliInvocation {
                modifiers: Modifiers {
                    model: Some("anthropic/haiku-4.5".to_string()),
                    read_only: true,
                    ..Modifiers::default()
                },
                command: Command::Prompt {
                    prompt: "Document the API".to_string()
                },
            }
        );
        assert_eq!(
            parse(&["tau", "-q", "How do generators work?"]).command,
            Command::Query {
                query: "How do generators work?".to_string()
            }
        );
        assert_eq!(
            parse(&["tau", "-f", "test_filter.md"]).command,
            Command::File {
                path: PathBuf::from("test_filter.md")
            }
        );
    }

    #[test]
    fn parses_help_and_version_forms() {
        assert_eq!(
            parse(&["tau", "-h"]).command,
            Command::Help { question: None }
        );
        assert_eq!(
            parse(&["tau", "-h", "How do I search old conversations?"]).command,
            Command::Help {
                question: Some("How do I search old conversations?".to_string())
            }
        );
        assert_eq!(
            parse(&["tau", "help", "How", "do", "I", "search?"]).command,
            Command::Help {
                question: Some("How do I search?".to_string())
            }
        );
        assert_eq!(parse(&["tau", "-v"]).command, Command::Version);
        assert_eq!(parse(&["tau", "version"]).command, Command::Version);
    }

    #[test]
    fn parses_global_modifiers_before_or_after_subcommand() {
        assert_eq!(
            parse(&["tau", "cl", "-j"]).modifiers,
            Modifiers {
                json: true,
                ..Modifiers::default()
            }
        );
        assert_eq!(
            parse(&["tau", "-j", "-r", "-w", "cf", "next"]).modifiers,
            Modifiers {
                json: true,
                read_only: true,
                writes_allowed: true,
                ..Modifiers::default()
            }
        );
    }

    #[test]
    fn rejects_missing_command() {
        assert!(parse_cli_from(["tau"]).is_err());
    }

    #[test]
    fn rejects_invalid_thinking_level() {
        assert!(parse_cli_from(["tau", "pt", "extreme"]).is_err());
    }
}
