use std::{ffi::OsString, path::PathBuf};

use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use crate::provider_config::ConfigurableProvider;

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
    pub alias: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Message {
        contents: Option<String>,
    },
    MessageAdd {
        contents: Option<String>,
    },
    MessageInclude {
        paths: Vec<PathBuf>,
    },
    MessagePaste,
    MessageReference {
        paths: Vec<PathBuf>,
    },
    MessageSend,
    MessageEcho,
    MessageClear,
    MessageUndo {
        offset: isize,
    },

    Status,
    Conversation {
        alias: Option<String>,
    },
    ConversationAlias {
        alias: String,
    },
    ConversationHistory,
    ConversationCompact,
    ConversationFork {
        offset: Option<isize>,
        alias: Option<String>,
    },
    ConversationSwitch {
        alias: String,
    },
    ConversationList,
    ConversationDelete {
        alias: String,
    },

    Provider {
        model: String,
    },
    ProviderThinking {
        level: ThinkingLevel,
    },
    ProviderConfig {
        provider: Option<ConfigurableProvider>,
    },
    ProviderList,

    Query {
        query: String,
    },
    Prompt {
        prompt: String,
    },
    File {
        path: PathBuf,
    },
    Version,
    Help {
        question: Option<String>,
    },
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

    /// Specifies an alias for commands that create or rename things.
    #[arg(
        long,
        short = 'a',
        global = true,
        value_name = "ALIAS",
        id = "alias_modifier"
    )]
    alias: Option<String>,

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
    #[command(name = "message-undo", alias = "mu")]
    MessageUndo {
        #[arg(value_name = "OFFSET", allow_hyphen_values = true)]
        offset: Option<isize>,
    },

    #[command(name = "conversation", alias = "c")]
    Conversation,
    #[command(name = "conversation-alias", alias = "ca")]
    ConversationAlias { alias: String },
    #[command(name = "conversation-history", alias = "ch")]
    ConversationHistory,
    #[command(name = "conversation-compact", alias = "cc")]
    ConversationCompact,
    #[command(name = "conversation-fork", alias = "cf")]
    ConversationFork {
        #[arg(value_name = "OFFSET", allow_hyphen_values = true)]
        offset: Option<isize>,
    },
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
    #[command(name = "provider-config")]
    ProviderConfig {
        #[arg(value_name = "PROVIDER")]
        provider: Option<ConfigurableProvider>,
    },
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
            alias: raw.alias.clone(),
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

        if top_level_command_count > 1 {
            return Err(RawCli::command().error(
                clap::error::ErrorKind::ArgumentConflict,
                "expected at most one Tau command",
            ));
        }

        let command = if top_level_command_count == 0 {
            Command::Status
        } else if let Some(query) = raw.query {
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
            command_from_raw(raw.command.expect("checked above"), raw.alias)
        };

        validate_modifiers(&modifiers, &command, command.allowed_modifiers())?;

        Ok(Self { modifiers, command })
    }
}

#[derive(Debug, Clone, Copy)]
struct AllowedModifiers {
    yes: bool,
    conversation: bool,
    model: bool,
    read_only: bool,
    writes_allowed: bool,
    json: bool,
    alias: bool,
}

impl AllowedModifiers {
    const NONE: Self = Self {
        yes: false,
        conversation: false,
        model: false,
        read_only: false,
        writes_allowed: false,
        json: false,
        alias: false,
    };

    const fn yes(mut self) -> Self {
        self.yes = true;
        self
    }

    const fn conversation(mut self) -> Self {
        self.conversation = true;
        self
    }

    const fn model(mut self) -> Self {
        self.model = true;
        self
    }

    const fn read_only(mut self) -> Self {
        self.read_only = true;
        self
    }

    const fn writes_allowed(mut self) -> Self {
        self.writes_allowed = true;
        self
    }

    const fn json(mut self) -> Self {
        self.json = true;
        self
    }

    const fn alias(mut self) -> Self {
        self.alias = true;
        self
    }
}

impl Command {
    fn allowed_modifiers(&self) -> AllowedModifiers {
        match self {
            Command::Status => AllowedModifiers::NONE,

            Command::Message { .. }
            | Command::MessageAdd { .. }
            | Command::MessageInclude { .. }
            | Command::MessagePaste
            | Command::MessageReference { .. }
            | Command::MessageSend
            | Command::MessageEcho => AllowedModifiers::NONE.conversation(),

            Command::MessageClear | Command::MessageUndo { .. } => {
                AllowedModifiers::NONE.conversation().yes()
            }

            Command::Conversation { .. } => AllowedModifiers::NONE.alias().read_only(),
            Command::ConversationAlias { .. } => AllowedModifiers::NONE,
            Command::ConversationHistory => AllowedModifiers::NONE.json(),
            Command::ConversationCompact => AllowedModifiers::NONE.yes(),
            Command::ConversationFork { .. } => {
                AllowedModifiers::NONE.alias().read_only().writes_allowed()
            }
            Command::ConversationSwitch { .. } => AllowedModifiers::NONE,
            Command::ConversationList => AllowedModifiers::NONE.json(),
            Command::ConversationDelete { .. } => AllowedModifiers::NONE.yes(),

            Command::Provider { .. } => AllowedModifiers::NONE.yes(),
            Command::ProviderThinking { .. } => AllowedModifiers::NONE,
            Command::ProviderConfig { .. } => AllowedModifiers::NONE,
            Command::ProviderList => AllowedModifiers::NONE.json(),

            Command::Query { .. } => AllowedModifiers::NONE.model(),
            Command::Prompt { .. } | Command::File { .. } => {
                AllowedModifiers::NONE.model().read_only()
            }
            Command::Version | Command::Help { .. } => AllowedModifiers::NONE,
        }
    }
}

fn validate_modifiers(
    modifiers: &Modifiers,
    command: &Command,
    allowed: AllowedModifiers,
) -> Result<(), clap::Error> {
    if modifiers.yes && !allowed.yes {
        return Err(unsupported_modifier("--yes/-y", command));
    }
    if modifiers.conversation.is_some() && !allowed.conversation {
        return Err(unsupported_modifier("--conversation/-c", command));
    }
    if modifiers.model.is_some() && !allowed.model {
        return Err(unsupported_modifier("--model/-m", command));
    }
    if modifiers.read_only && !allowed.read_only {
        return Err(unsupported_modifier("--read-only/-r", command));
    }
    if modifiers.writes_allowed && !allowed.writes_allowed {
        return Err(unsupported_modifier("--writes-allowed/-w", command));
    }
    if modifiers.json && !allowed.json {
        return Err(unsupported_modifier("--json/-j", command));
    }
    if modifiers.alias.is_some() && !allowed.alias {
        return Err(unsupported_modifier("--alias/-a", command));
    }

    Ok(())
}

fn unsupported_modifier(modifier: &str, _command: &Command) -> clap::Error {
    RawCli::command().error(
        clap::error::ErrorKind::ArgumentConflict,
        format!("modifier {modifier} is not supported by this command"),
    )
}

fn command_from_raw(command: RawCommand, alias_modifier: Option<String>) -> Command {
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
        RawCommand::MessageUndo { offset } => Command::MessageUndo {
            offset: offset.unwrap_or(-1),
        },
        RawCommand::Conversation => Command::Conversation {
            alias: alias_modifier,
        },
        RawCommand::ConversationAlias { alias } => Command::ConversationAlias { alias },
        RawCommand::ConversationHistory => Command::ConversationHistory,
        RawCommand::ConversationCompact => Command::ConversationCompact,
        RawCommand::ConversationFork { offset } => Command::ConversationFork {
            offset,
            alias: alias_modifier,
        },
        RawCommand::ConversationSwitch { alias } => Command::ConversationSwitch { alias },
        RawCommand::ConversationList => Command::ConversationList,
        RawCommand::ConversationDelete { alias } => Command::ConversationDelete { alias },
        RawCommand::Provider { model_ref } => Command::Provider { model: model_ref },
        RawCommand::ProviderThinking { level } => Command::ProviderThinking { level },
        RawCommand::ProviderConfig { provider } => Command::ProviderConfig { provider },
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
        assert_eq!(
            parse(&["tau", "mu", "-2"]).command,
            Command::MessageUndo { offset: -2 }
        );
        assert_eq!(
            parse(&["tau", "mu"]).command,
            Command::MessageUndo { offset: -1 }
        );
    }

    #[test]
    fn parses_conversation_commands() {
        assert_eq!(
            parse(&["tau", "c", "-a", "Feature work phase 2"]).command,
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
            parse(&["tau", "cf", "-2", "-a", "forked"]).command,
            Command::ConversationFork {
                offset: Some(-2),
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
        assert_eq!(
            parse(&["tau", "provider-config", "anthropic-api"]).command,
            Command::ProviderConfig {
                provider: Some(ConfigurableProvider::AnthropicApi)
            }
        );
        assert_eq!(
            parse(&["tau", "provider-config"]).command,
            Command::ProviderConfig { provider: None }
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
            parse(&["tau", "-r", "-w", "cf", "-a", "next"]).modifiers,
            Modifiers {
                read_only: true,
                writes_allowed: true,
                alias: Some("next".to_string()),
                ..Modifiers::default()
            }
        );
    }

    #[test]
    fn parses_status_when_no_command_is_given() {
        assert_eq!(parse(&["tau"]).command, Command::Status);
    }

    #[test]
    fn rejects_invalid_thinking_level() {
        assert!(parse_cli_from(["tau", "pt", "extreme"]).is_err());
    }

    #[test]
    fn rejects_unsupported_modifiers() {
        assert!(parse_cli_from(["tau", "-a", "not-for-list", "pl"]).is_err());
        assert!(parse_cli_from(["tau", "-j", "cf"]).is_err());
        assert!(parse_cli_from(["tau", "-r", "q", "What?"]).is_err());
        assert!(parse_cli_from(["tau", "-y", "ch"]).is_err());
        assert!(parse_cli_from(["tau", "-j", "provider-config", "anthropic-api"]).is_err());
    }
}
