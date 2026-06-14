use std::{ffi::OsString, path::PathBuf};

use clap::{ArgAction, CommandFactory, Parser, Subcommand};

use crate::provider_config::ConfigurableProvider;

#[derive(Debug, Clone)]
pub struct CliInvocation {
    pub modifiers: Modifiers,
    pub command: Command,
    pub(crate) definition: &'static CommandDefinition,
}

impl PartialEq for CliInvocation {
    fn eq(&self, other: &Self) -> bool {
        self.modifiers == other.modifiers && self.command == other.command
    }
}

impl Eq for CliInvocation {}

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
    Message(MessageCommand),
    Status,
    Conversation(ConversationCommand),
    Provider(ProviderCommand),
    Query { query: String },
    Prompt { prompt: String },
    File { path: PathBuf },
    Version,
    Help { question: Option<String> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MessageCommand {
    Message { contents: Option<String> },
    Add { contents: Option<String> },
    Include { paths: Vec<PathBuf> },
    Paste,
    Reference { paths: Vec<PathBuf> },
    Send,
    Echo,
    Clear,
    Undo { offset: isize },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversationCommand {
    Create {
        alias: Option<String>,
    },
    Alias {
        alias: String,
    },
    History,
    Compact,
    Fork {
        offset: Option<isize>,
        alias: Option<String>,
    },
    Switch {
        alias: String,
    },
    List,
    Delete {
        alias: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProviderCommand {
    Select {
        model: String,
    },
    Thinking {
        level: ThinkingLevel,
    },
    Config {
        provider: Option<ConfigurableProvider>,
    },
    List,
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

    #[command(name = "status")]
    Status,
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

        let (command, definition) = if top_level_command_count == 0 {
            (Command::Status, &STATUS_COMMAND)
        } else if let Some(query) = raw.query {
            (Command::Query { query }, &QUERY_COMMAND)
        } else if let Some(prompt) = raw.prompt {
            (Command::Prompt { prompt }, &PROMPT_COMMAND)
        } else if let Some(path) = raw.file {
            (Command::File { path }, &FILE_COMMAND)
        } else if raw.version {
            (Command::Version, &VERSION_COMMAND)
        } else if let Some(question) = raw.help {
            (Command::Help { question }, &HELP_COMMAND)
        } else {
            command_from_raw(raw.command.expect("checked above"), raw.alias)
        };

        validate_modifiers(&modifiers, &command, definition.allowed_modifiers)?;

        Ok(Self {
            modifiers,
            command,
            definition,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct CommandDefinition {
    pub(crate) name: &'static str,
    pub(crate) aliases: &'static [&'static str],
    pub(crate) usage: &'static str,
    pub(crate) summary: &'static str,
    pub(crate) category: CommandCategory,
    pub(crate) allowed_modifiers: &'static [Modifier],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CommandCategory {
    General,
    Message,
    Conversation,
    Provider,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Modifier {
    Yes,
    Conversation,
    Model,
    ReadOnly,
    WritesAllowed,
    Json,
    Alias,
}

const STATUS_COMMAND: CommandDefinition = CommandDefinition {
    name: "status",
    aliases: &[],
    usage: "tau status",
    summary: "Show the current Tau status.",
    category: CommandCategory::General,
    allowed_modifiers: &[],
};

const QUERY_COMMAND: CommandDefinition = CommandDefinition {
    name: "query",
    aliases: &[],
    usage: "tau query <QUERY>",
    summary: "Ask a general question with no project context.",
    category: CommandCategory::General,
    allowed_modifiers: &[Modifier::Model],
};

const PROMPT_COMMAND: CommandDefinition = CommandDefinition {
    name: "prompt",
    aliases: &[],
    usage: "tau prompt <PROMPT>",
    summary: "Execute a prompt with project context.",
    category: CommandCategory::General,
    allowed_modifiers: &[Modifier::Model, Modifier::ReadOnly],
};

const FILE_COMMAND: CommandDefinition = CommandDefinition {
    name: "file",
    aliases: &[],
    usage: "tau file <PATH>",
    summary: "Execute a prompt read from a file.",
    category: CommandCategory::General,
    allowed_modifiers: &[Modifier::Model, Modifier::ReadOnly],
};

const VERSION_COMMAND: CommandDefinition = CommandDefinition {
    name: "version",
    aliases: &[],
    usage: "tau version",
    summary: "Show the current Tau version.",
    category: CommandCategory::General,
    allowed_modifiers: &[],
};

const HELP_COMMAND: CommandDefinition = CommandDefinition {
    name: "help",
    aliases: &["h"],
    usage: "tau help [QUESTION]",
    summary: "Show help, optionally asking a question about Tau.",
    category: CommandCategory::General,
    allowed_modifiers: &[],
};

const MESSAGE_COMMAND: CommandDefinition = CommandDefinition {
    name: "message",
    aliases: &["m"],
    usage: "tau message [CONTENTS]...",
    summary: "Set the current message contents.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_ADD_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-add",
    aliases: &["ma"],
    usage: "tau message-add [CONTENTS]...",
    summary: "Append text to the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_INCLUDE_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-include",
    aliases: &["mi"],
    usage: "tau message-include [PATH]...",
    summary: "Include files in the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_PASTE_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-paste",
    aliases: &["mp"],
    usage: "tau message-paste",
    summary: "Paste clipboard contents into the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_REFERENCE_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-reference",
    aliases: &["mr"],
    usage: "tau message-reference [PATH]...",
    summary: "Reference files from the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_SEND_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-send",
    aliases: &["ms"],
    usage: "tau message-send",
    summary: "Send the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_ECHO_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-echo",
    aliases: &["me"],
    usage: "tau message-echo",
    summary: "Print the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation],
};

const MESSAGE_CLEAR_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-clear",
    aliases: &["mc"],
    usage: "tau message-clear",
    summary: "Clear the current message.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation, Modifier::Yes],
};

const MESSAGE_UNDO_COMMAND: CommandDefinition = CommandDefinition {
    name: "message-undo",
    aliases: &["mu"],
    usage: "tau message-undo [OFFSET]",
    summary: "Undo messages from the current conversation.",
    category: CommandCategory::Message,
    allowed_modifiers: &[Modifier::Conversation, Modifier::Yes],
};

const CONVERSATION_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation",
    aliases: &["c"],
    usage: "tau conversation",
    summary: "Create a conversation.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[Modifier::Alias, Modifier::ReadOnly],
};

const CONVERSATION_ALIAS_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-alias",
    aliases: &["ca"],
    usage: "tau conversation-alias <ALIAS>",
    summary: "Rename the current conversation.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[],
};

const CONVERSATION_HISTORY_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-history",
    aliases: &["ch"],
    usage: "tau conversation-history",
    summary: "Show the current conversation history.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[Modifier::Json],
};

const CONVERSATION_COMPACT_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-compact",
    aliases: &["cc"],
    usage: "tau conversation-compact",
    summary: "Compact the current conversation.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[Modifier::Yes],
};

const CONVERSATION_FORK_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-fork",
    aliases: &["cf"],
    usage: "tau conversation-fork [OFFSET]",
    summary: "Fork the current conversation.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[Modifier::Alias, Modifier::ReadOnly, Modifier::WritesAllowed],
};

const CONVERSATION_SWITCH_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-switch",
    aliases: &["cs"],
    usage: "tau conversation-switch <ALIAS>",
    summary: "Switch to another conversation.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[],
};

const CONVERSATION_LIST_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-list",
    aliases: &["cl"],
    usage: "tau conversation-list",
    summary: "List conversations.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[Modifier::Json],
};

const CONVERSATION_DELETE_COMMAND: CommandDefinition = CommandDefinition {
    name: "conversation-delete",
    aliases: &["cd"],
    usage: "tau conversation-delete <ALIAS>",
    summary: "Delete a conversation.",
    category: CommandCategory::Conversation,
    allowed_modifiers: &[Modifier::Yes],
};

const PROVIDER_COMMAND: CommandDefinition = CommandDefinition {
    name: "provider",
    aliases: &["p"],
    usage: "tau provider <PROVIDER/MODEL>",
    summary: "Select the current provider and model.",
    category: CommandCategory::Provider,
    allowed_modifiers: &[Modifier::Yes],
};

const PROVIDER_THINKING_COMMAND: CommandDefinition = CommandDefinition {
    name: "provider-thinking",
    aliases: &["pt"],
    usage: "tau provider-thinking <LEVEL>",
    summary: "Set the current thinking level.",
    category: CommandCategory::Provider,
    allowed_modifiers: &[],
};

const PROVIDER_CONFIG_COMMAND: CommandDefinition = CommandDefinition {
    name: "provider-config",
    aliases: &[],
    usage: "tau provider-config [PROVIDER]",
    summary: "Configure a provider.",
    category: CommandCategory::Provider,
    allowed_modifiers: &[],
};

const PROVIDER_LIST_COMMAND: CommandDefinition = CommandDefinition {
    name: "provider-list",
    aliases: &["pl"],
    usage: "tau provider-list",
    summary: "List configured providers.",
    category: CommandCategory::Provider,
    allowed_modifiers: &[Modifier::Json],
};

fn validate_modifiers(
    modifiers: &Modifiers,
    command: &Command,
    allowed: &[Modifier],
) -> Result<(), clap::Error> {
    if modifiers.yes && !allowed.contains(&Modifier::Yes) {
        return Err(unsupported_modifier("--yes/-y", command));
    }
    if modifiers.conversation.is_some() && !allowed.contains(&Modifier::Conversation) {
        return Err(unsupported_modifier("--conversation/-c", command));
    }
    if modifiers.model.is_some() && !allowed.contains(&Modifier::Model) {
        return Err(unsupported_modifier("--model/-m", command));
    }
    if modifiers.read_only && !allowed.contains(&Modifier::ReadOnly) {
        return Err(unsupported_modifier("--read-only/-r", command));
    }
    if modifiers.writes_allowed && !allowed.contains(&Modifier::WritesAllowed) {
        return Err(unsupported_modifier("--writes-allowed/-w", command));
    }
    if modifiers.json && !allowed.contains(&Modifier::Json) {
        return Err(unsupported_modifier("--json/-j", command));
    }
    if modifiers.alias.is_some() && !allowed.contains(&Modifier::Alias) {
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

fn command_from_raw(
    command: RawCommand,
    alias_modifier: Option<String>,
) -> (Command, &'static CommandDefinition) {
    match command {
        RawCommand::Message { contents } => (
            Command::Message(MessageCommand::Message {
                contents: join_optional(contents),
            }),
            &MESSAGE_COMMAND,
        ),
        RawCommand::MessageAdd { contents } => (
            Command::Message(MessageCommand::Add {
                contents: join_optional(contents),
            }),
            &MESSAGE_ADD_COMMAND,
        ),
        RawCommand::MessageInclude { paths } => (
            Command::Message(MessageCommand::Include { paths }),
            &MESSAGE_INCLUDE_COMMAND,
        ),
        RawCommand::MessagePaste => (
            Command::Message(MessageCommand::Paste),
            &MESSAGE_PASTE_COMMAND,
        ),
        RawCommand::MessageReference { paths } => (
            Command::Message(MessageCommand::Reference { paths }),
            &MESSAGE_REFERENCE_COMMAND,
        ),
        RawCommand::MessageSend => (
            Command::Message(MessageCommand::Send),
            &MESSAGE_SEND_COMMAND,
        ),
        RawCommand::MessageEcho => (
            Command::Message(MessageCommand::Echo),
            &MESSAGE_ECHO_COMMAND,
        ),
        RawCommand::MessageClear => (
            Command::Message(MessageCommand::Clear),
            &MESSAGE_CLEAR_COMMAND,
        ),
        RawCommand::MessageUndo { offset } => (
            Command::Message(MessageCommand::Undo {
                offset: offset.unwrap_or(-1),
            }),
            &MESSAGE_UNDO_COMMAND,
        ),
        RawCommand::Conversation => (
            Command::Conversation(ConversationCommand::Create {
                alias: alias_modifier,
            }),
            &CONVERSATION_COMMAND,
        ),
        RawCommand::ConversationAlias { alias } => (
            Command::Conversation(ConversationCommand::Alias { alias }),
            &CONVERSATION_ALIAS_COMMAND,
        ),
        RawCommand::ConversationHistory => (
            Command::Conversation(ConversationCommand::History),
            &CONVERSATION_HISTORY_COMMAND,
        ),
        RawCommand::ConversationCompact => (
            Command::Conversation(ConversationCommand::Compact),
            &CONVERSATION_COMPACT_COMMAND,
        ),
        RawCommand::ConversationFork { offset } => (
            Command::Conversation(ConversationCommand::Fork {
                offset,
                alias: alias_modifier,
            }),
            &CONVERSATION_FORK_COMMAND,
        ),
        RawCommand::ConversationSwitch { alias } => (
            Command::Conversation(ConversationCommand::Switch { alias }),
            &CONVERSATION_SWITCH_COMMAND,
        ),
        RawCommand::ConversationList => (
            Command::Conversation(ConversationCommand::List),
            &CONVERSATION_LIST_COMMAND,
        ),
        RawCommand::ConversationDelete { alias } => (
            Command::Conversation(ConversationCommand::Delete { alias }),
            &CONVERSATION_DELETE_COMMAND,
        ),
        RawCommand::Provider { model_ref } => (
            Command::Provider(ProviderCommand::Select { model: model_ref }),
            &PROVIDER_COMMAND,
        ),
        RawCommand::ProviderThinking { level } => (
            Command::Provider(ProviderCommand::Thinking { level }),
            &PROVIDER_THINKING_COMMAND,
        ),
        RawCommand::ProviderConfig { provider } => (
            Command::Provider(ProviderCommand::Config { provider }),
            &PROVIDER_CONFIG_COMMAND,
        ),
        RawCommand::ProviderList => (
            Command::Provider(ProviderCommand::List),
            &PROVIDER_LIST_COMMAND,
        ),
        RawCommand::Status => (Command::Status, &STATUS_COMMAND),
        RawCommand::Query { query } => (Command::Query { query }, &QUERY_COMMAND),
        RawCommand::Prompt { prompt } => (Command::Prompt { prompt }, &PROMPT_COMMAND),
        RawCommand::File { path } => (Command::File { path }, &FILE_COMMAND),
        RawCommand::Version => (Command::Version, &VERSION_COMMAND),
        RawCommand::Help { question } => (
            Command::Help {
                question: join_optional(question),
            },
            &HELP_COMMAND,
        ),
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

    fn invocation(modifiers: Modifiers, command: Command) -> CliInvocation {
        CliInvocation {
            modifiers,
            command,
            definition: &STATUS_COMMAND,
        }
    }

    #[test]
    fn parses_message_command_with_alias_and_contents() {
        assert_eq!(
            parse(&["tau", "m", "Also", "check", "tests"]),
            invocation(
                Modifiers::default(),
                Command::Message(MessageCommand::Message {
                    contents: Some("Also check tests".to_string())
                })
            )
        );
    }

    #[test]
    fn parses_message_without_contents_for_editor() {
        assert_eq!(
            parse(&["tau", "message"]),
            invocation(
                Modifiers::default(),
                Command::Message(MessageCommand::Message { contents: None })
            )
        );
    }

    #[test]
    fn parses_message_building_commands() {
        assert_eq!(
            parse(&["tau", "-c", "bug-1234", "mi", "a.rs", "image.png"]),
            invocation(
                Modifiers {
                    conversation: Some("bug-1234".to_string()),
                    ..Modifiers::default()
                },
                Command::Message(MessageCommand::Include {
                    paths: vec![PathBuf::from("a.rs"), PathBuf::from("image.png")]
                })
            )
        );

        assert_eq!(
            parse(&["tau", "mp"]).command,
            Command::Message(MessageCommand::Paste)
        );
        assert_eq!(
            parse(&["tau", "mr", "src/main.rs"]).command,
            Command::Message(MessageCommand::Reference {
                paths: vec![PathBuf::from("src/main.rs")]
            })
        );
        assert_eq!(
            parse(&["tau", "ms"]).command,
            Command::Message(MessageCommand::Send)
        );
        assert_eq!(
            parse(&["tau", "me"]).command,
            Command::Message(MessageCommand::Echo)
        );
        assert_eq!(
            parse(&["tau", "-y", "mc"]).command,
            Command::Message(MessageCommand::Clear)
        );
        assert_eq!(
            parse(&["tau", "mu", "-2"]).command,
            Command::Message(MessageCommand::Undo { offset: -2 })
        );
        assert_eq!(
            parse(&["tau", "mu"]).command,
            Command::Message(MessageCommand::Undo { offset: -1 })
        );
    }

    #[test]
    fn parses_conversation_commands() {
        assert_eq!(
            parse(&["tau", "c", "-a", "Feature work phase 2"]).command,
            Command::Conversation(ConversationCommand::Create {
                alias: Some("Feature work phase 2".to_string())
            })
        );
        assert_eq!(
            parse(&["tau", "ca", "My awesome feature"]).command,
            Command::Conversation(ConversationCommand::Alias {
                alias: "My awesome feature".to_string()
            })
        );
        assert_eq!(
            parse(&["tau", "ch"]).command,
            Command::Conversation(ConversationCommand::History)
        );
        assert_eq!(
            parse(&["tau", "cc"]).command,
            Command::Conversation(ConversationCommand::Compact)
        );
        assert_eq!(
            parse(&["tau", "cf", "-2", "-a", "forked"]).command,
            Command::Conversation(ConversationCommand::Fork {
                offset: Some(-2),
                alias: Some("forked".to_string())
            })
        );
        assert_eq!(
            parse(&["tau", "cs", "bug-1234"]).command,
            Command::Conversation(ConversationCommand::Switch {
                alias: "bug-1234".to_string()
            })
        );
        assert_eq!(
            parse(&["tau", "cl"]).command,
            Command::Conversation(ConversationCommand::List)
        );
        assert_eq!(
            parse(&["tau", "cd", "old-feature"]).command,
            Command::Conversation(ConversationCommand::Delete {
                alias: "old-feature".to_string()
            })
        );
    }

    #[test]
    fn parses_provider_commands() {
        assert_eq!(
            parse(&["tau", "-y", "p", "anthropic/opus-4.7"]),
            invocation(
                Modifiers {
                    yes: true,
                    ..Modifiers::default()
                },
                Command::Provider(ProviderCommand::Select {
                    model: "anthropic/opus-4.7".to_string()
                })
            )
        );
        assert_eq!(
            parse(&["tau", "pt", "xhigh"]).command,
            Command::Provider(ProviderCommand::Thinking {
                level: ThinkingLevel::XHigh
            })
        );
        assert_eq!(
            parse(&["tau", "provider-config", "anthropic-api"]).command,
            Command::Provider(ProviderCommand::Config {
                provider: Some(ConfigurableProvider::AnthropicApi)
            })
        );
        assert_eq!(
            parse(&["tau", "provider-config"]).command,
            Command::Provider(ProviderCommand::Config { provider: None })
        );
        assert_eq!(
            parse(&["tau", "pl"]).command,
            Command::Provider(ProviderCommand::List)
        );
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
            invocation(
                Modifiers {
                    model: Some("anthropic/haiku-4.5".to_string()),
                    read_only: true,
                    ..Modifiers::default()
                },
                Command::Prompt {
                    prompt: "Document the API".to_string()
                }
            )
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
        assert_eq!(parse(&["tau", "status"]).command, Command::Status);
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
