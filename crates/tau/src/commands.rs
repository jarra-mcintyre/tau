use std::{error::Error, path::PathBuf};

use crate::{
    cli::{CliInvocation, Command},
    output::OutputStyle,
};

mod conversation;
mod message;
mod provider;

pub(crate) async fn dispatch(
    invocation: CliInvocation,
    output: OutputStyle,
) -> Result<(), Box<dyn Error>> {
    let modifiers = invocation.modifiers;
    match invocation.command {
        Command::Message(command) => message::dispatch(modifiers, command, output).await,
        Command::Conversation(command) => conversation::dispatch(modifiers, command),
        Command::Provider(command) => provider::dispatch(modifiers, command).await,
        Command::Version => {
            println!("{}", env!("CARGO_PKG_VERSION"));
            Ok(())
        }
        other => Err(format!("command not implemented yet: {other:?}").into()),
    }
}

pub(super) fn state_path() -> Result<PathBuf, Box<dyn Error>> {
    let Some(home) = std::env::var_os("HOME") else {
        return Err("cannot open Tau state because HOME is not set".into());
    };
    Ok(PathBuf::from(home).join(".tau/state.db"))
}
