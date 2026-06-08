use std::error::Error;

use crate::{
    cli::{Modifiers, ProviderCommand},
    provider_config::ConfigurableProvider,
};

pub(super) async fn dispatch(
    modifiers: Modifiers,
    command: ProviderCommand,
) -> Result<(), Box<dyn Error>> {
    match command {
        ProviderCommand::Config { provider } => configure(provider).await,
        ProviderCommand::List => list(modifiers),
        other => Err(format!("command not implemented yet: {other:?}").into()),
    }
}

async fn configure(provider: Option<ConfigurableProvider>) -> Result<(), Box<dyn Error>> {
    crate::provider_config::configure_provider(provider).await
}

fn list(modifiers: Modifiers) -> Result<(), Box<dyn Error>> {
    crate::config::list_providers(modifiers.json)
}
