use crate::context::{TauContext, ToolRegistrationError};

pub mod bash;
pub mod edit_file;
pub mod read_file;
pub mod write_file;

pub fn register_builtin_tools(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    bash::register(context)?;
    read_file::register(context)?;
    edit_file::register(context)?;
    write_file::register(context)?;
    Ok(())
}
