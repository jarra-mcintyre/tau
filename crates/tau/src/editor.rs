use std::{fs, process::Command};

pub(crate) fn edit_message() -> Result<String, Box<dyn std::error::Error>> {
    let editor = std::env::var_os("EDITOR").ok_or("EDITOR is not set")?;
    let path = std::env::temp_dir().join(format!("tau-message-{}.md", uuid::Uuid::new_v4()));
    fs::write(&path, "")?;
    let status = Command::new(editor).arg(&path).status()?;
    if !status.success() {
        return Err("editor exited unsuccessfully".into());
    }
    let message = fs::read_to_string(&path)?;
    let _ = fs::remove_file(path);
    Ok(message)
}
