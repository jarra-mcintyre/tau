use std::path::{Path, PathBuf};

use ignore::{DirEntry, WalkBuilder};

pub fn find_files(
    directory: &Path,
    include_hidden: bool,
    include_ignored: bool,
) -> impl Iterator<Item = Result<DirEntry, ignore::Error>> {
    let mut builder = WalkBuilder::new(directory);
    builder
        .hidden(false)
        .ignore(!include_ignored)
        .git_ignore(!include_ignored)
        .git_global(!include_ignored)
        .git_exclude(!include_ignored)
        .require_git(false)
        .add_custom_ignore_filename(".hgignore")
        .add_custom_ignore_filename(".svnignore");

    if !include_hidden {
        builder.filter_entry(|entry| !is_hidden_directory(entry));
    }

    builder.build()
}

fn is_hidden_directory(entry: &DirEntry) -> bool {
    entry
        .file_type()
        .is_some_and(|file_type| file_type.is_dir())
        && entry
            .file_name()
            .to_str()
            .is_some_and(|name| name.starts_with('.') && name != ".")
}

pub fn error_path(_error: &ignore::Error) -> PathBuf {
    PathBuf::from(".")
}
