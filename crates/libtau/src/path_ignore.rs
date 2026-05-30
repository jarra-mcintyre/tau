use std::{
    fs, io,
    path::{Path, PathBuf},
};

use regex::Regex;

const IGNORE_FILES: &[&str] = &[".gitignore", ".hgignore", ".svnignore"];

#[derive(Debug, Clone, Default)]
pub struct IgnoreRules {
    rules: Vec<IgnoreRule>,
}

#[derive(Debug, Clone)]
struct IgnoreRule {
    base: PathBuf,
    pattern: Pattern,
    negated: bool,
    directory_only: bool,
}

#[derive(Debug, Clone)]
enum Pattern {
    Basename(Regex),
    Path(Regex),
    Regex(Regex),
}

impl IgnoreRules {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn from_directory_chain(directory: &Path) -> Result<Self, io::Error> {
        let root = repository_root(directory).unwrap_or(directory);
        let mut chain = directory
            .ancestors()
            .take_while(|ancestor| *ancestor != root)
            .collect::<Vec<_>>();
        chain.push(root);
        chain.reverse();

        let mut rules = Self::new();
        for directory in chain {
            rules.load_from_directory(directory)?;
        }

        Ok(rules)
    }

    pub fn with_rules_from_directory(&self, directory: &Path) -> Result<Self, io::Error> {
        let mut rules = self.clone();
        rules.load_from_directory(directory)?;
        Ok(rules)
    }

    pub fn is_ignored(&self, path: &Path, is_directory: bool) -> bool {
        let mut ignored = false;

        for rule in &self.rules {
            if rule.matches(path, is_directory) {
                ignored = !rule.negated;
            }
        }

        ignored
    }

    fn load_from_directory(&mut self, directory: &Path) -> Result<(), io::Error> {
        for file_name in IGNORE_FILES {
            let path = directory.join(file_name);
            match fs::read_to_string(&path) {
                Ok(contents) => self.parse_file(directory, file_name, &contents),
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => return Err(error),
            }
        }

        Ok(())
    }

    fn parse_file(&mut self, directory: &Path, file_name: &str, contents: &str) {
        let mut hg_syntax = HgSyntax::Regexp;

        for line in contents.lines() {
            let Some(mut pattern) = normalize_line(line) else {
                continue;
            };

            if file_name == ".hgignore" {
                if let Some(syntax) = pattern.strip_prefix("syntax:") {
                    hg_syntax = if syntax.trim() == "glob" {
                        HgSyntax::Glob
                    } else {
                        HgSyntax::Regexp
                    };
                    continue;
                }
            }

            let negated = pattern.starts_with('!');
            if negated {
                pattern.remove(0);
            } else if pattern.starts_with("\\!") {
                pattern.remove(0);
            }

            if pattern.is_empty() {
                continue;
            }

            let directory_only = pattern.ends_with('/');
            while pattern.ends_with('/') {
                pattern.pop();
            }

            if pattern.is_empty() {
                continue;
            }

            let pattern = if file_name == ".hgignore" && hg_syntax == HgSyntax::Regexp {
                match Regex::new(&pattern) {
                    Ok(regex) => Pattern::Regex(regex),
                    Err(_) => continue,
                }
            } else {
                let rooted = pattern.starts_with('/');
                if rooted {
                    pattern.remove(0);
                }

                let has_slash = pattern.contains('/');
                let regex = glob_regex(&pattern);
                match Regex::new(&regex) {
                    Ok(regex) if has_slash || rooted => Pattern::Path(regex),
                    Ok(regex) => Pattern::Basename(regex),
                    Err(_) => continue,
                }
            };

            self.rules.push(IgnoreRule {
                base: directory.to_path_buf(),
                pattern,
                negated,
                directory_only,
            });
        }
    }
}

impl IgnoreRule {
    fn matches(&self, path: &Path, is_directory: bool) -> bool {
        if self.directory_only && !is_directory {
            return false;
        }

        let relative_path = match path.strip_prefix(&self.base) {
            Ok(path) => path,
            Err(_) => return false,
        };
        let relative_path = path_to_string(relative_path);
        let basename = path.file_name().map(|name| name.to_string_lossy());

        match &self.pattern {
            Pattern::Basename(regex) => basename.is_some_and(|name| regex.is_match(&name)),
            Pattern::Path(regex) => regex.is_match(&relative_path),
            Pattern::Regex(regex) => regex.is_match(&relative_path),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum HgSyntax {
    Glob,
    Regexp,
}

fn repository_root(directory: &Path) -> Option<&Path> {
    directory.ancestors().find(|ancestor| {
        [".git", ".hg", ".svn"]
            .iter()
            .any(|name| ancestor.join(name).exists())
    })
}

fn normalize_line(line: &str) -> Option<String> {
    let line = line.trim_end_matches('\r').trim_end();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }

    if let Some(line) = line.strip_prefix("\\#") {
        return Some(format!("#{line}"));
    }

    Some(line.to_string())
}

fn glob_regex(pattern: &str) -> String {
    let mut regex = String::from("^");
    let mut chars = pattern.chars().peekable();

    while let Some(character) = chars.next() {
        match character {
            '*' if chars.peek() == Some(&'*') => {
                chars.next();
                if chars.peek() == Some(&'/') {
                    chars.next();
                    regex.push_str("(?:.*/)?");
                } else {
                    regex.push_str(".*");
                }
            }
            '*' => regex.push_str("[^/]*"),
            '?' => regex.push_str("[^/]"),
            '[' => {
                regex.push('[');
                if chars.peek() == Some(&'!') {
                    chars.next();
                    regex.push('^');
                }
                for character in chars.by_ref() {
                    regex.push(character);
                    if character == ']' {
                        break;
                    }
                }
            }
            '\\' => {
                if let Some(character) = chars.next() {
                    regex.push_str(&regex::escape(&character.to_string()));
                }
            }
            character => regex.push_str(&regex::escape(&character.to_string())),
        }
    }

    regex.push('$');
    regex
}

fn path_to_string(path: &Path) -> String {
    path.components()
        .map(|component| component.as_os_str().to_string_lossy())
        .collect::<Vec<_>>()
        .join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_directory(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!(
            "tau-ignore-rules-test-{}-{name}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&path);
        fs::create_dir_all(&path).unwrap();
        path
    }

    #[test]
    fn gitignore_matches_basename_patterns() {
        let root = test_directory("git-basename");
        fs::write(root.join(".gitignore"), "target\n*.log\n").unwrap();
        let rules = IgnoreRules::new().with_rules_from_directory(&root).unwrap();

        assert!(rules.is_ignored(&root.join("target"), true));
        assert!(rules.is_ignored(&root.join("nested").join("debug.log"), false));
        assert!(!rules.is_ignored(&root.join("debug.txt"), false));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn later_negated_patterns_unignore_paths() {
        let root = test_directory("negation");
        fs::write(root.join(".gitignore"), "*.log\n!important.log\n").unwrap();
        let rules = IgnoreRules::new().with_rules_from_directory(&root).unwrap();

        assert!(rules.is_ignored(&root.join("debug.log"), false));
        assert!(!rules.is_ignored(&root.join("important.log"), false));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn hgignore_supports_regexp_and_glob_syntax() {
        let root = test_directory("hg");
        fs::write(root.join(".hgignore"), "target$\nsyntax: glob\n*.tmp\n").unwrap();
        let rules = IgnoreRules::new().with_rules_from_directory(&root).unwrap();

        assert!(rules.is_ignored(&root.join("target"), true));
        assert!(rules.is_ignored(&root.join("scratch.tmp"), false));
        assert!(!rules.is_ignored(&root.join("scratch.txt"), false));

        fs::remove_dir_all(root).unwrap();
    }
}
