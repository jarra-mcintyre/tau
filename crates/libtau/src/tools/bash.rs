//! This implements the bash tool. The principal behaviours are:
//! - Runs command in bash shell. Buffering output stdout and stderr into a temp file
//! - If the file contents are not overly long, then returns the contents as is
//! - Otherwise moves the temp file to a permanent location and returns a message pointing the model towards it

use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    thread,
    time::{Duration, SystemTime},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::TauContext,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "bash";
pub const DESCRIPTION: &str = "Run a command in a bash shell (with an optional timeout). Output is truncated to the last 2000 lines or 50 KiB. Full output is saved to a file when truncated";

const MAX_OUTPUT_LINES: usize = 2000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct BashInput {
    /// Command to run
    pub command: String,
    /// Timeout in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BashStatus {
    Success,
    Error,
    TimedOut,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct BashOutput {
    pub status: BashStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_code: Option<i32>,
    pub output: BashStreamOutput,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<BashError>,
}

#[derive(Debug, Clone, Default, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct BashStreamOutput {
    pub output: String,
    pub truncated: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub full_output_path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct BashError {
    pub kind: BashErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum BashErrorKind {
    SpawnFailed,
    Io,
    TimedOut,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(definition()?)
}

pub fn definition() -> Result<ToolDefinition, ToolRegistrationError> {
    ToolDefinition::new::<BashInput>(NAME, DESCRIPTION, false, callback)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: BashInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    Ok(bash(input).into_tool_output())
}

pub fn bash(input: BashInput) -> BashOutput {
    let mut child = match Command::new("bash")
        .arg("-lc")
        .arg(&input.command)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    {
        Ok(child) => child,
        Err(error) => return error_output(BashError::new(BashErrorKind::SpawnFailed, error)),
    };

    let output = Arc::new(Mutex::new(Vec::new()));
    let sequence = Arc::new(AtomicUsize::new(0));
    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdout_handle = read_stream(stdout, Arc::clone(&output), Arc::clone(&sequence));
    let stderr_handle = read_stream(stderr, Arc::clone(&output), Arc::clone(&sequence));

    let timeout = input.timeout_seconds.map(Duration::from_secs);
    let start = SystemTime::now();
    let mut timed_out = false;
    let mut timeout_error = None;
    let exit_status = loop {
        match child.try_wait() {
            Ok(Some(status)) => break Some(status),
            Ok(None) => {
                if let Some(timeout) = timeout
                    && start.elapsed().unwrap_or_default() >= timeout
                {
                    timed_out = true;
                    timeout_error = Some(BashError {
                        kind: BashErrorKind::TimedOut,
                        message: format!(
                            "command exceeded timeout of {} seconds",
                            timeout.as_secs()
                        ),
                    });
                    let _ = child.kill();
                    break child.wait().ok();
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return error_output(BashError::new(BashErrorKind::Other, error)),
        }
    };

    let stdout_error = join_stream(stdout_handle);
    let stderr_error = join_stream(stderr_handle);
    let error = stdout_error.or(stderr_error).or(timeout_error);

    let exit_code = exit_status.and_then(|status| status.code());
    let status = if timed_out {
        BashStatus::TimedOut
    } else if exit_status.is_some_and(|status| status.success()) {
        BashStatus::Success
    } else {
        BashStatus::Error
    };

    let output = match build_stream_output(output) {
        Ok(output) => output,
        Err(error) => {
            return BashOutput {
                status: BashStatus::Error,
                exit_code,
                output: BashStreamOutput::default(),
                error: Some(BashError::new(BashErrorKind::Io, error)),
            };
        }
    };

    BashOutput {
        status,
        exit_code,
        output,
        error,
    }
}

fn read_stream(
    mut pipe: impl Read + Send + 'static,
    output: Arc<Mutex<Vec<StreamChunk>>>,
    sequence: Arc<AtomicUsize>,
) -> thread::JoinHandle<io::Result<()>> {
    thread::spawn(move || {
        let mut buffer = [0_u8; 8192];
        loop {
            let bytes_read = pipe.read(&mut buffer)?;
            if bytes_read == 0 {
                break;
            }

            output
                .lock()
                .expect("output mutex poisoned")
                .push(StreamChunk {
                    sequence: sequence.fetch_add(1, Ordering::SeqCst),
                    bytes: buffer[..bytes_read].to_vec(),
                });
        }
        Ok(())
    })
}

fn build_stream_output(output: Arc<Mutex<Vec<StreamChunk>>>) -> io::Result<BashStreamOutput> {
    let mut chunks = Arc::try_unwrap(output)
        .map_err(|_| io::Error::other("output stream is still shared"))?
        .into_inner()
        .map_err(|_| io::Error::other("output mutex poisoned"))?;
    chunks.sort_by_key(|chunk| chunk.sequence);

    let mut full_output = tempfile::Builder::new()
        .prefix("tau-bash-output-")
        .suffix(".txt")
        .tempfile_in(tau_home_dir()?)?;
    let mut tail = TailBuffer::new();

    for chunk in chunks {
        full_output.write_all(&chunk.bytes)?;
        tail.push(&chunk.bytes);
    }

    if tail.truncated {
        let (_file, temp_path) = full_output.keep().map_err(|error| error.error)?;
        let path = move_output_to_working_directory(&temp_path)?;
        Ok(BashStreamOutput {
            output: tail.output(),
            truncated: true,
            full_output_path: Some(path),
        })
    } else {
        Ok(BashStreamOutput {
            output: tail.output(),
            truncated: false,
            full_output_path: None,
        })
    }
}

#[derive(Debug)]
struct StreamChunk {
    sequence: usize,
    bytes: Vec<u8>,
}

fn tau_home_dir() -> io::Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set"))?;
    let path = home.join(".tau");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn move_output_to_working_directory(temp_path: &std::path::Path) -> io::Result<PathBuf> {
    let mut output_file = tempfile::Builder::new()
        .prefix("tau-bash-output-")
        .suffix(".txt")
        .tempfile_in(std::env::current_dir()?)?;
    let mut temp_file = fs::File::open(temp_path)?;
    io::copy(&mut temp_file, &mut output_file)?;
    fs::remove_file(temp_path)?;

    let (_file, path) = output_file.keep().map_err(|error| error.error)?;
    Ok(path)
}

fn join_stream(handle: thread::JoinHandle<io::Result<()>>) -> Option<BashError> {
    match handle.join() {
        Ok(Ok(())) => None,
        Ok(Err(error)) => Some(BashError::new(BashErrorKind::Io, error)),
        Err(_) => Some(BashError::new(
            BashErrorKind::Other,
            "stream reader thread panicked",
        )),
    }
}

#[derive(Debug, Default)]
struct TailBuffer {
    bytes: Vec<u8>,
    newline_count: usize,
    truncated: bool,
}

impl TailBuffer {
    fn new() -> Self {
        Self::default()
    }

    fn push(&mut self, bytes: &[u8]) {
        self.newline_count += bytes.iter().filter(|byte| **byte == b'\n').count();
        self.bytes.extend_from_slice(bytes);
        self.trim_bytes();
        self.trim_lines();
    }

    fn output(&self) -> String {
        String::from_utf8_lossy(&self.bytes).into_owned()
    }

    fn trim_bytes(&mut self) {
        if self.bytes.len() <= MAX_OUTPUT_BYTES {
            return;
        }

        let overflow = self.bytes.len() - MAX_OUTPUT_BYTES;
        self.newline_count -= self.bytes[..overflow]
            .iter()
            .filter(|byte| **byte == b'\n')
            .count();
        self.bytes.drain(..overflow);
        self.truncated = true;
    }

    fn trim_lines(&mut self) {
        while self.newline_count > MAX_OUTPUT_LINES {
            let Some(newline_index) = self.bytes.iter().position(|byte| *byte == b'\n') else {
                self.bytes.clear();
                self.newline_count = 0;
                self.truncated = true;
                return;
            };
            self.bytes.drain(..=newline_index);
            self.newline_count -= 1;
            self.truncated = true;
        }
    }
}

fn error_output(error: BashError) -> BashOutput {
    BashOutput {
        status: BashStatus::Error,
        exit_code: None,
        output: BashStreamOutput::default(),
        error: Some(error),
    }
}

impl BashOutput {
    fn into_tool_output(self) -> ToolOutput {
        let mut text = self.output.output;

        if self.output.truncated {
            let path = self
                .output
                .full_output_path
                .map(|path| path.display().to_string())
                .unwrap_or_else(|| "unknown".to_string());
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&format!(
                "(output was truncated; full output saved to {path})"
            ));
        }

        if let Some(error) = &self.error {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            text.push_str(&error.message);
        }

        if !matches!(self.status, BashStatus::Success) {
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
            let code = self.exit_code.unwrap_or(-1);
            text.push_str(&format!("(return code was {code})"));
        }

        if text.is_empty() {
            text = "done".to_string();
        }

        if matches!(self.status, BashStatus::Success) && self.error.is_none() {
            ToolOutput::text(text)
        } else {
            ToolOutput::error(text)
        }
    }
}

impl BashError {
    fn new(kind: BashErrorKind, error: impl ToString) -> Self {
        Self {
            kind,
            message: error.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_command_in_bash() {
        let output = bash(BashInput {
            command: "name=Tau; echo hello-$name".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Success);
        assert_eq!(output.exit_code, Some(0));
        assert_eq!(output.output.output.trim(), "hello-Tau");
        assert!(!output.output.truncated);
    }

    #[test]
    fn returns_non_zero_exit_status() {
        let output = bash(BashInput {
            command: "echo nope >&2; exit 7".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Error);
        assert_eq!(output.exit_code, Some(7));
        assert!(output.output.output.contains("nope"));
    }

    #[test]
    fn times_out() {
        let output = bash(BashInput {
            command: "sleep 2".to_string(),
            timeout_seconds: Some(0),
        });

        assert_eq!(output.status, BashStatus::TimedOut);
        assert_eq!(output.error.unwrap().kind, BashErrorKind::TimedOut);
    }

    #[test]
    fn keeps_tail_without_recounting_lines_from_scratch() {
        let mut tail = TailBuffer::new();
        for index in 0..(MAX_OUTPUT_LINES + 100) {
            tail.push(format!("line-{index}\n").as_bytes());
        }

        assert!(tail.truncated);
        assert!(tail.output().starts_with("line-100\n"));
        assert_eq!(tail.newline_count, MAX_OUTPUT_LINES);
    }

    #[test]
    fn truncates_and_saves_full_output() {
        let output = bash(BashInput {
            command: "python3 - <<'PY'\nprint('a' * 60000)\nPY".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Success);
        assert!(output.output.truncated);
        assert!(output.output.output.len() <= MAX_OUTPUT_BYTES);
        let path = output.output.full_output_path.expect("full output path");
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("tau-bash-output-")
        );
        let full = std::fs::read_to_string(&path).unwrap();
        assert!(full.len() > MAX_OUTPUT_BYTES);
        std::fs::remove_file(path).unwrap();
    }
}
