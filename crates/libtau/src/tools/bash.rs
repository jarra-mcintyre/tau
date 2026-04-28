use std::{
    fs,
    io::{self, Read, Write},
    path::PathBuf,
    process::{Command, Stdio},
    thread,
    time::{Duration, SystemTime},
};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::context::{
    TauContext, ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError,
};

pub const NAME: &str = "bash";
pub const DESCRIPTION: &str = "Run a command in a bash shell (with an optional timeout).  Both stdout and stderr are truncated to the last 2000 lines or 50 KiB. Full stream output is saved to a file when truncated";

const MAX_OUTPUT_LINES: usize = 2000;
const MAX_OUTPUT_BYTES: usize = 50 * 1024;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct BashInput {
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
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
    pub stdout: BashStreamOutput,
    pub stderr: BashStreamOutput,
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
    ToolDefinition::new::<BashInput>(NAME, DESCRIPTION, callback)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: BashInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let output = bash(input);
    let value = serde_json::to_value(output)
        .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))?;
    Ok(ToolOutput::json(value))
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

    let stdout = child.stdout.take().expect("stdout was piped");
    let stderr = child.stderr.take().expect("stderr was piped");
    let stdout_handle = thread::spawn(move || read_stream("stdout", stdout));
    let stderr_handle = thread::spawn(move || read_stream("stderr", stderr));

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
                    // FIXME(FUTURE): This might leave dependent processes
                    let _ = child.kill();
                    break child.wait().ok();
                }
                thread::sleep(Duration::from_millis(25));
            }
            Err(error) => return error_output(BashError::new(BashErrorKind::Other, error)),
        }
    };

    let (stdout, stdout_error) = join_stream(stdout_handle);
    let (stderr, stderr_error) = join_stream(stderr_handle);
    let error = stdout_error.or(stderr_error).or(timeout_error);

    let exit_code = exit_status.and_then(|status| status.code());
    let status = if timed_out {
        BashStatus::TimedOut
    } else if exit_status.is_some_and(|status| status.success()) {
        BashStatus::Success
    } else {
        BashStatus::Error
    };

    BashOutput {
        status,
        exit_code,
        stdout,
        stderr,
        error,
    }
}

fn read_stream(stream_name: &'static str, mut pipe: impl Read) -> io::Result<BashStreamOutput> {
    let mut full_output = tempfile::Builder::new()
        .prefix(&format!("tau-bash-{stream_name}-"))
        .suffix(".txt")
        .tempfile_in(tau_home_dir()?)?;
    let mut tail = TailBuffer::new();
    let mut chunk = [0_u8; 8192];

    loop {
        let bytes_read = pipe.read(&mut chunk)?;
        if bytes_read == 0 {
            break;
        }

        let bytes = &chunk[..bytes_read];
        full_output.write_all(bytes)?;
        tail.push(bytes);
    }

    if tail.truncated {
        let (_file, temp_path) = full_output.keep().map_err(|error| error.error)?;
        let path = move_output_to_working_directory(&temp_path, stream_name)?;
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

fn tau_home_dir() -> io::Result<PathBuf> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "HOME is not set"))?;
    let path = home.join(".tau");
    fs::create_dir_all(&path)?;
    Ok(path)
}

fn move_output_to_working_directory(
    temp_path: &std::path::Path,
    stream_name: &str,
) -> io::Result<PathBuf> {
    let mut output_file = tempfile::Builder::new()
        .prefix(&format!("tau-bash-{stream_name}-"))
        .suffix(".txt")
        .tempfile_in(std::env::current_dir()?)?;
    let mut temp_file = fs::File::open(temp_path)?;
    io::copy(&mut temp_file, &mut output_file)?;
    fs::remove_file(temp_path)?;

    let (_file, path) = output_file.keep().map_err(|error| error.error)?;
    Ok(path)
}

fn join_stream(
    handle: thread::JoinHandle<io::Result<BashStreamOutput>>,
) -> (BashStreamOutput, Option<BashError>) {
    match handle.join() {
        Ok(Ok(output)) => (output, None),
        Ok(Err(error)) => (
            BashStreamOutput::default(),
            Some(BashError::new(BashErrorKind::Io, error)),
        ),
        Err(_) => (
            BashStreamOutput::default(),
            Some(BashError::new(
                BashErrorKind::Other,
                "stream reader thread panicked",
            )),
        ),
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
        stdout: BashStreamOutput::default(),
        stderr: BashStreamOutput::default(),
        error: Some(error),
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
        assert_eq!(output.stdout.output.trim(), "hello-Tau");
        assert_eq!(output.stderr.output, "");
        assert!(!output.stdout.truncated);
        assert!(!output.stderr.truncated);
    }

    #[test]
    fn returns_non_zero_exit_status() {
        let output = bash(BashInput {
            command: "echo nope >&2; exit 7".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Error);
        assert_eq!(output.exit_code, Some(7));
        assert!(output.stderr.output.contains("nope"));
        assert_eq!(output.stdout.output, "");
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
        assert!(output.stdout.truncated);
        assert!(!output.stderr.truncated);
        assert!(output.stdout.output.len() <= MAX_OUTPUT_BYTES);
        let path = output.stdout.full_output_path.expect("full output path");
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("tau-bash-stdout-")
        );
        let full = std::fs::read_to_string(&path).unwrap();
        assert!(full.len() > MAX_OUTPUT_BYTES);
        std::fs::remove_file(path).unwrap();
    }
}
