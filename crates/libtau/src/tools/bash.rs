//! This implements the bash tool. The principal behaviours are:
//! - Runs command in bash shell. Buffering output stdout and stderr into a single temp file.
//!   The two streams are interleaved as they would show on the terminal.
//! - If the file contents are not overly long (less than max lines and max size), then returns the contents as is
//! - Otherwise moves the temp file to a permanent location and returns a message pointing the model towards it and
//!   informing the model of the size/number of lines in the file
//! - If the command returns an error code this is appended on the end as "(return code was -1)" or similar
//! - If the command does not output anything at all then the string "(Completed with no output)" is returned
//! - Large outputs are never read into memory.
//!
//! Future work:
//! - Currently this supports the bash shell. In the future I expect to rename this to the "shell" tool and support other shells
//! - Currently just does UTF8 encoding

use std::{
    fs::File,
    io::{Read, Write},
    path::PathBuf,
    process::{Child, Command, Stdio},
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tempfile::NamedTempFile;

use crate::{
    context::TauContext,
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "bash";
pub const DESCRIPTION: &str = "Run a command in a bash shell (with an optional timeout). \
stdout and stderr output are shown together. \
If the output is too long it will be automatically saved to a file.";

const MAX_OUTPUT_LINES: u64 = 2000;
const MAX_OUTPUT_BYTES: u64 = 50 * 1024;
const READ_BUFFER_BYTES: usize = 8192;
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct BashInput {
    /// Command to run
    pub command: String,
    /// Timeout in seconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_seconds: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BashOutput {
    status: BashStatus,
    exit_code: Option<i32>,
    output: CommandOutput,
    error: Option<BashError>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BashStatus {
    Success,
    Error,
    TimedOut,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CommandOutput {
    Inline(String),
    Saved {
        path: PathBuf,
        bytes: u64,
        lines: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct BashError {
    kind: BashErrorKind,
    message: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum BashErrorKind {
    SpawnFailed,
    Io,
    TimedOut,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(ToolDefinition::new::<BashInput>(
        NAME,
        DESCRIPTION,
        false,
        callback,
    )?)
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: BashInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    Ok(bash(input).into_tool_output())
}

pub fn bash(input: BashInput) -> BashOutput {
    match run_bash(input) {
        Ok(output) => output,
        Err(error) => error_output(error),
    }
}

fn run_bash(input: BashInput) -> Result<BashOutput, BashError> {
    let output_file = tempfile::Builder::new()
        .prefix("tau-bash-output-")
        .suffix(".txt")
        .tempfile_in(current_dir()?)
        .map_err(BashError::io)?;
    let output_writer = File::options()
        .write(true)
        .open(output_file.path())
        .map_err(BashError::io)?;

    let mut child = spawn_bash(&input.command)?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| BashError::other("bash stdout was not piped"))?;

    let capture = Arc::new(Mutex::new(OutputCapture::default()));
    let reader = read_output(stdout, output_writer, Arc::clone(&capture));

    let (status, timeout_error) = wait_for_child(&mut child, input.timeout_seconds)?;
    let reader_error = join_reader(reader);
    let capture = Arc::try_unwrap(capture)
        .map_err(|_| BashError::other("output capture is still shared"))?
        .into_inner()
        .map_err(|_| BashError::other("output capture mutex poisoned"))?;

    if let Some(error) = capture.error.or(reader_error) {
        return Ok(BashOutput {
            status: BashStatus::Error,
            exit_code: status.and_then(|status| status.code()),
            output: CommandOutput::Inline(String::new()),
            error: Some(error),
        });
    }

    let output = finish_output(output_file, capture.summary)?;
    let exit_code = status.and_then(|status| status.code());
    let bash_status = if timeout_error.is_some() {
        BashStatus::TimedOut
    } else if status.is_some_and(|status| status.success()) {
        BashStatus::Success
    } else {
        BashStatus::Error
    };

    Ok(BashOutput {
        status: bash_status,
        exit_code,
        output,
        error: timeout_error,
    })
}

fn spawn_bash(command: &str) -> Result<Child, BashError> {
    let mut process = Command::new("bash");
    process
        .arg("-lc")
        .arg(format!("exec 2>&1\n{command}"))
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());

    #[cfg(unix)]
    process.process_group(0);

    process.spawn().map_err(BashError::spawn_failed)
}

fn read_output(
    mut output: impl Read + Send + 'static,
    mut output_file: File,
    capture: Arc<Mutex<OutputCapture>>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let mut buffer = [0_u8; READ_BUFFER_BYTES];
        loop {
            let bytes_read = match output.read(&mut buffer) {
                Ok(0) => return,
                Ok(bytes_read) => bytes_read,
                Err(error) => {
                    capture_error(&capture, BashError::io(error));
                    return;
                }
            };

            if let Err(error) = output_file.write_all(&buffer[..bytes_read]) {
                capture_error(&capture, BashError::io(error));
                return;
            }

            capture
                .lock()
                .expect("output capture mutex poisoned")
                .summary
                .push(&buffer[..bytes_read]);
        }
    })
}

fn capture_error(capture: &Mutex<OutputCapture>, error: BashError) {
    capture.lock().expect("output capture mutex poisoned").error = Some(error);
}

fn wait_for_child(
    child: &mut Child,
    timeout_seconds: Option<u64>,
) -> Result<(Option<std::process::ExitStatus>, Option<BashError>), BashError> {
    let timeout = timeout_seconds.map(Duration::from_secs);
    let start = Instant::now();

    loop {
        match child.try_wait() {
            Ok(Some(status)) => return Ok((Some(status), None)),
            Ok(None) => {}
            Err(error) => return Err(BashError::other(error)),
        }

        if let Some(timeout) = timeout
            && start.elapsed() >= timeout
        {
            kill_child(child);
            let status = child.wait().ok();
            return Ok((
                status,
                Some(BashError {
                    kind: BashErrorKind::TimedOut,
                    message: format!("command exceeded timeout of {} seconds", timeout.as_secs()),
                }),
            ));
        }

        thread::sleep(WAIT_POLL_INTERVAL);
    }
}

fn kill_child(child: &mut Child) {
    #[cfg(unix)]
    {
        let process_group = format!("-{}", child.id());
        let _ = Command::new("kill")
            .arg("-TERM")
            .arg(process_group)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
    }

    let _ = child.kill();
}

fn join_reader(reader: thread::JoinHandle<()>) -> Option<BashError> {
    match reader.join() {
        Ok(()) => None,
        Err(_) => Some(BashError::other("output reader thread panicked")),
    }
}

fn finish_output(
    output_file: NamedTempFile,
    summary: OutputSummary,
) -> Result<CommandOutput, BashError> {
    if summary.too_large {
        let (_file, path) = output_file
            .keep()
            .map_err(|error| BashError::io(error.error))?;
        Ok(CommandOutput::Saved {
            path,
            bytes: summary.bytes,
            lines: summary.lines(),
        })
    } else {
        Ok(CommandOutput::Inline(summary.text()))
    }
}

fn current_dir() -> Result<PathBuf, BashError> {
    std::env::current_dir().map_err(BashError::io)
}

#[derive(Debug, Default)]
struct OutputCapture {
    summary: OutputSummary,
    error: Option<BashError>,
}

#[derive(Debug, Default)]
struct OutputSummary {
    inline: Vec<u8>,
    bytes: u64,
    newline_count: u64,
    last_byte: Option<u8>,
    too_large: bool,
}

impl OutputSummary {
    fn push(&mut self, bytes: &[u8]) {
        self.bytes += bytes.len() as u64;
        self.newline_count += bytes.iter().filter(|byte| **byte == b'\n').count() as u64;
        self.last_byte = bytes.last().copied().or(self.last_byte);

        if self.too_large {
            return;
        }

        self.inline.extend_from_slice(bytes);
        if self.bytes > MAX_OUTPUT_BYTES || self.lines() > MAX_OUTPUT_LINES {
            self.inline.clear();
            self.too_large = true;
        }
    }

    fn lines(&self) -> u64 {
        self.newline_count + u64::from(self.bytes > 0 && self.last_byte != Some(b'\n'))
    }

    fn text(self) -> String {
        String::from_utf8_lossy(&self.inline).into_owned()
    }
}

fn error_output(error: BashError) -> BashOutput {
    BashOutput {
        status: BashStatus::Error,
        exit_code: None,
        output: CommandOutput::Inline(String::new()),
        error: Some(error),
    }
}

impl BashOutput {
    fn into_tool_output(self) -> ToolOutput {
        let failed_tool_call = self.error.as_ref().is_some_and(|error| {
            matches!(
                error.kind,
                BashErrorKind::SpawnFailed | BashErrorKind::Io | BashErrorKind::Other
            )
        });
        let text = self.into_text();

        if failed_tool_call {
            ToolOutput::error(text)
        } else {
            ToolOutput::text(text)
        }
    }

    fn into_text(self) -> String {
        let mut text = match self.output {
            CommandOutput::Inline(output) if output.is_empty() => {
                "(Completed with no output)".to_string()
            }
            CommandOutput::Inline(output) => output,
            CommandOutput::Saved { path, bytes, lines } => format!(
                "Full output saved to: {}\nFile size: {bytes} bytes\nLines: {lines}",
                path.display()
            ),
        };

        if let Some(error) = self.error {
            append_line(&mut text, error.message);
        }

        if !matches!(self.status, BashStatus::Success) {
            append_line(
                &mut text,
                format!("(return code was {})", self.exit_code.unwrap_or(-1)),
            );
        }

        text
    }
}

fn append_line(text: &mut String, line: impl AsRef<str>) {
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(line.as_ref());
}

impl BashError {
    fn spawn_failed(error: impl ToString) -> Self {
        Self::new(BashErrorKind::SpawnFailed, error)
    }

    fn io(error: impl ToString) -> Self {
        Self::new(BashErrorKind::Io, error)
    }

    fn other(error: impl ToString) -> Self {
        Self::new(BashErrorKind::Other, error)
    }

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
    use crate::context::ContentPart;

    #[test]
    fn runs_command_in_bash() {
        let output = bash(BashInput {
            command: "name=Tau; echo hello-$name".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Success);
        assert_eq!(output.exit_code, Some(0));
        assert_eq!(
            output.output,
            CommandOutput::Inline("hello-Tau\n".to_string())
        );
    }

    #[test]
    fn captures_stdout_and_stderr_in_one_stream() {
        let output = bash(BashInput {
            command: "echo out; echo err >&2".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Success);
        assert_eq!(
            output.output,
            CommandOutput::Inline("out\nerr\n".to_string())
        );
    }

    #[test]
    fn returns_non_zero_exit_status() {
        let output = bash(BashInput {
            command: "echo nope >&2; exit 7".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Error);
        assert_eq!(output.exit_code, Some(7));
        assert_eq!(output.output, CommandOutput::Inline("nope\n".to_string()));
    }

    #[test]
    fn formats_empty_output_as_completed_with_no_output() {
        let output = BashOutput {
            status: BashStatus::Success,
            exit_code: Some(0),
            output: CommandOutput::Inline(String::new()),
            error: None,
        };

        assert_eq!(output.into_text(), "(Completed with no output)");
    }

    #[test]
    fn formats_non_zero_exit_status() {
        let output = BashOutput {
            status: BashStatus::Error,
            exit_code: Some(7),
            output: CommandOutput::Inline("nope\n".to_string()),
            error: None,
        };

        assert_eq!(output.into_text(), "nope\n(return code was 7)");
    }

    #[test]
    fn non_zero_exit_status_is_not_a_failed_tool_call() {
        let output = BashOutput {
            status: BashStatus::Error,
            exit_code: Some(7),
            output: CommandOutput::Inline(String::new()),
            error: None,
        }
        .into_tool_output();

        assert_eq!(
            output.content,
            vec![ContentPart::text(
                "(Completed with no output)\n(return code was 7)"
            )]
        );
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
    fn kills_process_group_on_timeout() {
        let started = Instant::now();
        let output = bash(BashInput {
            command: "sleep 10 & wait".to_string(),
            timeout_seconds: Some(0),
        });

        assert_eq!(output.status, BashStatus::TimedOut);
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn saves_large_output_without_returning_contents() {
        let output = bash(BashInput {
            command: "python3 - <<'PY'\nprint('a' * 60000)\nPY".to_string(),
            timeout_seconds: Some(5),
        });

        assert_eq!(output.status, BashStatus::Success);
        let CommandOutput::Saved { path, bytes, lines } = &output.output else {
            panic!("expected saved output");
        };
        assert_eq!(*bytes, 60001);
        assert_eq!(*lines, 1);
        assert!(
            path.file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("tau-bash-output-")
        );

        let text = output.clone().into_text();
        assert!(text.contains("File size: 60001 bytes"));
        assert!(text.contains("Lines: 1"));
        assert!(!text.contains(&"a".repeat(100)));

        let full = std::fs::read_to_string(path).unwrap();
        assert!(full.len() > MAX_OUTPUT_BYTES as usize);
        std::fs::remove_file(path).unwrap();
    }
}
