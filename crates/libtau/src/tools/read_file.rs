use std::{
    fs::{self, File},
    io::{self, Read, Seek, SeekFrom},
    path::PathBuf,
};

use base64::{Engine as _, engine::general_purpose::STANDARD as BASE64};
use chardetng::EncodingDetector;
use encoding_rs::{CoderResult, Encoding, UTF_8};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::{
    context::{ContentPart, MediaData, TauContext},
    tools::{ToolCallError, ToolDefinition, ToolOutput, ToolRegistrationError},
};

pub const NAME: &str = "read_file";
pub const DESCRIPTION: &str = "Read text and image files from disk. Content encoding is detected automatically. Text files larger than 100 KiB return only size and line count unless ignore_soft_limit=true. Text reads can specify line ranges.";

const DEFAULT_MAX_BYTES: u64 = 20 * 1024 * 1024;
const TEXT_SOFT_LIMIT_BYTES: u64 = 100 * 1024;
const DETECTION_BYTES: u64 = 8192;
const STREAM_BUFFER_BYTES: usize = 8192;

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileInput {
    pub path: PathBuf,
    /// Optional 1-based first line to read. Defaults to the first line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_line: Option<i64>,
    /// Optional 1-based last line to read, inclusive. Defaults to the last line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_line: Option<i64>,
    /// Optional content type override, such as text/plain or image/png.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub content_type: Option<String>,
    /// Allow reading text files larger than the soft limit.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ignore_soft_limit: Option<bool>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileOutput {
    pub okay: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub contents: Option<String>,
    #[serde(default, skip_serializing_if = "is_false")]
    pub soft_limited: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_lines: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<ReadFileError>,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
pub struct ReadFileError {
    pub kind: ReadFileErrorKind,
    pub message: String,
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ReadFileErrorKind {
    NotFound,
    PermissionDenied,
    InvalidInput,
    Other,
}

pub fn register(context: &mut TauContext) -> Result<(), ToolRegistrationError> {
    context.register_tool(ToolDefinition::new::<ReadFileInput>(
        NAME,
        DESCRIPTION,
        true,
        callback,
    )?)
}

fn is_false(value: &bool) -> bool {
    !*value
}

fn callback(input: Value) -> Result<ToolOutput, ToolCallError> {
    let input: ReadFileInput = serde_json::from_value(input)
        .map_err(|error| ToolCallError::InvalidInput(error.to_string()))?;
    let path = input.path.display().to_string();

    match read_file_result(input) {
        ReadFileResult::Image {
            media_type,
            data,
            total_size_bytes,
        } => Ok(ToolOutput {
            content: vec![
                ContentPart::text(format!(
                    "Read image file {path} (media_type={media_type}, size_bytes={total_size_bytes})"
                )),
                ContentPart::Image {
                    media_type,
                    data: MediaData::Base64(data),
                    metadata: None,
                },
            ],
        }),
        ReadFileResult::Output(output) => {
            if output.okay {
                if let Some(contents) = &output.contents {
                    return Ok(ToolOutput::text(contents.clone()));
                }
            }

            let value = serde_json::to_value(output)
                .map_err(|error| ToolCallError::OutputSerializationFailed(error.to_string()))?;
            Ok(ToolOutput::json(value))
        }
    }
}

pub fn read_file(input: ReadFileInput) -> ReadFileOutput {
    match read_file_result(input) {
        ReadFileResult::Output(output) => output,
        ReadFileResult::Image {
            media_type,
            total_size_bytes,
            ..
        } => ReadFileOutput {
            okay: true,
            contents: None,
            soft_limited: false,
            total_size_bytes: Some(total_size_bytes),
            total_lines: None,
            message: Some(format!("read image file ({media_type})")),
            error: None,
        },
    }
}

enum ReadFileResult {
    Output(ReadFileOutput),
    Image {
        media_type: String,
        data: String,
        total_size_bytes: u64,
    },
}

fn read_file_result(input: ReadFileInput) -> ReadFileResult {
    if let Err(error) = validate_line_range(input.first_line, input.last_line) {
        return ReadFileResult::Output(ReadFileOutput::error(error));
    }

    let has_line_range = input.first_line.is_some() || input.last_line.is_some();
    let length = match fs::metadata(&input.path) {
        Ok(metadata) => metadata.len(),
        Err(error) => {
            return ReadFileResult::Output(ReadFileOutput::error(ReadFileError::from_io_error(
                error,
            )));
        }
    };
    if length > DEFAULT_MAX_BYTES {
        return ReadFileResult::Output(ReadFileOutput::error(ReadFileError::invalid_input(
            format!(
                "file is too large to read ({length} bytes; maximum is {DEFAULT_MAX_BYTES} bytes)"
            ),
        )));
    }

    let sample = match read_prefix(&input.path, DETECTION_BYTES) {
        Ok(sample) => sample,
        Err(error) => {
            return ReadFileResult::Output(ReadFileOutput::error(ReadFileError::from_io_error(
                error,
            )));
        }
    };

    let content_type = input
        .content_type
        .as_deref()
        .map(str::trim)
        .filter(|content_type| !content_type.is_empty());
    let image_type = match content_type {
        Some(content_type) if is_image_content_type(content_type) => Some(content_type.to_string()),
        Some(_) => None,
        None => detect_image_type(&sample),
    };

    if let Some(media_type) = image_type {
        if has_line_range {
            return ReadFileResult::Output(ReadFileOutput::error(ReadFileError::invalid_input(
                "line ranges can only be used with text files".to_string(),
            )));
        }
        return match fs::read(&input.path) {
            Ok(bytes) => ReadFileResult::Image {
                media_type,
                data: BASE64.encode(bytes),
                total_size_bytes: length,
            },
            Err(error) => {
                ReadFileResult::Output(ReadFileOutput::error(ReadFileError::from_io_error(error)))
            }
        };
    }

    let (encoding, bom_length) = detect_text_encoding(&sample);

    if length > TEXT_SOFT_LIMIT_BYTES && !input.ignore_soft_limit.unwrap_or(false) {
        return match count_lines_streaming(&input.path, encoding, bom_length) {
            Ok(total_lines) => ReadFileResult::Output(ReadFileOutput {
                okay: true,
                contents: None,
                soft_limited: true,
                total_size_bytes: Some(length),
                total_lines: Some(total_lines),
                message: Some(format!(
                    "text file is larger than the soft limit ({length} bytes; soft limit is {TEXT_SOFT_LIMIT_BYTES} bytes). Request again with ignore_soft_limit=true to read it."
                )),
                error: None,
            }),
            Err(error) => {
                ReadFileResult::Output(ReadFileOutput::error(ReadFileError::from_io_error(error)))
            }
        };
    }

    let contents = if has_line_range {
        read_line_range_streaming(
            &input.path,
            encoding,
            bom_length,
            input.first_line,
            input.last_line,
        )
    } else {
        read_text_file(&input.path, encoding, bom_length)
    };

    match contents {
        Ok(contents) => ReadFileResult::Output(ReadFileOutput {
            okay: true,
            contents: Some(contents),
            soft_limited: false,
            total_size_bytes: Some(length),
            total_lines: None,
            message: None,
            error: None,
        }),
        Err(error) => {
            ReadFileResult::Output(ReadFileOutput::error(ReadFileError::from_io_error(error)))
        }
    }
}

fn read_prefix(path: &PathBuf, limit: u64) -> io::Result<Vec<u8>> {
    let mut file = File::open(path)?;
    let mut bytes = Vec::new();
    file.by_ref().take(limit).read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn detect_image_type(bytes: &[u8]) -> Option<String> {
    let kind = infer::get(bytes)?;
    if kind.mime_type().starts_with("image/") {
        Some(kind.mime_type().to_string())
    } else {
        None
    }
}

fn is_image_content_type(content_type: &str) -> bool {
    content_type
        .split(';')
        .next()
        .map(str::trim)
        .unwrap_or(content_type)
        .to_ascii_lowercase()
        .starts_with("image/")
}

fn detect_text_encoding(bytes: &[u8]) -> (&'static Encoding, usize) {
    if let Some((encoding, length)) = Encoding::for_bom(bytes) {
        return (encoding, length);
    }

    if std::str::from_utf8(bytes).is_ok() {
        return (UTF_8, 0);
    }

    let mut detector = EncodingDetector::new();
    detector.feed(bytes, true);
    (detector.guess(None, true), 0)
}

fn read_text_file(
    path: &PathBuf,
    encoding: &'static Encoding,
    bom_length: usize,
) -> io::Result<String> {
    let bytes = fs::read(path)?;
    Ok(decode_bytes(
        &bytes[bom_length.min(bytes.len())..],
        encoding,
    ))
}

fn decode_bytes(bytes: &[u8], encoding: &'static Encoding) -> String {
    if encoding == UTF_8 {
        String::from_utf8_lossy(bytes).into_owned()
    } else {
        encoding.decode_without_bom_handling(bytes).0.into_owned()
    }
}

fn count_lines_streaming(
    path: &PathBuf,
    encoding: &'static Encoding,
    bom_length: usize,
) -> io::Result<u64> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(bom_length as u64))?;

    let mut decoder = encoding.new_decoder_without_bom_handling();
    let mut bytes = vec![0; STREAM_BUFFER_BYTES];
    let mut lines = 0;
    let mut has_pending = false;

    loop {
        let read = file.read(&mut bytes)?;
        let last = read == 0;
        let decoded = decode_chunk(&mut decoder, &bytes[..read], last);

        if !decoded.is_empty() {
            lines += decoded.matches('\n').count() as u64;
            has_pending = !decoded.ends_with('\n');
        }

        if last {
            if has_pending {
                lines += 1;
            }
            return Ok(lines);
        }
    }
}

fn read_line_range_streaming(
    path: &PathBuf,
    encoding: &'static Encoding,
    bom_length: usize,
    first_line: Option<i64>,
    last_line: Option<i64>,
) -> io::Result<String> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(bom_length as u64))?;

    let first_line = first_line.unwrap_or(1) as usize;
    let last_line = last_line.map(|line| line as usize);
    let mut decoder = encoding.new_decoder_without_bom_handling();
    let mut bytes = vec![0; STREAM_BUFFER_BYTES];
    let mut output = String::new();
    let mut pending = String::new();
    let mut line_number = 1;

    loop {
        let read = file.read(&mut bytes)?;
        let last = read == 0;
        let decoded = decode_chunk(&mut decoder, &bytes[..read], last);

        if !decoded.is_empty() {
            pending.push_str(&decoded);
            while let Some(index) = pending.find('\n') {
                let line: String = pending.drain(..=index).collect();
                if line_number >= first_line
                    && last_line.is_none_or(|last_line| line_number <= last_line)
                {
                    output.push_str(&line);
                }
                line_number += 1;
                if last_line.is_some_and(|last_line| line_number > last_line) {
                    return Ok(output);
                }
            }
        }

        if last {
            if !pending.is_empty()
                && line_number >= first_line
                && last_line.is_none_or(|last_line| line_number <= last_line)
            {
                output.push_str(&pending);
            }
            return Ok(output);
        }
    }
}

fn decode_chunk(decoder: &mut encoding_rs::Decoder, bytes: &[u8], last: bool) -> String {
    let mut decoded = String::new();
    let mut offset = 0;
    loop {
        let (result, consumed, _) = decoder.decode_to_string(&bytes[offset..], &mut decoded, last);
        offset += consumed;
        if result == CoderResult::InputEmpty {
            break;
        }
        decoded.reserve(STREAM_BUFFER_BYTES);
    }
    decoded
}

fn validate_line_range(
    first_line: Option<i64>,
    last_line: Option<i64>,
) -> Result<(), ReadFileError> {
    if first_line.is_some_and(|line| line <= 0) || last_line.is_some_and(|line| line <= 0) {
        return Err(ReadFileError::invalid_input(
            "line numbers are 1-based and must be greater than zero".to_string(),
        ));
    }

    if let (Some(first_line), Some(last_line)) = (first_line, last_line) {
        if first_line > last_line {
            return Err(ReadFileError::invalid_input(format!(
                "first_line ({first_line}) must be less than or equal to last_line ({last_line})"
            )));
        }
    }

    Ok(())
}

impl ReadFileOutput {
    fn error(error: ReadFileError) -> Self {
        Self {
            okay: false,
            contents: None,
            soft_limited: false,
            total_size_bytes: None,
            total_lines: None,
            message: None,
            error: Some(error),
        }
    }
}

impl ReadFileError {
    fn from_io_error(error: io::Error) -> Self {
        Self {
            kind: ReadFileErrorKind::from_io_error_kind(error.kind()),
            message: error.to_string(),
        }
    }

    fn invalid_input(message: String) -> Self {
        Self {
            kind: ReadFileErrorKind::InvalidInput,
            message,
        }
    }
}

impl ReadFileErrorKind {
    fn from_io_error_kind(kind: io::ErrorKind) -> Self {
        match kind {
            io::ErrorKind::NotFound => Self::NotFound,
            io::ErrorKind::PermissionDenied => Self::PermissionDenied,
            io::ErrorKind::InvalidInput => Self::InvalidInput,
            _ => Self::Other,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_file_contents() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-success.txt",
            std::process::id()
        ));
        fs::write(&path, "hello tau").unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, Some("hello tau".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn returns_not_found_error() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-missing.txt",
            std::process::id()
        ));
        let output = read_file(ReadFileInput {
            path,
            first_line: None,
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(!output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::NotFound);
    }

    #[test]
    fn reads_inclusive_line_range() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-range.txt",
            std::process::id()
        ));
        fs::write(&path, "one\ntwo\nthree\nfour\n").unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: Some(2),
            last_line: Some(3),
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, Some("two\nthree\n".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_from_start_when_only_last_line_is_set() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-end-line.txt",
            std::process::id()
        ));
        fs::write(&path, "one\ntwo\nthree").unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: Some(2),
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, Some("one\ntwo\n".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn returns_invalid_input_for_invalid_line_range() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-invalid-range.txt",
            std::process::id()
        ));

        let output = read_file(ReadFileInput {
            path,
            first_line: Some(3),
            last_line: Some(2),
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(!output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::InvalidInput);
    }

    #[test]
    fn returns_invalid_input_for_negative_line_number() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-negative-line.txt",
            std::process::id()
        ));

        let output = read_file(ReadFileInput {
            path,
            first_line: Some(-1),
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(!output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::InvalidInput);
    }

    #[test]
    fn reads_detected_image() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-image.png",
            std::process::id()
        ));
        let bytes = b"\x89PNG\r\n\x1a\nimage bytes";
        fs::write(&path, bytes).unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.total_size_bytes, Some(bytes.len() as u64));
        assert!(output.message.unwrap().contains("image/png"));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_image_with_content_type_override() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-image.bin",
            std::process::id()
        ));
        let bytes = b"not detectable but caller knows";
        fs::write(&path, bytes).unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
            content_type: Some("image/jpeg".to_string()),
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.total_size_bytes, Some(bytes.len() as u64));
        assert!(output.message.unwrap().contains("image/jpeg"));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn decodes_non_utf8_text() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-windows-1252.txt",
            std::process::id()
        ));
        fs::write(&path, b"caf\xe9").unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, Some("café".to_string()));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_large_file_without_line_range() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-large.txt",
            std::process::id()
        ));
        fs::write(&path, vec![b'a'; DEFAULT_MAX_BYTES as usize + 1]).unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(!output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::InvalidInput);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn rejects_large_file_with_line_range() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-large-range.txt",
            std::process::id()
        ));
        let mut contents = b"one\ntwo\n".to_vec();
        contents.extend(vec![b'a'; DEFAULT_MAX_BYTES as usize + 1]);
        fs::write(&path, contents).unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: Some(2),
            last_line: Some(2),
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(!output.okay);
        assert_eq!(output.contents, None);
        assert_eq!(output.error.unwrap().kind, ReadFileErrorKind::InvalidInput);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn returns_summary_for_text_over_soft_limit() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-soft-limit.txt",
            std::process::id()
        ));
        let contents = format!("one\ntwo\n{}", "a".repeat(TEXT_SOFT_LIMIT_BYTES as usize));
        fs::write(&path, &contents).unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: None,
            last_line: None,
            content_type: None,
            ignore_soft_limit: None,
        });

        assert!(output.okay);
        assert_eq!(output.contents, None);
        assert!(output.soft_limited);
        assert_eq!(output.total_size_bytes, Some(contents.len() as u64));
        assert_eq!(output.total_lines, Some(3));
        assert!(output.message.unwrap().contains("ignore_soft_limit=true"));
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn reads_text_over_soft_limit_when_requested() {
        let path = std::env::temp_dir().join(format!(
            "tau-read-file-test-{}-ignore-soft-limit.txt",
            std::process::id()
        ));
        let contents = format!("one\ntwo\n{}", "a".repeat(TEXT_SOFT_LIMIT_BYTES as usize));
        fs::write(&path, &contents).unwrap();

        let output = read_file(ReadFileInput {
            path: path.clone(),
            first_line: Some(2),
            last_line: Some(2),
            content_type: None,
            ignore_soft_limit: Some(true),
        });

        assert!(output.okay);
        assert_eq!(output.contents, Some("two\n".to_string()));
        assert!(!output.soft_limited);
        assert_eq!(output.error, None);

        fs::remove_file(path).unwrap();
    }
}
