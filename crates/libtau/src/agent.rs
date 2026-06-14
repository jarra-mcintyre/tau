use std::{future::Future, pin::Pin};

use crate::{
    context::{ContentPart, ResponsePart, TauResponse, TauSession, ToolResult, ToolUse},
    providers::{ProviderError, TokenUsage},
};

pub type AgentError = Box<dyn std::error::Error>;
pub type HookFuture<'a, T> = Pin<Box<dyn Future<Output = T> + 'a>>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestReason {
    UserMessage,
    ToolResults,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderErrorAction {
    Retry,
    Fail,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AgentTurn {
    pub token_usage: Option<TokenUsage>,
}

pub trait AgentHooks {
    fn request_start(
        &mut self,
        _session: &TauSession,
        _reason: RequestReason,
    ) -> Result<(), AgentError> {
        Ok(())
    }

    fn response_part(
        &mut self,
        _session: &TauSession,
        _part: &ResponsePart,
    ) -> Result<(), AgentError> {
        Ok(())
    }

    fn tool_call_start(
        &mut self,
        _session: &TauSession,
        _call: &ToolUse,
    ) -> Result<(), AgentError> {
        Ok(())
    }

    fn tool_result(
        &mut self,
        _session: &TauSession,
        _result: &ToolResult,
    ) -> Result<(), AgentError> {
        Ok(())
    }

    fn provider_error<'a>(
        &'a mut self,
        _session: &'a mut TauSession,
        _error: &'a ProviderError,
    ) -> HookFuture<'a, Result<ProviderErrorAction, AgentError>> {
        Box::pin(async { Ok(ProviderErrorAction::Fail) })
    }
}

#[derive(Default)]
pub struct NoopAgentHooks;

impl AgentHooks for NoopAgentHooks {}

pub async fn run_agent_turn(
    session: &mut TauSession,
    user_message: impl Into<String>,
    hooks: &mut impl AgentHooks,
) -> Result<AgentTurn, AgentError> {
    run_agent_turn_with_content(session, vec![ContentPart::text(user_message)], hooks).await
}

pub async fn run_agent_turn_with_content(
    session: &mut TauSession,
    user_content: Vec<ContentPart>,
    hooks: &mut impl AgentHooks,
) -> Result<AgentTurn, AgentError> {
    session.push_user_content(user_content);
    let mut response =
        request_response_with_hooks(session, hooks, RequestReason::UserMessage).await?;

    loop {
        let mut tool_calls = Vec::new();
        for part in &response.parts {
            hooks.response_part(session, part)?;
            if let ResponsePart::ToolUse { call } = part {
                tool_calls.push(call.clone());
            }
        }

        if tool_calls.is_empty() {
            return Ok(AgentTurn { token_usage: session.total_token_usage().cloned() });
        }

        run_tools(session, &tool_calls, hooks).await?;
        response = request_response_with_hooks(session, hooks, RequestReason::ToolResults).await?;
    }
}

async fn request_response_with_hooks(
    session: &mut TauSession,
    hooks: &mut impl AgentHooks,
    reason: RequestReason,
) -> Result<TauResponse, AgentError> {
    loop {
        hooks.request_start(session, reason)?;
        match session.request_response().await {
            Ok(response) => return Ok(response),
            Err(error) => match hooks.provider_error(session, &error).await? {
                ProviderErrorAction::Retry => continue,
                ProviderErrorAction::Fail => return Err(error.into()),
            },
        }
    }
}

async fn run_tools(
    session: &mut TauSession,
    tool_calls: &[ToolUse],
    hooks: &mut impl AgentHooks,
) -> Result<(), AgentError> {
    for call in tool_calls {
        hooks.tool_call_start(session, call)?;
    }

    let results = session.call_tools_parallel_and_record(tool_calls);

    for result in &results {
        hooks.tool_result(session, result)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        context::{ConversationItem, ResponseStop, ResponseStopReason, TauContext},
        providers::ModelApi,
        tools::{ToolDefinition, ToolOutput},
    };
    use std::{collections::VecDeque, sync::Mutex};

    #[derive(Debug)]
    struct ResponsesProvider {
        responses: Mutex<VecDeque<TauResponse>>,
    }

    #[async_trait::async_trait]
    impl ModelApi for ResponsesProvider {
        fn name(&self) -> &'static str {
            "stub"
        }

        async fn respond(&self, _session: &mut TauSession) -> Result<TauResponse, ProviderError> {
            Ok(self.responses.lock().unwrap().pop_front().unwrap())
        }
    }

    #[derive(Default)]
    struct RecordingHooks {
        response_parts: usize,
        tool_starts: usize,
        tool_results: usize,
    }

    impl AgentHooks for RecordingHooks {
        fn response_part(
            &mut self,
            _session: &TauSession,
            _part: &ResponsePart,
        ) -> Result<(), AgentError> {
            self.response_parts += 1;
            Ok(())
        }

        fn tool_call_start(
            &mut self,
            _session: &TauSession,
            _call: &ToolUse,
        ) -> Result<(), AgentError> {
            self.tool_starts += 1;
            Ok(())
        }

        fn tool_result(
            &mut self,
            _session: &TauSession,
            _result: &ToolResult,
        ) -> Result<(), AgentError> {
            self.tool_results += 1;
            Ok(())
        }
    }

    #[derive(Debug)]
    struct RetryProvider {
        failed_once: Mutex<bool>,
    }

    #[async_trait::async_trait]
    impl ModelApi for RetryProvider {
        fn name(&self) -> &'static str {
            "stub"
        }

        async fn respond(&self, _session: &mut TauSession) -> Result<TauResponse, ProviderError> {
            let mut failed_once = self.failed_once.lock().unwrap();
            if !*failed_once {
                *failed_once = true;
                return Err(ProviderError::Configuration("retry me".to_string()));
            }

            Ok(TauResponse {
                parts: vec![ResponsePart::Content {
                    content: ContentPart::text("done"),
                }],
                usage: Some(TokenUsage {
                    uncached_input_tokens: Some(1),
                    cache_read_input_tokens: None,
                    cache_creation_input_tokens: None,
                    output_tokens: Some(2),
                    total_tokens: Some(3),
                }),
            })
        }
    }

    #[test]
    fn agent_loop_executes_tool_round_and_returns_turn_usage() {
        fn echo(input: serde_json::Value) -> Result<ToolOutput, crate::tools::ToolCallError> {
            Ok(ToolOutput::json(input))
        }

        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let call = ToolUse {
                    id: "call_1".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({"ok": true}),
                };
                let provider = ResponsesProvider {
                    responses: Mutex::new(VecDeque::from([
                        TauResponse {
                            parts: vec![ResponsePart::ToolUse { call }],
                            usage: Some(TokenUsage {
                                uncached_input_tokens: Some(1),
                                cache_read_input_tokens: None,
                                cache_creation_input_tokens: None,
                                output_tokens: Some(2),
                                total_tokens: Some(3),
                            }),
                        },
                        TauResponse {
                            parts: vec![
                                ResponsePart::Content {
                                    content: ContentPart::text("done"),
                                },
                                ResponsePart::Stop {
                                    stop: ResponseStop {
                                        reason: ResponseStopReason::EndTurn,
                                        metadata: None,
                                    },
                                },
                            ],
                            usage: Some(TokenUsage {
                                uncached_input_tokens: Some(4),
                                cache_read_input_tokens: None,
                                cache_creation_input_tokens: None,
                                output_tokens: Some(5),
                                total_tokens: Some(9),
                            }),
                        },
                    ])),
                };
                let mut context = TauContext::default();
                context
                    .register_tool(ToolDefinition {
                        name: "echo".to_string(),
                        description: "echo".to_string(),
                        readonly: true,
                        input_schema: serde_json::json!({"type":"object"}),
                        callback: echo,
                    })
                    .unwrap();
                let mut session = context.session(provider, "gpt-test");
                let mut hooks = RecordingHooks::default();

                let turn = run_agent_turn(&mut session, "hello", &mut hooks)
                    .await
                    .unwrap();

                assert_eq!(turn.token_usage.unwrap().total_tokens, Some(12));
                assert_eq!(session.total_token_usage().unwrap().total_tokens, Some(12));
                assert_eq!(hooks.response_parts, 3);
                assert_eq!(hooks.tool_starts, 1);
                assert_eq!(hooks.tool_results, 1);
                assert!(
                    session
                        .conversation()
                        .items
                        .iter()
                        .any(|item| matches!(item, ConversationItem::ToolResult { .. }))
                );
            });
    }

    #[test]
    fn provider_error_hook_can_retry_request() {
        #[derive(Default)]
        struct RetryHooks {
            errors: usize,
        }

        impl AgentHooks for RetryHooks {
            fn provider_error<'a>(
                &'a mut self,
                _session: &'a mut TauSession,
                error: &'a ProviderError,
            ) -> HookFuture<'a, Result<ProviderErrorAction, AgentError>> {
                Box::pin(async move {
                    if matches!(error, ProviderError::Configuration(_)) {
                        self.errors += 1;
                        Ok(ProviderErrorAction::Retry)
                    } else {
                        Ok(ProviderErrorAction::Fail)
                    }
                })
            }
        }

        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                let provider = RetryProvider {
                    failed_once: Mutex::new(false),
                };
                let mut session = TauContext::default().session(provider, "gpt-test");
                let mut hooks = RetryHooks::default();

                let turn = run_agent_turn(&mut session, "hello", &mut hooks)
                    .await
                    .unwrap();

                assert_eq!(hooks.errors, 1);
                assert_eq!(turn.token_usage.unwrap().total_tokens, Some(3));
                assert_eq!(session.total_token_usage().unwrap().total_tokens, Some(3));
            });
    }
}
