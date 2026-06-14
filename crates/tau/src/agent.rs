use libtau::{
    agent::{AgentError, AgentHooks, AgentTurn, HookFuture, ProviderErrorAction, RequestReason},
    context::{ResponsePart, TauSession, ToolResult, ToolUse},
};

use crate::{
    config::{CliConfig, OAuthRefreshRequest},
    output::{
        OutputStyle, Style, compact_json, format_server_tool_use, print_content,
        print_server_tool_result,
    },
};

pub(crate) async fn run_turn(
    session: &mut TauSession,
    cli_config: &mut CliConfig,
    user_message: &str,
    output: &OutputStyle,
) -> Result<AgentTurn, AgentError> {
    let mut hooks = CliAgentHooks { cli_config, output };
    libtau::agent::run_agent_turn(session, user_message, &mut hooks).await
}

struct CliAgentHooks<'a> {
    cli_config: &'a mut CliConfig,
    output: &'a OutputStyle,
}

impl AgentHooks for CliAgentHooks<'_> {
    fn request_start(
        &mut self,
        _session: &TauSession,
        reason: RequestReason,
    ) -> Result<(), AgentError> {
        match reason {
            RequestReason::UserMessage => {
                self.output
                    .println_styled(Style::Muted, "[Sending message]");
                self.output
                    .println_styled(Style::Muted, "[Message sent. Waiting for model response]");
            }
            RequestReason::ToolResults => self.output.println_styled(
                Style::Muted,
                "[Sending tool results. Waiting for model response]",
            ),
        }
        Ok(())
    }

    fn response_part(
        &mut self,
        _session: &TauSession,
        part: &ResponsePart,
    ) -> Result<(), AgentError> {
        match part {
            ResponsePart::Content { content } => print_content(content, self.output),
            ResponsePart::ServerToolUse { call } => self
                .output
                .println_indented_styled(Style::Tool, &format_server_tool_use(call)),
            ResponsePart::ServerToolResult { result } => {
                print_server_tool_result(result, self.output)
            }
            ResponsePart::ToolUse { .. } | ResponsePart::Stop { .. } => {}
        }
        Ok(())
    }

    fn tool_call_start(&mut self, _session: &TauSession, call: &ToolUse) -> Result<(), AgentError> {
        self.output.println_indented_styled(
            Style::Tool,
            &format!("[tool] {}({})", call.name, compact_json(&call.input)),
        );
        Ok(())
    }

    fn tool_result(
        &mut self,
        _session: &TauSession,
        result: &ToolResult,
    ) -> Result<(), AgentError> {
        match &result.error {
            Some(error) => self.output.println_indented_styled(
                Style::Tool,
                &format!("[tool] {} failed: {error}", result.name),
            ),
            None => self
                .output
                .println_indented_styled(Style::Tool, &format!("[tool] {} completed", result.name)),
        }
        Ok(())
    }

    fn provider_error<'a>(
        &'a mut self,
        session: &'a mut TauSession,
        error: &'a libtau::api::ProviderError,
    ) -> HookFuture<'a, Result<ProviderErrorAction, AgentError>> {
        Box::pin(async move {
            if let libtau::api::ProviderError::ReauthenticationRequired {
                access,
                refresh,
                expires,
            } = error
            {
                self.cli_config
                    .refresh_oauth_provider(
                        &session.provider().name().to_string(),
                        OAuthRefreshRequest {
                            access: access.clone(),
                            refresh: refresh.clone(),
                            expires: *expires,
                        },
                    )
                    .await?;
                let provider = self.cli_config.build_provider_for_current_model()?;
                session.refresh_provider(provider);
                Ok(ProviderErrorAction::Retry)
            } else {
                Ok(ProviderErrorAction::Fail)
            }
        })
    }
}
