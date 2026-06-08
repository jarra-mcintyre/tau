use libtau::{
    context::{ContentPart, ResponsePart, TauSession, ToolResult, ToolUse},
    providers::TokenUsage,
};

use crate::{
    config::{CliConfig, OAuthRefreshRequest},
    output::{
        OutputStyle, Style, compact_json, format_server_tool_use, print_content,
        print_server_tool_result,
    },
};

pub(crate) async fn run_turn(
    context: &mut TauSession,
    cli_config: &mut CliConfig,
    user_message: &str,
    output: &OutputStyle,
) -> Result<Option<TokenUsage>, Box<dyn std::error::Error>> {
    output.println_styled(Style::Muted, "[Sending message]");
    context.push_user_content(vec![ContentPart::text(user_message)]);
    output.println_styled(Style::Muted, "[Message sent. Waiting for model response]");
    let mut response = request_response_with_reauth(context, cli_config).await?;
    let mut total_usage = context.last_token_usage().cloned();

    loop {
        let mut tool_calls = Vec::new();
        for part in response.parts {
            match part {
                ResponsePart::Content { content } => print_content(&content, output),
                ResponsePart::ToolUse { call } => tool_calls.push(call),
                ResponsePart::ServerToolUse { call } => {
                    output.println_indented_styled(Style::Tool, &format_server_tool_use(&call));
                }
                ResponsePart::ServerToolResult { result } => {
                    print_server_tool_result(&result, output)
                }
                ResponsePart::Stop { .. } => {}
            }
        }

        if tool_calls.is_empty() {
            return Ok(total_usage);
        }

        run_tools(context, &tool_calls, output);
        output.println_styled(
            Style::Muted,
            "[Sending tool results. Waiting for model response]",
        );
        response = request_response_with_reauth(context, cli_config).await?;
        add_usage(&mut total_usage, context.last_token_usage());
    }
}

async fn request_response_with_reauth(
    context: &mut TauSession,
    cli_config: &mut CliConfig,
) -> Result<libtau::context::TauResponse, Box<dyn std::error::Error>> {
    match context.request_response().await {
        Ok(response) => Ok(response),
        Err(libtau::api::ProviderError::ReauthenticationRequired {
            access,
            refresh,
            expires,
        }) => {
            cli_config
                .refresh_oauth_provider(
                    &context.provider().name().to_string(),
                    OAuthRefreshRequest {
                        access,
                        refresh,
                        expires,
                    },
                )
                .await?;
            let provider = cli_config.build_provider_for_current_model()?;
            context.refresh_provider(provider);
            context.request_response().await.map_err(Into::into)
        }
        Err(error) => Err(error.into()),
    }
}

fn run_tools(
    context: &mut TauSession,
    tool_calls: &[ToolUse],
    output: &OutputStyle,
) -> Vec<ToolResult> {
    for call in tool_calls {
        output.println_indented_styled(
            Style::Tool,
            &format!("[tool] {}({})", call.name, compact_json(&call.input)),
        );
    }

    let results = context.call_tools_parallel_and_record(tool_calls);

    for result in &results {
        match &result.error {
            Some(error) => output.println_indented_styled(
                Style::Tool,
                &format!("[tool] {} failed: {error}", result.name),
            ),
            None => output
                .println_indented_styled(Style::Tool, &format!("[tool] {} completed", result.name)),
        }
    }

    results
}

fn add_usage(total: &mut Option<TokenUsage>, usage: Option<&TokenUsage>) {
    let Some(usage) = usage else {
        return;
    };
    match total {
        Some(total) => {
            total.input_tokens = add_optional(total.input_tokens, usage.input_tokens);
            total.output_tokens = add_optional(total.output_tokens, usage.output_tokens);
            total.total_tokens = add_optional(total.total_tokens, usage.total_tokens);
        }
        None => *total = Some(usage.clone()),
    }
}

fn add_optional(left: Option<u64>, right: Option<u64>) -> Option<u64> {
    match (left, right) {
        (Some(left), Some(right)) => Some(left + right),
        _ => left.or(right),
    }
}
