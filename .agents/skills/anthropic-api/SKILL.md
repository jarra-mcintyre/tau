---
name: anthropic-api
description: documents latest Anthropic API capabilities and behaviours
---

Latest available Anthropic models are: Opus 4.7, Opus 4.6, Sonnet 4.6, Haiku 4.6


# Working with messages

Source: https://platform.claude.com/docs/en/build-with-claude/working-with-messages

- Messages API is stateless. Always send the full conversation history.
- Add cache control headers to control caching

```bash
#!/bin/sh
curl https://api.anthropic.com/v1/messages \
     --header "x-api-key: $ANTHROPIC_API_KEY" \
     --header "anthropic-version: 2023-06-01" \
     --header "content-type: application/json" \
     --data \
'{
    "model": "claude-opus-4-7",
    "max_tokens": 1024,
    "messages": [
        {"role": "user", "content": "Hello, Claude"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Can you describe LLMs to me?"}

    ]
}'
```

## Response format

The following gives a complete example of an API response (as per (Anthropic API Create Message)[https://platform.claude.com/docs/en/api/beta/messages/create]). Note the response ID is a unique response assigned to the message for QC and support purposes and should be recorded. 

```json
{
  "id": "msg_013Zva2CMHLNnXjNJJKqJ2EF",
  "container": {
    "id": "id",
    "expires_at": "2019-12-27T18:11:19.117Z",
    "skills": [
      {
        "skill_id": "pdf",
        "type": "anthropic",
        "version": "latest"
      }
    ]
  },
  "content": [
    {
      "citations": [
        {
          "cited_text": "cited_text",
          "document_index": 0,
          "document_title": "document_title",
          "end_char_index": 0,
          "file_id": "file_id",
          "start_char_index": 0,
          "type": "char_location"
        }
      ],
      "text": "Hi! My name is Claude.",
      "type": "text"
    }
  ],
  "context_management": {
    "applied_edits": [
      {
        "cleared_input_tokens": 0,
        "cleared_tool_uses": 0,
        "type": "clear_tool_uses_20250919"
      }
    ]
  },
  "model": "claude-opus-4-6",
  "role": "assistant",
  "stop_details": {
    "category": "cyber",
    "explanation": "explanation",
    "type": "refusal"
  },
  "stop_reason": "end_turn",
  "stop_sequence": null,
  "type": "message",
  "usage": {
    "cache_creation": {
      "ephemeral_1h_input_tokens": 0,
      "ephemeral_5m_input_tokens": 0
    },
    "cache_creation_input_tokens": 2051,
    "cache_read_input_tokens": 2051,
    "inference_geo": "inference_geo",
    "input_tokens": 2095,
    "iterations": [
      {
        "cache_creation": {
          "ephemeral_1h_input_tokens": 0,
          "ephemeral_5m_input_tokens": 0
        },
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "type": "message"
      }
    ],
    "output_tokens": 503,
    "server_tool_use": {
      "web_fetch_requests": 2,
      "web_search_requests": 0
    },
    "service_tier": "standard",
    "speed": "standard"
  }
}
```

## Caching

If a cache control header is added then the system caches prompts
- For automatic caching add a top-level single `cache_control` block to the request. The cache breakpoint is moved forward automatically
- For fine-grained cache add `cache_control` block to each message where caching should be run up-to

By default cache has a 5-minute life-time. This is automatically refreshed on each requests. Add `"ttl": "1h"` to the cache block to switch to 1 hour caching. This has an additionally cost.

Automatic caching example (as per Anthropic docs):
```json
{
  "model": "claude-opus-4-7",
  "max_tokens": 1024,
  "cache_control": {"type": "ephemeral"},
  "system": "You are an AI assistant tasked with analyzing literary works. Your goal is to provide insightful commentary on themes, characters, and writing style.",
  "messages": [
    {
      "role": "user",
      "content": "Analyze the major themes in Pride and Prejudice."
    }
  ]
}
```

## Stop reasons

Every response to a successful message will have a `stop_reason` field. This is an enum that will be set to:
- `end_turn` - the model naturally ended its turn (possibly sending an empty message)
- `max_tokens` - the maximum token limit was encountered
- `stop_sequence` - a custom stop sequence (specified with the `stop_sequence` parameter) was encountered
- `tool_use` - a tool use was requested
- `pause_turn` - server sampling loop reached its iteration limit. Response may contain a `server_tool_use` block without a corresponding `server_tool_result`. send back the response as is to signal to the model to keep processing.
- `refusal` - model refused to generate response

# More information

- For control of model thinking see: references/thinking.md
- For tool use see: references/tool-use.md
- For web-search see: references/web-search-tool.md
- For streaming responses see: references/streaming-responses.md

Note that these pages generally summarise all required information. None-the-less links to the Anthropic documentation are included for completeness. Only follow the links if the information is definitively insufficient.
