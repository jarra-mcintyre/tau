# Basics

Examples of basic request/response are given below. This info is summarised from [OpenAI /responses endpoint](https://developers.openai.com/api/reference/resources/responses/methods/create#responses_create-input-input_item_list-item-function_tool_call_output-output). If and only if you need more information then follow this link.

- In multi-turn conversations always set `previous_response_id` or echo back the entire, unmodified conversation history
- To avoid invalidating the cache, instructions, tool blocks and so on must remain constant between turns
- Use strict model for tool calls and enable parallel tool calls

# Common request-response flows

## Text Input

```json
{
  "model": "gpt-5.5",
  "instructions": "You are a helpfull assistant..",
  "max_output_tokens": 9999, // optional
  "max_tool_calls": 99, // optional
  "parallel_tool_calls": true,
  "prompt_cache_retention": "24h", // optional; one of `24h` or `in_memory` (only older models support in_memory prompt caching)
  "reasoning": {
    "effort": "high",  // optional; one of `none`, `minimal`, `low`, `medium`, `high` or `xhigh`
    "summary": "detailed", // optional; one of `auto`, `concise` or `detailed`
  },
  "service_tier": "auto", // optional; one of `auto`, `default`, `flex` or `priority`
  "input": "Hello"
}
```

```json
{
  "id": "resp_abc", // for follow up turns set `previous_response_id` to match this
  "object": "response",
  "created_at": 1741476542,
  "status": "completed",
  "completed_at": 1741476543,
  "error": null,
  "incomplete_details": null,
  "instructions": null,
  "max_output_tokens": null,
  "model": "gpt-5.5",
  "output": [
    {
      "type": "message",
      "id": "msg_123",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "Hello. How may I help you?",
          "annotations": []
        }
      ]
    }
  ],
  "parallel_tool_calls": true,
  "previous_response_id": null,
  "reasoning": {
    "effort": null,
    "summary": null
  },
  "store": true,
  "temperature": 1.0,
  "text": {
    "format": {
      "type": "text"
    }
  },
  "tool_choice": "auto",
  "tools": [],
  "top_p": 1.0,
  "truncation": "disabled",
  "usage": {
    "input_tokens": 36,
    "input_tokens_details": {
      "cached_tokens": 0
    },
    "output_tokens": 87,
    "output_tokens_details": {
      "reasoning_tokens": 0
    },
    "total_tokens": 123
  },
  "user": null,
  "metadata": {}
}
```

## Images

Images are sent with `input_image` type content blocks. `image_url` may be a link to the image or a data url.

```json
{
  "model": "gpt-5.5",
  "input": [
    {
      "role": "user",
      "content": [
        {"type": "input_text", "text": "Describe this image"},
        {
          "type": "input_image",
          "image_url": "data:image/jpeg;base64,abc123", // either a PNG, JPEG, WEB-P or non-animated GIF image
          "detail": "auto" // one of "low", "high", "original" or "auto"
        }
      ]
    }
  ]
}
```

## Tool use

Request:
```json
{
  "model": "gpt-5.5",
  "tools": [
    { "type": "web_search" },
    {
      "type": "function",
      "name": "get_current_weather",
      "description": "Get the current weather in a given location",
      "strict": true,
      "parameters": {
        "type": "object",
        "properties": {
          "location": {
            "type": "string",
            "description": "The city, state and country, e.g. Sydney, NSW, Australia"
          },
          "unit": {
            "type": ["string", "null"],
            "enum": ["celsius", "fahrenheit"]
          }
        },
        "required": ["location", "unit"],
        "additionalProperties": false
      }
    }
  ],
  "input": "Search how to use tools in the OpenAI Responses API. Also tell me the current weather in Tokyo"
}
```

Response:
```json
{
  "id": "resp_abc",
  "object": "response",
  
  // ... other standard response content ...

  "output": [
    {
      "type": "web_search_call",
      "id": "ws_456",
      "status": "completed"
    },
    {
      "type": "message",
      "id": "msg_789",
      "status": "completed",
      "role": "assistant",
      "content": [
        {
          "type": "output_text",
          "text": "You can find more information on tool usage here..",
          "annotations": [
            {
              "type": "url_citation",
              "start_index": 442,
              "end_index": 557,
              "url": "https://.../?utm_source=chatgpt.com",
              "title": "..."
            }
          ]
        }
      ]
    }
    {
      "type": "function_call",
      "id": "fc_123",
      "call_id": "call_unLAR8MvFNptuiZK6K6HCy5k",
      "name": "get_current_weather",
      "arguments": "{\"location\":\"Tokyo, Japan\",\"unit\":\"celsius\"}",
      "status": "completed"
    }
  ],
  
  // ... other response content ...

  "tools": [
    {
      "type": "web_search_preview",
      "domains": [],
      "search_context_size": "medium",
      "user_location": {
        "type": "approximate",
        "city": null,
        "country": "Japan",
        "region": null,
        "timezone": null
      }
    },
    {
      // ... echos back definition ...
    }
  ],
  // ... other response parameters ...
}
```

Function call results are sent with a `function_call_output` block.

## Annotations/Citations

```json
{
  // ... standard fields ...
  "content": [
    {
      "type": "output_text",
      "text": "On June 7, 2026, alients landed in Antarctica ..",
      "annotations": [
        {
          "type": "url_citation",
          "start_index": 2606,
          "end_index": 2758,
          "url": "https://...",
          "title": "Title..."
        }
      ]
    }
  ]
}
```

# Response Types

The API may return the following response types, as identified by the `type` field:
- `message`
- `reasoning`
- `compaction`
- Tool use related: `function_call`, `function_call_output`, `web_search_call`
- MCP related: `mcp_call`, `mcp_list_tools`, `mcp_approval_request`, `mcp_approval_response`
- Additional out off scope: `file_search_call`, `computer_call`, `computer_call_output`, `tool_search_call`, `tool_search_call_output`, `additional_tools`, `image_generation_call`, `code_interpreter_call`, `local_shell_call`, `local_shell_call_output`, `shell_call`, `shell_call_output`, `apply_patch_call`, `apply_patch_call_output`, `custom_tool_call`


## `reasoning` type output

```json
{
  "id": "123",
  "type": "reasoning",
  "status": "complete",
  "summary": [
    {
      "text": "The user has asked me to tell them the weather in Tokyo, Japan",
      "type": "summary_text" // always summary_text
    }
  ],
  // optional array of additional content
  "content": [
    {
      "text": "The user has asked me to tell them the weather in Tokyo, Japan",
      "type": "reasoning_text" // always reasoning_text
    }
  ],
  "encrypted_content": "afwqgt" // optional
}
```

## `web_search_call` type output

The `web_search_call` type output includes an `action` item. The action `type` will be one of `search`, `open_page` or `find_in_page`. Search type blocks will (usually) include a `queries` field. This is an array of strings, or a single `query` field.


```json
  {
    "type": "web_search_call",
    "id": "ws_123",
    "status": "completed",
    "action": {
      "type": "search",
      "query": "latest news from Greenland" // either this or queries
    }
  }
```

The `web_search_call` type item

A web_search_call output item with the ID of the search call, along with the action taken in web_search_call.action. The action is one of:

    search, which represents a web search. It will usually (but not always) includes the search queries which were searched. Search actions incur a tool call cost (see pricing).
    open_page, which represents a page being opened. Supported in reasoning models.
    find_in_page, which represents searching within a page. Supported in reasoning models.


## `compaction` type output

```json
{
  "id": "cmp_001",
  "type": "compaction",
  "encrypted_content": "gAAAAABpM0Yj-...=",
  "created_by": "somebody" // optional
}
```

Annotations will generally be added to further messages to cite web-search results.

# Compaction

To enable automatic compaction set `context_management` as follows:

```json
{
  "type": "compaction",
  "compact_threshold": 100000
}
```

Otherwise conversation history can be compacted by calling the `/v1/responses/compact` endpoint with either `previous_response_id` or the conversation history.

The format here matches the `/v1/responses` endpoint but with a more limited set of fields (`model`, `previous_response_id`, `input`).

# More information

The following links give more information. Don't waste tokens opening these unless strictly necessary.

- https://developers.openai.com/api/docs/guides/text
- https://developers.openai.com/api/docs/guides/images-vision
- https://developers.openai.com/api/docs/guides/function-calling
- https://developers.openai.com/api/docs/guides/structured-outputs
- https://developers.openai.com/api/docs/guides/tools-web-search

