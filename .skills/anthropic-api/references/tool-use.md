Official documentation is available at: (Anthropic strict tool use documentation)[https://platform.claude.com/docs/en/agents-and-tools/tool-use/strict-tool-use] and (Anthropic parallel tool use documentation)[https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use]. This document summarise official documentation. Only use those links if strictly unsure.

- Strict tool use enforces tool compliance to specification grammars and should always be used.
- Claude can make multiple tool requests to answer a user query.

Strict tool use works with the `anthropic-version` header set to `2023-06-01` and does not require any beta features. 

# Worked example

Request format:
```json
{
  "model": "claude-opus-4-7",
  "max_tokens": 1024,
  "messages": [
    {"role": "user", "content": "What is the weather in Tokyo ?"}
  ],
  "tools": [{
    "name": "get_weather",
    "description": "Get the current weather in a given location",
    "strict": true,
    "input_schema": {
    "type": "object",
    "properties": {
      "location": {
        "type": "string",
        "description": "The city and state, e.g. Melbourne, Australia"
      },
      "unit": {
        "type": "string",
        "enum": ["celsius", "fahrenheit"]
      }
    },
    "required": ["location"],
    "additionalProperties": false
    }
  }]
}
```

Response format:
```json
{
  "id": "msg_01Aq9w938a90dw8q",
  "model": "claude-opus-4-7",
  "stop_reason": "tool_use",
  "role": "assistant",
  "content": [
    {
      "type": "text",
      "text": "I'll check the current weather in Tokyo for you."
    },
    {
      "type": "tool_use",
      "id": "toolu_01A09q90qw90lq917835lq9",
      "name": "get_weather",
      "input": { "location": "Tokyo, Japan", "unit": "celsius" }
    }
  ]
}
```

When the model requests a tool use:
- Immediately respond with the tool use result (no interleaving user messages)
- If the response include tool uses and user messages, always place the tool use blocks first

Example tool use results include:

## Basic response

The text content block can be left empty, if there's no textual content needed.

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": "15 degrees"
    }
  ]
}
```

## With an image

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": [
        { "type": "text", "text": "15 degrees" },
        {
          "type": "image",
          "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": "/9j/4AAQSkZJRg..."
          }
        }
      ]
    }
  ]
}
```

## With a document

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": [
        { "type": "text", "text": "The weather is" },
        {
          "type": "document",
          "source": {
            "type": "text",
            "media_type": "text/plain",
            "data": "15 degrees"
          }
        }
      ]
    }
  ]
}
```

## Errors

Set `is_error` to indicate that the tool call failed.

```json
{
  "role": "user",
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_01A09q90qw90lq917835lq9",
      "content": "ConnectionError: the weather service API is not available (HTTP 500)",
      "is_error": true
    }
  ]
}
```

# Tool use schemas

## Supported schema features

The following JSON schema features are supported:
- All basic types: object, array, string, integer, number, boolean, null
- enum (strings, numbers, bools, or nulls only - no complex types)
- const
- anyOf and allOf (with limitations - allOf with $ref not supported)
- $ref, $def, and definitions (external $ref not supported)
- default property for all supported types
- required and additionalProperties (must be set to false for objects)
- String formats: date-time, time, date, duration, email, hostname, uri, ipv4, ipv6, uuid
- Array minItems (only values 0 and 1 supported)

The following JSON schema features are not supported (and will result in a 400):
- Recursive schemas
- Complex types within enums
- External $ref (e.g., '$ref': 'http://...')
- Numerical constraints (minimum, maximum, multipleOf, etc.)
- String constraints (minLength, maxLength)
- Array constraints beyond minItems of 0 or 1
- additionalProperties set to anything other than false

The following regex features are supported:
- Full matching (^...$) and partial matching
- Quantifiers: *, +, ?, simple {n,m} cases
- Character classes: [], ., \d, \w, \s
- Groups: (...)

The following regex features are not supported:
- Backreferences to groups (e.g., \1, \2)
- Lookahead/lookbehind assertions (e.g., (?=...), (?!...))
- Word boundaries: \b, \B
- Complex {n,m} quantifiers with large ranges

As a rule of thumb avoid complex regexes. These may result in a 400.

## Limits

There are limits on total schema complexity, some documented, some undocumented. Documented limits include:
- No more than 20 strict tool calls per request
- No more than 24 optional parameters
- No more than 16 parameters with union 

If complexity limits are hit the API returns a 400.

# Input examples

The `input_examples` field can optionally be set on the tool definition to include input examples:

```json
{
  "name": "get_weather",
  "input_schema": {...},
  "input_examples": [
    {"location": "Tokyo, Japan", "unit": "celsius"},
    {"location": "Melbourne, Australia"}
  ]
}
```

# Cache control

- Place `cache_control: {"type": "ephemeral"}` on the last tool in the `tools` array (assuming it doesn't change frequently). This caches the entire tool definition prefix up to and including the marked tool.
- Cache is invalidated when tool definitions change. The `tool_choice` header changes. Thinking parameters are changed and so on. TLDR; tool definitions should be constant

```
{
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather in a given location",
      "input_schema": { ... }
    },
    {
      "name": "get_time",
      "description": "Get the current time in a given time zone",
      "input_schema": { ... },
      "cache_control": { "type": "ephemeral" }
    }
  ]
}
```

# Forcing tool use

Set `"tool_choice"` to control model tool use. Permissible options are:
- `"auto"` - allow the model to decide
- `"any"` - model must use at least one of the specified tools
- `"none"` - model cannot use any tools
- `"tool"` - model must use the specified tool (e.g. `{"type": "tool", "name": "get_weather"}`)
-

# Parallel tool use

The model response may include multiple tool use requests. This is paralle tool use. Always respond in a single message.
Send tool results in a single message:

```
// ❌ This reduces parallel tool use
[
  {"role": "assistant", "content": [tool_use_1, tool_use_2]},
  {"role": "user", "content": [tool_result_1]},
  {"role": "user", "content": [tool_result_2]}  // Separate message
]

// ✅ This maintains parallel tool use
[
  {"role": "assistant", "content": [tool_use_1, tool_use_2]},
  {"role": "user", "content": [tool_result_1, tool_result_2]}  // Single message
]
```

# Server tools

Server tool use is automatic. The server response will include blocks showing server tool use results. The tool use request will include the type `server_tool_use` and will be followed by an appropriate tool use result

Server tool uses may fail (e.g. `too_many_requests`, `unavailable`, `max_uses_exceeded` etc)

