This summarises information from Anthropic documentation:
- Basic documentation: (Extended Thinking)[https://platform.claude.com/docs/en/build-with-claude/extended-thinking]
- (Adaptive thinking)[https://platform.claude.com/docs/en/build-with-claude/adaptive-thinking]
- (Thinking effort)[https://platform.claude.com/docs/en/build-with-claude/effort]


Enable thinking by setting the `thinking` parameter:
- For Opus 4.7 and later only adaptive thinking can be used: `thinking: {type: "adaptive"}`
- For earlier models a maximum number of thinking tokens can be set: `thinking: {type: "enabled", budget_tokens: N}`
- For Opus 4.6 and Sonnet 4.6 adaptive thinking is recommended but not required

```json
{
  "model": "claude-sonnet-4-6",
  "max_tokens": 16000,
  "thinking": {
    "type": "enabled",
    "budget_tokens": 10000,
    "display": "summarized"
  },
  "messages": [
    {
      "role": "user",
      "content": "Are there an infinite number of prime numbers such that n mod 4 == 3?"
    }
  ]
}
```

When thinking (aka. "extended thinking") is enabled the model will add thinknig blocks to its responses. If the `display` parameter is set to `summarized` these blocks will include a summary of thinking (this is the default on some, but not all models)

```json
{
  "content": [
    {
      "type": "thinking",
      "thinking": "Let me analyze this step by step...",
      "signature": "WaUjzkypQ2mUEVM36O2TxuC06KN8xyfbJwyem2dw3URve/op91XWHOEBLLqIOMfFG/UvLEczmEsUjavL...."
    },
    {
      "type": "text",
      "text": "Based on my analysis..."
    }
  ]
}
```

### Redacted thinking blocks

From time to time thinking block contents may be redacted in which case the response type is `redacted`.

```json
{
  "type": "redacted_thinking",
  "data": "..."
}
```

## Caching

The complete unmodified tool use blocks should be included in any subsequent requests to the model. 

## Streaming

When streaming is used with thinking enabled results are received as `thinking_delta` events.

```
event: message_start
data: {"type": "message_start", "message": {"id": "msg_01...", "type": "message", "role": "assistant", "content": [], "model": "claude-sonnet-4-6", "stop_reason": null, "stop_sequence": null}}

event: content_block_start
data: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": "", "signature": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "I need to find the GCD of 1071 and 462 using the Euclidean algorithm.\n\n1071 = 2 × 462 + 147"}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "\n462 = 3 × 147 + 21\n147 = 7 × 21 + 0\n\nSo GCD(1071, 462) = 21"}}

// Additional thinking deltas...

event: content_block_delta
data: {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "EqQBCgIYAhIM1gbcDa9GJwZA2b3hGgxBdjrkzLoky3dl1pkiMOYds..."}}

event: content_block_stop
data: {"type": "content_block_stop", "index": 0}

event: content_block_start
data: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}

event: content_block_delta
data: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "The greatest common divisor of 1071 and 462 is **21**."}}

// Additional text deltas...

event: content_block_stop
data: {"type": "content_block_stop", "index": 1}

event: message_delta
data: {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null}}

event: message_stop
data: {"type": "message_stop"}
```

## Adaptive thinking effort

When using adaptive thinking the amount of effort the model expends on thinking is controlled with the `effort` parameter. Available options are `max`, `xhigh`, `high`, `medium` and `low`.


```json
{
    "model": "claude-opus-4-7",
    "max_tokens": 16000,
    "thinking": {
        "type": "adaptive"
    },
    "output_config": {
        "effort": "medium"
    },
    "messages": [
        {
            "role": "user",
            "content": "What is the capital of France?"
        }
    ]
}
```