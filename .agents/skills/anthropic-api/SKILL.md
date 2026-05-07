---
name: anthropic-api
description: documents latest Anthropic API capabilities and behaviours
---

Latest available Anthropic models are: Opus 4.7, Opus 4.6, Sonnet 4.6, Haiku 4.6


# Working with messages

Source: https://platform.claude.com/docs/en/build-with-claude/working-with-messages

- Messages API is stateless. Always send the full conversation history.
- Add cache control headers to control caching

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

# More information

- For control of model thinking see: references/thinking.md
- For tool use see: references/tool-use.md
- For web-search see: references/web-search-tool.md
- For streaming responses see: references/streaming-responses.md

Note that these pages generally summarise all required information. None-the-less links to the Anthropic documentation are included for completeness. Only follow the links if the information is definitively insufficient.
