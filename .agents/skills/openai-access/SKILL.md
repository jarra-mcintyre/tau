---
name: openai-access
description: documents the OpenAI API formats as well authentication behaviour for the OpenAI API provider.
---

There are two OpenAI APIs. The legacy Chat Completions API and modern Response API. When accessing the services provided by OpenAI the Responses API is used. However, other providers often implement the Chat Completions API (or minor variants of it).

The OpenAI provider can be accessed in two ways. Either by API key or by OAuth authentication with Codex. Both use the Responses API but with different base URLs.

- `references/codex.md` describes codex Responses API and how to authenticate to it
- `references/responses.md` documents the responses API