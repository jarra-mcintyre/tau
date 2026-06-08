Tau - Just another coding agent and library

Tau is implemented in Rust. The repository root is a Cargo workspace with two packages:
- `crates/libtau` - library of reusable logic for implementing coding agents (tools, skill handling, prompt generation, model access etc)
- `crates/tau` - Command for running the Tau agent

Use normal Cargo commands for building, formatting, testing and running. Use `find crates/ -name '*.rs'` to get a quick idea of the source layout.

## Design Considerations
- We're targetting Linux and OS-X. However, future support for other platforms isn't out of the question.
- This is pre-alpha code. Legacy support/migrations should not be considered unless instructed otherwise

## Coding Style

- Don't choose defaults on your own. Ask.
- Before implementing complex functionality always consider whether using a third-party library would be more appropriate.
- Don't add new dependencies on your own but do be proactive in considering and suggesting them.

### Clean and simple code

The following guidelines are intended to keep the code clean and simpler to read. They are not optional
- You must prefer imports over-qualified references (e.g. import `Error` rather than writing `std::error::Error`).
- Function names must decribe what the function does, not how. Do not put implementation details in the name (e.g. do not add `_arc` to a function name just because it takes an Arc).
- You must avoid one-line or unnecessary helper functions. Never create a function that just calls `default`. One line helper functions which don't encode any complex logic are just unnecessary line-noise.
- Never add new public functions unless you're actually using them or were told to. They just add noise.
- For structures with multiple fields, derive `Builder` (from `bon`) and use builders to instantiate them. This reduces noise line-noise especially for structures with large numbers of optional parameters.

## Architecture

- "Context" tracks global configuration such as tool and provider definitions
- "Sessions": track a conversation with a model and include user message, agent message, tools use etc. Each session is tied to a provider.
- "API": implements support for common model provider APIs (currently Anthropic Messages and OpenAI Requests)
- "Providers": services that provide a model. Each provider uses a specific API (e.g. Anthropic Messages for Anthropic)
- "Models": Models (e.g. GPT-5.5, Opus 4.7) are provided by a Provider.

APIs track the complete conversation state in an API specific format suitable for cache-friendly, multi-turn conversation. Sessions track the conversation history in a model agnostic format suitable for display to users etc

## Files

- `~/.tau/providers.json` - provider configuration
- `~/.tau/state.db` - sqlite database tracking current state
- `~/.tau/sessions/` - folder of sessions (encoded in JSON)
- `~/.tau/docs/` - documentation folder

Providers are configured in both `~/.tau/providers.json` and `~/.tau/state.db`