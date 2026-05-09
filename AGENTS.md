Tau - Just another coding agent and library

Tau is implemented in Rust. The repository root is a Cargo workspace with two packages:
- `crates/libtau` - library of reusable logic for implementing coding agents (tools, skill handling, prompt generation, model access etc)
- `crates/tau` - Command for running the Tau agent

Use normal Cargo commands for building, formatting, testing and running. Use `find crates/ -name '*.rs'` to get a quick idea of the source layout.

## Design Considerations
- We're targetting Linux and OS-X. However, future support for other platforms isn't out of the question

## Architecture

- "Context" tracks global configuration such as tool and provider definitions
- "Sessions" track a conversation with a model and include user message, agent message, tools use etc. Each session is tied to a provider.
- "Providers": implement support for a specific provider API. Currently we support Anthropic Messages and OpenAI Requests.
- "Models": Models (e.g. GPT-5.5, Opus 4.7) are provided by a Provider.

Providers track the complete conversation history in a provider specific format and optionally any resume IDs etc (e.g. OpenAI `previous_response_id`). Sessions track the conversation history in a model agnostic format suitable for display to users etc

Sessions can be switced between different models.