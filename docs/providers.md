Access to different models in Tau is achieved by configuring providers. Providers are any service which provide access to models. Tau has built-in support for a number of popular providers and also allows you to define your own. Each provider is accessed through a specific API. For instance the Anthropic provider is accessed through the Anthropic Messages API. Tau supports a number of different APIs for accessing models.

You can also configure a customer provider to access self-hosted models on tools such as llama.cpp and Ollama, or to Providers that Tau does not have in-built support for.

# Providers

The following providers are supported:
- OpenAI (`openai-api` and `openai-codex`). Both API key and Codex (OAuth) account access is supported.
- Anthropic (`anthropic-api`). Only API key access is supported.
- Google (`google-api`) for Gemini series models
- DeepSeek (`deepseek-api`) (to be implemented)
- AWS Bedrock (`anthropic-aws`) (to be implemented)
- Google Vertex AI (`anthropic-google`) (to be implemented)

Access to built-in providers can be configured by running `tau provider-config` (e.g. `tau provider-config openai-api` or `tau provider-config openai-codex`). Providers can also be configured by editing `~/.tau/providers.json` directly.

# Model APIs

The following model APIs are supported:
- Anthropic Messages
- OpenAI Requests
- OpenAI Completions (to be implemented)
- Gemini API (to be implemented) 

Support for these two APIs is sufficient to enable access to most major models from most major providers as well as self hosted models. Models can be accessed directly from Anthropic, OpenAI and Google, DeepSeek and so on. They can also be access through services like OpenRouter, AWS Bedrock, Google Vertex AI and Azure Model Foundry.

# Custom providers

## llama.cpp

Tau can be configured to connect to a `llama-server` process. llama.cpp supports both the Anthropic Messages and OpenAI Requests API. To get started you will need to edit `~/.tau/providers.json` and define a new, custom provider. The following gives a sample configuration.

```json
{
  "providers": [
    // ...
    {
      "type": "openai-api",
      "config": {
        "api_key": "sk..."
      }
    },
    {
      "type": "anthropic-api",
      "config": {
        "api_key": "..."
      }
    },
    {
      "type": "
    }
    {
      "type": "custom",
      "config": {
        // You can set name to anything. This will be the provider name in Tau
        "name": "llamacpp",
        // Set api to either anthropic_messages or openai_requests
        "api": "anthropic_messages",
        // API Key configuration is optional
        "api_key": "none",
        // Set base_url as appropriate. Do not include any path
        "base_url": "http://localhost:8080",
        "options": null,
        // List each model that is available. Typically you will want to specify a model name in llama.cpp
        "models": ["the-model"]
      }
    }
    // ..
  ]
}
```
