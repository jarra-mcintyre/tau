Tau works differently to other agentic coding tools. There's no TUI. You just interact with Tau straight from the command line. Tau commands are designed to be composable and embeddable. That means you can naturally compose Tau with other tools such as grep, rg, find, fzf, less and so on.

# Basic usage

## Coding with Tau

You code with Tau by sending messages on the command line. Messages are added to the current conversation history.

```bash
tau # shows the current stage (conversation alias, message offset, token usage .etc.)

# You build messages with message commands. These use the 'm' alias.
tau m "Also ..."  # Send a message to the model. This sends the message straight away
tau m # or you can use $EDITOR to write your message
tau ma "Also ..." # Add text to the message without sending it. This allows you to build more complex interactions
tau mi instructions.md screenshot.png # you can also add files to a message (without sending it)
tau mi $(fzf)  # this can of course be composed with other tools
tau mi .  # shell expansions will also work
tau mp # you can also paste text and images from the system clipboard as a message
tau mr source/file/path_1.rs source/file/path_2.rs # you can add paths to files as extra references to the current message. Only the path is passed to the model. It can choose to read it or not.
tau ms # once you've built up all relevant instructions, run `tau ms` to send the current message
tau me # "echo" the current message
tau mu -2 # rewind ("undo") the last two messages in the conversation
tau -y mc # Or use mc to clear the current message

# You manage the current conversation with conversation commands. These use the 'c' alias
tau c # start a new conversation
tau c -a "Feature work phase 2" # start a new conversation, specifying an alias
tau ca "My awesome feature" # change the alias for the current conversation
tau cf -2 -a "My awesome feature 2" # fork the current conversation at the second last message
tau ch | less # view the history of the current conversation
tau ch | grep "Launch the missiles" # or search it
tau -j ch | jq '.' # Output can also be in JSON
tau -y cc # compact the current conversation 
tau cs "Ticket #5431" # switch to a different conversation
tau cl # list previous conversations
tau cl | grep "Let's get that readme fixed up" # or search them
tau -j cl | jq '.' # output can also be in JSON
tau cl | tau -p "Find conversations relating to the API feature" # tau commands can be composed
tau -y cd "My dud feature" # delete some old conversations

# You manage which provider (e.g. OpenAI or Anthropic) and model to use with provider commands. These use the 'p' alias
tau -y p anthropic/opus-4.7  # change the current model (starts a new conversation)
tau pt xhigh  # set the thinking level for the current conversation
tau pl # list available providers
```

## "Non-interactive" mode

```bash
tau -p "Tell me how authentication works" # compatible with the CC or Codex -p flag
tau -r -p "Document the API data structures" # you can add the `-r` flag to put the model in read-only mode

cargo test | tau -m anthropic/haiku-4.5 -r -p "You will be given test results. They're a bit noisy. Highlight only the useful results (build failures, test failures)" # you can pipe additional inputs in as well
cargo test | tau -m anthropic/haiku-4.5 -r -f test_filter.md  # you can also use the -f command to read the prompt from a file.
```

## Getting help

```bash
tau -h # prints out basic help
tau -h "How do I search old conversations?" # you can also just ask questions about how to use Tau
tau -q "How do I type generator functions in Python?" # or you can ask general questions with the -q flag
tau -r -p "How do I compile this project?" # or you can ask project-specific questions with the -p flag. Add the -r flag to stop the model from trying to make changes
```

`tau -h` prompts have access to the Tau docs. `tau -p` prompts have access to the project context and tools. `tau -q` prompts have no extra context at all.

## Commands

You interact with Tau through commands, which are listed below. Each command has one or more aliases (these are shown in brackets).

## Modifiers

Some commands accept extra modifiers/flags. These should be placed before the command to avoid ambiguity. Flags also have aliases (these are shown in brackets).

- `--yes` (`-y`): "yes" modifier. Skips any confirmation prompts
- `--conversation <alias>` (`-c <alias>`): conversation modifier. Specifies which conversation a message command should act on.
- `--model <provider>/<model>` (`-m`): model modifier. Specifies which model to use
- `--alias <alias>` (`-a`): specify an alias
- `--read-only` (`-r`): read-only modifier. Disables the editor and shell tools.
- `--writes-allowed` (`-w`): write modifier. Enables the editor and shell tools.
- `--json` (`-j`): JSON modifier. Formats output as JSON. This formats Tau output, not the model output.

### Message commands

All message commands accept the `-c <alias>` flag to specify the conversation. If not specified, they act on the current conversation.

For example: `tau -c bug-1234 m "Check the current git diff and see if this is fixed"`

- `message <contents>` (`m`): Send a message to the current conversation. The message is sent immediately. If `<contents>` is not specified, your configured editor (`$EDITOR`) is launched.
- `message-add (<contents>)` (`ma`): Add to the current message. Nothing is sent. If `<contents>` is not specified, your configured editor is launched.
- `message-include <paths>` (`mi`): Include the contents of the specified file(s) in the current message. Nothing is sent. Files can be images, text, etc.
- `message-paste` (`mp`): Paste the current system clipboard contents into the current message. Text and images are supported.
- `message-reference <paths>` (`mr`): Include the specified paths as references in the current conversation. The model is told about the paths. However, whether it reads the paths is up to the model.
- `message-send` (`ms`): Send the current message to the current conversation.
- `message-echo` (`me`): "Echo" the current message (prints it on `stdout`).
- `message-clear` (`mc`): Clear the current message. By default, this will prompt you to confirm the action. Use the `-y` flag to execute automatically.
- `message-undo (<offset>)` (`mu`): Undo/Rewind the conversation to `<offset>`. 1 is the first message in the conversation. -1 is the last message in the conversation. If `<offset>` is not specified it defaults to -1. Use the `-y` flag to execute automatically

### Conversation commands

- `conversation` (`c`): Start a new conversation. You can optionally specify an alias. If you don't specify one, an alias will be automatically generated. Accepts the `-r` flag to start a read-only conversation. Accepts the `-a` flag to specify an alias.
- `conversation-alias <alias>` (`ca`): Set the alias of the current conversation.
- `conversation-history` (`ch`): Show the history of messages in the current conversation (accepts the `-j` flag).
- `conversation-compact` (`cc`): Compact the current conversation.
- `conversation-fork (<offset>)` (`cf`): Fork the conversation. You can optionally specify an offset from the first or last message to fork at. Accepts the `-r` and `-w` flags to switch the fork to read/write mode. Accepts the `-a` flag to specify the conversation alias.
- `conversation-switch <alias>` (`cs`): Switch to a different conversation.
- `conversation-list` (`cl`): List all conversations (accepts the `-j` flag to print in JSON).
- `conversation-delete <alias>` (`cd`): Delete the specified conversation.

### Provider commands

Provider commands allow you to update the current provider/model configuration.

- `provider <provider>/<model>`  (`p`): Set the current provider and model.
- `provider-thinking <level>` (`pt`): Change the model thinking level. Supported levels are `disabled`, `low`, `medium`, `high`, `xhigh`, and `max`.
- `provider-list`  (`pl`): List configured providers and models (accepts the `-j` flag to print in JSON).
- `provider-config <provider> `: Configure a provider 

### Misc. commands

- `query <query>` (`-q`): Ask a general question. No project context or edit tools are provided to the model. Use `prompt` (`-p`) if you need these. Accepts the `--model`/`-m` flags.
- `prompt <prompt>` (`-p`): Execute a prompt (like `-p` in other coding agents). You can optionally stream additional content on `stdin` to allow piped usage. Accepts the `--read-only`/`-r` and `--model`/`-m` flags.
- `file <path>` (`-f`): Same as `prompt` (`-p`), but reads the prompt from a file. Accepts the `--read-only`/`-r` and `--model`/`-m` flags.
- `version` (`-v`): Show the current Tau version.
- `help (<question>)` (`h`, `-h`): Show the help. You can optionally specify a question.
