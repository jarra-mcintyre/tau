Tau - Just another coding agent and library

Tau is implemented in Rust. The repository root is a Cargo workspace with two packages:
- `crates/libtau` - library of reusable logic for implementing coding agents (tools, skill handling, prompt generation, model access etc)
- `crates/tau` - Command for running the Tau agent

Use normal Cargo commands for building, formatting, testing and running.

## Design Considerations
- We're targetting Linux and OS-X. However, future support for other platforms isn't out of the question

