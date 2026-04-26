default:
    @just --list

# Format all Rust code in the workspace.
format:
    cargo fmt --all

# Run all unit tests in the workspace. Additional args are passed to Cargo
test *args:
    cargo test --workspace {{args}}

# Build the workspace. Additional args are passed to Cargo. Usage: `just build` or `just build --release`
build *args:
    cargo build --workspace {{args}}

# Run the Tau CLI in debug mode. Pass CLI args after `--`, e.g. `just run -- --help`.
run *args:
    cargo run -p tau-cli -- {{args}}

# Run the Tau CLI in release mode.
run-release *args:
    cargo run -p tau-cli --release -- {{args}}
