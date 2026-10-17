#!/bin/sh
set -eu
if ! command -v cc >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then
  apt-get update
  apt-get install -y --no-install-recommends build-essential pkg-config ca-certificates curl
fi
curl --proto '=https' --tlsv1.2 -fsS https://sh.rustup.rs -o /tmp/nc2000-rustup.sh
sh /tmp/nc2000-rustup.sh -y --profile minimal --default-toolchain 1.98.1
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_TARGET_DIR=/tmp/nc2000-budget-target
rustc --version
cargo --version
cargo build --locked --release -p nc2000-bot --example budget_resume
