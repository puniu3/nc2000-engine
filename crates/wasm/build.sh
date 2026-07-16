#!/usr/bin/env bash
# Canonical wasm build (M9c tuned profile).
#
# wasm-pack 0.13 has no custom-profile flag, so the wasm-only rustc profile
# (fat LTO + 1 codegen unit; native `release` untouched) rides in as cargo
# env overrides. wasm-opt level comes from [package.metadata.wasm-pack]
# in Cargo.toml (-O3; -O4 measured equal, default -O slower).
#
#   crates/wasm/build.sh          # web target  -> pkg-web  (the Vite app)
#   crates/wasm/build.sh nodejs   # node target -> pkg-node (tests-node/*)
set -euo pipefail
cd "$(dirname "$0")"
target="${1:-web}"
case "$target" in
  web) out=pkg-web ;;
  nodejs) out=pkg-node ;;
  *) echo "usage: build.sh [web|nodejs]" >&2; exit 2 ;;
esac
export CARGO_PROFILE_RELEASE_LTO=fat
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
exec wasm-pack build --release --target "$target" --out-dir "$out"
