#!/usr/bin/env bash
# Seed-paired duel: threshold-preserving + damage-bookkeeping-free node key
# against the shipped uniform-16 key, at equal iterations.
#
# The mechanistic probe (`examples/key_shape`) says the abstraction halves the
# chance fan-out of a joint action (11.9 -> 6.0 distinct successors) but buys
# only +10% descent depth (1.53 -> 1.69 plies), because the joint-action
# product, not chance, is what the budget goes on. This duel is the strength
# reading on that: if depth is not the binding constraint, it should come back
# parity, and the line closes with both mechanism and strength evidence.
#
# Run under cx from the repo root:
#   cx submit -c 16 -T 21600 -n key-abs-duel -d <repo> -b tmp/key-abs -- bash tools/cx-key-abs.sh
set -u
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "NO CARGO on PATH"; exit 127; }
test -f Cargo.toml || { echo "NO CARGO.TOML in $PWD"; exit 66; }

cargo build --release -q -p nc2000-bot --example arena || exit 65
mkdir -p tmp/key-abs
rc=0
for iters in 3000 10000; do
  echo "=== arena skuctabs:$iters vs skuct:$iters  $(date -Is)"
  timeout 9000 ./target/release/examples/arena "skuctabs:$iters" "skuct:$iters" \
    --games 800 --seed 7 --threads 16 > "tmp/key-abs/duel-$iters.txt" 2>&1
  s=$?
  if [ $s -ne 0 ]; then
    echo "duel-$iters exited $s" >> tmp/key-abs/FAILURES
    rc=1
  fi
  cp -r tmp/key-abs/* "$CX_OUT"/ 2>/dev/null
done
cp -r tmp/key-abs/* "$CX_OUT"/
exit $rc
