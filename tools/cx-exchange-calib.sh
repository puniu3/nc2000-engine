#!/usr/bin/env bash
# Phase A calibration sweep for the exchange term (`eval::exchange_margin`):
# the race computation generalised from the 1v1 endgame to every living pair.
#
# Corpus positions, not self-play: the same run that calibrated Spikes put r at
# 0.580 on corpus positions against ~0.78 on self-play, so self-play is the
# wrong distribution to fit a matchup feature on.
#
# `/tmp` is gitignored, so the spectator logs do not rsync into the workspace —
# they are unpacked here from the tracked archive instead.
#
# Run under cx from the repo root:
#   cx submit -c 16 -T 21600 -n exchange-calib -d <repo> -b tmp/exchange-calib -- bash tools/cx-exchange-calib.sh
set -u
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "NO CARGO on PATH"; exit 127; }
test -f Cargo.toml || { echo "NO CARGO.TOML in $PWD"; exit 66; }

CORPUS=tmp/corpus-spectator
if [ ! -d "$CORPUS" ] || [ -z "$(ls -A "$CORPUS" 2>/dev/null)" ]; then
  test -f data/corpus-spectator-logs.zip || { echo "NO CORPUS ARCHIVE"; exit 67; }
  mkdir -p "$CORPUS"
  # The cx image has python3 but no unzip (measured: exit 68, "unzip: command
  # not found"), so unpack through the stdlib rather than adding a dependency.
  python3 -m zipfile -e data/corpus-spectator-logs.zip "$CORPUS" || {
    echo "UNZIP FAILED"
    exit 68
  }
  # the archive may carry a top directory; flatten if so
  if [ -z "$(ls "$CORPUS"/*.raw.log 2>/dev/null)" ]; then
    inner=$(find "$CORPUS" -name '*.raw.log' -exec dirname {} \; 2>/dev/null | head -1)
    [ -n "$inner" ] && mv "$inner"/*.raw.log "$CORPUS"/
  fi
fi
n=$(ls "$CORPUS"/*.raw.log 2>/dev/null | wc -l)
echo "corpus battles: $n"
[ "$n" -ge 500 ] || { echo "CORPUS TOO SMALL ($n)"; exit 69; }

cargo build --release -q -p nc2000-bot --example eval_calibration || exit 65
mkdir -p tmp/exchange-calib
rc=0
echo "=== exchange sweep, 570 corpus positions $(date -Is)"
timeout 20000 ./target/release/examples/eval_calibration \
  --games 0 --corpus "$CORPUS" --battles 0-569 \
  --playouts 32 --gt-iters 300 --threads 16 --seed 1 \
  --ab --exchange-sweep > tmp/exchange-calib/sweep.txt 2>&1
s=$?
[ $s -ne 0 ] && { echo "sweep exited $s" >> tmp/exchange-calib/FAILURES; rc=1; }
cp -r tmp/exchange-calib/* "$CX_OUT"/
exit $rc
