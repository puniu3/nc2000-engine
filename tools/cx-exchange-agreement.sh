#!/usr/bin/env bash
# The shipping question for the exchange term, once the duel came back parity
# at both candidate weights and both budgets (A-scores 0.494 / 0.485 / 0.504 /
# 0.490, all straddling 0.5, think time identical).
#
# At equal strength the owner's criterion is which bot reads as more sensible
# to a human, because that is what does or does not generate false-positive bug
# reports. Corpus agreement is the wrong instrument for "is the eval right" —
# the M16b switch cluster turned out to be an argmax artifact — but it is
# exactly the right instrument for "would a human watching agree".
#
# Product budget, because a 3k measurement describes a bot twice as noisy as
# the shipped one. The 30k baseline for these same battles and seed already
# exists locally as tmp/ha-30k-base2.jsonl.
#
# Run under cx from the repo root:
#   cx submit -c 16 -T 21600 -n exchange-agreement -d <repo> -b tmp/exchange-agree -- bash tools/cx-exchange-agreement.sh
set -u
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "NO CARGO on PATH"; exit 127; }
test -f Cargo.toml || { echo "NO CARGO.TOML in $PWD"; exit 66; }

CORPUS=tmp/corpus-spectator
if [ ! -d "$CORPUS" ] || [ -z "$(ls -A "$CORPUS" 2>/dev/null)" ]; then
  test -f data/corpus-spectator-logs.zip || { echo "NO CORPUS ARCHIVE"; exit 67; }
  mkdir -p "$CORPUS"
  # no unzip on the cx image; python3 is there
  python3 -m zipfile -e data/corpus-spectator-logs.zip "$CORPUS" || {
    echo "UNZIP FAILED"
    exit 68
  }
  if [ -z "$(ls "$CORPUS"/*.raw.log 2>/dev/null)" ]; then
    inner=$(find "$CORPUS" -name '*.raw.log' -exec dirname {} \; 2>/dev/null | head -1)
    [ -n "$inner" ] && mv "$inner"/*.raw.log "$CORPUS"/
  fi
fi
n=$(ls "$CORPUS"/*.raw.log 2>/dev/null | wc -l)
echo "corpus battles: $n"
[ "$n" -ge 500 ] || { echo "CORPUS TOO SMALL ($n)"; exit 69; }

cargo build --release -q -p nc2000-bot --example human_agreement || exit 65
mkdir -p tmp/exchange-agree
rc=0
# baseline re-run too: the local one predates the reveal-channel fixes, and a
# shipping decision should not rest on arms measured through different code.
for arm in "base:" "w0.5:--exchange 0.5" "w0.75:--exchange 0.75"; do
  tag="${arm%%:*}"
  flag="${arm#*:}"
  echo "=== $tag  $(date -Is)"
  # shellcheck disable=SC2086
  timeout 9000 ./target/release/examples/human_agreement \
    --corpus "$CORPUS" --battles 0-59 --iters 30000 --threads 16 --seed 1 \
    $flag --out "tmp/exchange-agree/$tag.jsonl" > "tmp/exchange-agree/$tag.log" 2>&1
  s=$?
  [ $s -ne 0 ] && { echo "$tag exited $s" >> tmp/exchange-agree/FAILURES; rc=1; }
  cp -r tmp/exchange-agree/* "$CX_OUT"/ 2>/dev/null
done
cp -r tmp/exchange-agree/* "$CX_OUT"/
exit $rc
