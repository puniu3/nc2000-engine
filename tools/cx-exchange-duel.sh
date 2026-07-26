#!/usr/bin/env bash
# The sovereign reading on the exchange term: seed-paired duel, shipped default
# vs the candidate weight, same agent, same budget.
#
# The 570-position corpus sweep improved calibration (r 0.585 -> 0.644, Brier
# 0.2116 -> 0.2047) but the three criteria disagree on the weight -- r peaks at
# 0.75-1.0, Brier and MSE at 0.5 -- so both are duelled. Calibration is
# necessary-not-sufficient here by measured precedent: M17c's heal-blind
# variant fit the anchors twice as well and lost its duel 0.39.
#
# Two budgets, because an eval term can be budget-dependent (M17b's whole
# point): 3000, and 10000 as the closest affordable step toward the shipped
# 30000.
#
# Run under cx from the repo root:
#   cx submit -c 16 -T 21600 -n exchange-duel -d <repo> -b tmp/exchange-duel -- bash tools/cx-exchange-duel.sh
set -u
export PATH="$HOME/.cargo/bin:$PATH"
command -v cargo >/dev/null || { echo "NO CARGO on PATH"; exit 127; }
test -f Cargo.toml || { echo "NO CARGO.TOML in $PWD"; exit 66; }

cargo build --release -q -p nc2000-bot --example eval_ab_duel || exit 65
mkdir -p tmp/exchange-duel
rc=0
for w in 0.5 0.75; do
  for iters in 3000 10000; do
    tag="w${w}-i${iters}"
    echo "=== exchange $w @ $iters  $(date -Is)"
    timeout 9000 ./target/release/examples/eval_ab_duel \
      --games 800 --iters "$iters" --seed 11 --exchange "$w" \
      > "tmp/exchange-duel/$tag.txt" 2>&1
    s=$?
    if [ $s -ne 0 ]; then
      echo "$tag exited $s" >> tmp/exchange-duel/FAILURES
      rc=1
    fi
    cp -r tmp/exchange-duel/* "$CX_OUT"/ 2>/dev/null
  done
done
cp -r tmp/exchange-duel/* "$CX_OUT"/
exit $rc
