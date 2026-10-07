# Hidden Power category in the bot damage estimate

The ordinary damage estimate resolved Hidden Power type and power from the user, but selected Attack/Defense from the placeholder move's static Physical category. The native engine independently selects Physical/Special after its move-modification callback. Special Hidden Power therefore used the wrong stats in evaluation and rollout damage estimates.

The fix resolves category alongside type and power. It is applied to the local default bot; public deployment is unchanged. No feature gate, weight tuning, or human calibration is involved. The previously frozen 104-game information trial remains a measurement of the old estimator; it does not establish the effect of this fix or prove that this bug caused its outcome differences.

The new regression uses all 16 types at powers 35 and 70. It perturbs the attacker's and defender's relevant/irrelevant stored stats separately and compares full native one-turn damage distributions. These are artificial dependency checks, not legal-team strength fixtures. Native enumeration runs the real move-modification callback; it does not copy the estimator's category into a synthetic move. All types deal nonzero damage in the fixture, avoiding vacuous immunity checks.

Before the fix, 64 of 128 estimator dependency comparisons contradicted the engine, covering every special type at both powers. After the fix all 128 agree. The 16 physical cases and 160 ordinary-move estimates remain identical. Native damage distributions before/after are identical; each measurement used 4,702 native runs. Four data mutations are rejected by the archive checker. The existing 230 bot unit tests and the new regression test pass.

In the Alakazam-to-Mew Ice fixture, the power-35 estimate changes from 10.1804 to 18.5098 damage; native mean damage including critical hits is 19.0657. At power 70 the estimate changes from 20.3608 to 36.0941; native mean is 37.7548. The estimator remains approximate and excludes critical hits, so exact mean-damage agreement is not claimed.

`before.json` and `after.json` preserve distributions and perturbations. `before-source.json` identifies the baseline, and `summary.json` records hashes and checks. `tools/hp-category-audit` rechecks the archive. General strength, superiority over the deployed bot, and optimality remain unproved.
