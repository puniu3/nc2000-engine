# EarlyRest independent confirmation

The frozen 64-pair stress confirmation scored 36 wins for EarlyPerish and
40 for EarlyRest: 14 improved, 10 worsened, 40 tied pairs. Two additional
on-pool pairs had identical full trajectories. All 132 games terminated.
All 3,823 decisions replayed legally with matching state keys, seeds and
outcomes. All 96 first-divergence searches reproduced recorded choices.

The predeclared equal-weight eight-case Dirichlet model gives
P(improvement) = 80.197%, clearing its 80% criterion. Posterior mean is
+5.54 percentage points, with 95% credible interval [-7.32, +18.30].
Prior sensitivity is 81.03%, 80.20%, 76.53%, 69.33%, 61.41% at masses
0.5, 1, 4, 16, 64. This is a model-conditional benchmark decision, not
evidence of expert-human superiority or proximity to optimal play.

Rest cases scored 18 versus 17; no-Rest cases scored 18 versus 23.
The intended recovery-defense mechanism is unproven. Four improved
Snorlax/Umbreon no-Rest pairs first change Pursuit to Toxic against
Machamp after Rest enters its sampled loadouts. Other changes involve
bench beliefs even when active-foe sampled kits match. These examples
identify exposure and first divergence, not a causal decomposition of
whole-game wins.

The secondary review found no no-Rest regression, capped outcome or
control mismatch. The fixed primary criterion passed without adding
trials, changing priors or pooling the 32-pair pilot. EarlyRest is now the local source default. All 170 relevant tests and
workspace/all-targets compilation pass. A source comparison confirms the
runtime implementation differs from the frozen candidate only in which
enum variant is default. The constructor matches the explicit candidate
on 2,000 sampling seeds. Public deployment remains unchanged.

summary.json records default_changed=false at experiment time;
decision.json and adoption-verification.json record subsequent adoption.

## Reproduction

The candidate source was frozen at commit 7b457c9. Protocol, candidate
hashes and analysis plan were committed before outcomes. The archived
pilot-manifest / pilot-verification / pilot-source filenames are shared
analyzer conventions; in this directory they contain confirmation data.
The stage field in summary.json distinguishes this independent sample.

Run tools/early-rest-confirmation-report with the frozen native candidate
sources and the original temporary game/corpus inputs. The script checks
source hashes before replay and aggregation. records.json.gz also contains
the anonymous generated game traces and diagnostic corpus rows. No raw
spectator logs or player handles are included.
