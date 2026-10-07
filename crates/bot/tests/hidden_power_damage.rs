#[path = "support/hidden_power_damage.rs"]
mod support;

use support::*;

#[test]
fn category_matches_native_damage_dependencies_for_every_type_and_power() {
    let dex = conformance::load_dex();
    let mut failures = Vec::new();
    for index in 0..16 {
        for high in [false, true] {
            let b = fixture(&dex, index, high);
            let baseline = estimate(&dex, &b);
            let (native, _) = damage(&dex, &b);
            assert!(native.keys().any(|damage| *damage > 0));
            for attacker in [true, false] {
                let unchanged = perturb(&b, index, attacker, false);
                assert_eq!(native, damage(&dex, &unchanged).0);
                if estimate(&dex, &unchanged) != baseline {
                    failures.push((TYPES[index], high, attacker, "unused stat changes estimate"));
                }
                let changed = perturb(&b, index, attacker, true);
                assert_ne!(native, damage(&dex, &changed).0);
                if estimate(&dex, &changed) == baseline {
                    failures.push((TYPES[index], high, attacker, "used stat ignored"));
                }
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}
