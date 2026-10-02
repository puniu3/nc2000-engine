use nc2000_engine::dex::{CondId, MoveId};
use nc2000_engine::fxhash::{FxHasher, SearchHasher};
use nc2000_engine::state::{DataBag, EffectState, Scalar, DK};
use std::hash::{Hash, Hasher};

fn hash<T: Hash, H: Hasher + Default>(value: &T) -> u64 {
    let mut h = H::default();
    value.hash(&mut h);
    h.finish()
}

fn check(bag: &DataBag, values: &[(DK, Scalar)]) {
    let mut entries = [None; 4];
    for (i, &entry) in values.iter().enumerate() {
        entries[i] = Some(entry);
    }
    let old = (entries, values.len() as u8);
    assert_eq!(hash::<_, FxHasher>(bag), hash::<_, FxHasher>(&old));
    assert_eq!(hash::<_, SearchHasher>(bag), hash::<_, SearchHasher>(&old));
    assert_eq!(
        hash::<_, std::collections::hash_map::DefaultHasher>(bag),
        hash::<_, std::collections::hash_map::DefaultHasher>(&old)
    );
    assert_eq!(
        format!("{bag:?}"),
        format!("DataBag {{ entries: {entries:?}, n: {} }}", values.len())
    );
}

#[test]
fn scalar_storage_preserves_all_variants_and_float_bits() {
    let values = [
        Scalar::Int(i64::MIN),
        Scalar::Int(i64::MAX),
        Scalar::Int(-1),
        Scalar::Int(0),
        Scalar::Float(-0.0),
        Scalar::Float(f64::INFINITY),
        Scalar::Float(f64::NEG_INFINITY),
        Scalar::Float(f64::from_bits(0x7ff8_1234_5678_abcd)),
        Scalar::Float(0.125),
        Scalar::Bool(false),
        Scalar::Bool(true),
        Scalar::MoveK(MoveId(u16::MAX)),
        Scalar::CondK(CondId(u16::MAX)),
        Scalar::Slot(0, 0),
        Scalar::Slot(255, 255),
    ];
    let keys = [DK::Counter, DK::Move, DK::TargetSlot, DK::Hp];
    for start in 0..values.len() {
        let mut bag = DataBag::default();
        let mut expected = Vec::new();
        check(&bag, &expected);
        for (i, key) in keys.into_iter().enumerate() {
            let value = values[(start + i) % values.len()];
            bag.push(key, value);
            expected.push((key, value));
            check(&bag, &expected);
        }
        for ((_, actual), (_, expected)) in bag.iter().zip(&expected) {
            if let (Scalar::Float(a), Scalar::Float(b)) = (actual, expected) {
                assert_eq!(a.to_bits(), b.to_bits());
            } else {
                assert_eq!(actual, *expected);
            }
        }
        bag.retain(|key| key != DK::Move && key != DK::Hp);
        expected.retain(|(key, _)| *key != DK::Move && *key != DK::Hp);
        check(&bag, &expected);
        bag.push(DK::Move, Scalar::MoveK(MoveId(2)));
        expected.push((DK::Move, Scalar::MoveK(MoveId(2))));
        check(&bag, &expected);
    }
}

#[test]
fn effect_updates_replace_values_and_preserve_scalar_equality() {
    let mut effect = EffectState::default();
    effect.set(DK::Counter, Scalar::Float(-0.0));
    let mut other = effect;
    other.set(DK::Counter, Scalar::Float(0.0));
    assert_eq!(effect.data, other.data);
    effect.set_int(DK::Counter, i64::MAX);
    assert_eq!(effect.data.iter().count(), 1);
    assert_eq!(effect.get_int(DK::Counter), i64::MAX);
    effect.set(DK::Move, Scalar::MoveK(MoveId(7)));
    assert_eq!(effect.get_move(), Some(MoveId(7)));
    effect.remove(DK::Counter);
    assert_eq!(effect.get_int(DK::Counter), 0);
    effect.set(DK::Counter, Scalar::Float(f64::NAN));
    assert_ne!(effect.data, effect.data);
}
