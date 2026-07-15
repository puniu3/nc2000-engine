//! Bit-exact parity of the Rust PRNG against reference vectors generated from
//! PS's sim/prng.ts by tools/gen-prng-vectors.js.

use conformance::fixture::repo_root;
use nc2000_engine::prng::Prng;
use serde_json::Value;

fn vectors() -> Vec<Value> {
    let path = repo_root().join("fixtures/prng-vectors.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path:?}: {e}"));
    serde_json::from_str(&text).unwrap()
}

fn prng_for(v: &Value) -> Prng {
    let seed = v["seed"].as_str().unwrap();
    // Vector "next" seeds are limb arrays serialized as "a,b,c,d".
    Prng::from_seed_str(seed).unwrap_or_else(|| panic!("unparseable seed {seed}"))
}

#[test]
fn prng_matches_reference_vectors() {
    let mut kinds_seen = 0;
    for v in vectors() {
        match v["kind"].as_str().unwrap() {
            "next" => {
                let mut p = prng_for(&v);
                for (i, expected) in v["values"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(
                        p.next_u32() as u64,
                        expected.as_u64().unwrap(),
                        "next #{i} from seed {}",
                        v["seed"]
                    );
                }
            }
            "random_n" => {
                let mut p = prng_for(&v);
                let n = v["n"].as_u64().unwrap() as u32;
                for (i, expected) in v["values"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(p.random(n) as u64, expected.as_u64().unwrap(), "random({n}) #{i}");
                }
            }
            "random_range" => {
                let mut p = prng_for(&v);
                let (from, to) = (v["from"].as_u64().unwrap() as u32, v["to"].as_u64().unwrap() as u32);
                for (i, expected) in v["values"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(
                        p.random_range(from, to) as u64,
                        expected.as_u64().unwrap(),
                        "random({from},{to}) #{i}"
                    );
                }
            }
            "random_chance" => {
                let mut p = prng_for(&v);
                let (num, den) = (v["num"].as_u64().unwrap() as u32, v["den"].as_u64().unwrap() as u32);
                for (i, expected) in v["values"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(
                        p.random_chance(num, den),
                        expected.as_bool().unwrap(),
                        "randomChance({num},{den}) #{i}"
                    );
                }
            }
            "shuffle" => {
                let mut p = prng_for(&v);
                let size = v["size"].as_u64().unwrap() as usize;
                for (r, expected) in v["runs"].as_array().unwrap().iter().enumerate() {
                    let mut arr: Vec<u64> = (0..size as u64).collect();
                    p.shuffle(&mut arr, 0, size);
                    let want: Vec<u64> =
                        expected.as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                    assert_eq!(arr, want, "shuffle run #{r}");
                }
            }
            "sample" => {
                let mut p = prng_for(&v);
                let size = v["size"].as_u64().unwrap() as usize;
                for (i, expected) in v["values"].as_array().unwrap().iter().enumerate() {
                    assert_eq!(
                        p.sample_index(size) as u64,
                        expected.as_u64().unwrap(),
                        "sample #{i}"
                    );
                }
            }
            "seed_after" => {
                let mut p = prng_for(&v);
                for _ in 0..v["draws"].as_u64().unwrap() {
                    p.random(16);
                }
                assert_eq!(p.seed_str(), v["endSeed"].as_str().unwrap(), "seed after draws");
            }
            other => panic!("unknown vector kind {other}"),
        }
        kinds_seen += 1;
    }
    assert!(kinds_seen >= 8, "expected all vector sets to run, got {kinds_seen}");
}
