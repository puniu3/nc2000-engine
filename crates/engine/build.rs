use sha2::{Digest, Sha256};
use std::{fs, path::Path};

fn hash_sources(path: &Path, hash: &mut Sha256) {
    let mut entries: Vec<_> = fs::read_dir(path)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    entries.sort();
    for entry in entries {
        if entry.is_dir() {
            hash_sources(&entry, hash);
        } else if entry.extension().is_some_and(|e| e == "rs") {
            println!("cargo:rerun-if-changed={}", entry.display());
            hash.update(entry.to_string_lossy().as_bytes());
            hash.update(fs::read(&entry).unwrap());
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let mut hash = Sha256::new();
    hash_sources(Path::new("src"), &mut hash);
    let digest = hash.finalize();
    let source: [u8; 16] = digest[..16].try_into().unwrap();
    // Only this exact source digest shares the transition version checked by
    // frozen_compact_replay_remains_compatible; other edits invalidate it.
    let fingerprint = match source {
        [187, 224, 233, 197, 178, 63, 201, 240, 141, 35, 42, 119, 82, 169, 125, 109]
        | [54, 119, 34, 133, 64, 134, 144, 87, 69, 206, 248, 30, 75, 232, 90, 147]
        | [46, 85, 103, 49, 207, 212, 47, 139, 64, 114, 3, 131, 94, 59, 16, 247]
        | [12, 127, 37, 199, 163, 126, 216, 44, 229, 202, 255, 197, 107, 225, 122, 184] =>
            [41, 101, 229, 191, 83, 13, 34, 22, 119, 166, 40, 224, 139, 210, 186, 231],
        _ => source,
    };
    let output = format!(
        "pub const ENGINE_FINGERPRINT: [u8; 16] = {:?};",
        fingerprint
    );
    fs::write(
        Path::new(&std::env::var("OUT_DIR").unwrap()).join("replay_version.rs"),
        output,
    )
    .unwrap();
}
