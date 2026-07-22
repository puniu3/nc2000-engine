use std::fs;
use std::path::{Path, PathBuf};

fn fnv1a64_update(hash: u64, bytes: &[u8]) -> u64 {
    bytes.iter().fold(hash, |hash, &byte| {
        (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3)
    })
}

fn fingerprint_part(hash: u64, bytes: &[u8]) -> u64 {
    let hash = fnv1a64_update(hash, &(bytes.len() as u64).to_le_bytes());
    fnv1a64_update(hash, bytes)
}

fn rust_files(dir: &Path, files: &mut Vec<PathBuf>) {
    let mut entries = fs::read_dir(dir)
        .unwrap_or_else(|error| panic!("read {}: {error}", dir.display()))
        .map(|entry| entry.expect("read source entry").path())
        .collect::<Vec<_>>();
    entries.sort();
    for path in entries {
        if path.is_dir() {
            rust_files(&path, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs") {
            files.push(path);
        }
    }
}

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let root = manifest.parent().unwrap().parent().unwrap();
    let mut files = vec![
        root.join("Cargo.toml"),
        root.join("Cargo.lock"),
        manifest.join("Cargo.toml"),
        manifest.join("build.rs"),
        root.join("crates/engine/Cargo.toml"),
        root.join("crates/conformance/Cargo.toml"),
        manifest.join("examples/endgame_exactness_corpus.rs"),
        manifest.join("examples/anchor_gate.rs"),
    ];
    rust_files(&manifest.join("src"), &mut files);
    rust_files(&root.join("crates/engine/src"), &mut files);
    rust_files(&root.join("crates/conformance/src"), &mut files);
    files.sort();
    files.dedup();

    let mut hash = 0xcbf2_9ce4_8422_2325;
    hash = fingerprint_part(hash, b"nc2000-m17e-solver-build-v3");
    for path in files {
        println!("cargo:rerun-if-changed={}", path.display());
        let relative = path.strip_prefix(root).unwrap_or(&path).to_string_lossy();
        let contents = fs::read(&path).unwrap_or_else(|error| {
            panic!("read build identity source {}: {error}", path.display())
        });
        hash = fingerprint_part(hash, relative.as_bytes());
        hash = fingerprint_part(hash, &contents);
    }
    println!(
        "cargo:rustc-env=M17E_SOLVER_BUILD_FINGERPRINT=fnv1a64:{hash:016x}:m17e-solver-build-v3"
    );
}
