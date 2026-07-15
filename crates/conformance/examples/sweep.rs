//! M2 grind-loop tool: run every fixture in a corpus, catching per-battle
//! panics (unported callbacks), and print one line per battle plus a
//! frequency table of missing callbacks. Usage:
//!   cargo run -p conformance --example sweep [-- puredata|full]

use std::collections::BTreeMap;
use std::panic;

use conformance::fixture::{corpus_files, repo_root, Fixture};
use conformance::{load_dex, replay};

fn main() {
    let corpus = std::env::args().nth(1).unwrap_or_else(|| "full".into());
    let dex = load_dex();
    let files = corpus_files(&repo_root().join("fixtures/corpus-v1").join(&corpus));
    let mut missing: BTreeMap<String, u32> = BTreeMap::new();
    let mut pass = 0;
    panic::set_hook(Box::new(|_| {})); // silence the default backtrace spam
    for path in &files {
        let fx = Fixture::load(path).unwrap();
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| replay(&dex, &fx)));
        match result {
            Ok(Ok(())) => {
                pass += 1;
                println!("{name}: OK");
            }
            Ok(Err(e)) => {
                let s = format!("{e:?}");
                println!("{name}: DIVERGED {}", &s[..s.len().min(300)]);
            }
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "<non-string panic>".into());
                *missing.entry(msg.clone()).or_default() += 1;
                println!("{name}: PANIC {msg}");
            }
        }
    }
    let _ = panic::take_hook();
    println!("\n{pass}/{} OK", files.len());
    if !missing.is_empty() {
        println!("\nmissing callbacks (battles blocked):");
        for (msg, n) in &missing {
            println!("  {n:2}x {msg}");
        }
    }
}
