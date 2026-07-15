//! Divergence reporting: locate the first point where the Rust engine's
//! snapshot differs from the golden fixture, with enough context for triage
//! (turn, log window, JSON path).

use serde_json::Value;

#[derive(Debug)]
pub struct Divergence {
    pub snapshot_index: usize,
    pub turn: u32,
    pub path: String,
    pub expected: String,
    pub actual: String,
    /// Golden log lines of the diverging snapshot — human/expert readable.
    pub log_context: Vec<String>,
}

impl std::fmt::Display for Divergence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(
            f,
            "snapshot #{} (turn {}): {} — expected {}, got {}",
            self.snapshot_index, self.turn, self.path, self.expected, self.actual
        )?;
        for line in &self.log_context {
            writeln!(f, "    {line}")?;
        }
        Ok(())
    }
}

/// First difference between two JSON values, as a path + value pair.
pub fn first_diff(expected: &Value, actual: &Value, path: &str) -> Option<(String, String, String)> {
    match (expected, actual) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, va) in a {
                let sub = format!("{path}.{k}");
                match b.get(k) {
                    Some(vb) => {
                        if let Some(d) = first_diff(va, vb, &sub) {
                            return Some(d);
                        }
                    }
                    None => return Some((sub, va.to_string(), "<missing>".into())),
                }
            }
            for k in b.keys() {
                if !a.contains_key(k) {
                    return Some((format!("{path}.{k}"), "<missing>".into(), b[k].to_string()));
                }
            }
            None
        }
        (Value::Array(a), Value::Array(b)) => {
            for (i, (va, vb)) in a.iter().zip(b.iter()).enumerate() {
                if let Some(d) = first_diff(va, vb, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            if a.len() != b.len() {
                return Some((
                    format!("{path}.len"),
                    a.len().to_string(),
                    b.len().to_string(),
                ));
            }
            None
        }
        _ => {
            if expected == actual {
                None
            } else {
                Some((path.to_string(), expected.to_string(), actual.to_string()))
            }
        }
    }
}
