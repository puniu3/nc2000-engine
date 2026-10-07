# Native region solver v1

Protocol registered at `98386a8`. The experimental patch applies to the native serialization adapter's bounds.rs whose base hash is bound by `parent_bounds_source_sha256`. The resulting source hash and copied binary are also bound. Main bounds.rs was not modified.

```sh
python3 tools/native-region-solver-audit tmp/native-region-solver-v1
```

Each arm JSON in scratch is the protocol plus its fixed capacity, arm name and pass identifier. Raw output archives use zero gzip timestamps; summary hashes are of uncompressed bytes. Ordinary bot defaults are unaffected.
