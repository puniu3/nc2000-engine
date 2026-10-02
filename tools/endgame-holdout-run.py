import hashlib
import json
import pathlib
import shutil
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[1]
protocol_path = root / "findings/endgame-integration-v1/holdout-generation-protocol.json"
protocol_bytes = protocol_path.read_bytes()
protocol = json.loads(protocol_bytes)
for name in ("catalog", "prior"):
    assert hashlib.sha256((root / protocol[name]).read_bytes()).hexdigest() == protocol[f"{name}_sha256"]
destination = pathlib.Path(sys.argv[1]).resolve()
destination.mkdir(parents=True, exist_ok=False)
binary = destination / "selfplay_endgames"
shutil.copy2(root / "target/release/examples/selfplay_endgames", binary)
shutil.copy2(protocol_path, destination / "protocol.json")
output = destination / "games.jsonl"
command = [str(binary), "generate-weighted", str(output), str(protocol["games"]), str(protocol["iterations"]), str(protocol["seed"])]
manifest = {
    "protocol_sha256": hashlib.sha256(protocol_bytes).hexdigest(),
    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
    "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
    "dirty": subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip(),
    "command": command,
}
(destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
with (destination / "generation.log").open("x") as log:
    subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
games = [json.loads(line) for line in output.read_text().splitlines()]
assert len(games) == protocol["games"]
assert [game["game"] for game in games] == list(range(protocol["games"]))
(destination / "integrity.json").write_text(json.dumps({"games": len(games), "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, indent=2) + "\n")
print(f"Generated {len(games)} independent source games; endgame evaluation remains sealed until candidate selection.")
