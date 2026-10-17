"""Verify every model file against the pinned Hugging Face tree."""

import argparse
import hashlib
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--tree", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
options = parser.parse_args()
results = []
for entry in json.loads(options.tree.read_text()):
    if entry["type"] != "file":
        continue
    name = Path(entry["path"])
    if name.is_absolute() or ".." in name.parts:
        raise ValueError("Invalid path in model tree")
    path = options.model / name
    row = {"path": str(name), "expected_bytes": entry["size"], "pass": False}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
                digest.update(block)
        row.update(bytes=path.stat().st_size, sha256=digest.hexdigest())
        row["pass"] = row["bytes"] == entry["size"]
        if entry.get("lfs"):
            row["expected_sha256"] = entry["lfs"]["oid"]
            row["pass"] = row["pass"] and row["sha256"] == row["expected_sha256"]
    results.append(row)
passed = bool(results) and all(row["pass"] for row in results)
options.output.write_text(json.dumps({"status": "PASS" if passed else "FAIL", "files": results}, indent=2) + "\n")
raise SystemExit(0 if passed else 1)
