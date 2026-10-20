"""Check the SHA-256 inventory of a local release or a downloaded subset."""

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=["full", "available"], default="full")
    args = parser.parse_args()
    checked, missing, errors = 0, [], []
    for line in (ROOT / "MANIFEST.sha256").read_text().splitlines():
        expected, relative = line.split("  ", 1)
        path = ROOT / relative
        if not path.resolve().is_relative_to(ROOT):
            raise ValueError(f"Invalid inventory path: {relative}")
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256(path) != expected:
            errors.append(relative)
        checked += 1
        if checked % 250 == 0:
            print(f"Checked {checked} files", flush=True)
    print(json.dumps({"checked": checked, "missing": len(missing), "mismatched": errors,
                      "scope": args.scope}, indent=2))
    if errors or (missing and args.scope == "full"):
        if missing:
            print("Missing files: " + ", ".join(missing[:12]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
