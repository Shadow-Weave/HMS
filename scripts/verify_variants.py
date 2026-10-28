#!/usr/bin/env python3
"""Check original HMS Pro and Elite files against their imported SHA-256 snapshots."""

import hashlib
import json
import sys
from pathlib import Path


def verify(root, snapshots):
    errors = []
    for name, snapshot in snapshots.items():
        base = root / snapshot["root"]
        for relative, expected in snapshot["files"].items():
            path = base / relative
            if not path.is_file():
                errors.append(f"{name}: missing {relative}")
            elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                errors.append(f"{name}: changed {relative}")
    return errors


def main():
    root = Path(__file__).resolve().parents[1]
    snapshots = json.loads((root / "variants" / "sources.json").read_text(encoding="utf-8"))
    errors = verify(root, snapshots)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    for name, snapshot in snapshots.items():
        print(f"{name}: {len(snapshot['files'])} original files unchanged ({snapshot['revision'][:12]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
