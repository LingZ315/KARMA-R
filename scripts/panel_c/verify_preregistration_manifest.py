#!/usr/bin/env python3
"""Verify the local preregistration tree and deterministic archive."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import verify_file_manifest, verify_preregistration_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    local = verify_file_manifest(args.manifest, args.root)
    archived = verify_preregistration_archive(args.archive, manifest_path=args.manifest)
    if sorted(local, key=lambda row: row["path"]) != sorted(archived, key=lambda row: row["path"]):
        raise ValueError("local and archived manifest verification differ")
    print(json.dumps({"status": "PASS", "files": len(local)}, sort_keys=True))


if __name__ == "__main__":
    main()
