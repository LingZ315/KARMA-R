#!/usr/bin/env python3
"""Verify a real remote preregistration record and downloaded archive bytes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import write_json_new
from karmavl.panel_c.external import verify_external_timestamp_receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify_external_timestamp_receipt(args.receipt, args.archive)
    write_json_new(args.output, result)
    print(json.dumps({"status": result["status"], "provider": result["provider"]}, sort_keys=True))


if __name__ == "__main__":
    main()
