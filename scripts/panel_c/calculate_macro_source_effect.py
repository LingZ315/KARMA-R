#!/usr/bin/env python3
"""Calculate preregistered macro, median, and source-sign summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    load_jsonl,
    require_analysis_input_freeze,
    require_frozen_execution_state,
    write_json_new,
)
from karmavl.panel_c.statistics import primary_endpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-target", type=Path, required=True)
    parser.add_argument("--analysis-input-freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    args = parser.parse_args()
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    require_analysis_input_freeze(args.analysis_input_freeze, paired_rows=args.paired_target)
    result = primary_endpoint(load_jsonl(args.paired_target))
    write_json_new(args.output, result)
    print(json.dumps({"sources": len(result["source_effects"])}, sort_keys=True))


if __name__ == "__main__":
    main()
