#!/usr/bin/env python3
"""Run the preregistered source-stratified paired bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    load_json,
    load_jsonl,
    require_analysis_input_freeze,
    require_frozen_execution_state,
    write_json_new,
)
from karmavl.panel_c.statistics import primary_endpoint, source_stratified_paired_bootstrap


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-target", type=Path, required=True)
    parser.add_argument("--analysis-input-freeze", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
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
    contract = load_json(args.analysis_contract)
    rows = load_jsonl(args.paired_target)
    estimate = primary_endpoint(rows)
    result = source_stratified_paired_bootstrap(
        rows,
        replicates=int(contract["inference"]["replicates"]),
        seed=int(contract["inference"]["seed"]),
        confidence=float(contract["inference"]["confidence_level"]),
    )
    threshold = 0.015
    result["point_estimate"] = estimate["pooled_query_weighted_effect"]
    result["statistical_superiority"] = result["pooled_ci"][0] > 0.0
    result["practical_superiority"] = result["point_estimate"] >= threshold
    result["confirmatory_success"] = bool(
        result["statistical_superiority"] and result["practical_superiority"]
    )
    write_json_new(args.output, result)
    print(json.dumps({"replicates": result["replicates"]}, sort_keys=True))


if __name__ == "__main__":
    main()
