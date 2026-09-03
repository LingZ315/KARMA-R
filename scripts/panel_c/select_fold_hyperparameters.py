#!/usr/bin/env python3
"""Select one fold's H2 margin for Gate 3 accuracy or Gate 4 utility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from karmavl.panel_c.common import (
    add_frozen_execution_arguments,
    load_json,
    load_jsonl,
    require_frozen_execution_from_args,
    write_json_new,
)
from karmavl.panel_c.nested import select_fold_margin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--policy-margin-rows", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--objective", choices=("accuracy", "utility"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    contract = load_json(args.analysis_contract)
    result = select_fold_margin(
        load_jsonl(args.policy_margin_rows),
        fold_id=args.fold_id,
        held_out_source=args.held_out_source,
        margin_grid=contract["policy"]["margin_grid"],
        objective=args.objective,
        cost_coefficient=float(contract["policy"]["cost_coefficient_per_gpu_second"]),
        fixed_margin_order=contract["policy"]["fixed_margin_tie_break_order"],
    )
    write_json_new(args.output, result)
    print(
        json.dumps(
            {
                "fold": args.fold_id,
                "objective": args.objective,
                "selected_margin": result["selected_margin"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
