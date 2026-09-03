#!/usr/bin/env python3
"""Select fold-local B_A* or B_U*; global pooling is prohibited."""

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
from karmavl.panel_c.nested import select_fold_b_star


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-id", required=True)
    parser.add_argument("--held-out-source", required=True)
    parser.add_argument("--policy-baseline-rows", type=Path, required=True)
    parser.add_argument("--analysis-contract", type=Path, required=True)
    parser.add_argument("--objective", choices=("accuracy", "utility"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    add_frozen_execution_arguments(parser)
    args = parser.parse_args()
    require_frozen_execution_from_args(args)
    contract = load_json(args.analysis_contract)
    result = select_fold_b_star(
        load_jsonl(args.policy_baseline_rows),
        fold_id=args.fold_id,
        held_out_source=args.held_out_source,
        cost_coefficient=float(contract["policy"]["cost_coefficient_per_gpu_second"]),
        objective=args.objective,
        method_order=contract["simple_controls"]["fixed_method_tie_break_order"],
    )
    write_json_new(args.output, result)
    print(
        json.dumps(
            {
                "fold": args.fold_id,
                "objective": args.objective,
                "B_f_star": result["selected_B_f_star"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
