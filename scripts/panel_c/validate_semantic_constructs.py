#!/usr/bin/env python3
"""Compare frozen semantic-classifier labels with retained human constructs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from karmavl.panel_c.common import load_jsonl, write_json_new


HUMAN_TO_PRIMARY_SUBTYPE = {
    "general_vqa": ("general_visual_reasoning", "none"),
    "visual_reasoning": ("general_visual_reasoning", "none"),
    "fine_grained_perception": ("fine_grained_perception", "none"),
    "ocr": ("ocr_text_reading", "none"),
    "document_understanding": ("structured_artifact_reasoning", "document"),
    "chart_reasoning": ("structured_artifact_reasoning", "chart_graph"),
    "spatial_reasoning": ("spatial_reasoning", "none"),
    "visual_counting": ("quantitative_reasoning", "visual_counting"),
    "mathematical_reasoning": ("quantitative_reasoning", "mathematical_reasoning"),
    "science_reasoning": ("science_reasoning", "none"),
}


def _macro_f1(truth: list[str], predicted: list[str]) -> tuple[float, dict[str, float], dict[str, float]]:
    labels = sorted(set(truth) | set(predicted))
    f1: dict[str, float] = {}
    recall: dict[str, float] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall[label] = tp / (tp + fn) if tp + fn else 0.0
        f1[label] = (
            2 * precision * recall[label] / (precision + recall[label])
            if precision + recall[label]
            else 0.0
        )
    return sum(f1.values()) / len(f1), f1, recall


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-audit", type=Path, required=True)
    parser.add_argument("--classifier-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    with args.human_audit.open(encoding="utf-8-sig", newline="") as stream:
        human_rows = list(csv.DictReader(stream))
    human = {str(row["hashed_item_id"]): row for row in human_rows}
    if len(human) != len(human_rows):
        raise ValueError("duplicate human audit identifier")
    classifier_rows = load_jsonl(args.classifier_output)
    forbidden = {"correct", "correctness", "candidate_outcome", "utility_outcome", "route_id"}
    if any(forbidden & {key.casefold() for key in row} for row in classifier_rows):
        raise PermissionError("candidate outcome information reached construct validation")
    classifier = {
        str(row.get("hashed_item_id", row.get("query_id"))): row for row in classifier_rows
    }
    if set(classifier) != set(human):
        raise ValueError("classifier and human construct identifiers are not an exact match")
    truth_primary: list[str] = []
    predicted_primary: list[str] = []
    truth_subtype: list[str] = []
    predicted_subtype: list[str] = []
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    ambiguity = Counter()
    for item_id in sorted(human):
        human_primary, human_subtype = HUMAN_TO_PRIMARY_SUBTYPE[human[item_id]["adjudicated_label"]]
        machine = classifier[item_id]
        machine_primary = str(machine["primary_class"])
        machine_subtype = str(machine.get("subtype") or "none")
        truth_primary.append(human_primary)
        predicted_primary.append(machine_primary)
        truth_subtype.append(human_subtype)
        predicted_subtype.append(machine_subtype)
        confusion[human_primary][machine_primary] += 1
        human_ambiguous = human[item_id]["ambiguity_flag"].casefold() == "true"
        machine_ambiguous = bool(machine.get("ambiguity"))
        ambiguity[f"human_{int(human_ambiguous)}_machine_{int(machine_ambiguous)}"] += 1
    macro, per_class_f1, recall = _macro_f1(truth_primary, predicted_primary)
    subtype_mask = [index for index, value in enumerate(truth_subtype) if value != "none"]
    result = {
        "schema_version": 1,
        "artifact_role": "panel_c_pre_target_semantic_construct_validation",
        "target_candidate_outcomes_used": False,
        "n": len(truth_primary),
        "primary_class_accuracy": sum(t == p for t, p in zip(truth_primary, predicted_primary, strict=True))
        / len(truth_primary),
        "primary_class_macro_f1": macro,
        "per_class_f1": per_class_f1,
        "per_class_recall": recall,
        "confusion_matrix": {truth: dict(counts) for truth, counts in sorted(confusion.items())},
        "compatible_subtype_n": len(subtype_mask),
        "compatible_subtype_accuracy": (
            sum(truth_subtype[i] == predicted_subtype[i] for i in subtype_mask) / len(subtype_mask)
            if subtype_mask
            else None
        ),
        "ambiguity_behavior": dict(ambiguity),
    }
    write_json_new(args.output, result)
    print(json.dumps({"n": result["n"], "target_candidate_outcomes_used": False}, sort_keys=True))


if __name__ == "__main__":
    main()
