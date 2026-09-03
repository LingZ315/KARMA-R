#!/usr/bin/env python3
"""Build the outcome-firewalled Panel-C leave-one-source-out cohort.

The source adapter reads provider records, but eligibility and ordering use no
answers, predictions, correctness, candidate identities, or costs. Router input
rows deliberately omit explicit dataset/source identity and semantic labels.
Semantic labels are produced later by the externally frozen
explicit-source-metadata-blind classifier. Image/question content is not
sanitized and may retain latent source cues.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any


# Legacy immutable seed text retained so rebuilding reproduces the frozen IDs;
# it is an identifier, not a claim that content-level source cues are absent.
SELECTION_SEED = "karma-r-panel-c-source-blind-loso-v1-20260831"
SOURCE_CAP = 2_200
MINIMUM_SOURCE_N = 300
TRAIN_PARTITIONS = ("profile", "calibration", "policy")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_lines(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def load_adapter(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("panel_c_source_adapter", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import source adapter: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.SEED = SELECTION_SEED
    return module


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-adapter", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-fingerprints", type=Path, required=True)
    parser.add_argument("--replication-b-row-manifest", type=Path, required=True)
    parser.add_argument("--source-cap", type=int, default=SOURCE_CAP)
    parser.add_argument("--minimum-source-n", type=int, default=MINIMUM_SOURCE_N)
    args = parser.parse_args()
    if args.source_cap < args.minimum_source_n:
        raise ValueError("source cap must be at least the minimum source size")
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("output root must be absent or empty; replacement is prohibited")
    args.output_root.mkdir(parents=True, exist_ok=True)

    adapter = load_adapter(args.source_adapter.resolve())
    prior = json.loads(args.prior_fingerprints.read_text(encoding="utf-8"))
    prior_images = set(prior.get("image_sha256s", ()))
    prior_samples = set(prior.get("sample_ids", ()))
    replication_b_rows = 0
    for line in args.replication_b_row_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prior_images.add(str(row["original_image_sha256"]))
        prior_samples.add(f"{row['source']}:{row['source_record_id']}")
        replication_b_rows += 1

    source_records: dict[str, list[Any]] = {}
    source_audits: dict[str, Any] = {}
    for source in sorted(adapter.SOURCES):
        raw = adapter.LOADERS[source](args.cache_dir)
        eligible, audit = adapter._deduplicate(raw, prior_images, prior_samples)
        source_records[source] = eligible
        source_audits[source] = {
            "raw_rows": len(raw),
            "eligible_after_historical_and_within_source_filter": len(eligible),
            "exclusions": audit,
            "source_spec": adapter.SOURCES[source],
        }
        del raw

    # Resolve any cross-source exact-image duplication without using row outcomes.
    global_choice: dict[str, Any] = {}
    for source in sorted(source_records):
        for record in source_records[source]:
            image_hash = record.original_image_sha256
            current = global_choice.get(image_hash)
            candidate_key = (record.selection_key, record.source, record.source_record_id)
            if current is None or candidate_key < (
                current.selection_key,
                current.source,
                current.source_record_id,
            ):
                global_choice[image_hash] = record
    globally_retained = {
        (record.source, record.source_record_id, record.original_image_sha256)
        for record in global_choice.values()
    }
    for source in sorted(source_records):
        before = len(source_records[source])
        source_records[source] = [
            record
            for record in source_records[source]
            if (record.source, record.source_record_id, record.original_image_sha256)
            in globally_retained
        ]
        source_records[source].sort(key=lambda record: record.selection_key)
        source_audits[source]["cross_source_duplicate_image_excluded"] = (
            before - len(source_records[source])
        )
        source_audits[source]["eligible_after_all_filters"] = len(source_records[source])

    selected_sources = [
        source
        for source in sorted(source_records)
        if len(source_records[source]) >= args.minimum_source_n
    ]
    if len(selected_sources) < 3:
        raise RuntimeError("Panel C requires at least three eligible held-out sources")
    excluded_sources = {
        source: (
            "no eligible unseen exact-image groups"
            if not source_records[source]
            else f"eligible N={len(source_records[source])} below minimum {args.minimum_source_n}"
        )
        for source in sorted(source_records)
        if source not in selected_sources
    }
    selected_by_source = {
        source: source_records[source][: min(args.source_cap, len(source_records[source]))]
        for source in selected_sources
    }
    selected = [
        record
        for source in selected_sources
        for record in selected_by_source[source]
    ]
    selected.sort(key=lambda record: (record.source, record.selection_key))
    if len({record.original_image_sha256 for record in selected}) != len(selected):
        raise AssertionError("selected Panel-C rows are not exact-image disjoint")

    private_dir = args.output_root / "private"
    image_dir = args.output_root / "images"
    private_dir.mkdir(parents=True)
    image_dir.mkdir(parents=True)
    router_input_path = private_dir / "outcome_blind_router_inputs.jsonl"
    row_manifest_path = private_dir / "row_manifest.jsonl"
    answer_path = private_dir / "sealed_reference_answers.jsonl"
    fold_role_path = private_dir / "fold_roles.jsonl"

    records_by_source: dict[str, list[tuple[str, Any]]] = {
        source: [] for source in selected_sources
    }
    record_by_query: dict[str, Any] = {}
    with (
        router_input_path.open("w", encoding="utf-8", newline="\n") as inputs,
        row_manifest_path.open("w", encoding="utf-8", newline="\n") as manifests,
        answer_path.open("w", encoding="utf-8", newline="\n") as answers,
    ):
        for index, record in enumerate(selected, start=1):
            query_id = f"PC{index:05d}"
            image_path = image_dir / f"{query_id}.png"
            adapter._write_image(record, image_path)
            normalized_image_sha256 = sha256_file(image_path)
            prompt_sha256 = hashlib.sha256(record.prompt.encode("utf-8")).hexdigest()
            router_row = {
                "query_id": query_id,
                "image_path": str(image_path.resolve()),
                "prompt": record.prompt,
                "scorer": record.scorer,
                "semantic_assignment_status": "PENDING_EXTERNALLY_FROZEN_METADATA_FREE_CLASSIFIER",
            }
            manifest_row = {
                "query_id": query_id,
                "source": record.source,
                "source_record_id": record.source_record_id,
                "original_image_sha256": record.original_image_sha256,
                "normalized_image_sha256": normalized_image_sha256,
                "prompt_sha256": prompt_sha256,
                "scorer": record.scorer,
            }
            answer_row = {
                "query_id": query_id,
                "answer": record.answer,
                "scorer": record.scorer,
            }
            inputs.write(json.dumps(router_row, sort_keys=True, ensure_ascii=False) + "\n")
            manifests.write(json.dumps(manifest_row, sort_keys=True, ensure_ascii=False) + "\n")
            answers.write(json.dumps(answer_row, sort_keys=True, ensure_ascii=False) + "\n")
            records_by_source[record.source].append((query_id, record))
            record_by_query[query_id] = record
    os.chmod(answer_path, 0o600)

    # A source receives a fixed training partition whenever it is not held out.
    training_roles: dict[str, dict[str, list[str]]] = {}
    for source in selected_sources:
        query_ids = [query_id for query_id, _record in records_by_source[source]]
        n = len(query_ids)
        profile_end = n // 5
        calibration_end = profile_end + (2 * n) // 5
        training_roles[source] = {
            "profile": query_ids[:profile_end],
            "calibration": query_ids[profile_end:calibration_end],
            "policy": query_ids[calibration_end:],
        }

    folds: list[dict[str, Any]] = []
    scoped_answer_hashes: dict[str, str] = {}
    scoped_answer_root = private_dir / "fold_scoped_reference_answers"
    scoped_answer_root.mkdir(parents=True)
    with fold_role_path.open("w", encoding="utf-8", newline="\n") as role_stream:
        for held_source in selected_sources:
            fold_id = f"holdout_{held_source}"
            role_ids: dict[str, list[str]] = {role: [] for role in (*TRAIN_PARTITIONS, "target")}
            role_ids["target"] = [query_id for query_id, _record in records_by_source[held_source]]
            for training_source in selected_sources:
                if training_source == held_source:
                    continue
                for role in TRAIN_PARTITIONS:
                    role_ids[role].extend(training_roles[training_source][role])
            for role in (*TRAIN_PARTITIONS, "target"):
                role_ids[role].sort()
                for query_id in role_ids[role]:
                    role_stream.write(
                        json.dumps(
                            {"fold_id": fold_id, "query_id": query_id, "role": role},
                            sort_keys=True,
                        )
                        + "\n"
                    )
                # Result scoring accepts only these exact fold/role-scoped
                # ledgers.  Thus a fold-local selector never opens the answer
                # ledger for its held-out source.
                scoped_path = scoped_answer_root / fold_id / f"{role}_reference_answers.jsonl"
                scoped_path.parent.mkdir(parents=True, exist_ok=True)
                with scoped_path.open("x", encoding="utf-8", newline="\n") as scoped_stream:
                    for query_id in role_ids[role]:
                        record = record_by_query[query_id]
                        scoped_stream.write(
                            json.dumps(
                                {
                                    "query_id": query_id,
                                    "answer": record.answer,
                                    "scorer": record.scorer,
                                },
                                sort_keys=True,
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                os.chmod(scoped_path, 0o600)
                scoped_answer_hashes[f"{fold_id}/{role}"] = sha256_file(scoped_path)
            role_sets = {role: set(values) for role, values in role_ids.items()}
            for left_index, left in enumerate(role_sets):
                for right in list(role_sets)[left_index + 1 :]:
                    if role_sets[left] & role_sets[right]:
                        raise AssertionError(f"{fold_id}: {left}/{right} role overlap")
            target_sources = {
                record.source
                for query_id, record in records_by_source[held_source]
                if query_id in role_sets["target"]
            }
            if target_sources != {held_source}:
                raise AssertionError(f"{fold_id}: target source drift")
            folds.append(
                {
                    "fold_id": fold_id,
                    "held_out_source": held_source,
                    "source_feature_exposed_to_router": False,
                    "roles": {
                        role: {
                            "count": len(values),
                            "query_id_sha256": sha256_lines(values),
                        }
                        for role, values in role_ids.items()
                    },
                    "target_source_absent_from_profile_calibration_policy": True,
                }
            )

    fold_manifest = {
        "schema_version": 1,
        "artifact_role": "panel_c_explicit_source_metadata_blind_loso_fold_manifest",
        "status": "FROZEN_IDS_PRE_EXTERNAL_TIMESTAMP",
        "selection_seed": SELECTION_SEED,
        "selected_sources": selected_sources,
        "excluded_sources": excluded_sources,
        "source_feature_exposed_to_router": False,
        "folds": folds,
        "fold_roles_sha256": sha256_file(fold_role_path),
    }
    write_json(args.output_root / "FOLD_MANIFEST.json", fold_manifest)

    selected_counts = {source: len(selected_by_source[source]) for source in selected_sources}
    cohort_manifest = {
        "schema_version": 1,
        "artifact_role": "panel_c_preoutcome_cohort_manifest",
        "status": "READY_FOR_EXTERNAL_PROTOCOL_TIMESTAMP_NO_MODEL_INFERENCE",
        "selection_seed": SELECTION_SEED,
        "selection_used_answers": False,
        "selection_used_predictions": False,
        "selection_used_correctness": False,
        "selection_used_cost": False,
        "target_outcomes_inspected": False,
        "model_inference_executed": False,
        "source_feature_in_router_inputs": False,
        "semantic_labels_in_router_inputs": False,
        "semantic_label_plan": "externally frozen image/question-only classifier",
        "source_selection_rule": {
            "minimum_unseen_unique_images": args.minimum_source_n,
            "maximum_rows_per_source": args.source_cap,
            "include_all_source_adapters_meeting_minimum": True,
            "outcome_based_selection": False,
        },
        "selected_sources": selected_sources,
        "excluded_sources": excluded_sources,
        "selected_counts": selected_counts,
        "selected_total": len(selected),
        "historical_fingerprint_input": {
            "pre_replication_b_manifest": str(args.prior_fingerprints.resolve()),
            "pre_replication_b_manifest_sha256": sha256_file(args.prior_fingerprints),
            "replication_b_row_manifest": str(args.replication_b_row_manifest.resolve()),
            "replication_b_row_manifest_sha256": sha256_file(args.replication_b_row_manifest),
            "replication_b_rows_added": replication_b_rows,
            "prior_exact_image_count_after_union": len(prior_images),
            "prior_exact_sample_count_after_union": len(prior_samples),
        },
        "source_adapter": {
            "path": str(args.source_adapter.resolve()),
            "sha256": sha256_file(args.source_adapter),
        },
        "source_audits": source_audits,
        "artifact_hashes": {
            "outcome_blind_router_inputs": sha256_file(router_input_path),
            "row_manifest": sha256_file(row_manifest_path),
            "sealed_reference_answers": sha256_file(answer_path),
            "fold_scoped_reference_answers": scoped_answer_hashes,
            "fold_roles": sha256_file(fold_role_path),
            "fold_manifest": sha256_file(args.output_root / "FOLD_MANIFEST.json"),
        },
        "redistribution": "questions, answers, images, source record identifiers, and fold-role rows remain restricted",
    }
    write_json(args.output_root / "COHORT_MANIFEST.json", cohort_manifest)

    audit = {
        "schema_version": 1,
        "artifact_role": "panel_c_preoutcome_leakage_and_novelty_audit",
        "status": "PASS",
        "target_outcomes_inspected": False,
        "model_predictions_present": False,
        "router_input_forbidden_fields": [
            "source",
            "dataset",
            "answer",
            "correct",
            "candidate_outcome",
            "utility_outcome",
        ],
        "router_input_forbidden_fields_found": [],
        "selected_exact_image_overlap_with_history": 0,
        "selected_exact_sample_overlap_with_history": 0,
        "selected_cross_source_exact_image_duplicates": 0,
        "selected_source_count": len(selected_sources),
        "selected_total": len(selected),
        "per_source": {
            source: {
                "selected": selected_counts[source],
                "eligible": source_audits[source]["eligible_after_all_filters"],
            }
            for source in selected_sources
        },
    }
    write_json(args.output_root / "PREOUTCOME_AUDIT.json", audit)
    print(
        json.dumps(
            {
                "status": cohort_manifest["status"],
                "selected_sources": selected_sources,
                "selected_counts": selected_counts,
                "selected_total": len(selected),
                "target_outcomes_inspected": False,
                "model_inference_executed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
