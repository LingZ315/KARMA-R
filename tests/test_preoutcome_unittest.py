from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from karmavl.panel_c.common import require_authorization_lock, sha256_file, verify_file_manifest
from karmavl.panel_c.costs import deployment_utility, method_serving_cost, validate_method_requirements
from karmavl.panel_c.external import verify_external_timestamp_receipt
from karmavl.panel_c.features import (
    build_arm_feature_rows,
    build_fold_profiles,
    semantic_query_vector,
)
from karmavl.panel_c.nested import select_fold_b_star, select_fold_margin, verify_nested_loso
from karmavl.panel_c.routing import fit_router, predict_router
from karmavl.panel_c.scoring import score_response
from karmavl.panel_c.statistics import primary_endpoint, source_stratified_paired_bootstrap


def split_fixture() -> dict:
    folds = []
    for held in "abc":
        other = [value for value in "abc" if value != held]
        folds.append(
            {
                "fold_id": f"holdout_{held}",
                "held_out_source": held,
                "roles": {
                    "profile": {"query_ids": [f"{other[0]}1"]},
                    "calibration": {"query_ids": [f"{other[1]}1"]},
                    "policy": {"query_ids": [f"{other[0]}2", f"{other[1]}2"]},
                    "target": {"query_ids": [f"{held}1", f"{held}2"]},
                },
            }
        )
    return {"folds": folds}


def contract_fixture() -> dict:
    return {
        "cohort": {"pooled_target_n": 6},
        "outer_fold_selection": {
            "B_star_scope": "fold_local_policy_only",
            "h2_hyperparameter_scope": "fold_local_non_target_only",
            "global_cross_fold_selection_for_primary": False,
        },
    }


class PreOutcomeFreezeTests(unittest.TestCase):
    def test_nested_loso_passes_actual_membership(self) -> None:
        sources = {f"{source}{index}": source for source in "abc" for index in (1, 2)}
        result = verify_nested_loso(
            split_manifest=split_fixture(), source_by_query=sources, analysis_contract=contract_fixture()
        )
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["cross_fold_reuse"]["acknowledged"])
        self.assertTrue(all(not row["B_star_selected_with_heldout_source"] for row in result["folds"]))

    def test_nested_loso_rejects_heldout_training_source(self) -> None:
        split = split_fixture()
        split["folds"][0]["roles"]["profile"]["query_ids"] = ["a1"]
        sources = {f"{source}{index}": source for source in "abc" for index in (1, 2)}
        result = verify_nested_loso(
            split_manifest=split, source_by_query=sources, analysis_contract=contract_fixture()
        )
        self.assertEqual(result["status"], "FAIL")

    def test_b_star_is_fold_local(self) -> None:
        rows = [
            {
                "fold_id": "holdout_a",
                "role": "policy",
                "source": "b",
                "query_id": query,
                "baseline_id": baseline,
                "correct": baseline == "cheap",
                "serving_gpu_seconds": 1.0 if baseline == "cheap" else 2.0,
                "target_outcome": False,
            }
            for query in ("b1", "b2")
            for baseline in ("cheap", "static")
        ]
        result = select_fold_b_star(
            rows, fold_id="holdout_a", held_out_source="a", cost_coefficient=0.01
        )
        self.assertEqual(result["selected_B_f_star"], "cheap")
        with self.assertRaises(ValueError):
            select_fold_b_star(
                [*rows, {**rows[0], "query_id": "a1", "source": "a"}],
                fold_id="holdout_a",
                held_out_source="a",
                cost_coefficient=0.01,
            )

    def test_margin_is_fold_local(self) -> None:
        rows = [
            {
                "fold_id": "holdout_a",
                "role": "policy",
                "source": "b",
                "query_id": f"b{index}",
                "margin": margin,
                "realized_utility": 0.5,
                "target_outcome": False,
            }
            for index, margin in enumerate((0.0, 0.01), start=1)
        ]
        result = select_fold_margin(
            rows, fold_id="holdout_a", held_out_source="a", margin_grid=(0.0, 0.01)
        )
        self.assertEqual(result["selected_margin"], 0.01)
        with self.assertRaises(ValueError):
            select_fold_margin(
                [{**rows[0], "role": "target"}, rows[1]],
                fold_id="holdout_a",
                held_out_source="a",
                margin_grid=(0.0, 0.01),
            )

    def test_source_metadata_is_rejected(self) -> None:
        vector = semantic_query_vector(
            {"primary_class": "general", "subtype": "none", "ambiguity": False},
            class_order=["general"],
            subtype_order=["count"],
        )
        self.assertEqual(vector.tolist(), [1.0, 0.0, 0.0])
        with self.assertRaises(ValueError):
            semantic_query_vector(
                {
                    "primary_class": "general",
                    "subtype": "none",
                    "ambiguity": False,
                    "source_id": "a",
                },
                class_order=["general"],
                subtype_order=["count"],
            )

    def test_method_cost_requirements(self) -> None:
        names = (
            "incumbent_only",
            "cheapest_candidate",
            "static_calibration_global_best",
            "static_class_conditional_best",
            "logistic_raw",
            "nearest_profile",
            "H1",
            "H1_5",
            "H2",
        )
        methods = {
            name: {
                "needs_semantic_classifier": name
                in {"static_class_conditional_best", "logistic_raw", "nearest_profile", "H1", "H1_5", "H2"},
                "needs_learned_router": name in {"logistic_raw", "H1", "H1_5", "H2"},
            }
            for name in names
        }
        validate_method_requirements(methods)
        h2 = method_serving_cost(
            [{"feature_gpu_seconds": 2.0, "router_gpu_seconds": 0.0, "selected_vlm_gpu_seconds": 3.0}],
            requirements=methods["H2"],
        )
        incumbent = method_serving_cost(
            [{"feature_gpu_seconds": 0.0, "router_gpu_seconds": 0.0, "selected_vlm_gpu_seconds": 3.0}],
            requirements=methods["incumbent_only"],
        )
        self.assertEqual((h2, incumbent), (5.0, 3.0))
        self.assertAlmostEqual(
            deployment_utility(
                accuracy=0.8,
                onboarding_gpu_seconds=100.0,
                serving_gpu_seconds=2.0,
                queries=1000,
                cost_coefficient=0.01,
            ),
            0.779,
        )

    def test_fake_receipt_and_missing_lock_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "archive.zip"
            archive.write_bytes(b"not a real preregistration archive")
            receipt = root / "receipt.json"
            receipt.write_text(
                json.dumps(
                    {
                        "provider": "fake",
                        "record_id": "123",
                        "persistent_url": "https://example.invalid/123",
                        "timestamp_utc": "2026-01-01T00:00:00Z",
                        "local_archive_sha256": sha256_file(archive),
                        "remote_archive_sha256": sha256_file(archive),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaises(PermissionError):
                verify_external_timestamp_receipt(receipt, archive)
            with self.assertRaises(FileNotFoundError):
                require_authorization_lock(
                    root / "missing.lock",
                    preregistration_archive=archive,
                    preregistration_manifest=root / "manifest.json",
                    code_manifest=root / "code.json",
                )

    def test_manifest_detects_drift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            member = root / "member.txt"
            member.write_text("frozen\n", encoding="utf-8")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "files": [
                            {
                                "path": "member.txt",
                                "bytes": member.stat().st_size,
                                "sha256": sha256_file(member),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(len(verify_file_manifest(manifest, root)), 1)
            member.write_text("drift\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify_file_manifest(manifest, root)

    def test_profile_feature_router_is_executable(self) -> None:
        routes = ["incumbent", "candidate_a", "candidate_b"]
        semantics = {
            "q1": {"primary_class": "general", "subtype": "none", "ambiguity": False},
            "q2": {"primary_class": "count", "subtype": "visual_counting", "ambiguity": False},
            "q3": {"primary_class": "general", "subtype": "none", "ambiguity": True},
        }
        profile_rows = [
            {
                "query_id": query,
                "route_id": route,
                "role": "profile",
                "correct": (query == "q1") == (route != "candidate_b"),
                "generation_gpu_seconds": 1.0,
                "target_outcome": False,
            }
            for query in semantics
            for route in routes
        ]
        profiles = build_fold_profiles(
            profile_rows,
            semantics,
            candidate_routes=["candidate_a", "candidate_b"],
            all_routes=routes,
            class_order=["general", "count"],
            subtype_order=["visual_counting"],
            minimum_support=1,
        )
        features = build_arm_feature_rows(
            query_ids=semantics, semantics=semantics, profiles=profiles, arm="H2"
        )
        correctness = {
            (row["query_id"], row["route_id"]): row["route_id"] == "candidate_a"
            for row in features
        }
        model = fit_router(
            features,
            correctness,
            learner="bilinear_logistic",
            l2=10.0,
            maximum_iterations=20,
        )
        predictions = predict_router(model, features)
        self.assertEqual(len(predictions), len(features))
        self.assertTrue(all(np.isfinite(row["predicted_accuracy"]) for row in predictions))

    def test_frozen_scorers(self) -> None:
        cases = [
            ("yes", "yes", "yes_no", "Is it visible?", True),
            ("42", "42", "numeric_short", "How many?", True),
            ("The cat", "cat", "normalized_short", "What?", True),
            ("B", "blue", "multiple_choice_or_normalized_short", "A. red\nB. blue", True),
            ("", "yes", "yes_no", "Is it visible?", False),
        ]
        for response, reference, scorer, prompt, expected in cases:
            with self.subTest(scorer=scorer, response=response):
                self.assertIs(score_response(response, reference, scorer, prompt), expected)

    def test_primary_and_source_stratified_bootstrap(self) -> None:
        rows = [
            {
                "query_id": f"{source}{index}",
                "source": source,
                "h2_correct": index % 2 == 0,
                "b_f_star_correct": index == 0,
            }
            for source in ("a", "b")
            for index in range(4)
        ]
        point = primary_endpoint(rows)
        boot = source_stratified_paired_bootstrap(rows, replicates=200, seed=7, confidence=0.95)
        self.assertEqual(point["target_n"], 8)
        self.assertEqual(set(boot["source_specific_descriptive_ci"]), {"a", "b"})
        with self.assertRaises(ValueError):
            primary_endpoint([rows[0], rows[0]])

    def test_required_result_placeholders_absent(self) -> None:
        candidates = list(Path(__file__).resolve().parents)
        root = next(
            path
            for path in candidates
            if (path / "configs" / "panel_c_analysis_contract.json").exists()
            or (path / "configs" / "kbs_v7_0_1" / "panel_c_analysis_contract.json").exists()
        )
        self.assertFalse((root / "PANEL_C_EXTERNAL_TIMESTAMP_RECEIPT.json").exists())
        self.assertFalse((root / "PANEL_C_TARGET_EXECUTION_AUTHORIZED.lock").exists())
        self.assertFalse((root / "revision_outputs" / "panel_c_results.csv").exists())
        contract_path = root / "configs" / "panel_c_analysis_contract.json"
        if not contract_path.exists():
            contract_path = root / "configs" / "kbs_v7_0_1" / "panel_c_analysis_contract.json"
        contract = json.loads(
            contract_path.read_text(encoding="utf-8")
        )
        self.assertIs(contract["chronology"]["target_execution_authorized"], False)


if __name__ == "__main__":
    unittest.main()
