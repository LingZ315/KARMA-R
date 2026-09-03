#!/usr/bin/env python3
"""Run the frozen image/question-only Panel-C semantic classifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

from karmavl.panel_c.common import (
    load_json,
    require_frozen_execution_state,
    require_policy_bundle_freeze,
    sha256_file,
    write_json_new,
)


def _runtime_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"```text\r?\n(.*?)\r?\n```", text, flags=re.DOTALL)
    if len(matches) != 1:
        raise ValueError("semantic prompt file must contain exactly one text fence")
    return matches[0]


def _parse_response(text: str, classes: set[str], subtypes: set[str]) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("classifier response contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    if set(value) != {"primary_class", "subtype", "ambiguity", "rationale"}:
        raise ValueError("classifier response schema mismatch")
    if value["primary_class"] not in classes or value["subtype"] not in subtypes | {"none"}:
        raise ValueError("classifier response contains an unknown label")
    if not isinstance(value["ambiguity"], bool) or not isinstance(value["rationale"], str):
        raise ValueError("classifier response has invalid field types")
    if len(value["rationale"].split()) > 20:
        raise ValueError("classifier rationale exceeds 20 words")
    if value["primary_class"] != "structured_artifact_reasoning" and value["subtype"] in {
        "document",
        "chart_graph",
    }:
        raise ValueError("structured subtype is incompatible with the primary class")
    if value["primary_class"] != "quantitative_reasoning" and value["subtype"] in {
        "visual_counting",
        "mathematical_reasoning",
    }:
        raise ValueError("quantitative subtype is incompatible with the primary class")
    return value


def _smoke_image() -> Image.Image:
    image = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((30, 30, 150, 155), outline="navy", width=8)
    draw.text((35, 190), "COUNT 1", fill="black")
    return image


def _completed(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as stream:
        return {str(json.loads(line)["query_id"]) for line in stream if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--role", choices=("profile", "calibration", "policy", "target"), required=True)
    parser.add_argument("--fold-id")
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--policy-bundle-lock", type=Path)
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    args = parser.parse_args()
    execution_started = datetime.now(timezone.utc)
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    if args.role == "target":
        if (
            args.policy_bundle_lock is None
            or args.policy_bundle is None
            or args.execution_receipt is None
            or not args.fold_id
            or args.split_manifest is None
        ):
            raise PermissionError(
                "target semantic inference requires fold ID, complete policy-bundle lock, bundle, and execution receipt"
            )
        require_policy_bundle_freeze(
            args.policy_bundle_lock,
            bundle_path=args.policy_bundle,
            expected_fold_id=args.fold_id,
        )
    elif any(value is not None for value in (args.policy_bundle_lock, args.policy_bundle)):
        raise ValueError("non-target semantic inference must not receive target policy artifacts")
    config = load_json(args.config)
    if config.get("artifact_role") != "panel_c_explicit_source_metadata_blind_semantic_classifier_freeze":
        raise ValueError("semantic classifier config role mismatch")
    if sha256_file(args.prompt) != config["prompt_sha256"]:
        raise ValueError("semantic classifier prompt hash drift")
    prompt = _runtime_prompt(args.prompt)
    classes = {
        "general_visual_reasoning",
        "fine_grained_perception",
        "ocr_text_reading",
        "structured_artifact_reasoning",
        "spatial_reasoning",
        "quantitative_reasoning",
        "science_reasoning",
    }
    subtypes = {"document", "chart_graph", "visual_counting", "mathematical_reasoning"}
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    forbidden = {value.casefold() for value in config["forbidden_request_fields"]} | {
        "source_id",
        "dataset_id",
        "source_one_hot",
    }
    found = sorted({key for row in rows for key in row if key.casefold() in forbidden})
    if found:
        raise ValueError(f"explicit source/outcome fields reached semantic inference: {found}")
    ids = [str(row["query_id"]) for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("semantic input contains duplicate query IDs")
    if args.role == "target":
        split = load_json(args.split_manifest)
        fold = next(
            (row for row in split.get("folds", []) if str(row.get("fold_id")) == args.fold_id),
            None,
        )
        if fold is None:
            raise ValueError("target semantic fold is absent from the frozen split manifest")
        expected_ids = {str(value) for value in fold["roles"]["target"]["query_ids"]}
        if set(ids) != expected_ids:
            raise PermissionError("target semantic input is not the exact frozen fold target set")
    runtime = config.get("runtime", {})
    required_gpu_count = int(runtime.get("visible_gpu_count", 4))
    if torch.cuda.device_count() != required_gpu_count:
        raise RuntimeError(
            f"semantic runtime requires exactly {required_gpu_count} visible GPUs; "
            f"found {torch.cuda.device_count()}"
        )
    required_gpu_model = str(runtime.get("gpu_model", "NVIDIA GeForce RTX 5090"))
    names = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if any(name != required_gpu_model for name in names):
        raise RuntimeError(f"semantic runtime GPU drift: {names}")
    max_memory_gib = int(runtime.get("max_memory_gib_per_visible_gpu", 30))
    max_memory = {index: f"{max_memory_gib}GiB" for index in range(required_gpu_count)}
    common = {
        "revision": config["revision"],
        "cache_dir": args.cache_dir,
        "local_files_only": bool(runtime.get("local_files_only", True)),
        "trust_remote_code": True,
    }
    processor = AutoProcessor.from_pretrained(config["repository"], **common)
    model = AutoModelForImageTextToText.from_pretrained(
        config["repository"],
        **common,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        attn_implementation="sdpa",
        low_cpu_mem_usage=True,
    ).eval()
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved not in (None, config["revision"]):
        raise RuntimeError("semantic model revision drift")
    occupied_devices = {
        int(value)
        for value in getattr(model, "hf_device_map", {}).values()
        if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
    }
    if len(occupied_devices) != required_gpu_count:
        raise RuntimeError(
            f"semantic model must occupy all {required_gpu_count} frozen GPUs; "
            f"used {sorted(occupied_devices)}"
        )

    def infer(image: Image.Image, question: str) -> str:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": prompt}]},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Question: {question}"},
                ],
            },
        ]
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.get_input_embeddings().weight.device, dtype=torch.bfloat16)
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(config["max_new_tokens"]),
        )
        continuation = output[:, inputs["input_ids"].shape[1] :]
        return processor.batch_decode(continuation, skip_special_tokens=True)[0]

    warmup = _smoke_image()
    _ = infer(warmup, "How many navy rectangles are visible?")
    warmup.close()
    torch.cuda.synchronize()
    completed = _completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "x"
    with args.output.open(mode, encoding="utf-8", newline="\n") as stream:
        for index, row in enumerate(rows):
            query_id = str(row["query_id"])
            if query_id in completed:
                continue
            question = str(row.get("question", row.get("prompt", "")))
            if not question:
                raise ValueError(f"missing question for {query_id}")
            image_path = Path(str(row["image_path"]))
            fallback = False
            raw_response = ""
            parsed: dict[str, Any] | None = None
            elapsed = 0.0
            error: str | None = None
            attempts = 0
            for attempts in range(1, int(config["maximum_attempts"]) + 1):
                try:
                    torch.manual_seed(int(config["seed_base"]) + index)
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    raw_response = infer(image, question).strip()
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    image.close()
                    parsed = _parse_response(raw_response, classes, subtypes)
                    break
                except Exception as exc:  # frozen identical retry/fallback policy
                    torch.cuda.synchronize()
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
            if parsed is None:
                fallback = True
                parsed = {
                    "primary_class": "general_visual_reasoning",
                    "subtype": "none",
                    "ambiguity": True,
                    "rationale": "Frozen schema fallback after deterministic attempts.",
                }
            record = {
                "query_id": query_id,
                **parsed,
                "feature_generation_wall_seconds": elapsed,
                "feature_gpu_count": required_gpu_count,
                "feature_gpu_seconds": required_gpu_count * elapsed,
                "feature_gpu_seconds_rule": "occupied_gpu_count multiplied by CUDA-synchronized generation wall seconds",
                "attempts": attempts,
                "fallback": fallback,
                "error": error,
                "model_revision": config["revision"],
                "image_sha256": sha256_file(image_path),
                "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
                "raw_response": raw_response,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    if args.execution_receipt is not None:
        receipt = {
            "schema_version": 1,
            "artifact_role": "panel_c_semantic_inference_execution_receipt",
            "status": "COMPLETED",
            "role": args.role,
            "fold_id": args.fold_id,
            "started_at_utc": execution_started.isoformat().replace("+00:00", "Z"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "target_candidate_responses_generated": False,
            "target_outcomes_accessed": False,
            "bindings": {
                "output": {"path": str(args.output.resolve()), "sha256": sha256_file(args.output)},
                "config": {"path": str(args.config.resolve()), "sha256": sha256_file(args.config)},
                "prompt": {"path": str(args.prompt.resolve()), "sha256": sha256_file(args.prompt)},
                "authorization_lock": {
                    "path": str(args.authorization_lock.resolve()),
                    "sha256": sha256_file(args.authorization_lock),
                },
            },
        }
        if args.role == "target":
            receipt["bindings"]["policy_bundle_lock"] = {
                "path": str(args.policy_bundle_lock.resolve()),
                "sha256": sha256_file(args.policy_bundle_lock),
            }
            receipt["bindings"]["split_manifest"] = {
                "path": str(args.split_manifest.resolve()),
                "sha256": sha256_file(args.split_manifest),
            }
        write_json_new(args.execution_receipt, receipt)
    print(
        json.dumps(
            {"status": "SEMANTIC_FEATURES_WRITTEN", "role": args.role, "rows": len(rows)},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
