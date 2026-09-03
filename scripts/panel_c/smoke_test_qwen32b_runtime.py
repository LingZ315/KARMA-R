#!/usr/bin/env python3
"""Outcome-free, synthetic execution smoke test for frozen Qwen3-VL-32B runtime.

This executable never reads Panel-C queries, images, outcomes, routes, or source
identifiers.  A PASS requires real model generation and strict JSON parsing; the
production classifier's deterministic schema fallback is deliberately disabled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor


CLASSES = {
    "general_visual_reasoning",
    "fine_grained_perception",
    "ocr_text_reading",
    "structured_artifact_reasoning",
    "spatial_reasoning",
    "quantitative_reasoning",
    "science_reasoning",
}
SUBTYPES = {"document", "chart_graph", "visual_counting", "mathematical_reasoning"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def runtime_prompt(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    matches = re.findall(r"```text\r?\n(.*?)\r?\n```", text, flags=re.DOTALL)
    if len(matches) != 1:
        raise ValueError("semantic prompt file must contain exactly one text fence")
    return matches[0]


def parse_response(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", stripped, flags=re.IGNORECASE)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("classifier response contains no JSON object")
    value = json.loads(stripped[start : end + 1])
    if set(value) != {"primary_class", "subtype", "ambiguity", "rationale"}:
        raise ValueError("classifier response schema mismatch")
    if value["primary_class"] not in CLASSES or value["subtype"] not in SUBTYPES | {"none"}:
        raise ValueError("classifier response contains an unknown label")
    if not isinstance(value["ambiguity"], bool) or not isinstance(value["rationale"], str):
        raise ValueError("classifier response has invalid field types")
    if len(value["rationale"].split()) > 20:
        raise ValueError("classifier rationale exceeds 20 words")
    if value["primary_class"] != "structured_artifact_reasoning" and value["subtype"] in {
        "document",
        "chart_graph",
    }:
        raise ValueError("structured subtype is incompatible with primary_class")
    if value["primary_class"] != "quantitative_reasoning" and value["subtype"] in {
        "visual_counting",
        "mathematical_reasoning",
    }:
        raise ValueError("quantitative subtype is incompatible with primary_class")
    return value


def build_synthetic_image(path: Path) -> Image.Image:
    image = Image.new("RGB", (512, 320), "white")
    draw = ImageDraw.Draw(image)
    for left in (45, 195, 345):
        draw.rectangle((left, 55, left + 100, 185), outline="navy", width=10)
    draw.text((42, 245), "SYNTHETIC SHAPES", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)
    return image


def package_version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


def nvidia_smi_rows() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,name,driver_version,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    rows: list[dict[str, Any]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 7:
            continue
        rows.append(
            {
                "physical_index": int(fields[0]),
                "uuid": fields[1],
                "name": fields[2],
                "driver_version": fields[3],
                "memory_total_mib": int(fields[4]),
                "memory_used_mib": int(fields[5]),
                "memory_free_mib": int(fields[6]),
            }
        )
    return rows


def synchronize_all() -> None:
    for index in range(torch.cuda.device_count()):
        torch.cuda.synchronize(index)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--synthetic-image", type=Path, required=True)
    parser.add_argument("--max-memory-gib", type=int, default=30)
    args = parser.parse_args()

    started_utc = datetime.now(timezone.utc).isoformat()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    prompt = runtime_prompt(args.prompt)
    if sha256_file(args.prompt) != config["prompt_sha256"]:
        raise ValueError("frozen prompt SHA-256 mismatch")
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("multi-GPU CUDA runtime is unavailable")

    visible_physical = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if not visible_physical:
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be explicitly frozen")
    gpu_before = nvidia_smi_rows()
    max_memory = {index: f"{args.max_memory_gib}GiB" for index in range(torch.cuda.device_count())}
    for index in range(torch.cuda.device_count()):
        # PyTorch 2.11 requires a materialized context before resetting the
        # allocator statistics on some CUDA 13 / Blackwell installations.
        with torch.cuda.device(index):
            torch.empty(1, device=f"cuda:{index}")
            torch.cuda.reset_peak_memory_stats()

    common = {
        "revision": config["revision"],
        "cache_dir": str(args.cache_dir),
        "local_files_only": True,
        "trust_remote_code": True,
    }
    load_started = time.perf_counter()
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
    synchronize_all()
    model_load_seconds = time.perf_counter() - load_started
    resolved_revision = getattr(model.config, "_commit_hash", None)
    if resolved_revision not in (None, config["revision"]):
        raise RuntimeError(f"model revision drift: {resolved_revision}")

    hf_device_map = {str(key): str(value) for key, value in getattr(model, "hf_device_map", {}).items()}
    occupied_local_indices = sorted(
        {
            int(value)
            for value in hf_device_map.values()
            if isinstance(value, int) or (isinstance(value, str) and value.isdigit())
        }
    )
    if not occupied_local_indices:
        occupied_local_indices = sorted({parameter.device.index for parameter in model.parameters() if parameter.device.type == "cuda"})
    if len(occupied_local_indices) < 2:
        raise RuntimeError("model was not distributed across multiple GPUs")

    image = build_synthetic_image(args.synthetic_image)
    question = "How many navy rectangles are visible?"
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
    preprocess_started = time.perf_counter()
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    input_device = model.get_input_embeddings().weight.device
    inputs = inputs.to(input_device, dtype=torch.bfloat16)
    preprocess_seconds = time.perf_counter() - preprocess_started

    torch.manual_seed(int(config["seed_base"]))
    torch.cuda.manual_seed_all(int(config["seed_base"]))
    synchronize_all()
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=int(config["max_new_tokens"]),
        )
    synchronize_all()
    generation_seconds = time.perf_counter() - generation_started
    continuation = output[:, inputs["input_ids"].shape[1] :]
    raw_response = processor.batch_decode(continuation, skip_special_tokens=True)[0].strip()
    parsed = parse_response(raw_response)
    image.close()

    gpu_after = nvidia_smi_rows()
    peak_memory = []
    for local_index in range(torch.cuda.device_count()):
        peak_memory.append(
            {
                "local_index": local_index,
                "physical_index": visible_physical[local_index] if local_index < len(visible_physical) else None,
                "name": torch.cuda.get_device_name(local_index),
                "peak_allocated_mib": round(torch.cuda.max_memory_allocated(local_index) / 1024**2, 2),
                "peak_reserved_mib": round(torch.cuda.max_memory_reserved(local_index) / 1024**2, 2),
            }
        )
    occupied_gpu_count = len(occupied_local_indices)
    receipt = {
        "schema_version": 1,
        "artifact_role": "qwen32b_outcome_free_synthetic_execution_smoke_receipt",
        "status": "PASS",
        "started_utc": started_utc,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "outcome_firewall": {
            "input_origin": "programmatically generated synthetic image and fixed synthetic question",
            "panel_c_queries_read": False,
            "panel_c_images_read": False,
            "panel_c_outcomes_read": False,
            "panel_c_routes_read": False,
            "source_or_dataset_identifiers_supplied": False,
            "synthetic_image_sha256": sha256_file(args.synthetic_image),
            "synthetic_question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest(),
        },
        "model": {
            "repository": config["repository"],
            "requested_revision": config["revision"],
            "resolved_revision": resolved_revision,
            "dtype": "bfloat16",
            "quantization": "none",
            "attention_implementation": "sdpa",
            "device_map_strategy": "auto",
            "hf_device_map": hf_device_map,
        },
        "runtime": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
            "safetensors": package_version("safetensors"),
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": visible_physical,
            "visible_gpu_count": torch.cuda.device_count(),
            "occupied_local_gpu_indices": occupied_local_indices,
            "occupied_gpu_count": occupied_gpu_count,
            "max_memory_per_visible_gpu": max_memory,
            "gpu_inventory_before": gpu_before,
            "gpu_inventory_after": gpu_after,
            "peak_memory": peak_memory,
        },
        "timing": {
            "model_load_seconds": round(model_load_seconds, 6),
            "preprocess_seconds": round(preprocess_seconds, 6),
            "generation_wall_seconds": round(generation_seconds, 6),
            "generation_gpu_seconds_rule": "occupied_gpu_count multiplied by synchronized generation wall seconds",
            "generation_gpu_seconds": round(occupied_gpu_count * generation_seconds, 6),
        },
        "frozen_inputs": {
            "config_path": str(args.config.resolve()),
            "config_sha256": sha256_file(args.config),
            "prompt_path": str(args.prompt.resolve()),
            "prompt_sha256": sha256_file(args.prompt),
            "seed": int(config["seed_base"]),
            "do_sample": False,
            "max_new_tokens": int(config["max_new_tokens"]),
        },
        "generation": {
            "raw_response": raw_response,
            "parsed_response": parsed,
            "strict_schema_parse": True,
            "fallback_used": False,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "receipt": str(args.output), "generation_gpu_seconds": receipt["timing"]["generation_gpu_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
