#!/usr/bin/env python3
"""Run one pinned Panel-C route without loading any outcome ledger.

The runner is restartable and writes one JSON object per query.  It deliberately
accepts only the outcome-free input ledger.  Answers, correctness, router arms,
and target outcomes are neither accepted nor imported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from PIL import Image, ImageDraw
from transformers import (
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    AutoProcessor,
    GenerationConfig,
    LlavaForConditionalGeneration,
)

from karmavl.panel_c.common import (
    load_json,
    require_frozen_execution_state,
    require_policy_bundle_freeze,
    require_target_route_bundle_freeze,
    sha256_file as common_sha256_file,
    write_json_new,
)


ROUTES: dict[str, dict[str, Any]] = {
    "smolvlm_incumbent": {
        "repository": "HuggingFaceTB/SmolVLM-Instruct",
        "revision": "81cd9a775a4d644f2faf4e7becff4559b46b14c7",
        "adapter": "smol",
    },
    "phi4_mm": {
        "repository": "microsoft/Phi-4-multimodal-instruct",
        "revision": "93f923e1a7727d1c4f446756212d9d3e8fcc5d81",
        "adapter": "phi4",
    },
    "granite4_vision": {
        "repository": "ibm-granite/granite-4.0-3b-vision",
        "revision": "bf108f36960fb4df79bf035e506c592f4ee3c2d3",
        "adapter": "granite",
    },
    "ovis25_9b": {
        "repository": "AIDC-AI/Ovis2.5-9B",
        "revision": "d73b2283ae2a930b7762f8d7b8b8a3f0f3b5c3bd",
        "adapter": "ovis",
    },
    "qwen3vl_4b": {
        "repository": "Qwen/Qwen3-VL-4B-Instruct",
        "revision": "ebb281ec70b05090aa6165b016eac8ec08e71b17",
        "adapter": "qwen",
    },
    "internvl35_4b": {
        "repository": "OpenGVLab/InternVL3_5-4B-HF",
        "revision": "6bd4487402110ef9889ba50eb7aefeb302526fed",
        "adapter": "internvl",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _messages(image: Image.Image, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]


def _load(route: dict[str, Any], cache_dir: Path) -> tuple[Any, Any, Callable[[Image.Image, str, int], str]]:
    repo = route["repository"]
    revision = route["revision"]
    common = {"revision": revision, "cache_dir": cache_dir, "local_files_only": False}
    adapter = route["adapter"]
    if adapter in {"smol", "qwen", "internvl"}:
        processor = AutoProcessor.from_pretrained(
            repo, **common, trust_remote_code=(adapter == "internvl")
        )
        model = AutoModelForImageTextToText.from_pretrained(
            repo,
            **common,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
            trust_remote_code=(adapter == "internvl"),
        ).eval()

        def infer(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
            inputs = processor.apply_chat_template(
                _messages(image, prompt),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device, dtype=torch.bfloat16)
            output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            continuation = output[:, inputs["input_ids"].shape[1] :]
            return processor.batch_decode(continuation, skip_special_tokens=True)[0]

    elif adapter == "phi4":
        processor = AutoProcessor.from_pretrained(repo, **common, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            **common,
            device_map="cuda",
            torch_dtype="auto",
            trust_remote_code=True,
            _attn_implementation="sdpa",
        ).eval()
        generation_config = GenerationConfig.from_pretrained(repo, **common)

        def infer(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
            text = f"<|user|><|image_1|>{prompt}<|end|><|assistant|>"
            inputs = processor(text=text, images=image, return_tensors="pt").to(model.device)
            output = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                generation_config=generation_config,
            )
            continuation = output[:, inputs["input_ids"].shape[1] :]
            return processor.batch_decode(
                continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

    elif adapter == "granite":
        processor = AutoProcessor.from_pretrained(repo, **common, trust_remote_code=True)
        model = AutoModelForImageTextToText.from_pretrained(
            repo,
            **common,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        ).eval()
        if hasattr(model, "merge_lora_adapters"):
            model.merge_lora_adapters()

        def infer(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
            conv = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
            text = processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True, do_pad=True).to(
                model.device
            )
            output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
            continuation = output[:, inputs["input_ids"].shape[1] :]
            return processor.batch_decode(continuation, skip_special_tokens=True)[0]

    elif adapter == "pixtral":
        processor = AutoProcessor.from_pretrained(repo, **common)
        model = LlavaForConditionalGeneration.from_pretrained(
            repo,
            **common,
            torch_dtype=torch.bfloat16,
            device_map="cuda",
            attn_implementation="sdpa",
        ).eval()

        def infer(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
            chat = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "content": prompt}],
                }
            ]
            text = processor.apply_chat_template(chat, add_generation_prompt=True)
            inputs = processor(text=text, images=[image], return_tensors="pt").to(
                model.device, dtype=torch.bfloat16
            )
            output = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
            continuation = output[:, inputs["input_ids"].shape[1] :]
            return processor.batch_decode(
                continuation, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

    elif adapter == "ovis":
        processor = None
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            **common,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="cuda",
        ).eval()

        def infer(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
            inputs, pixels, grids = model.preprocess_inputs(
                messages=_messages(image, prompt),
                add_generation_prompt=True,
                enable_thinking=False,
                max_pixels=1024 * 1024,
            )
            inputs = inputs.to(model.device)
            pixels = pixels.to(model.device) if pixels is not None else None
            grids = grids.to(model.device) if grids is not None else None
            output = model.generate(
                inputs=inputs,
                pixel_values=pixels,
                grid_thws=grids,
                enable_thinking=False,
                do_sample=False,
                max_new_tokens=max_new_tokens,
            )
            # Ovis returns continuation tokens rather than prompt+continuation.
            return model.text_tokenizer.decode(output[0], skip_special_tokens=True)

    elif adapter == "step3":
        processor = AutoProcessor.from_pretrained(repo, **common, trust_remote_code=True)
        key_mapping = {
            "^vision_model": "model.vision_model",
            r"^model(?!\.(language_model|vision_model))": "model.language_model",
            "vit_large_projector": "model.vit_large_projector",
        }
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            **common,
            trust_remote_code=True,
            device_map="cuda",
            torch_dtype="auto",
            key_mapping=key_mapping,
            attn_implementation="sdpa",
        ).eval()

        def infer(image: Image.Image, prompt: str, max_new_tokens: int) -> str:
            inputs = processor.apply_chat_template(
                _messages(image, prompt),
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
            continuation = output[:, inputs["input_ids"].shape[1] :]
            return processor.decode(continuation[0], skip_special_tokens=True)

    else:  # pragma: no cover - guarded by the frozen registry
        raise ValueError(f"unknown adapter {adapter}")
    resolved = getattr(model.config, "_commit_hash", None)
    if resolved not in (None, revision):
        raise RuntimeError(f"model revision drift: resolved {resolved}, expected {revision}")
    return model, processor, infer


def _smoke_image() -> Image.Image:
    image = Image.new("RGB", (384, 256), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((25, 25, 150, 150), outline="navy", width=8)
    draw.ellipse((210, 40, 330, 160), outline="darkred", width=8)
    draw.text((35, 190), "TEST 17", fill="black")
    return image


def _load_completed(output: Path) -> set[str]:
    if not output.exists():
        return set()
    completed: set[str] = set()
    with output.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                completed.add(str(json.loads(line)["query_id"]))
    return completed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-id", choices=sorted(ROUTES), required=True)
    parser.add_argument("--role", choices=("profile", "calibration", "policy", "target", "smoke"), required=True)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--model-pool", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--authorization-lock", type=Path, required=True)
    parser.add_argument("--preregistration-archive", type=Path, required=True)
    parser.add_argument("--preregistration-manifest", type=Path, required=True)
    parser.add_argument("--code-manifest", type=Path, required=True)
    parser.add_argument("--policy-bundle", type=Path)
    parser.add_argument("--policy-bundle-lock", type=Path)
    parser.add_argument("--target-routes", type=Path)
    parser.add_argument("--target-route-freeze", type=Path)
    parser.add_argument("--execution-receipt", type=Path)
    args = parser.parse_args()
    if not args.smoke and (args.input is None or args.output is None):
        parser.error("--input and --output are required unless --smoke is used")
    require_frozen_execution_state(
        args.authorization_lock,
        preregistration_archive=args.preregistration_archive,
        preregistration_manifest=args.preregistration_manifest,
        code_manifest=args.code_manifest,
    )
    frozen_target_routes: list[dict[str, Any]] | None = None
    if args.role == "target":
        required_target_artifacts = (
            args.policy_bundle,
            args.policy_bundle_lock,
            args.target_routes,
            args.target_route_freeze,
            args.execution_receipt,
        )
        if any(value is None for value in required_target_artifacts):
            raise PermissionError(
                "target inference requires policy bundle/lock, route ledger/lock, and execution receipt"
            )
        require_policy_bundle_freeze(args.policy_bundle_lock, bundle_path=args.policy_bundle)
        _, frozen_target_routes = require_target_route_bundle_freeze(
            args.target_route_freeze,
            routes_path=args.target_routes,
            policy_bundle_lock=args.policy_bundle_lock,
        )
        if args.output is not None and args.output.exists():
            raise PermissionError(
                "target output must not pre-exist this post-route-freeze execution; fail closed"
            )
    elif any(
        value is not None
        for value in (
            args.policy_bundle,
            args.policy_bundle_lock,
            args.target_routes,
            args.target_route_freeze,
        )
    ):
        raise ValueError("non-target inference must not receive target freeze artifacts")
    execution_started = datetime.now(timezone.utc)
    pool = load_json(args.model_pool)
    pool_rows = [pool["incumbent"], *pool["candidates"]]
    frozen = next((row for row in pool_rows if row["route_id"] == args.route_id), None)
    if frozen is None:
        raise ValueError("route is absent from the frozen model pool")
    route = ROUTES[args.route_id]
    if route["repository"] != frozen["repository"] or route["revision"] != frozen["revision"]:
        raise ValueError("runner registry differs from the frozen model-pool config")
    inference_contract = pool["common_inference_contract"]
    max_new_tokens = int(inference_contract["max_new_tokens"])
    if args.max_new_tokens is not None and args.max_new_tokens != max_new_tokens:
        raise ValueError("max-new-tokens differs from the frozen model-pool contract")
    answer_instruction = str(inference_contract["prompt"])
    rows: list[dict[str, Any]] = []
    if not args.smoke:
        rows = [
            json.loads(line)
            for line in args.input.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        forbidden = {
            "source",
            "dataset",
            "answer",
            "correct",
            "correctness",
            "candidate_outcome",
            "utility_outcome",
            "selected_route",
            "route",
        }
        found = sorted({key for row in rows for key in row if key.casefold() in forbidden})
        if found:
            raise ValueError(f"outcome/source/route fields reached candidate inference: {found}")
        input_ids = [str(row["query_id"]) for row in rows]
        if len(input_ids) != len(set(input_ids)):
            raise ValueError("candidate inference input contains duplicate query IDs")
        if args.role == "target":
            selected_fields = (
                "h2_gate3_accuracy_selected_route",
                "b_A_star_selected_route",
                "h2_gate4_utility_selected_route",
                "b_U_star_selected_route",
            )
            required_ids = {
                str(row["query_id"])
                for row in frozen_target_routes
                if args.route_id in {str(row[field]) for field in selected_fields}
            }
            if set(input_ids) != required_ids:
                raise PermissionError(
                    "target input is not the exact minimal query set selected for this frozen route"
                )
            if not required_ids:
                raise PermissionError("the frozen confirmatory policies do not require this target route")
    model, _, infer = _load(route, args.cache_dir)
    warmup = _smoke_image()
    _ = infer(warmup, "Return only the number printed in the image.", 16)
    warmup.close()
    torch.cuda.synchronize()
    if args.smoke:
        print(json.dumps({"route_id": args.route_id, "loaded": True, "authorized": True}))
        return

    completed = _load_completed(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if args.output.exists() else "w"
    processed = 0
    with args.output.open(mode, encoding="utf-8", newline="\n") as stream:
        for row in rows:
            query_id = str(row["query_id"])
            if query_id in completed:
                continue
            image_path = Path(row["image_path"])
            prompt = str(row["prompt"])
            inference_prompt = f"{answer_instruction}\n\n{prompt}"
            success = True
            error = None
            response = ""
            elapsed = 0.0
            attempts = 0
            for attempts in range(1, 4):
                try:
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")
                    torch.cuda.synchronize()
                    start = time.perf_counter()
                    response = infer(image, inference_prompt, max_new_tokens).strip()
                    torch.cuda.synchronize()
                    elapsed += time.perf_counter() - start
                    image.close()
                    if response:
                        break
                    raise RuntimeError("empty deterministic response")
                except Exception as exc:  # terminal failures remain in the denominator
                    torch.cuda.synchronize()
                    success = False
                    error = f"{type(exc).__name__}: {str(exc)[:300]}"
            else:
                response = ""
            success = bool(response)
            record = {
                "query_id": query_id,
                "role": args.role,
                "route_id": args.route_id,
                "repository": route["repository"],
                "revision": route["revision"],
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "inference_prompt_sha256": hashlib.sha256(
                    inference_prompt.encode("utf-8")
                ).hexdigest(),
                "image_sha256": sha256_file(image_path),
                "response": response,
                "generation_gpu_seconds": elapsed,
                "success": success,
                "attempts": attempts,
                "error": error,
            }
            stream.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
            processed += 1
            if processed % 50 == 0:
                print(json.dumps({"route_id": args.route_id, "new_rows": processed, "remaining": len(rows) - len(completed) - processed}))
    if args.execution_receipt is not None:
        receipt = {
            "schema_version": 1,
            "artifact_role": "panel_c_candidate_inference_execution_receipt",
            "status": "COMPLETED",
            "role": args.role,
            "route_id": args.route_id,
            "started_at_utc": execution_started.isoformat().replace("+00:00", "Z"),
            "completed_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "rows": processed,
            "target_outcomes_accessed": False,
            "bindings": {
                "input": {"path": str(args.input.resolve()), "sha256": common_sha256_file(args.input)},
                "output": {"path": str(args.output.resolve()), "sha256": common_sha256_file(args.output)},
                "authorization_lock": {
                    "path": str(args.authorization_lock.resolve()),
                    "sha256": common_sha256_file(args.authorization_lock),
                },
            },
        }
        if args.role == "target":
            receipt["bindings"].update(
                {
                    "policy_bundle_lock": {
                        "path": str(args.policy_bundle_lock.resolve()),
                        "sha256": common_sha256_file(args.policy_bundle_lock),
                    },
                    "target_routes": {
                        "path": str(args.target_routes.resolve()),
                        "sha256": common_sha256_file(args.target_routes),
                    },
                    "target_route_freeze": {
                        "path": str(args.target_route_freeze.resolve()),
                        "sha256": common_sha256_file(args.target_route_freeze),
                    },
                }
            )
        write_json_new(args.execution_receipt, receipt)
    print(
        json.dumps(
            {
                "route_id": args.route_id,
                "new_rows": processed,
                "total_rows": len(rows),
                "output_sha256": sha256_file(args.output),
                "python": platform.python_version(),
                "torch": torch.__version__,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
