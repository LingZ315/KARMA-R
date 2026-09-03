# Qwen3-VL-32B outcome-free execution smoke test

Status: **PASS**

The test used one programmatically generated 512×320 image containing three navy rectangles and the fixed question “How many navy rectangles are visible?”. It used no Panel-C query ID, benchmark image, source/dataset identifier, candidate response, answer ledger, correctness, score, route, or target outcome. PASS required a real model generation whose raw output passed the strict frozen JSON parser; the production fallback was disabled for this test.

## Frozen model and runtime

| Item | Verified value |
| --- | --- |
| Model | `Qwen/Qwen3-VL-32B-Instruct` |
| Requested/resolved revision | `0cfaf48183f594c314753d30a4c4974bc75f3ccb` / exact match |
| Processor | repository processor at the same pinned revision |
| Server | `ky-lz-8-0`, Linux 5.15 |
| Python / PyTorch / Transformers | 3.12.3 / 2.11.0+cu130 / 5.8.0 |
| GPU allocation | four simultaneously occupied NVIDIA GeForce RTX 5090 GPUs; physical devices 0, 1, 4, 5 for this run |
| Precision / quantization | BF16 / none |
| Device mapping | Hugging Face `device_map="auto"`, 30 GiB maximum per visible GPU, SDPA |
| Batch size / decoding | 1 / deterministic (`do_sample=false`, 192 maximum new tokens) |

The final runtime freezes **four visible RTX 5090 devices**, but not their physical server indices; the orchestration layer must expose four idle devices through `CUDA_VISIBLE_DEVICES`. The model must occupy all four logical devices, otherwise the runner fails closed.

## Measured execution

| Measurement | Value |
| --- | ---: |
| Processor + model load | 73.015064 s |
| Image/text preprocessing | 0.053781 s |
| CUDA-synchronized generation wall time | 3.307659 s |
| Occupied GPU count | 4 |
| Generation GPU-seconds | 13.230635 GPU-s |
| Peak allocated memory, logical GPU 0 | 15,209.45 MiB |
| Peak allocated memory, logical GPU 1 | 17,240.37 MiB |
| Peak allocated memory, logical GPU 2 | 17,240.37 MiB |
| Peak allocated memory, logical GPU 3 | 18,402.32 MiB |
| Maximum peak reserved memory | 19,064.00 MiB |

The frozen multi-GPU accounting rule is:

\[
C_{\mathrm{GPU-sec}} = G\,t,
\]

where `G` is the number of simultaneously occupied devices and `t` is CUDA-synchronized generation wall time. Thus this run is `4 × 3.30765875 = 13.230635` GPU-seconds, not 3.307659 GPU-seconds.

## Functional checks

- Model and processor loaded from the pinned local cache: PASS.
- Synthetic image preprocessing: PASS.
- Combined image-and-text inference: PASS.
- Deterministic generation: PASS.
- Strict four-key semantic JSON schema parse: PASS.
- Known class/subtype and compatibility checks: PASS.
- No OOM: PASS.
- Timing and multi-GPU accounting: PASS.
- Fallback or manual repair used: **No**.

The raw model response classified the task as `quantitative_reasoning` / `visual_counting`, with `ambiguity=false`. This synthetic result was not used to change any class definition, H2 parameter, source split, or scientific endpoint.

## Smoke-test-driven correction

The v7.0.1 runner used the naïve single-device `device_map="cuda"`, which is not a defensible deployment plan for a 32B BF16 model on one 32-GB GPU. v7.0.2 changes only the operational runtime: four RTX 5090 GPUs, `device_map="auto"`, a 30-GiB per-device ceiling, input placement on the embedding device, and `G × t` cost accounting. The model, revision, prompt, schema, decoding, retry/fallback rule, data split, and scientific parameters are unchanged.

Machine-readable evidence: `qwen32b_execution_smoke_receipt.json`, 11,642 bytes, SHA-256 `19947f5259ef7ad9814cc609461f380ff38bf12da4a6c3d3698db15d1126f94d`.
