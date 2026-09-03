#!/usr/bin/env python3
"""Build the outcome-firewalled Replication-B cohort from pinned sources.

Selection uses only source identity, outcome-blind metadata, questions, image
hashes, and a frozen SHA-256 ordering key.  Answers are copied into physically
separate ledgers but never influence eligibility, deduplication, semantic
assignment, partitioning, or row order.  The script prints aggregate counts
only and never prints questions or answers.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image


SEED = "karma-r-replication-b-v1-20260830"
PARTITION_ORDER = ("profile", "calibration", "policy", "target")
CLASSES = (
    "general_visual_reasoning",
    "fine_grained_perception",
    "ocr_text_reading",
    "structured_artifact_reasoning",
    "spatial_reasoning",
    "quantitative_reasoning",
    "science_reasoning",
)
SOURCE_IMAGE_PIXEL_LIMIT = 25_000_000


SOURCES: dict[str, dict[str, Any]] = {
    "iconqa": {
        "repository": "lmms-lab-encoder/ICON-QA",
        "revision": "94f9ba83a935851992dfa0e9b66b7ba2aba4cb7e",
        "files": (
            "data/test-00000-of-00001-e3c690d531807508.parquet",
            "data/val-00000-of-00001-f9787027db08d358.parquet",
        ),
        "license": "UPSTREAM_CARD_UNDECLARED_EVALUATION_ONLY_NOT_REDISTRIBUTED",
        "quota": {"profile": 86, "calibration": 298, "policy": 256, "target": 1560},
    },
    "tallyqa": {
        "repository": "nimapourjafar/mm_tallyqa",
        "revision": "a62372c438b3527b51c71eb0f136c6ae4d1d81c9",
        "files": ("data/train-00000-of-00010.parquet",),
        "license": "UPSTREAM_CARD_UNDECLARED_EVALUATION_ONLY_NOT_REDISTRIBUTED",
        "quota": {"profile": 72, "calibration": 180, "policy": 168, "target": 980},
    },
    "wemath2_standard": {
        "repository": "We-Math/We-Math2.0-Standard",
        "revision": "19064a4b3be262db1fadc952d03b3102acccf942",
        "files": ("data/standard-00000-of-00001-6bb245f2ae7687a3.parquet",),
        "license": "UPSTREAM_CARD_UNDECLARED_EVALUATION_ONLY_NOT_REDISTRIBUTED",
        "quota": {"profile": 108, "calibration": 216, "policy": 216, "target": 1260},
    },
    "chartmuseum": {
        "repository": "lytang/ChartMuseum",
        "revision": "462d46deb187d8a40c5a9de4e69e14f1df982e58",
        "files": (
            "data/dev-00000-of-00001-6dfc215b35b5fc35.parquet",
            "data/test-00000-of-00001-d549f7069f75a077.parquet",
        ),
        "license": "cc-by-sa-4.0",
        "quota": {"profile": 48, "calibration": 96, "policy": 96, "target": 566},
    },
    "tablevqa_bench": {
        "repository": "terryoo/TableVQA-Bench",
        "revision": "17e291af2b1cbc9340509a0412a6f1c05b0a76b5",
        "files": (
            "data/fintabnetqa-00000-of-00001-c337fe9eb7a70460.parquet",
            "data/vtabfact-00000-of-00001-ecd1dbae37761ddd.parquet",
            "data/vwtq-00000-of-00001-764eb826ab450a91.parquet",
            "data/vwtq_syn-00000-of-00001-2daaa7285aca2c1d.parquet",
        ),
        "license": "cc-by-4.0",
        "quota": {"profile": 54, "calibration": 108, "policy": 108, "target": 624},
    },
    "visonlyqa": {
        "repositories": (
            (
                "ryokamoi/VisOnlyQA_Eval_Real_v1.1",
                "3747d1370919a6ed0b7ee09896ef6dbffc8aa6d9",
            ),
            (
                "ryokamoi/VisOnlyQA_Eval_Synthetic",
                "ef0b5fbcc40a35fdd2947f9b65bed8a8ee99e86d",
            ),
        ),
        "license": "gpl-3.0",
        "quota": {"profile": 60, "calibration": 84, "policy": 96, "target": 560},
    },
    "mme": {
        "repository": "lmms-lab-encoder/MME",
        "revision": "d6c9023f017b564f7b3ccccf5348166bce8fdbcd",
        "files": (
            "data/test-00000-of-00004-a25dbe3b44c4fda6.parquet",
            "data/test-00001-of-00004-7d22c7f1aba6fca4.parquet",
            "data/test-00002-of-00004-594798fd3f5b029c.parquet",
            "data/test-00003-of-00004-53ae1794f93b1e35.parquet",
        ),
        "license": "UPSTREAM_CARD_UNDECLARED_EVALUATION_ONLY_NOT_REDISTRIBUTED",
        "quota": {"profile": 84, "calibration": 42, "policy": 84, "target": 450},
    },
}


@dataclass
class Record:
    source: str
    source_record_id: str
    question: str
    answer: str
    image_bytes: bytes
    prompt: str
    scorer: str
    primary_class: str
    subtype: str | None = None
    secondary_classes: tuple[str, ...] = ()

    @property
    def original_image_sha256(self) -> str:
        return hashlib.sha256(self.image_bytes).hexdigest()

    @property
    def selection_key(self) -> str:
        material = f"{SEED}|{self.source}|{self.source_record_id}|{self.original_image_sha256}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _download(repo: str, revision: str, filename: str, cache_dir: Path) -> Path:
    return Path(
        hf_hub_download(
            repo_id=repo,
            repo_type="dataset",
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
    )


def _direct_resolve_download(
    repo: str,
    revision: str,
    filename: str,
    output_root: Path,
    retries: int = 12,
) -> Path:
    """Download one pinned dataset file without the snapshot/Xet token API.

    Public Hugging Face resolve URLs are immutable when addressed by commit.
    This deliberately serial path avoids snapshot_download's concurrent Xet
    token refreshes, which can trigger a shared-IP 429 on the execution host.
    """
    destination = output_root / revision / filename
    if destination.is_file() and destination.stat().st_size:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    quoted = urllib.parse.quote(filename, safe="/")
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{quoted}"
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "KARMA-R-Replication-B/1.0"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    stream.write(block)
            if temporary.stat().st_size == 0:
                raise IOError(f"empty download for {filename}")
            temporary.replace(destination)
            return destination
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt == retries:
                break
            time.sleep(min(60, 5 * attempt))
    raise RuntimeError(f"failed pinned direct download for {filename}") from last_error


def _rows(repo: str, revision: str, filenames: Iterable[str], cache_dir: Path) -> Iterable[tuple[str, dict[str, Any]]]:
    for filename in filenames:
        path = _download(repo, revision, filename, cache_dir)
        for index, row in enumerate(pq.read_table(path).to_pylist()):
            yield f"{filename}:{index}", row


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, dict) and isinstance(value.get("bytes"), (bytes, bytearray)):
        return bytes(value["bytes"])
    raise ValueError("row lacks embedded image bytes")


def _options_prompt(question: str, options: Iterable[Any] | None = None) -> str:
    lines = [
        "Answer the question using the image.",
        "Return only the final answer, with no explanation.",
        f"Question: {question.strip()}",
    ]
    values = [str(value).strip() for value in (options or ()) if str(value).strip()]
    if values:
        lines.append("Options:")
        lines.extend(f"{chr(65 + index)}. {value}" for index, value in enumerate(values))
        lines.append("Return only the option letter.")
    return "\n".join(lines)


def _iconqa(cache_dir: Path) -> list[Record]:
    spec = SOURCES["iconqa"]
    result: list[Record] = []
    for row_id, row in _rows(spec["repository"], spec["revision"], spec["files"], cache_dir):
        question_type = str(row.get("ques_type", ""))
        if question_type not in {"choose_txt", "fill_in_blank"}:
            continue
        skills = {value.strip().casefold() for value in str(row.get("skills", "")).split(",") if value.strip()}
        if skills & {"counting", "fraction", "measurement", "estimation", "algebra", "probability", "time"}:
            primary = "quantitative_reasoning"
            subtype = "visual_counting" if "counting" in skills else "mathematical_reasoning"
        elif skills & {"spatial", "geometry"}:
            primary, subtype = "spatial_reasoning", None
        else:
            primary, subtype = "general_visual_reasoning", None
        options: list[Any] = []
        if question_type == "choose_txt":
            raw = row.get("choices")
            try:
                parsed = ast.literal_eval(str(raw))
                options = list(parsed) if isinstance(parsed, (list, tuple)) else []
            except (SyntaxError, ValueError):
                options = []
        question = str(row["question"])
        result.append(
            Record(
                "iconqa",
                f"{row_id}:{row.get('question_id', '')}",
                question,
                str(row["answer"]),
                _image_bytes(row["query_image"]),
                _options_prompt(question, options),
                "multiple_choice_or_normalized_short",
                primary,
                subtype,
            )
        )
    return result


def _tallyqa(cache_dir: Path) -> list[Record]:
    spec = SOURCES["tallyqa"]
    result: list[Record] = []
    for row_id, row in _rows(spec["repository"], spec["revision"], spec["files"], cache_dir):
        messages = list(row.get("data") or ())
        images = list(row.get("images") or ())
        if len(messages) < 3 or not images:
            continue
        question = str(messages[1].get("data", ""))
        answer = str(messages[2].get("data", ""))
        if not question or not answer:
            continue
        result.append(
            Record(
                "tallyqa",
                row_id,
                question,
                answer,
                _image_bytes(images[0]),
                _options_prompt(question),
                "numeric_short",
                "quantitative_reasoning",
                "visual_counting",
            )
        )
    return result


def _wemath(cache_dir: Path) -> list[Record]:
    spec = SOURCES["wemath2_standard"]
    return [
        Record(
            "wemath2_standard",
            f"{row_id}:{row.get('id', row.get('idx', ''))}",
            str(row["question"]),
            str(row["answer"]),
            _image_bytes(row["image"]),
            _options_prompt(str(row["question"])),
            "multiple_choice_or_normalized_short",
            "quantitative_reasoning",
            "mathematical_reasoning",
        )
        for row_id, row in _rows(spec["repository"], spec["revision"], spec["files"], cache_dir)
    ]


def _chartmuseum(cache_dir: Path) -> list[Record]:
    spec = SOURCES["chartmuseum"]
    snapshot = (
        cache_dir
        / f"datasets--{spec['repository'].replace('/', '--')}"
        / "snapshots"
        / spec["revision"]
    )
    direct_cache = cache_dir / "pinned-direct" / "chartmuseum"
    result: list[Record] = []
    for row_id, row in _rows(spec["repository"], spec["revision"], spec["files"], cache_dir):
        relative = Path(str(row["image"]))
        candidates = (snapshot / relative, snapshot / "images" / relative.name)
        image_path = next((value for value in candidates if value.is_file()), None)
        if image_path is None:
            filename = str(relative).replace("\\", "/")
            if not filename.startswith("images/"):
                filename = f"images/{relative.name}"
            image_path = _direct_resolve_download(
                spec["repository"],
                spec["revision"],
                filename,
                direct_cache,
            )
        question = str(row["question"])
        result.append(
            Record(
                "chartmuseum",
                f"{row_id}:{row.get('hash', '')}",
                question,
                str(row["answer"]),
                image_path.read_bytes(),
                _options_prompt(question),
                "numeric_or_normalized_short",
                "structured_artifact_reasoning",
                "chart_graph",
            )
        )
    return result


def _tablevqa(cache_dir: Path) -> list[Record]:
    spec = SOURCES["tablevqa_bench"]
    result: list[Record] = []
    for row_id, row in _rows(spec["repository"], spec["revision"], spec["files"], cache_dir):
        question = str(row["question"])
        result.append(
            Record(
                "tablevqa_bench",
                f"{row_id}:{row.get('qa_id', '')}",
                question,
                str(row["gt"]),
                _image_bytes(row["image"]),
                _options_prompt(question),
                "numeric_or_normalized_short",
                "structured_artifact_reasoning",
                "document",
            )
        )
    return result


def _visonlyqa(cache_dir: Path) -> list[Record]:
    result: list[Record] = []
    for repo, revision in SOURCES["visonlyqa"]["repositories"]:
        from huggingface_hub import dataset_info

        info = dataset_info(repo, revision=revision)
        files = sorted(s.rfilename for s in info.siblings if s.rfilename.startswith("data/") and s.rfilename.endswith(".parquet"))
        for row_id, row in _rows(repo, revision, files, cache_dir):
            category = f"{row.get('image_category', '')}|{row.get('task_category', '')}|{row_id}".casefold()
            if "chemistry" in category:
                primary, subtype = "science_reasoning", None
            elif "chart" in category:
                primary, subtype = "structured_artifact_reasoning", "chart_graph"
            else:
                primary, subtype = "spatial_reasoning", None
            options = list(row.get("response_options") or ())
            question = str(row["question"])
            result.append(
                Record(
                    "visonlyqa",
                    f"{repo}:{row_id}:{row.get('id', '')}",
                    question,
                    str(row["answer"]),
                    _image_bytes(row["decoded_image"]),
                    _options_prompt(question, options),
                    "multiple_choice_or_normalized_short",
                    primary,
                    subtype,
                )
            )
    return result


def _mme_class(category: str) -> tuple[str, str | None]:
    key = category.casefold()
    if key in {"ocr", "text_translation"}:
        return "ocr_text_reading", None
    if key == "position":
        return "spatial_reasoning", None
    if key in {"count", "numerical_calculation"}:
        return "quantitative_reasoning", "visual_counting" if key == "count" else "mathematical_reasoning"
    if key in {"color", "celebrity", "landmark", "posters", "artwork"}:
        return "fine_grained_perception", None
    return "general_visual_reasoning", None


def _mme(cache_dir: Path) -> list[Record]:
    spec = SOURCES["mme"]
    result: list[Record] = []
    for row_id, row in _rows(spec["repository"], spec["revision"], spec["files"], cache_dir):
        primary, subtype = _mme_class(str(row["category"]))
        question = str(row["question"])
        result.append(
            Record(
                "mme",
                f"{row_id}:{row.get('question_id', '')}",
                question,
                str(row["answer"]),
                _image_bytes(row["image"]),
                _options_prompt(question),
                "yes_no",
                primary,
                subtype,
            )
        )
    return result


LOADERS = {
    "iconqa": _iconqa,
    "tallyqa": _tallyqa,
    "wemath2_standard": _wemath,
    "chartmuseum": _chartmuseum,
    "tablevqa_bench": _tablevqa,
    "visonlyqa": _visonlyqa,
    "mme": _mme,
}


def _deduplicate(records: list[Record], prior_images: set[str], prior_samples: set[str]) -> tuple[list[Record], dict[str, int]]:
    by_image: dict[str, Record] = {}
    audit = Counter()
    for record in records:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(record.image_bytes)) as image:
                    pixel_count = int(image.width) * int(image.height)
            if pixel_count > SOURCE_IMAGE_PIXEL_LIMIT:
                audit["source_image_pixel_budget_excluded"] += 1
                continue
        except (Image.DecompressionBombError, OSError, ValueError):
            audit["source_image_unreadable_or_decompression_risk_excluded"] += 1
            continue
        image_hash = record.original_image_sha256
        sample_id = f"{record.source}:{record.source_record_id}"
        if image_hash in prior_images:
            audit["prior_exact_image_overlap_excluded"] += 1
            continue
        if sample_id in prior_samples:
            audit["prior_exact_sample_overlap_excluded"] += 1
            continue
        existing = by_image.get(image_hash)
        if existing is None or record.selection_key < existing.selection_key:
            if existing is not None:
                audit["within_source_duplicate_image_excluded"] += 1
            by_image[image_hash] = record
        else:
            audit["within_source_duplicate_image_excluded"] += 1
    return sorted(by_image.values(), key=lambda row: row.selection_key), dict(audit)


def _select_profile(source: str, records: list[Record], count: int) -> list[Record]:
    required: dict[str, int] = {}
    if source == "mme":
        required = {
            "ocr_text_reading": 25,
            "fine_grained_perception": 25,
            "general_visual_reasoning": 25,
        }
    elif source == "visonlyqa":
        required = {"science_reasoning": 25, "spatial_reasoning": 25}
    selected: list[Record] = []
    selected_keys: set[str] = set()
    for class_name, minimum in required.items():
        candidates = [row for row in records if row.primary_class == class_name]
        if len(candidates) < minimum:
            raise RuntimeError(f"{source}: only {len(candidates)} rows support {class_name}; need {minimum}")
        for row in candidates[:minimum]:
            selected.append(row)
            selected_keys.add(row.selection_key)
    for row in records:
        if len(selected) >= count:
            break
        if row.selection_key not in selected_keys:
            selected.append(row)
            selected_keys.add(row.selection_key)
    if len(selected) != count:
        raise RuntimeError(f"{source}: profile selection produced {len(selected)}, expected {count}")
    return selected


def _write_image(record: Record, path: Path) -> None:
    image = Image.open(io.BytesIO(record.image_bytes)).convert("RGB")
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", compress_level=6, optimize=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_revisions(spec: dict[str, Any]) -> list[dict[str, str]]:
    if "repositories" in spec:
        return [{"repository": repo, "revision": revision} for repo, revision in spec["repositories"]]
    return [{"repository": spec["repository"], "revision": spec["revision"]}]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prior-fingerprints", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("output root must be absent or empty; cohort replacement is prohibited")
    args.output_root.mkdir(parents=True, exist_ok=True)
    private_dir = args.output_root / "private"
    image_dir = args.output_root / "images"
    private_dir.mkdir(parents=True)
    prior = json.loads(args.prior_fingerprints.read_text(encoding="utf-8"))
    prior_images = set(prior.get("image_sha256s", ()))
    prior_samples = set(prior.get("sample_ids", ()))
    blocked_sources = {
        str(name).casefold()
        for values in prior.get("blocked_dataset_sources", {}).values()
        for name in values
    }
    if {source.casefold() for source in SOURCES} & blocked_sources:
        raise RuntimeError("selected source name intersects a historically blocked dataset")

    selected: list[tuple[str, Record]] = []
    source_audits: dict[str, Any] = {}
    global_images: set[str] = set()
    for source, spec in SOURCES.items():
        raw = LOADERS[source](args.cache_dir)
        eligible, audit = _deduplicate(raw, prior_images, prior_samples)
        eligible = [row for row in eligible if row.original_image_sha256 not in global_images]
        quota = spec["quota"]
        needed = sum(quota.values())
        if len(eligible) < needed:
            raise RuntimeError(f"{source}: {len(eligible)} unique eligible rows, need {needed}")
        profile = _select_profile(source, eligible, quota["profile"])
        profile_keys = {row.selection_key for row in profile}
        remaining = [row for row in eligible if row.selection_key not in profile_keys]
        cursor = 0
        allocations = {"profile": profile}
        for partition in ("calibration", "policy", "target"):
            size = quota[partition]
            allocations[partition] = remaining[cursor : cursor + size]
            cursor += size
        for partition in PARTITION_ORDER:
            if len(allocations[partition]) != quota[partition]:
                raise AssertionError(f"{source}/{partition} quota drift")
            selected.extend((partition, row) for row in allocations[partition])
            global_images.update(row.original_image_sha256 for row in allocations[partition])
        source_audits[source] = {
            "raw_rows": len(raw),
            "unique_eligible_rows": len(eligible),
            "selected_rows": needed,
            "exclusions": audit,
            "revisions": _source_revisions(spec),
            "license": spec["license"],
        }
        del raw, eligible, remaining, allocations

    selected.sort(key=lambda item: (PARTITION_ORDER.index(item[0]), item[1].source, item[1].selection_key))
    input_path = private_dir / "outcome_blind_inputs.jsonl"
    non_target_path = private_dir / "profile_calibration_policy_outcomes.jsonl"
    target_path = private_dir / "sealed_target_outcomes.jsonl"
    row_manifest_path = private_dir / "row_manifest.jsonl"
    support = {partition: Counter() for partition in PARTITION_ORDER}
    subtype_support = {partition: Counter() for partition in PARTITION_ORDER}
    source_partition = {partition: Counter() for partition in PARTITION_ORDER}
    with (
        input_path.open("w", encoding="utf-8", newline="\n") as inputs,
        non_target_path.open("w", encoding="utf-8", newline="\n") as non_target,
        target_path.open("w", encoding="utf-8", newline="\n") as target,
        row_manifest_path.open("w", encoding="utf-8", newline="\n") as rows_out,
    ):
        for index, (partition, record) in enumerate(selected, start=1):
            query_id = f"RB{index:05d}"
            image_path = image_dir / f"{query_id}.png"
            _write_image(record, image_path)
            normalized_image_hash = _sha256(image_path)
            prompt_hash = hashlib.sha256(record.prompt.encode("utf-8")).hexdigest()
            input_row = {
                "query_id": query_id,
                "partition": partition,
                "source": record.source,
                "image_path": str(image_path.resolve()),
                "prompt": record.prompt,
                "primary_class": record.primary_class,
                "secondary_classes": list(record.secondary_classes),
                "subtype": record.subtype,
                "ambiguity": False,
                "scorer": record.scorer,
            }
            outcome_row = {"query_id": query_id, "answer": record.answer, "scorer": record.scorer}
            manifest_row = {
                "query_id": query_id,
                "partition": partition,
                "source": record.source,
                "source_record_id": record.source_record_id,
                "group_id": record.original_image_sha256,
                "original_image_sha256": record.original_image_sha256,
                "normalized_image_sha256": normalized_image_hash,
                "prompt_sha256": prompt_hash,
                "primary_class": record.primary_class,
                "subtype": record.subtype,
            }
            inputs.write(json.dumps(input_row, ensure_ascii=False, sort_keys=True) + "\n")
            (target if partition == "target" else non_target).write(
                json.dumps(outcome_row, ensure_ascii=False, sort_keys=True) + "\n"
            )
            rows_out.write(json.dumps(manifest_row, ensure_ascii=False, sort_keys=True) + "\n")
            support[partition][record.primary_class] += 1
            if record.subtype:
                subtype_support[partition][record.subtype] += 1
            source_partition[partition][record.source] += 1
    os.chmod(target_path, 0o600)

    partition_counts = {partition: sum(source_partition[partition].values()) for partition in PARTITION_ORDER}
    expected = {"profile": 512, "calibration": 1024, "policy": 1024, "target": 6000}
    if partition_counts != expected:
        raise AssertionError(f"partition counts drifted: {partition_counts}")
    if any(support["profile"][name] < 25 for name in CLASSES):
        raise AssertionError(f"profile semantic support gate failed: {dict(support['profile'])}")
    for subtype in ("document", "chart_graph", "visual_counting", "mathematical_reasoning"):
        if subtype_support["profile"][subtype] < 25:
            raise AssertionError(f"profile subtype support gate failed for {subtype}")

    manifest = {
        "schema_version": 1,
        "artifact_role": "replication_b_frozen_cohort_manifest",
        "status": "READY_FOR_PRE_OUTCOME_FREEZE",
        "selection_seed": SEED,
        "selection_used_answers": False,
        "target_outcomes_inspected": False,
        "one_query_per_exact_image_group": True,
        "image_long_edge_maximum": 1024,
        "source_image_pixel_maximum": SOURCE_IMAGE_PIXEL_LIMIT,
        "image_encoding": "deterministic_rgb_png",
        "partition_counts": partition_counts,
        "source_partition_counts": {key: dict(value) for key, value in source_partition.items()},
        "semantic_support": {key: dict(value) for key, value in support.items()},
        "subtype_support": {key: dict(value) for key, value in subtype_support.items()},
        "sources": source_audits,
        "private_artifact_hashes": {
            "outcome_blind_inputs": _sha256(input_path),
            "non_target_outcomes": _sha256(non_target_path),
            "sealed_target_outcomes": _sha256(target_path),
            "row_manifest": _sha256(row_manifest_path),
        },
        "redistribution": "row-level questions, answers, and images excluded from public release",
    }
    prior_audit = {
        "schema_version": 1,
        "artifact_role": "replication_b_prior_fingerprint_audit",
        "status": "PASS",
        "target_outcomes_read": False,
        "historical_blocked_source_count": len(blocked_sources),
        "selected_source_name_overlap": [],
        "historical_exact_image_hash_count": len(prior_images),
        "historical_exact_sample_id_count": len(prior_samples),
        "selected_exact_image_overlap_after_filter": 0,
        "selected_exact_sample_overlap_after_filter": 0,
        "selected_unique_image_groups": len(global_images),
        "source_audits": source_audits,
    }
    support_manifest = {
        "schema_version": 1,
        "artifact_role": "replication_b_outcome_blind_schema_support",
        "status": "PASS",
        "annotation_identity": "deterministic_outcome_blind_source_metadata_mapping",
        "human_validation_claimed": False,
        "target_outcomes_used": False,
        "minimum_primary_support": 25,
        "minimum_subtype_support": 25,
        "partition_support": {key: dict(value) for key, value in support.items()},
        "partition_subtype_support": {key: dict(value) for key, value in subtype_support.items()},
    }
    for filename, payload in (
        ("COHORT_MANIFEST.json", manifest),
        ("PRIOR_FINGERPRINT_AUDIT.json", prior_audit),
        ("BLIND_SEMANTIC_SUPPORT.json", support_manifest),
    ):
        (args.output_root / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "status": "READY_FOR_PRE_OUTCOME_FREEZE",
                "partition_counts": partition_counts,
                "profile_semantic_support": dict(support["profile"]),
                "target_outcome_sha256": _sha256(target_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
