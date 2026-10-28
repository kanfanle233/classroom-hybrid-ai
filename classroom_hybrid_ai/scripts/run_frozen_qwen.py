"""Run closed eight-action Qwen classification on frozen student boxes.

This is an evaluation-only command.  It uses the frozen annotation *boxes* to
construct a static evidence packet.  A sample's gold label is never placed in
the prompt; Qwen sees only the complete fixed eight-action vocabulary and must
return one action ID or ``null``.

The command is intentionally outside the online package path and supports a
small ``--limit`` first run plus ``--resume`` for a longer audit.  It should be
run only after code, prompt and threshold hashes have been frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from classroom_hybrid_ai.config import load_config  # noqa: E402
from classroom_hybrid_ai.pipeline import _make_qwen_adapter  # noqa: E402
from classroom_hybrid_ai.taxonomy import (  # noqa: E402
    CORE8_CODES,
    FIXED_EIGHT_VISUAL_RULES,
    PROMPT_PRIORITY_CODES,
    decode_fixed_action,
    load_fixed_taxonomy,
)


def _classification_prompt(actions: Any) -> str:
    by_code = {action.code: action for action in actions}
    choices = [by_code[code].to_prompt_dict() for code in PROMPT_PRIORITY_CODES]
    return (
        "Analyze only the target student marked by the red box in this 480x416 still-image evidence packet. "
        "The top row shows target, hands/desk region, and expanded target context; the bottom row shows the full classroom. "
        "Choose exactly one action from this fixed vocabulary: "
        + json.dumps(choices, ensure_ascii=False, separators=(",", ":"))
        + ". Use visible pixels only. "
        + FIXED_EIGHT_VISUAL_RULES
        + "Do not infer identity, emotion, intent, attention, or discipline. "
        "If no class is visibly supported, set decision=uncertain and behavior=null. "
        "Return exactly one compact JSON object with only decision,behavior,observability. "
        "decision is confirmed or uncertain; behavior must be exactly one listed action_id or null; "
        "observability is clear, partial, or none. No Markdown, no extra fields, no rationale."
    )


def _score_module() -> Any:
    path = PROJECT_ROOT / "scripts" / "score_frozen_benchmark.py"
    spec = importlib.util.spec_from_file_location("frozen_score_runtime", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("frozen_score_module_load_failed")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"json_root_must_be_object:{path}")
    return value


def _normalise_phrase(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return "".join(text.split())


def _load_phrase_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    payload = _read_json(path.resolve())
    mapping = payload.get("mapping", payload)
    if not isinstance(mapping, dict):
        raise ValueError("phrase_map_must_be_object_or_mapping_field")
    allowed = {"dx", "dk", "tt", "zt", "js", "zl", "xt", "jz", "UNKNOWN"}
    output: dict[str, str] = {}
    for phrase, label in mapping.items():
        value = str(label).strip()
        if value not in allowed:
            raise ValueError(f"phrase_map_label_invalid:{label}")
        key = _normalise_phrase(phrase)
        if not key:
            raise ValueError("phrase_map_phrase_empty")
        output[key] = value
    return output


def _fit(image: Any, width: int, height: int) -> Any:
    from PIL import Image

    output = Image.new("RGB", (width, height), "white")
    copy = image.copy().convert("RGB")
    copy.thumbnail((width, height), Image.Resampling.LANCZOS)
    output.paste(copy, ((width - copy.width) // 2, (height - copy.height) // 2))
    copy.close()
    return output


def _build_static_composite(image_path: Path, box: Mapping[str, Any]) -> Any:
    """Create target, hand, expanded-context and classroom views without gold labels."""

    from PIL import Image, ImageDraw

    with Image.open(image_path) as source:
        image = source.convert("RGB")
        width, height = image.size
        try:
            x1 = max(0, min(width - 1, int(box["x1"])))
            y1 = max(0, min(height - 1, int(box["y1"])))
            x2 = max(x1 + 1, min(width, int(box["x2"])))
            y2 = max(y1 + 1, min(height, int(box["y2"])))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("annotation_box_invalid") from exc
        crop = image.crop((x1, y1, x2, y2))
        box_width = x2 - x1
        box_height = y2 - y1
        hand_top = y1 + int(box_height * 0.42)
        hand_crop = image.crop((x1, min(y2 - 1, hand_top), x2, y2))
        margin_x = int(box_width * 0.45)
        margin_y = int(box_height * 0.25)
        expanded = image.crop((
            max(0, x1 - margin_x), max(0, y1 - margin_y),
            min(width, x2 + margin_x), min(height, y2 + margin_y),
        ))
        composite = Image.new("RGB", (480, 416), "white")
        tile_width = 160
        tile_height = 208
        for index, view in enumerate((crop, hand_crop, expanded)):
            tile = _fit(view, tile_width, tile_height)
            composite.paste(tile, (index * tile_width, 0))
            tile.close()
        context = _fit(image, 480, 208)
        composite.paste(context, (0, 208))
        context.close()
        draw = ImageDraw.Draw(composite)
        sx = 480 / width
        sy = 208 / height
        draw.rectangle((x1 * sx, 208 + y1 * sy, x2 * sx, 208 + y2 * sy), outline=(220, 40, 40), width=3)
        draw.text((max(2, x1 * sx), 210 + max(0, y1 * sy)), "target", fill=(220, 40, 40))
        hand_crop.close()
        expanded.close()
        crop.close()
        image.close()
    return composite


def _iter_inputs(manifest: Mapping[str, Any], *, split: str, repository_root: Path) -> list[dict[str, Any]]:
    score = _score_module()
    _payload, samples = score.load_frozen_manifest(Path(manifest["_path"]))
    inputs: list[dict[str, Any]] = []
    for sample_id, sample in samples.items():
        if split != "all" and str(sample.get("split", "")) != split:
            continue
        annotation_path = score._annotation_path(sample, repository_root)
        annotation = _read_json(annotation_path)
        labels = annotation.get("labels")
        if not isinstance(labels, list):
            raise ValueError(f"annotation_labels_missing:{sample_id}")
        image_path = score._resolve_repository_path(str(sample["image_path"]), repository_root)
        for box_index, box in enumerate(labels):
            if not isinstance(box, dict):
                raise ValueError(f"annotation_box_invalid:{sample_id}:{box_index}")
            gold_label = str(box.get("name", "")).strip().lower()
            if gold_label not in CORE8_CODES:
                raise ValueError(f"annotation_label_invalid:{sample_id}:{box_index}:{gold_label}")
            inputs.append({
                "sample_id": sample_id,
                "box_index": box_index,
                "image_path": image_path,
                "box": {key: box.get(key) for key in ("x1", "y1", "x2", "y2")},
                "split": str(sample.get("split", "")),
                "source_group": sample.get("source_group"),
                "_gold_label": gold_label,
            })
    return inputs


def _stratified_inputs(inputs: list[dict[str, Any]], per_class: int) -> list[dict[str, Any]]:
    if per_class < 1:
        raise ValueError("per_class_must_be_positive")
    def rank(item: Mapping[str, Any]) -> str:
        key = f"fixed8-diverse-v1:{item.get('sample_id')}:{item.get('box_index')}"
        return hashlib.sha256(key.encode("utf-8")).hexdigest()

    selected: list[dict[str, Any]] = []
    counts: dict[str, int] = {}
    for code in CORE8_CODES:
        candidates = sorted(
            (item for item in inputs if item.get("_gold_label") == code),
            key=rank,
        )
        chosen: list[dict[str, Any]] = []
        used_groups: set[str] = set()
        # First pass maximizes source-video diversity. The second fills any
        # remaining slots deterministically without cherry-picking by size or
        # model outcome.
        for item in candidates:
            group = str(item.get("source_group") or item.get("sample_id"))
            if group in used_groups:
                continue
            chosen.append(item)
            used_groups.add(group)
            if len(chosen) >= per_class:
                break
        if len(chosen) < per_class:
            chosen_keys = {(item.get("sample_id"), item.get("box_index")) for item in chosen}
            for item in candidates:
                key = (item.get("sample_id"), item.get("box_index"))
                if key in chosen_keys:
                    continue
                chosen.append(item)
                if len(chosen) >= per_class:
                    break
        selected.extend(chosen)
        counts[code] = len(chosen)
    missing = {code: per_class - count for code, count in counts.items() if count < per_class}
    if missing:
        raise ValueError(f"stratified_inputs_insufficient:{missing}")
    return sorted(selected, key=rank)


def _prediction_fields(
    raw_behavior: Any,
    *,
    response_valid: bool,
    actions: Any,
    phrase_map: Mapping[str, str],
) -> tuple[str, str | None]:
    normalized = _normalise_phrase(raw_behavior)
    resolved = decode_fixed_action(raw_behavior, actions)
    label = (
        resolved.code
        if response_valid and resolved is not None
        else phrase_map.get(normalized, "UNKNOWN") if response_valid else "UNKNOWN"
    )
    mapped = next((item for item in actions if item.code == label), None)
    return label, None if mapped is None else mapped.name_zh


def _read_existing(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"prediction_row_not_object:{line_number}")
            key = (str(row.get("sample_id", "")), int(row.get("box_index", -1)))
            if key in output:
                raise ValueError(f"prediction_duplicate_existing:{key[0]}:{key[1]}")
            output[key] = row
    return output


def run(
    config_path: Path,
    manifest_path: Path,
    output_path: Path,
    *,
    split: str = "test",
    repository_root: Path | None = None,
    limit: int | None = None,
    per_class: int | None = None,
    resume: bool = False,
    phrase_map_path: Path | None = None,
    prepare_only: bool = False,
) -> dict[str, Any]:
    if limit is not None and limit < 1:
        raise ValueError("limit_must_be_positive")
    if per_class is not None and per_class < 1:
        raise ValueError("per_class_must_be_positive")
    if limit is not None and per_class is not None:
        raise ValueError("limit_and_per_class_are_mutually_exclusive")
    config = load_config(config_path.resolve())
    actions = load_fixed_taxonomy(config.behavior_taxonomy_path)
    manifest_payload = _read_json(manifest_path.resolve())
    manifest_payload["_path"] = str(manifest_path.resolve())
    repository_root = (repository_root or PROJECT_ROOT.parent).resolve()
    phrase_map = _load_phrase_map(phrase_map_path)
    inputs = _iter_inputs(manifest_payload, split=split, repository_root=repository_root)
    if per_class is not None:
        inputs = _stratified_inputs(inputs, per_class)
    elif limit is not None:
        inputs = inputs[:limit]

    existing = _read_existing(output_path.resolve()) if resume else {}
    if prepare_only:
        return {
            "schema_version": "classroom_hybrid_ai.frozen_qwen_prediction_plan.v1",
            "manifest": str(manifest_path.resolve()),
            "split": split,
            "requested_inputs": len(inputs),
            "skipped_existing": sum(
                (item["sample_id"], item["box_index"]) in existing for item in inputs
            ),
            "prompt_uses_fixed_eight_actions": True,
            "per_class": per_class,
            "selection_policy": "deterministic_hash_source_group_diverse_v1" if per_class else "manifest_order",
            "model_started": False,
        }
    output_path.resolve().parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not resume:
        output_path.unlink()
    prompt = _classification_prompt(actions)
    adapter: Any = None
    completed = 0
    skipped = 0
    failures = 0
    started = time.perf_counter()
    try:
        adapter = _make_qwen_adapter(config)
        warmup = getattr(adapter, "warmup", None)
        if callable(warmup):
            warmup()
        with output_path.open("a", encoding="utf-8", newline="\n") as handle:
            for item in inputs:
                key = (item["sample_id"], item["box_index"])
                if key in existing:
                    skipped += 1
                    continue
                composite = None
                response: dict[str, Any] | None = None
                response_valid = False
                error: str | None = None
                request_started = time.perf_counter()
                try:
                    composite = _build_static_composite(item["image_path"], item["box"])
                    response = adapter.complete(
                        prompt,
                        images=[composite],
                        max_new_tokens=64,
                    )
                    if not isinstance(response, dict):
                        raise ValueError("frozen_qwen_response_not_object")
                    if set(response) != {"decision", "behavior", "observability"}:
                        raise ValueError("frozen_qwen_response_fields_invalid")
                    if response.get("decision") not in {"confirmed", "uncertain"}:
                        raise ValueError("frozen_qwen_decision_invalid")
                    if response.get("observability") not in {"clear", "partial", "none"}:
                        raise ValueError("frozen_qwen_observability_invalid")
                    if response.get("behavior") is not None and not isinstance(response.get("behavior"), str):
                        raise ValueError("frozen_qwen_behavior_invalid")
                    resolved = decode_fixed_action(response.get("behavior"), actions)
                    if response.get("decision") == "confirmed" and (
                        resolved is None or response.get("behavior") != resolved.action_id
                    ):
                        raise ValueError("frozen_qwen_behavior_not_exact_fixed_action_id")
                    if response.get("decision") == "uncertain" and response.get("behavior") is not None:
                        raise ValueError("frozen_qwen_uncertain_behavior_must_be_null")
                    response_valid = True
                except Exception as exc:
                    failures += 1
                    error = f"{type(exc).__name__}:{exc}"[:2000]
                finally:
                    if composite is not None:
                        composite.close()
                raw_phrase = None if response is None else response.get("behavior")
                mapped_label, mapped_name = _prediction_fields(
                    raw_phrase,
                    response_valid=response_valid,
                    actions=actions,
                    phrase_map=phrase_map,
                )
                row = {
                    "sample_id": item["sample_id"],
                    "box_index": item["box_index"],
                    "split": item["split"],
                    "source_group": item["source_group"],
                    "predicted_label": mapped_label,
                    "predicted_action_name": mapped_name,
                    "raw_phrase": raw_phrase,
                    "schema_valid": response_valid,
                    "qwen_decision": None if response is None else response.get("decision"),
                    "observability": None if response is None else response.get("observability"),
                    "elapsed_seconds": round(time.perf_counter() - request_started, 3),
                    "error": error,
                }
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                handle.flush()
                completed += 1
    finally:
        if adapter is not None:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()
    return {
        "schema_version": "classroom_hybrid_ai.frozen_qwen_predictions.v1",
        "output": str(output_path.resolve()),
        "manifest": str(manifest_path.resolve()),
        "split": split,
        "requested_inputs": len(inputs),
        "completed": completed,
        "skipped_existing": skipped,
        "failures": failures,
        "phrase_map_entries": len(phrase_map),
        "per_class": per_class,
        "selection_policy": "deterministic_hash_source_group_diverse_v1" if per_class else "manifest_order",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "prompt_uses_fixed_eight_actions": True,
        "requires_posthoc_scoring": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run fixed eight-action Qwen classification on frozen classroom boxes")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "configs" / "blind_benchmark_manifest.v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "val", "train", "quarantine", "all"), default="test")
    parser.add_argument("--repository-root", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--per-class", type=int, help="select an equal number of gold boxes per class without exposing gold to Qwen")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phrase-map", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="enumerate inputs without loading Qwen")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = run(
        args.config,
        args.manifest,
        args.output,
        split=args.split,
        repository_root=args.repository_root,
        limit=args.limit,
        per_class=args.per_class,
        resume=args.resume,
        phrase_map_path=args.phrase_map,
        prepare_only=args.prepare_only,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run"]
