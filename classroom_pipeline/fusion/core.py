from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


ACTION_CODE = {
    "listen": 0,
    "write": 1,
    "read": 2,
    "turn_head": 3,
    "group_discussion": 4,
    "teacher_interaction": 5,
    "raise_hand": 6,
    "stand": 7,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().lstrip("\ufeff")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_taxonomy(path: Path | None = None) -> dict[str, dict[str, Any]]:
    fallback = {
        "tt": ("listen", "听课", "listen"),
        "dx": ("write", "写字", "write"),
        "dk": ("read", "看书", "read"),
        "zt": ("turn_head", "转头", "turn head"),
        "xt": ("group_discussion", "小组讨论", "group discussion"),
        "js": ("raise_hand", "举手", "raise hand"),
        "zl": ("stand", "站立", "stand"),
        "jz": ("teacher_interaction", "教师互动", "teacher interaction"),
    }
    data: dict[str, Any] = {}
    if path and path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            data = {}
    classes = data.get("classes", []) if isinstance(data, dict) else []
    out: dict[str, dict[str, Any]] = {}
    for code, item in fallback.items():
        sid, zh, en = item
        out[code] = {
            "behavior_code": code,
            "semantic_id": sid,
            "semantic_label_zh": zh,
            "semantic_label_en": en,
            "taxonomy_version": "smart_classroom_semantics_v1",
        }
        out[sid] = out[code]
    for cls in classes:
        if not isinstance(cls, dict):
            continue
        code = str(cls.get("behavior_code", "")).strip().lower()
        sid = str(cls.get("semantic_id", "")).strip().lower()
        if not code or not sid:
            continue
        item = {
            "behavior_code": code,
            "behavior_label_zh": cls.get("label_zh", ""),
            "behavior_label_en": cls.get("label_en", ""),
            "semantic_id": sid,
            "semantic_label_zh": cls.get("semantic_label_zh", ""),
            "semantic_label_en": cls.get("semantic_label_en", ""),
            "taxonomy_version": data.get("taxonomy_version", "smart_classroom_semantics_v1"),
        }
        out[code] = item
        out[sid] = item
        for alias in cls.get("aliases", []):
            out[str(alias).strip().lower()] = item
    return out


def _best_semantic(raw: dict[str, Any], taxonomy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = [
        raw.get("behavior_code"),
        raw.get("semantic_id"),
        raw.get("action"),
        raw.get("label"),
        raw.get("raw_action"),
    ]
    for key in keys:
        item = taxonomy.get(str(key or "").strip().lower())
        if item:
            return item
    return taxonomy.get("tt", {"behavior_code": "tt", "semantic_id": "listen"})


def enrich_action_semantics(row: dict[str, Any], taxonomy: dict[str, dict[str, Any]]) -> dict[str, Any]:
    enriched = dict(row)
    semantic = _best_semantic(enriched, taxonomy)
    for key in (
        "behavior_code",
        "behavior_label_zh",
        "behavior_label_en",
        "semantic_id",
        "semantic_label_zh",
        "semantic_label_en",
        "taxonomy_version",
    ):
        value = semantic.get(key, "")
        if value != "" and not str(enriched.get(key, "")).strip():
            enriched[key] = value
    semantic_id = str(enriched.get("semantic_id", enriched.get("action", "listen")) or "listen").strip().lower()
    enriched["semantic_id"] = semantic_id
    enriched["action"] = semantic_id
    enriched.setdefault("action_code", ACTION_CODE.get(semantic_id, -1))
    return enriched


def semanticize_behavior_rows(rows: list[dict[str, Any]], taxonomy: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        new_row = dict(row)
        behaviors = new_row.get("behaviors", new_row.get("persons", []))
        if isinstance(behaviors, list):
            new_items = []
            for item in behaviors:
                if not isinstance(item, dict):
                    continue
                enriched = dict(item)
                semantic = _best_semantic(enriched, taxonomy)
                enriched.update({k: v for k, v in semantic.items() if v != ""})
                enriched.setdefault("action", semantic.get("semantic_id", enriched.get("action", "listen")))
                new_items.append(enriched)
            if "persons" in new_row and "behaviors" not in new_row:
                new_row["persons"] = new_items
            else:
                new_row["behaviors"] = new_items
        out.append(new_row)
    return out


def semanticize_object_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    object_map = {
        "book": ("read", "book"),
        "cell phone": ("turn_head", "phone"),
        "phone": ("turn_head", "phone"),
        "laptop": ("read", "laptop"),
        "keyboard": ("write", "keyboard"),
    }
    for row in rows:
        new_row = dict(row)
        objects = new_row.get("objects", [])
        if isinstance(objects, list):
            enriched = []
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                item = dict(obj)
                name = str(item.get("name", item.get("label", "")) or "").strip().lower()
                semantic_id, object_type = object_map.get(name, ("object_context", name or "object"))
                item["object_type"] = object_type
                item["semantic_support"] = semantic_id
                enriched.append(item)
            new_row["objects"] = enriched
        out.append(new_row)
    return out


def behavior_rows_to_actions(rows: list[dict[str, Any]], fps: float = 25.0) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for row in rows:
        frame = int(row.get("frame", 0) or 0)
        t = float(row.get("t", frame / max(float(fps), 1e-6)) or 0.0)
        persons = row.get("persons")
        behaviors = row.get("behaviors")
        items = persons if isinstance(persons, list) else behaviors if isinstance(behaviors, list) else []
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                continue
            top = item.get("top_behaviors")
            source_item = item
            if isinstance(top, list) and top and isinstance(top[0], dict):
                source_item = {**item, **top[0]}
            action = str(source_item.get("semantic_id", source_item.get("action", "listen")) or "listen").strip().lower()
            track_id = int(item.get("track_id", item.get("behavior_track_id", idx + 1)) or (idx + 1))
            conf = float(source_item.get("conf", source_item.get("confidence", item.get("conf", 0.0))) or 0.0)
            duration = 1.0 / max(float(fps), 1e-6)
            row_out = {
                "track_id": track_id,
                "action": action,
                "action_code": ACTION_CODE.get(action, -1),
                "conf": round(max(0.0, min(1.0, conf)), 6),
                "start_time": round(t, 6),
                "end_time": round(t + duration, 6),
                "start_frame": frame,
                "end_frame": frame + 1,
                "frame": frame,
                "t": round(t, 6),
                "source": "behavior_detection",
                "duration": round(duration, 6),
                "behavior_code": source_item.get("behavior_code", item.get("behavior_code", item.get("label", ""))),
                "semantic_id": action,
            }
            for key in ("semantic_label_zh", "semantic_label_en", "taxonomy_version", "bbox"):
                if key in source_item:
                    row_out[key] = source_item[key]
                elif key in item:
                    row_out[key] = item[key]
            actions.append(row_out)
    return actions


def merge_action_rows(primary: list[dict[str, Any]], extra: list[dict[str, Any]], extra_track_offset: int = 100000) -> list[dict[str, Any]]:
    rows = [dict(r) for r in primary]
    for raw in extra:
        row = dict(raw)
        try:
            row["track_id"] = int(row.get("track_id", 0)) + int(extra_track_offset)
        except Exception:
            row["track_id"] = int(extra_track_offset)
        row["source"] = str(row.get("source", "behavior_detection")) + "+extra"
        rows.append(row)
    rows.sort(key=lambda r: (float(r.get("start_time", r.get("t", 0.0)) or 0.0), int(r.get("track_id", 0) or 0)))
    return rows


def merge_fusion_actions(
    actions: list[dict[str, Any]],
    behavior_actions: list[dict[str, Any]],
    objects: list[dict[str, Any]],
    *,
    taxonomy: dict[str, dict[str, Any]] | None = None,
    behavior_track_offset: int = 0,
    behavior_track_mode: str = "offset",
    object_beta: float = 0.2,
) -> list[dict[str, Any]]:
    taxonomy = taxonomy or load_taxonomy(None)
    merged = [enrich_action_semantics(r, taxonomy) for r in actions]
    seen = {
        (
            int(r.get("track_id", 0) or 0),
            int(r.get("start_frame", r.get("frame", 0)) or 0),
            str(r.get("action", "")),
        )
        for r in merged
    }
    for raw in behavior_actions:
        row = enrich_action_semantics(raw, taxonomy)
        if str(behavior_track_mode).lower() == "offset":
            row["track_id"] = int(row.get("track_id", 0) or 0) + int(behavior_track_offset)
        key = (
            int(row.get("track_id", 0) or 0),
            int(row.get("start_frame", row.get("frame", 0)) or 0),
            str(row.get("action", "")),
        )
        if key in seen:
            continue
        row["source"] = str(row.get("source", "behavior_detection")) + "+fusion_v2"
        merged.append(row)
        seen.add(key)

    object_rows = [r for r in objects if isinstance(r.get("objects", []), list)]
    if object_rows:
        for row in merged:
            t = float(row.get("t", row.get("start_time", 0.0)) or 0.0)
            support = 0.0
            evidence: list[dict[str, Any]] = []
            for obj_row in object_rows:
                ot = float(obj_row.get("t", obj_row.get("frame", 0)) or 0.0)
                if abs(ot - t) > 0.8:
                    continue
                for obj in obj_row.get("objects", []):
                    if not isinstance(obj, dict):
                        continue
                    evidence.append({
                        "name": obj.get("name", obj.get("label", "")),
                        "conf": obj.get("conf", obj.get("confidence", 0.0)),
                        "semantic_support": obj.get("semantic_support", ""),
                    })
                    if obj.get("semantic_support") == row.get("semantic_id", row.get("action")):
                        support = max(support, float(obj.get("conf", obj.get("confidence", 0.0)) or 0.0))
            if evidence:
                row["object_evidence"] = {
                    "support_score": round(float(support) * float(object_beta), 6),
                    "objects": evidence[:5],
                }
    merged.sort(key=lambda r: (float(r.get("start_time", r.get("t", 0.0)) or 0.0), int(r.get("track_id", 0) or 0)))
    return merged


def build_event_queries(asr_rows: list[dict[str, Any]], action_rows: list[dict[str, Any]], topk: int = 12, min_conf: float = 0.35) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, row in enumerate(asr_rows):
        t = float(row.get("t", row.get("start", row.get("start_time", 0.0))) or 0.0)
        text = str(row.get("text", row.get("trigger_text", "")) or "")
        out.append({
            "event_id": f"e_{idx:06d}",
            "query_id": f"e_{idx:06d}",
            "schema_version": "2026-04-01",
            "event_type": row.get("event_type", "unknown"),
            "query_text": text,
            "trigger_text": text,
            "trigger_words": [],
            "timestamp": round(t, 3),
            "t_center": round(t, 3),
            "start": round(float(row.get("start", row.get("start_time", t))), 3),
            "end": round(float(row.get("end", row.get("end_time", t + 3.0))), 3),
            "confidence": float(row.get("confidence", 0.25) or 0.25),
            "source": "asr",
        })
    visual = [r for r in action_rows if float(r.get("conf", 0.0) or 0.0) >= float(min_conf)]
    visual.sort(key=lambda r: float(r.get("conf", 0.0) or 0.0), reverse=True)
    for idx, row in enumerate(visual[: max(0, int(topk))]):
        action = str(row.get("semantic_id", row.get("action", "unknown")) or "unknown")
        t = float(row.get("t", row.get("start_time", 0.0)) or 0.0)
        out.append({
            "event_id": f"v_{idx:06d}",
            "query_id": f"v_{idx:06d}",
            "schema_version": "2026-04-01",
            "event_type": action,
            "query_text": action.replace("_", " "),
            "trigger_text": f"visual_fallback:{action}",
            "trigger_words": [action],
            "timestamp": round(t, 3),
            "t_center": round(t, 3),
            "start": round(float(row.get("start_time", t)), 3),
            "end": round(float(row.get("end_time", t + 3.0)), 3),
            "confidence": round(float(row.get("conf", 0.0) or 0.0), 4),
            "source": "visual_fallback",
            "track_id": row.get("track_id"),
            "behavior_code": row.get("behavior_code", ""),
            "semantic_id": action,
        })
    return out


def _jsonl_parse_errors(path: Path, limit: int = 5) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"missing: {path.name}"]
    if path.is_file() and path.stat().st_size == 0:
        return [f"empty: {path.name}"]
    if path.suffix.lower() != ".jsonl":
        return errors
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except Exception as exc:
                errors.append(f"{path.name}:{idx}: {exc}")
                if len(errors) >= limit:
                    break
    return errors


def write_contract_report(output_dir: Path, report: Path, names: list[str], strict: bool) -> int:
    errors: list[str] = []
    warnings: list[str] = []
    for name in names:
        path = output_dir / name
        file_errors = _jsonl_parse_errors(path)
        if file_errors:
            if strict:
                errors.extend(file_errors)
            else:
                warnings.extend(file_errors)
    payload = {
        "status": "ok" if not errors else "failed",
        "output_dir": str(output_dir),
        "checked": names,
        "errors": errors,
        "warnings": warnings,
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if not errors else 2


def command_semanticize_objects(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    count = write_jsonl(Path(args.out), semanticize_object_rows(read_jsonl(Path(args.src))))
    print(f"[DONE] semanticized object rows={count}: {args.out}")
    return 0


def command_semanticize_behavior(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--taxonomy", default="")
    p.add_argument("--strict", default="1")
    args = p.parse_args(argv)
    taxonomy = load_taxonomy(Path(args.taxonomy) if args.taxonomy else None)
    count = write_jsonl(Path(args.out), semanticize_behavior_rows(read_jsonl(Path(args.src)), taxonomy))
    print(f"[DONE] semanticized behavior rows={count}: {args.out}")
    return 0


def command_behavior_to_actions(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--iou_thres", default="0.3")
    p.add_argument("--max_gap_frames", default="3")
    args = p.parse_args(argv)
    count = write_jsonl(Path(args.out), behavior_rows_to_actions(read_jsonl(Path(args.src)), fps=float(args.fps)))
    print(f"[DONE] behavior actions rows={count}: {args.out}")
    return 0


def command_merge_sources(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--primary", required=True)
    p.add_argument("--extra", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--extra_track_offset", type=int, default=100000)
    args = p.parse_args(argv)
    rows = merge_action_rows(read_jsonl(Path(args.primary)), read_jsonl(Path(args.extra)), args.extra_track_offset)
    count = write_jsonl(Path(args.out), rows)
    print(f"[DONE] merged action rows={count}: {args.out}")
    return 0


def command_merge_fusion(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--actions", required=True)
    p.add_argument("--behavior_actions", default="")
    p.add_argument("--objects", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--taxonomy", default="")
    p.add_argument("--behavior_track_offset", type=int, default=0)
    p.add_argument("--behavior_track_mode", default="offset")
    p.add_argument("--object_window", default="0.8")
    p.add_argument("--object_beta", type=float, default=0.2)
    p.add_argument("--strict", default="1")
    args = p.parse_args(argv)
    actions = read_jsonl(Path(args.actions))
    behavior = read_jsonl(Path(args.behavior_actions)) if args.behavior_actions else []
    objects = read_jsonl(Path(args.objects)) if args.objects else []
    taxonomy = load_taxonomy(Path(args.taxonomy) if args.taxonomy else None)
    rows = merge_fusion_actions(
        actions,
        behavior,
        objects,
        taxonomy=taxonomy,
        behavior_track_offset=args.behavior_track_offset,
        behavior_track_mode=args.behavior_track_mode,
        object_beta=args.object_beta,
    )
    count = write_jsonl(Path(args.out), rows)
    print(f"[DONE] fusion action rows={count}: {args.out}")
    return 0


def command_build_event_queries(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--event_queries", required=True)
    p.add_argument("--actions", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--visual_out", default="")
    p.add_argument("--min_asr_queries", type=int, default=2)
    p.add_argument("--visual_topk", type=int, default=12)
    p.add_argument("--visual_min_conf", type=float, default=0.35)
    p.add_argument("--validate", default="1")
    args = p.parse_args(argv)
    asr_rows = read_jsonl(Path(args.event_queries))
    action_rows = read_jsonl(Path(args.actions))
    rows = build_event_queries(asr_rows, action_rows, topk=args.visual_topk, min_conf=args.visual_min_conf)
    visual_rows = [r for r in rows if r.get("source") == "visual_fallback"]
    count = write_jsonl(Path(args.out), rows)
    if args.visual_out:
        write_jsonl(Path(args.visual_out), visual_rows)
    print(f"[DONE] fusion event query rows={count}: {args.out}")
    return 0


def command_check_fusion_contract(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--strict", default="1")
    args = p.parse_args(argv)
    return write_contract_report(
        Path(args.output_dir),
        Path(args.report),
        ["actions.fusion_v2.jsonl", "event_queries.fusion_v2.jsonl", "align_multimodal.json", "verified_events.jsonl"],
        str(args.strict) == "1",
    )


def command_check_pipeline_contract(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--strict", default="1")
    p.add_argument("--require_timeline", default="1")
    p.add_argument("--track_backend", default="hybrid")
    args = p.parse_args(argv)
    names = [
        "pose_keypoints_v2.jsonl",
        "pose_tracks_smooth.jsonl",
        "actions.fusion_v2.jsonl",
        "event_queries.fusion_v2.jsonl",
        "verified_events.jsonl",
    ]
    if str(args.require_timeline) == "1":
        names.extend(["timeline_chart.png", "timeline_students.csv", "student_id_map.json"])
    return write_contract_report(Path(args.output_dir), Path(args.report), names, str(args.strict) == "1")
