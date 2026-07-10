# scripts/10_visualize_timeline.py
import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "SimSun",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False


ACTION_COLORS = {
    0: "#2ecc71",  # listen (tt)
    1: "#34495e",  # turn_head (zt)
    2: "#e74c3c",  # phone (legacy)
    3: "#3498db",  # doze (legacy)
    4: "#e67e22",  # group_discussion (xt)
    5: "#9b59b6",  # write (dx)
    6: "#f1c40f",  # raise_hand (js)
    7: "#1abc9c",  # stand (zl)
    8: "#d35400",  # read (dk)
    9: "#8e44ad",  # teacher_interaction (jz)
}

ACTION_LABELS = {
    0: "Listen/听课",
    1: "Turn Head/转头",
    2: "Phone/手机(旧)",
    3: "Doze/瞌睡(旧)",
    4: "Discuss/讨论",
    5: "Write/写字",
    6: "Raise Hand/举手",
    7: "Stand/站立",
    8: "Read/看书",
    9: "Teacher Interaction/教师互动",
}

ACTION_STR_TO_ID = {
    "tt": 0,
    "listening": 0,
    "listen": 0,
    "head-up listening": 0,
    "zt": 1,
    "turn_head": 1,
    "turning_head": 1,
    "turning head": 1,
    "distract": 1,
    "playing_phone": 2,
    "phone": 2,
    "cell phone": 2,
    "sleeping": 3,
    "doze": 3,
    "sleep": 3,
    "xt": 4,
    "group_discussion": 4,
    "group discussion": 4,
    "chatting": 4,
    "chat": 4,
    "dx": 5,
    "write": 5,
    "writing": 5,
    "reading_writing": 5,
    "note": 5,
    "js": 6,
    "raising_hand": 6,
    "raise_hand": 6,
    "raise": 6,
    "hand_raise": 6,
    "zl": 7,
    "standing": 7,
    "stand": 7,
    "dk": 8,
    "reading": 8,
    "read": 8,
    "book": 8,
    "jz": 9,
    "teacher_interaction": 9,
    "teacher interaction": 9,
    "teacher_guidance": 9,
    "teacher guidance": 9,
    "teacher_instruction": 9,
}

BEHAVIOR_CODE_TO_ID = {
    "tt": 0,
    "zt": 1,
    "xt": 4,
    "dx": 5,
    "js": 6,
    "zl": 7,
    "dk": 8,
    "jz": 9,
}

ACTION_ID_TO_SEMANTIC = {
    0: {"behavior_code": "tt", "semantic_id": "listen", "semantic_label_zh": "听课", "semantic_label_en": "listen"},
    1: {"behavior_code": "zt", "semantic_id": "turn_head", "semantic_label_zh": "转头", "semantic_label_en": "turn head"},
    2: {"behavior_code": "zt", "semantic_id": "turn_head", "semantic_label_zh": "转头", "semantic_label_en": "turn head"},
    3: {"behavior_code": "tt", "semantic_id": "listen", "semantic_label_zh": "听课", "semantic_label_en": "listen"},
    4: {
        "behavior_code": "xt",
        "semantic_id": "group_discussion",
        "semantic_label_zh": "小组讨论",
        "semantic_label_en": "group discussion",
    },
    5: {"behavior_code": "dx", "semantic_id": "write", "semantic_label_zh": "写字", "semantic_label_en": "write"},
    6: {"behavior_code": "js", "semantic_id": "raise_hand", "semantic_label_zh": "举手", "semantic_label_en": "raise hand"},
    7: {"behavior_code": "zl", "semantic_id": "stand", "semantic_label_zh": "站立", "semantic_label_en": "stand"},
    8: {"behavior_code": "dk", "semantic_id": "read", "semantic_label_zh": "看书", "semantic_label_en": "read"},
    9: {
        "behavior_code": "jz",
        "semantic_id": "teacher_interaction",
        "semantic_label_zh": "教师互动",
        "semantic_label_en": "teacher interaction",
    },
}

GROUP_COLORS = {
    "lecture": "#A8E6CF",
    "group_discuss": "#AEDFF7",
    "pair_chat": "#B4C7E7",
    "break": "#FFD3B6",
    "individual_work": "#E2DBBE",
    "transition": "#DDDDDD",
}

MLLM_COLORS = {
    "lecture": "#6CBF84",
    "group_discuss": "#72B7D9",
    "pair_chat": "#5FA3C7",
    "individual_work": "#B39B6A",
    "break": "#E79C6D",
    "transition": "#9FA7B2",
}


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: Any, default: int = 10**18) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _load_group_events(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except Exception:
                continue
    return rows


def _load_mllm(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and isinstance(obj.get("items"), list):
            return [x for x in obj["items"] if isinstance(x, dict)]
    except Exception:
        pass
    return []


def _load_verified_events(path: Optional[Path]) -> List[Dict[str, Any]]:
    if path is None or not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    rows.sort(key=lambda x: (_safe_float(x.get("query_time", 0.0)), _safe_int(x.get("track_id", 10**9))))
    return rows


def _people_dict(people_raw: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(people_raw, dict):
        return {str(k): v for k, v in people_raw.items() if isinstance(v, dict)}
    if isinstance(people_raw, list):
        out: Dict[str, Dict[str, Any]] = {}
        for idx, p in enumerate(people_raw):
            if not isinstance(p, dict):
                continue
            pid = p.get("track_id", p.get("person_id", p.get("id", idx)))
            out[str(pid)] = p
        return out
    return {}


def _extract_visual_sequence(person: Dict[str, Any]) -> List[dict]:
    for key in ("visual_sequence", "visual_actions", "actions"):
        seq = person.get(key)
        if isinstance(seq, list):
            return [x for x in seq if isinstance(x, dict)]
    return []


def _action_code(item: Dict[str, Any]) -> int:
    if item.get("corrected_action_code") is not None:
        return _safe_int(item.get("corrected_action_code"), 0)
    if item.get("action_code") is not None:
        return _safe_int(item.get("action_code"), 0)
    semantic_id = str(item.get("semantic_id", "")).strip().lower()
    if semantic_id:
        if semantic_id in ACTION_STR_TO_ID:
            return ACTION_STR_TO_ID[semantic_id]
    behavior_code = str(item.get("behavior_code", "")).strip().lower()
    if behavior_code in BEHAVIOR_CODE_TO_ID:
        return BEHAVIOR_CODE_TO_ID[behavior_code]
    label = str(item.get("action", item.get("label", ""))).strip().lower()
    return ACTION_STR_TO_ID.get(label, 0)


def _action_time_range(item: Dict[str, Any], fps: Optional[float], default_dur: float = 0.2) -> Optional[Tuple[float, float]]:
    st = item.get("start_time", item.get("start", None))
    ed = item.get("end_time", item.get("end", None))
    if st is not None or ed is not None:
        stf = _safe_float(st, _safe_float(ed, 0.0) - default_dur)
        edf = _safe_float(ed, stf + default_dur)
        if edf < stf:
            stf, edf = edf, stf
        if edf <= stf:
            edf = stf + default_dur
        return stf, edf

    if item.get("time") is not None or item.get("t") is not None:
        t = _safe_float(item.get("time", item.get("t", 0.0)))
        dur = _safe_float(item.get("duration", default_dur), default_dur)
        return t, t + max(default_dur, dur)

    if fps and fps > 0:
        sf = item.get("start_frame", item.get("frame", None))
        ef = item.get("end_frame", None)
        if sf is not None:
            stf = _safe_float(sf) / fps
            edf = _safe_float(ef, _safe_float(sf) + default_dur * fps) / fps
            if edf < stf:
                edf = stf + default_dur
            return stf, edf

    return None


def _merge_segments(segments: List[Tuple[float, float, int]], gap_tol: float, min_dur: float) -> List[Tuple[float, float, int]]:
    if not segments:
        return []
    segments = sorted(segments, key=lambda x: x[0])
    merged: List[List[Any]] = [[segments[0][0], segments[0][1], segments[0][2]]]
    for st, ed, code in segments[1:]:
        last = merged[-1]
        if code == last[2] and st <= last[1] + gap_tol:
            last[1] = max(last[1], ed)
        else:
            merged.append([st, ed, code])
    return [(float(st), float(ed), int(code)) for st, ed, code in merged if (ed - st) >= min_dur]


def _make_non_overlapping(segments: List[Tuple[float, float, int]], min_dur: float) -> List[Tuple[float, float, int]]:
    out: List[Tuple[float, float, int]] = []
    last_end = -1.0
    for st, ed, code in sorted(segments, key=lambda x: (x[0], x[1])):
        if st < last_end:
            st = last_end
        if ed - st < min_dur:
            continue
        out.append((float(st), float(ed), int(code)))
        last_end = float(ed)
    return out


def _build_student_id_map(track_ids: List[Any]) -> Dict[str, Any]:
    normalized: List[int] = []
    for tid in track_ids:
        try:
            value = int(tid)
        except Exception:
            continue
        if value >= 0 and value not in normalized:
            normalized.append(value)
    normalized.sort()
    return {
        "schema_version": "student_id_map_v1",
        "student_count": len(normalized),
        "students": [
            {"student_id": f"S{idx + 1:02d}", "track_id": tid}
            for idx, tid in enumerate(normalized)
        ],
    }


def _student_id_lookup(student_map: Dict[str, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for row in student_map.get("students", []) or []:
        if not isinstance(row, dict):
            continue
        try:
            out[int(row.get("track_id"))] = str(row.get("student_id"))
        except Exception:
            continue
    return out


def _write_student_outputs(
    *,
    out_path: Path,
    track_ids: List[Any],
    rows: List[Dict[str, Any]],
) -> None:
    student_map = _build_student_id_map(track_ids)
    lookup = _student_id_lookup(student_map)
    map_path = out_path.parent / "student_id_map.json"
    csv_path = out_path.parent / "timeline_students.csv"

    with map_path.open("w", encoding="utf-8") as f:
        json.dump(student_map, f, ensure_ascii=False, indent=2)

    fieldnames = [
        "student_id",
        "track_id",
        "start_time",
        "end_time",
        "behavior_code",
        "semantic_label_zh",
        "semantic_label_en",
        "confidence",
        "source",
    ]
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            try:
                tid = int(row.get("track_id", -1))
            except Exception:
                tid = -1
            writer.writerow(
                {
                    "student_id": lookup.get(tid, "UNMATCHED" if tid < 0 else f"ID_{tid}"),
                    "track_id": tid,
                    "start_time": row.get("start_time", row.get("start", "")),
                    "end_time": row.get("end_time", row.get("end", "")),
                    "behavior_code": row.get("behavior_code", ""),
                    "semantic_label_zh": row.get("semantic_label_zh", ""),
                    "semantic_label_en": row.get("semantic_label_en", ""),
                    "confidence": row.get("confidence", row.get("conf", row.get("match_score", ""))),
                    "source": row.get("source", ""),
                }
            )

    print(f"[DONE] student id map: {map_path}")
    print(f"[DONE] timeline students csv: {csv_path}")


def draw_timeline(
    data: Dict[str, Any],
    out_path: Path,
    group_events: List[Dict[str, Any]],
    mllm_items: List[Dict[str, Any]],
    fps: Optional[float],
    top_n: int,
    gap_tol: float,
    min_dur: float,
    t_min: Optional[float],
    t_max: Optional[float],
    student_track_max: int,
) -> None:
    people = _people_dict(data.get("people", {}))
    people = {
        pid: person
        for pid, person in people.items()
        if _safe_int(pid, 10**18) <= int(student_track_max)
    }
    if not people:
        raise ValueError("No people found in source JSON")

    fps_final = fps
    if fps_final is None:
        fps_final = _safe_float((data.get("meta", {}) or {}).get("fps", 25.0), 25.0)

    per_person_segments: Dict[str, List[Tuple[float, float, int]]] = {}
    per_person_peer_marks: Dict[str, List[Tuple[float, int]]] = {}
    per_person_total: Dict[str, float] = {}
    global_max_t = 0.0

    for pid, person in people.items():
        raw_seq = _extract_visual_sequence(person)
        segs = []
        peer_marks = []
        for item in raw_seq:
            tr = _action_time_range(item, fps_final)
            if tr is None:
                continue
            st, ed = tr
            if t_min is not None and ed < t_min:
                continue
            if t_max is not None and st > t_max:
                continue
            if t_min is not None:
                st = max(st, t_min)
            if t_max is not None:
                ed = min(ed, t_max)
            if ed <= st:
                continue
            code = _action_code(item)
            segs.append((st, ed, code))
            if bool(item.get("peer_correction_applied", False)):
                peer_marks.append(((st + ed) * 0.5, code))
            global_max_t = max(global_max_t, ed)

        merged = _make_non_overlapping(_merge_segments(segs, gap_tol=gap_tol, min_dur=min_dur), min_dur=min_dur)
        if merged:
            per_person_segments[pid] = merged
            per_person_peer_marks[pid] = peer_marks
            per_person_total[pid] = sum((ed - st) for st, ed, _ in merged)

    ranked = sorted(per_person_total.items(), key=lambda kv: (-kv[1], _safe_int(kv[0])))
    ranked = ranked[: max(1, top_n)]
    person_ids = [pid for pid, _ in ranked]

    has_group = len(group_events) > 0
    has_mllm = len(mllm_items) > 0

    row_index: Dict[str, int] = {pid: i for i, pid in enumerate(person_ids)}
    extra_rows = int(has_group) + int(has_mllm)
    fig_h = max(4.5, min(0.38 * (len(person_ids) + extra_rows) + 2.2, 32))
    fig, ax = plt.subplots(figsize=(19, fig_h))

    x_end = t_max if t_max is not None else max(global_max_t, 1.0)

    # Person bars
    for pid in person_ids:
        y = row_index[pid]
        for st, ed, code in per_person_segments.get(pid, []):
            ax.barh(
                y,
                ed - st,
                left=st,
                height=0.72,
                color=ACTION_COLORS.get(code, "#333333"),
                linewidth=0,
                zorder=2,
            )

        # Peer-aware correction markers
        for t_mid, code in per_person_peer_marks.get(pid, []):
            if t_mid > x_end:
                continue
            ax.scatter([t_mid], [y], marker="*", s=26, c=[ACTION_COLORS.get(code, "#ffffff")], edgecolors="black", zorder=4)

    student_map_for_axis = _build_student_id_map(person_ids)
    student_lookup_for_axis = _student_id_lookup(student_map_for_axis)
    y_ticks = [row_index[pid] for pid in person_ids]
    y_labels = []
    for pid in person_ids:
        try:
            tid = int(pid)
        except Exception:
            tid = -1
        y_labels.append(student_lookup_for_axis.get(tid, f"ID {pid}"))

    # Group row
    group_row = None
    if has_group:
        group_row = len(person_ids)
        y_ticks.append(group_row)
        y_labels.append("Group")
        for ev in group_events:
            st = _safe_float(ev.get("start_time", ev.get("start", 0.0)))
            ed = _safe_float(ev.get("end_time", ev.get("end", st)))
            if ed <= st:
                continue
            if t_min is not None and ed < t_min:
                continue
            if t_max is not None and st > t_max:
                continue
            st = max(st, t_min) if t_min is not None else st
            ed = min(ed, t_max) if t_max is not None else ed
            if ed <= st:
                continue
            label = str(ev.get("group_event", "transition"))
            ax.barh(
                group_row,
                ed - st,
                left=st,
                height=0.70,
                color=GROUP_COLORS.get(label, "#dddddd"),
                linewidth=0,
                alpha=0.85,
                zorder=1,
            )

            # Interaction arcs
            x_mid = (st + ed) * 0.5
            for p in ev.get("interaction_pairs", []) or []:
                try:
                    a_id = str(int(p.get("id_a")))
                    b_id = str(int(p.get("id_b")))
                except Exception:
                    continue
                if a_id not in row_index or b_id not in row_index:
                    continue
                y1 = row_index[a_id]
                y2 = row_index[b_id]
                if y1 == y2:
                    continue
                score = max(0.0, min(1.0, _safe_float(p.get("score", 0.3), 0.3)))
                arc = FancyArrowPatch(
                    (x_mid, y1),
                    (x_mid, y2),
                    arrowstyle="-",
                    connectionstyle="arc3,rad=0.35",
                    linewidth=0.7 + 1.5 * score,
                    color="#444444",
                    alpha=0.22 + 0.45 * score,
                    zorder=3,
                )
                ax.add_patch(arc)

    # MLLM row
    if has_mllm:
        mllm_row = len(person_ids) + int(has_group)
        y_ticks.append(mllm_row)
        y_labels.append("MLLM")
        for item in mllm_items:
            st = _safe_float(item.get("start_time", 0.0))
            ed = _safe_float(item.get("end_time", st))
            if ed <= st:
                continue
            if t_min is not None and ed < t_min:
                continue
            if t_max is not None and st > t_max:
                continue
            st = max(st, t_min) if t_min is not None else st
            ed = min(ed, t_max) if t_max is not None else ed
            if ed <= st:
                continue
            label = str(item.get("mllm_label", "transition"))
            ax.barh(
                mllm_row,
                ed - st,
                left=st,
                height=0.62,
                color=MLLM_COLORS.get(label, "#aab2bd"),
                linewidth=0,
                alpha=0.8,
                zorder=1,
            )

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Classroom Timeline (Actions + Group Interaction + MLLM)", fontsize=12)
    ax.set_xlim(0, x_end + max(1.0, 0.02 * x_end))
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)

    legend_items = [mpatches.Patch(color=ACTION_COLORS[k], label=ACTION_LABELS[k]) for k in sorted(ACTION_COLORS.keys())]
    legend_items.append(mpatches.Patch(color="#ffffff", label="* Peer Corrected"))
    if has_group:
        legend_items.append(mpatches.Patch(color="#dddddd", label="Group/Event"))
    if has_mllm:
        legend_items.append(mpatches.Patch(color="#aab2bd", label="MLLM Semantic"))
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=min(10, len(legend_items)),
        frameon=False,
        fontsize=8,
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close(fig)

    # Frontend JSON export
    frontend_items: List[Dict[str, Any]] = []
    student_rows: List[Dict[str, Any]] = []
    student_map = _build_student_id_map(person_ids)
    student_lookup = _student_id_lookup(student_map)
    for pid in person_ids:
        row = row_index[pid]
        for st, ed, code in per_person_segments.get(pid, []):
            semantic = ACTION_ID_TO_SEMANTIC.get(code, {})
            try:
                tid = int(pid)
            except Exception:
                tid = -1
            frontend_items.append(
                {
                    "type": "person",
                    "track_id": tid,
                    "student_id": student_lookup.get(tid, f"ID_{tid}"),
                    "row": int(row),
                    "start": round(float(st), 3),
                    "end": round(float(ed), 3),
                    "action_id": int(code),
                    "action_label": ACTION_LABELS.get(code, "Unknown"),
                    **semantic,
                    "color": ACTION_COLORS.get(code, "#333333"),
                }
            )
            student_rows.append(
                {
                    "track_id": tid,
                    "start_time": round(float(st), 3),
                    "end_time": round(float(ed), 3),
                    "behavior_code": semantic.get("behavior_code", ""),
                    "semantic_label_zh": semantic.get("semantic_label_zh", ""),
                    "semantic_label_en": semantic.get("semantic_label_en", ""),
                    "confidence": 1.0,
                    "source": "timeline_per_person",
                }
            )
    for ev in group_events:
        frontend_items.append(
            {
                "type": "group",
                "start": _safe_float(ev.get("start_time", ev.get("start", 0.0))),
                "end": _safe_float(ev.get("end_time", ev.get("end", 0.0))),
                "label": str(ev.get("group_event", "transition")),
                "interaction_pairs": ev.get("interaction_pairs", []),
            }
        )
    for item in mllm_items:
        frontend_items.append(
            {
                "type": "mllm",
                "track_id": item.get("track_id"),
                "start": _safe_float(item.get("start_time", 0.0)),
                "end": _safe_float(item.get("end_time", 0.0)),
                "label": str(item.get("mllm_label", "transition")),
                "confidence": _safe_float(item.get("mllm_confidence", 0.0)),
            }
        )

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"items": frontend_items}, f, ensure_ascii=False, indent=2)
    _write_student_outputs(out_path=out_path, track_ids=person_ids, rows=student_rows)

    print(f"[DONE] timeline png: {out_path}")
    print(f"[DONE] timeline json: {json_path}")


def draw_verified_timeline(
    verified_rows: List[Dict[str, Any]],
    out_path: Path,
    t_min: Optional[float] = None,
    t_max: Optional[float] = None,
) -> None:
    if not verified_rows:
        raise ValueError("No verified events found")

    label_colors = {
        "match": "#2ecc71",
        "uncertain": "#f1c40f",
        "mismatch": "#e74c3c",
    }

    track_ids = sorted({int(r.get("track_id", -1)) for r in verified_rows})
    row_index = {tid: i for i, tid in enumerate(track_ids)}

    fig_h = max(4.2, min(0.42 * len(track_ids) + 2.0, 28))
    fig, ax = plt.subplots(figsize=(18, fig_h))
    max_t = 1.0

    exported_items: List[Dict[str, Any]] = []
    student_rows: List[Dict[str, Any]] = []
    student_map = _build_student_id_map(track_ids)
    student_lookup = _student_id_lookup(student_map)
    for row in verified_rows:
        tid = int(row.get("track_id", -1))
        y = row_index[tid]
        win = row.get("window", {})
        if isinstance(win, dict):
            st = _safe_float(win.get("start", row.get("window_start", row.get("query_time", 0.0))))
            ed = _safe_float(win.get("end", row.get("window_end", st + 0.3)))
        else:
            st = _safe_float(row.get("window_start", row.get("query_time", 0.0)))
            ed = _safe_float(row.get("window_end", st + 0.3))
        if ed <= st:
            ed = st + 0.3
        if t_min is not None and ed < t_min:
            continue
        if t_max is not None and st > t_max:
            continue
        st = max(st, t_min) if t_min is not None else st
        ed = min(ed, t_max) if t_max is not None else ed
        if ed <= st:
            continue

        label = str(row.get("match_label", row.get("label", "mismatch")))
        reliability = max(
            0.0,
            min(1.0, _safe_float(row.get("reliability_score", row.get("reliability", 0.0)), 0.0)),
        )
        color = label_colors.get(label, "#95a5a6")
        alpha = 0.35 + 0.65 * reliability

        ax.barh(y, ed - st, left=st, height=0.66, color=color, alpha=alpha, linewidth=0, zorder=2)
        if label == "mismatch":
            q_t = _safe_float(row.get("query_time", st), st)
            ax.scatter([q_t], [y], c="#111111", s=12, marker="x", zorder=4)

        max_t = max(max_t, ed)
        exported_items.append(
            {
                "type": "verified_event",
                "track_id": tid,
                "student_id": student_lookup.get(tid, "UNMATCHED" if tid < 0 else f"ID_{tid}"),
                "start": round(st, 3),
                "end": round(ed, 3),
                "query_time": round(
                    _safe_float(
                        row.get(
                            "query_time",
                            row.get("window", {}).get("center", row.get("window_center", 0.0))
                            if isinstance(row.get("window"), dict)
                            else row.get("window_center", 0.0),
                        ),
                        0.0,
                    ),
                    3,
                ),
                "query_id": row.get("query_id", row.get("event_id")),
                "event_type": row.get("event_type"),
                "behavior_code": row.get("behavior_code", ""),
                "semantic_id": row.get("semantic_id", row.get("event_type", "")),
                "semantic_label_zh": row.get("semantic_label_zh", ""),
                "semantic_label_en": row.get("semantic_label_en", ""),
                "label": label,
                "reliability": round(reliability, 4),
                "match_score": round(_safe_float(row.get("p_match", row.get("match_score", 0.0)), 0.0), 4),
                "query_text": row.get("query_text", ""),
                "color": color,
            }
        )
        student_rows.append(
            {
                "track_id": tid,
                "start_time": round(st, 3),
                "end_time": round(ed, 3),
                "behavior_code": row.get("behavior_code", ""),
                "semantic_label_zh": row.get("semantic_label_zh", ""),
                "semantic_label_en": row.get("semantic_label_en", ""),
                "confidence": round(_safe_float(row.get("p_match", row.get("match_score", 0.0)), 0.0), 4),
                "source": "verified_events",
            }
        )

    ax.set_yticks([row_index[tid] for tid in track_ids])
    ax.set_yticklabels([student_lookup.get(tid, "UNMATCHED" if tid < 0 else f"ID_{tid}") for tid in track_ids], fontsize=8)
    ax.set_xlabel("Time (s)")
    ax.set_title("Verified Events Timeline (Reliability-Aware)", fontsize=12)
    ax.set_xlim(0, (t_max if t_max is not None else max_t) + 1.0)
    ax.grid(axis="x", linestyle="--", linewidth=0.4, alpha=0.35)

    legend_items = [
        mpatches.Patch(color=label_colors["match"], label="match"),
        mpatches.Patch(color=label_colors["uncertain"], label="uncertain"),
        mpatches.Patch(color=label_colors["mismatch"], label="mismatch"),
    ]
    ax.legend(handles=legend_items, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=3, frameon=False, fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=240)
    plt.close(fig)

    json_path = out_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"items": exported_items}, f, ensure_ascii=False, indent=2)
    _write_student_outputs(out_path=out_path, track_ids=track_ids, rows=student_rows)

    print(f"[DONE] verified timeline png: {out_path}")
    print(f"[DONE] verified timeline json: {json_path}")


def main():
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="per_person_sequences.json")
    parser.add_argument("--verified_src", default="", help="verified_events.jsonl (preferred)")
    parser.add_argument("--out", required=True, help="timeline png path")
    parser.add_argument("--group_src", default="", help="group_events.jsonl")
    parser.add_argument("--mllm_src", default="", help="mllm_verified_sequences.json")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--top_n", type=int, default=50)
    parser.add_argument("--gap_tol", type=float, default=0.12)
    parser.add_argument("--min_dur", type=float, default=0.35)
    parser.add_argument("--t_min", type=float, default=None)
    parser.add_argument("--t_max", type=float, default=None)
    parser.add_argument("--student_track_max", type=int, default=99999)
    args = parser.parse_args()

    src = Path(args.src)
    verified_src = Path(args.verified_src) if args.verified_src else None
    out = Path(args.out)
    group_src = Path(args.group_src) if args.group_src else None
    mllm_src = Path(args.mllm_src) if args.mllm_src else None

    if not src.is_absolute():
        src = (base_dir / src).resolve()
    if verified_src and not verified_src.is_absolute():
        verified_src = (base_dir / verified_src).resolve()
    if not out.is_absolute():
        out = (base_dir / out).resolve()
    if group_src and not group_src.is_absolute():
        group_src = (base_dir / group_src).resolve()
    if mllm_src and not mllm_src.is_absolute():
        mllm_src = (base_dir / mllm_src).resolve()

    verified_rows = _load_verified_events(verified_src)
    if verified_rows:
        draw_verified_timeline(
            verified_rows=verified_rows,
            out_path=out,
            t_min=args.t_min,
            t_max=args.t_max,
        )
        return

    with open(src, "r", encoding="utf-8") as f:
        data = json.load(f)

    group_events = _load_group_events(group_src)
    mllm_items = _load_mllm(mllm_src)

    draw_timeline(
        data=data,
        out_path=out,
        group_events=group_events,
        mllm_items=mllm_items,
        fps=args.fps,
        top_n=args.top_n,
        gap_tol=float(args.gap_tol),
        min_dur=float(args.min_dur),
        t_min=args.t_min,
        t_max=args.t_max,
        student_track_max=int(args.student_track_max),
    )


if __name__ == "__main__":
    main()
