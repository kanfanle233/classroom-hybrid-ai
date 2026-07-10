import os
import sys
import json
import csv
import re
import shutil
import subprocess
from pathlib import Path
from functools import lru_cache
import inspect
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.templating import Jinja2Templates

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS, TSNE
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Avoid loky probing warnings/errors in restricted Windows environments.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from server.evidence_schema import (
    CANONICAL_EVENT_ID, CANONICAL_QUERY_TEXT, CANONICAL_QUERY_SOURCE,
    CANONICAL_QUERY_TIME, CANONICAL_WINDOW_START, CANONICAL_WINDOW_END,
    CANONICAL_TRACK_ID, CANONICAL_LABEL, CANONICAL_P_MATCH, CANONICAL_P_MISMATCH,
    CANONICAL_RELIABILITY, CANONICAL_UNCERTAINTY,
    CANONICAL_VISUAL_SCORE, CANONICAL_TEXT_SCORE, CANONICAL_UQ_SCORE,
    CANONICAL_OVERLAP, CANONICAL_ACTION_CONFIDENCE,
    normalize_verified_event, normalize_event_query, normalize_align_candidate,
    _pick as _schema_pick, _safe_float as _schema_safe_float, _safe_int as _schema_safe_int,
)
from server.registry import (
    CaseRegistry, ArtifactResolver, GoldLoader, GoldBackedEvaluator, GoldReadinessAnalyzer,
    CaseManifestV1, GoldEventV1, VerifierEvalReportV1, ArtifactStatus,
    _load_gold_events, _validate_gold,
)
from server.pipeline_manifest import PipelineManifestManager, StepValidator

# 灏濊瘯瀵煎叆 Levenshtein锛屽鏋滄病鏈夊氨鐢ㄧ畝鍖栫増
try:
    import Levenshtein  # type: ignore
except Exception:
    Levenshtein = None


# =========================================================
# 1) 璺緞锛氫紭鍏堜娇鐢?paths.py锛堜笌浣?scripts/ 涓竴鑷达級
# =========================================================
def _try_import_paths():
    try:
        import paths  # type: ignore
        return paths
    except Exception:
        # 鍏佽浠?server 鐩綍瀵煎叆澶辫触鏃讹紝鎵嬪姩鍔犺矾寰?
        server_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(server_dir)
        try:
            import paths  # type: ignore
            return paths
        except Exception:
            return None


paths_mod = _try_import_paths()

# 娌℃湁 paths.py 灏辨寜椤圭洰缁撴瀯鎺ㄦ柇
BASE_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(getattr(paths_mod, "PROJECT_ROOT", BASE_DIR)).resolve()
OUTPUT_DIR = Path(getattr(paths_mod, "OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()
DATA_DIR = Path(getattr(paths_mod, "DATA_DIR", PROJECT_ROOT / "data")).resolve()
_CASE_FIXTURE_ROOT_ENV = os.environ.get("YOLOV11_CASE_FIXTURE_ROOT", "").strip()
CASE_FIXTURE_ROOT = Path(_CASE_FIXTURE_ROOT_ENV).resolve() if _CASE_FIXTURE_ROOT_ENV else None
VIDEO_DIR = (DATA_DIR / "videos").resolve()
DOCS_DIR = (PROJECT_ROOT / "docs").resolve()
PAPER_TABLE_DIR = (DOCS_DIR / "assets" / "tables" / "paper_d3_selected").resolve()
PAPER_CHART_DIR = (DOCS_DIR / "assets" / "charts" / "paper_d3_selected").resolve()
PAPER_VIDEO_DIR = (DOCS_DIR / "assets" / "videos").resolve()
ASSETS_DIR = (DOCS_DIR / "assets").resolve()
YOLO_PAPER_DIR = (DOCS_DIR / "paper").resolve()
YOLO_PAPER_PACKAGE_DIR = (YOLO_PAPER_DIR / "outline" / "paper_package_20260426").resolve()
YOLO_PAPER_DEMO_DIR = (YOLO_PAPER_PACKAGE_DIR / "06_demo_materials").resolve()
PAPER_MAINLINE_OUTPUT_DIR = (
    OUTPUT_DIR / "codex_reports" / "run_full_paper_mainline_001" / "full_integration_001"
).resolve()
PAPER_MAINLINE_CASE_ID = "paper_mainline_20260426"
PAPER_MAINLINE_ALIASES = {
    PAPER_MAINLINE_CASE_ID,
    "paper_mainline",
    "paper_demo_20260426",
    "paper_package_20260426",
    "run_full_paper_mainline_001",
}
LLM_JUDGE_PIPELINE_DIR = (OUTPUT_DIR / "llm_judge_pipeline").resolve()
STUDENT_FUSION_MODES = {"llm_distilled_student_v4", "llm_distilled_student"}

# =========================================================
# Video search configuration (for browser-compatible playback)
# =========================================================
# Priority order: .web.mp4 (H.264) > .mp4 (original codec)
VIDEO_CANDIDATE_NAMES = [
    "pose_behavior_fusion_yolo11x.web.mp4",
    "pose_demo_yolo11x.web.mp4",
    "video.web.mp4",
    "pose_behavior_fusion_yolo11x.mp4",
    "pose_demo_yolo11x.mp4",
    "video.mp4",
]

# Directories to search for video files (in order)
MEDIA_SEARCH_DIRS = [
    PROJECT_ROOT / "docs" / "assets" / "videos",
    PROJECT_ROOT / "output" / "codex_reports",
    PROJECT_ROOT / "output" / "frontend_bundle",
]
if CASE_FIXTURE_ROOT is not None and CASE_FIXTURE_ROOT.exists():
    MEDIA_SEARCH_DIRS.append(CASE_FIXTURE_ROOT)

# Frontend templates live with the backend service.
TEMPLATE_DIR = (PROJECT_ROOT / "server" / "templates")
if not TEMPLATE_DIR.exists():
    TEMPLATE_DIR = PROJECT_ROOT

# Static assets live with the backend service when present.
STATIC_DIR = (PROJECT_ROOT / "server" / "static")
if not STATIC_DIR.exists():
    STATIC_DIR = PROJECT_ROOT

app = FastAPI(title="Classroom Multi-Modal Viz Server")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================================================
# 2) 闈欐€佹枃浠舵寕杞斤紙鍏煎鏃ц矾寰勶細/output /data锛?
# =========================================================
# 鍏煎浣?.txt Flask 鐗堬細/output/<path:filename> -> output鐩綍 :contentReference[oaicite:2]{index=2}
if OUTPUT_DIR.exists():
    app.mount("/output", StaticFiles(directory=str(OUTPUT_DIR)), name="output")
    # 涔熺粰涓€涓埆鍚嶏細/outputs
    app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# 鍏煎浣?.txt Flask 鐗堬細/data/<path:filename> -> data鐩綍 :contentReference[oaicite:3]{index=3}
if DATA_DIR.exists():
    app.mount("/data", StaticFiles(directory=str(DATA_DIR)), name="data")

# 瑙嗛鍒悕锛堟洿鐩磋锛?
if VIDEO_DIR.exists():
    app.mount("/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")

# 闈欐€佽祫婧?
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=str(DOCS_DIR)), name="docs")

if ASSETS_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")

if YOLO_PAPER_DIR.exists():
    app.mount("/yolo_paper", StaticFiles(directory=str(YOLO_PAPER_DIR)), name="yolo_paper")

templates = Jinja2Templates(directory=str(TEMPLATE_DIR))


def _paper_case_candidates() -> List[Path]:
    candidates = [PAPER_MAINLINE_OUTPUT_DIR, YOLO_PAPER_DEMO_DIR]
    out: List[Path] = []
    for p in candidates:
        try:
            if p.exists() and p.is_dir() and _looks_like_case_dir(p):
                out.append(p)
        except Exception:
            continue
    return out


def _find_paper_case_dir(video_id: str) -> Optional[Path]:
    vid = str(video_id or "").strip()
    if not vid:
        return None
    if vid in PAPER_MAINLINE_ALIASES:
        candidates = _paper_case_candidates()
        return candidates[0] if candidates else None

    if YOLO_PAPER_DIR.exists():
        direct = YOLO_PAPER_DIR / vid
        if direct.exists() and direct.is_dir() and _looks_like_case_dir(direct):
            return direct
        for candidate in YOLO_PAPER_DIR.rglob(vid):
            if candidate.is_dir() and _looks_like_case_dir(candidate):
                return candidate
    return None


def _find_case_dir(video_id: str) -> Optional[Path]:
    """Find case directory by exact folder name, supporting nested view/case layout."""
    candidates: List[Path] = []
    paper_case = _find_paper_case_dir(video_id)
    if paper_case is not None:
        candidates.append(paper_case)
    if OUTPUT_DIR.exists():
        direct = OUTPUT_DIR / video_id
        if direct.exists() and direct.is_dir() and _looks_like_case_dir(direct):
            candidates.append(direct)
        for candidate in OUTPUT_DIR.rglob(video_id):
            if candidate.is_dir() and _looks_like_case_dir(candidate):
                candidates.append(candidate)
    if not candidates:
        return None
    uniq = list(dict.fromkeys(candidates))
    return sorted(uniq, key=lambda p: (-_case_dir_quality(p), len(str(p))))[0]


def _find_case_dir_for_analysis(video_id: str) -> Optional[Path]:
    candidates: List[Path] = []
    paper_case = _find_paper_case_dir(video_id)
    if paper_case is not None:
        candidates.append(paper_case)

    if OUTPUT_DIR.exists():
        direct = OUTPUT_DIR / video_id
        if direct.exists() and direct.is_dir() and _looks_like_case_dir(direct):
            candidates.append(direct)
        for candidate in OUTPUT_DIR.rglob(video_id):
            if candidate.is_dir() and _looks_like_case_dir(candidate):
                candidates.append(candidate)

    if not candidates:
        return None

    def _rank(p: Path) -> Tuple[int, int]:
        has_bundle = int((p / "analysis" / "bundle.json").exists())
        # prefer bundle=1 first, then shorter path
        return (-has_bundle, len(str(p)))

    return sorted(candidates, key=_rank)[0]


def _to_output_url(path: Path) -> str:
    rel = path.resolve().relative_to(OUTPUT_DIR)
    return f"/output/{rel.as_posix()}"


def _to_data_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(DATA_DIR.resolve())
    except Exception:
        return None
    return f"/data/{rel.as_posix()}"


def _to_assets_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(ASSETS_DIR.resolve())
    except Exception:
        return None
    return f"/assets/{rel.as_posix()}"


def _to_yolo_paper_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(YOLO_PAPER_DIR.resolve())
    except Exception:
        return None
    return f"/yolo_paper/{rel.as_posix()}"


def _path_to_public_url(path: Optional[Path]) -> Optional[str]:
    if path is None:
        return None
    if not path.exists():
        return None
    for fn in (_to_output_url, _to_assets_url, _to_docs_url, _to_yolo_paper_url, _to_data_url):
        try:
            u = fn(path)  # type: ignore[misc]
        except Exception:
            u = None
        if u:
            return u
    return None


def _ffmpeg_binary() -> Optional[str]:
    for candidate in (shutil.which("ffmpeg"), shutil.which("ffmpeg.exe")):
        if candidate:
            return candidate
    # Common local Windows install used in this workspace.
    for fallback in (
        Path(r"F:\ffmpeg\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"),
        Path(r"F:\ffmpeg\bin\ffmpeg.exe"),
    ):
        try:
            if fallback.exists():
                return str(fallback)
        except Exception:
            continue
    return None


def _browser_proxy_video_path(src: Path) -> Path:
    if src.name.lower().endswith(".web.mp4"):
        return src
    return src.with_name(f"{src.stem}.web.mp4")


def _ensure_browser_friendly_video(src: Optional[Path]) -> Optional[Path]:
    if src is None:
        return None
    try:
        if not src.exists() or not src.is_file():
            return None
    except Exception:
        return None

    if src.name.lower().endswith(".web.mp4"):
        return src

    proxy = _browser_proxy_video_path(src)
    if _is_nonempty_file(proxy, min_bytes=4096):
        return proxy

    ffmpeg_bin = _ffmpeg_binary()
    if ffmpeg_bin is None:
        return src

    tmp = proxy.with_name(f"{proxy.stem}.tmp.mp4")
    try:
        if tmp.exists():
            tmp.unlink()
    except Exception:
        pass

    cmd = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "high",
        "-preset",
        "veryfast",
        "-crf",
        "24",
        "-movflags",
        "+faststart",
        str(tmp),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tmp.replace(proxy)
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass
        return src

    return proxy if proxy.exists() else src


_VIEW_TO_DATA_SUBDIR: Dict[str, str] = {
    "front": "正方视角",
    "rear": "后方视角",
    "teacher": "教师视角",
    "top1": "斜上方视角1",
    "top2": "斜上方视角2",
}


@lru_cache(maxsize=2)
def _guess_dataset_root() -> Optional[Path]:
    if not DATA_DIR.exists():
        return None
    wanted = set(_VIEW_TO_DATA_SUBDIR.values())
    for p in DATA_DIR.iterdir():
        if not p.is_dir():
            continue
        try:
            child_names = {c.name for c in p.iterdir() if c.is_dir()}
        except Exception:
            continue
        if wanted.intersection(child_names):
            return p
    return None


def _resolve_dataset_view_dir(view_name: str) -> Optional[Path]:
    root = _guess_dataset_root()
    if root is None:
        return None
    key = str(view_name).strip().lower()
    mapped = _VIEW_TO_DATA_SUBDIR.get(key)
    if not mapped:
        return None
    p = root / mapped
    return p if p.exists() and p.is_dir() else None


def _candidate_case_stems(case_id: str) -> List[str]:
    raw = str(case_id).strip()
    out: List[str] = []

    def _push(x: str) -> None:
        x = str(x).strip()
        if x and x not in out:
            out.append(x)

    _push(raw)
    digits = re.sub(r"\D+", "", raw)
    if digits:
        _push(digits)
        try:
            _push(str(int(digits)))
        except Exception:
            pass
        _push(digits.zfill(3))
        _push(digits.zfill(4))
        _push(digits.zfill(6))
    return out


def _extract_case_tokens(video_id: str, case_dir: Optional[Path] = None) -> Tuple[str, str]:
    view_name = ""
    if case_dir is not None and case_dir.parent != OUTPUT_DIR:
        view_name = case_dir.parent.name
    case_id = str(video_id).split("__")[-1] if "__" in str(video_id) else str(video_id)
    return str(view_name), str(case_id)


def _find_docs_video_path(video_id: str, variant: str) -> Optional[Path]:
    if not PAPER_VIDEO_DIR.exists():
        return None
    vid = str(video_id).strip()
    if not vid:
        return None

    candidates: List[str] = []
    if variant == "overlay":
        candidates = [f"{vid}.overlay.mp4"]
    elif variant == "original":
        candidates = [f"{vid}.original.mp4", f"{vid}.mp4"]
    else:
        candidates = [f"{vid}.{variant}.mp4", f"{vid}.mp4"]

    for name in candidates:
        p = PAPER_VIDEO_DIR / name
        if p.exists():
            return p
    return None


def _find_output_original_video_path(case_dir: Path, video_id: str, case_id: str) -> Optional[Path]:
    preferred = [
        case_dir / f"{video_id}.mp4",
        case_dir / f"{video_id}.original.mp4",
        case_dir / f"{case_id}.mp4",
        case_dir / "original.mp4",
        case_dir / "source.mp4",
        case_dir / "video.mp4",
    ]
    for p in preferred:
        if p.exists():
            return p

    for p in sorted(case_dir.glob("*.mp4")):
        low = p.name.lower()
        if low.endswith("_overlay.mp4") or low.endswith(".overlay.mp4"):
            continue
        if low in {"pose_demo_out.mp4", "objects_demo_out.mp4"}:
            continue
        return p
    return None


def _find_dataset_original_video_path(view_name: str, case_id: str) -> Optional[Path]:
    search_dirs: List[Path] = []
    view_dir = _resolve_dataset_view_dir(view_name)
    if view_dir is not None:
        search_dirs.append(view_dir)
    else:
        root = _guess_dataset_root()
        if root is not None:
            for mapped in _VIEW_TO_DATA_SUBDIR.values():
                p = root / mapped
                if p.exists() and p.is_dir():
                    search_dirs.append(p)

    seen: set[str] = set()
    for view_dir in search_dirs:
        key = str(view_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        for stem in _candidate_case_stems(case_id):
            p = view_dir / f"{stem}.mp4"
            if p.exists():
                return p
    return None


def _resolve_overlay_video_path(video_id: str, case_dir: Path) -> Optional[Path]:
    overlay_candidates = sorted(case_dir.glob("*_overlay.mp4"))
    if overlay_candidates:
        return overlay_candidates[0]
    preferred = [
        case_dir / "pose_behavior_fusion_yolo11x.mp4",
        case_dir / "overlay.mp4",
        case_dir / "video_overlay.mp4",
        case_dir / "pose_behavior_video.mp4",
    ]
    for p in preferred:
        if p.exists():
            return p
    manifest = _read_json_file(case_dir / "pipeline_manifest.json", {})
    if isinstance(manifest, dict):
        artifacts = manifest.get("artifacts", {})
        if isinstance(artifacts, dict):
            for key in ("pose_behavior_video", "overlay_video"):
                raw = str(artifacts.get(key) or "").strip()
                if raw:
                    p = Path(raw)
                    if p.exists():
                        return p
    return _find_docs_video_path(video_id, "overlay")


def _case_metadata_video_candidates(case_dir: Path) -> Tuple[List[Path], List[str]]:
    video_paths: List[Path] = []
    case_ids: List[str] = []

    def _push_case_id(value: Any) -> None:
        text = str(value or "").strip()
        if text and text not in case_ids:
            case_ids.append(text)

    for meta_name in ("pipeline_manifest.json", "asr_quality_report.json"):
        meta_path = case_dir / meta_name
        if not meta_path.exists():
            continue
        try:
            obj = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        _push_case_id(obj.get("case_id"))
        _push_case_id(obj.get("video_id"))
        raw_video = obj.get("video") or obj.get("input_video")
        if isinstance(raw_video, str) and raw_video.strip():
            p = Path(raw_video)
            if p.exists() and p.is_file():
                video_paths.append(p)

    return video_paths, case_ids


def _resolve_original_video_path(video_id: str, case_dir: Path) -> Optional[Path]:
    view_name, case_id = _extract_case_tokens(video_id, case_dir)
    p = _find_output_original_video_path(case_dir, video_id, case_id)
    metadata_video_paths, metadata_case_ids = _case_metadata_video_candidates(case_dir)
    if p is None and metadata_video_paths:
        p = metadata_video_paths[0]
    if p is None:
        p = _find_docs_video_path(video_id, "original")
    if p is None:
        p = _find_dataset_original_video_path(view_name, case_id)
    if p is None:
        for meta_case_id in metadata_case_ids:
            p = _find_dataset_original_video_path(view_name, meta_case_id)
            if p is not None:
                break
    if p is None:
        p = _resolve_overlay_video_path(video_id, case_dir)
    return p if p and p.exists() else None


# =========================================================
# Browser-friendly video search (independent of case directory)
# =========================================================

def _find_playable_video(case_id: str) -> Optional[Path]:
    """Search for the best playable video using priority names and multi-directory search.

    Search order per directory:
      1. {dir}/{case_id}/{VIDEO_CANDIDATE_NAME}        (direct match)
      2. {dir}/{case_id}/assets/{VIDEO_CANDIDATE_NAME}  (frontend_bundle layout)
      3. {dir}/{case_id}.web.mp4                        (flat .web.mp4)
      4. {dir}/{case_id}.mp4                            (flat .mp4)

    The first .web.mp4 (H.264) found wins over any .mp4 (original codec).
    """
    for search_dir in MEDIA_SEARCH_DIRS:
        if not search_dir.exists():
            continue

        case_subdir = search_dir / case_id

        # ── Pattern A: {dir}/{case_id}/{name} ──
        if case_subdir.is_dir():
            for name in VIDEO_CANDIDATE_NAMES:
                p = case_subdir / name
                if p.is_file():
                    return p

            # ── Pattern B: {dir}/{case_id}/assets/{name} ──
            assets_dir = case_subdir / "assets"
            if assets_dir.is_dir():
                for name in VIDEO_CANDIDATE_NAMES:
                    p = assets_dir / name
                    if p.is_file():
                        return p

            # ── Pattern C: recursive fallback inside case dir ──
            for pattern in ("*.web.mp4", "*.mp4"):
                for p in sorted(case_subdir.rglob(pattern)):
                    return p

        # ── Pattern D: {dir}/{case_id}.web.mp4 ──
        p = search_dir / f"{case_id}.web.mp4"
        if p.is_file():
            return p

        # ── Pattern E: {dir}/{case_id}.mp4 ──
        p = search_dir / f"{case_id}.mp4"
        if p.is_file():
            return p

    return None


def _list_all_searched_paths(case_id: str) -> List[str]:
    """Enumerate all possible search paths for debug output."""
    paths: List[str] = []
    for search_dir in MEDIA_SEARCH_DIRS:
        paths.append(f"[dir] {search_dir}")
        if not search_dir.exists():
            paths.append(f"    (SKIPPED — directory does not exist)")
            continue
        case_subdir = search_dir / case_id
        for name in VIDEO_CANDIDATE_NAMES:
            paths.append(f"    {case_subdir / name}")
            paths.append(f"    {case_subdir / 'assets' / name}")
        paths.append(f"    {search_dir / f'{case_id}.web.mp4'}")
        paths.append(f"    {search_dir / f'{case_id}.mp4'}")
    return paths


def _parse_range(range_header: str, file_size: int) -> Optional[Tuple[int, int]]:
    """Parse HTTP Range header value (e.g. 'bytes=0-1023') into (start, end)."""
    try:
        match = re.match(r"bytes=(\d+)-(\d*)$", range_header.strip())
        if not match:
            return None
        start = int(match.group(1))
        end_str = match.group(2)
        end = int(end_str) if end_str else file_size - 1
        end = min(end, file_size - 1)
        if start >= file_size or start > end:
            return None
        return (start, end)
    except Exception:
        return None


def _is_nonempty_file(path: Path, min_bytes: int = 1) -> bool:
    try:
        return path.exists() and path.is_file() and int(path.stat().st_size) >= int(min_bytes)
    except Exception:
        return False


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def _case_pipeline_status(case_dir: Path) -> Dict[str, Any]:
    manifest = _read_json_file(case_dir / "pipeline_manifest.json", {})
    config = manifest.get("config_snapshot", {}) if isinstance(manifest, dict) else {}
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    if not isinstance(config, dict):
        config = {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    behavior_model = str(
        config.get("behavior_det_model")
        or manifest.get("behavior_det_model", "") if isinstance(manifest, dict) else ""
    )
    object_model = str(config.get("det_model") or config.get("object_model") or "")
    behavior_model_norm = behavior_model.replace("\\", "/").lower()
    object_model_norm = object_model.replace("\\", "/").lower()

    pipeline_report = _read_json_file(case_dir / "pipeline_contract_v2_report.json", {})
    fusion_report = _read_json_file(case_dir / "fusion_contract_report.json", {})
    pipeline_counts = pipeline_report.get("counts", {}) if isinstance(pipeline_report, dict) else {}
    fusion_counts = fusion_report.get("counts", {}) if isinstance(fusion_report, dict) else {}
    counts = pipeline_counts if isinstance(pipeline_counts, dict) and pipeline_counts else fusion_counts
    if not isinstance(counts, dict):
        counts = {}

    has_actions_v2 = _is_nonempty_file(case_dir / "actions.fusion_v2.jsonl", min_bytes=8)
    has_events_v2 = _is_nonempty_file(case_dir / "event_queries.fusion_v2.jsonl", min_bytes=8)
    has_verified = _is_nonempty_file(case_dir / "verified_events.jsonl", min_bytes=8)
    has_behavior_semantic = _is_nonempty_file(case_dir / "behavior_det.semantic.jsonl", min_bytes=8)
    has_timeline_students = _is_nonempty_file(case_dir / "timeline_students.csv", min_bytes=16)
    has_student_map = _is_nonempty_file(case_dir / "student_id_map.json", min_bytes=16)
    has_timeline_json = _is_nonempty_file(case_dir / "timeline_chart.json", min_bytes=8)

    uses_finetuned_behavior_model = (
        "official_yolo11s_detect_e150_v1" in behavior_model_norm
        or ("yolo11s" in behavior_model_norm and "/weights/best.pt" in behavior_model_norm)
    )
    uses_old_object_only_model = bool(object_model_norm) and "yolo11x.pt" in object_model_norm and not uses_finetuned_behavior_model
    fusion_v2_ready = (
        _truthy(config.get("fusion_contract_v2"))
        or bool(artifacts.get("actions_fusion_v2"))
        or (has_actions_v2 and has_events_v2)
    )
    timeline_ready = has_timeline_json and has_timeline_students and has_student_map
    contract_status = str(pipeline_report.get("status") or fusion_report.get("status") or "").lower()
    contract_ok = contract_status in {"ok", "pass", "passed"} or (
        _is_nonempty_file(case_dir / "pipeline_contract_v2_report.json", min_bytes=8)
        and contract_status != "failed"
    )
    mainline_ready = (
        uses_finetuned_behavior_model
        and fusion_v2_ready
        and has_actions_v2
        and has_events_v2
        and has_verified
        and has_behavior_semantic
        and contract_ok
    )

    reason = ""
    if not mainline_ready:
        missing: List[str] = []
        if not uses_finetuned_behavior_model:
            missing.append("fine_tuned_yolo11s_behavior_model")
        if not fusion_v2_ready:
            missing.append("fusion_v2")
        if not has_actions_v2:
            missing.append("actions.fusion_v2.jsonl")
        if not has_events_v2:
            missing.append("event_queries.fusion_v2.jsonl")
        if not has_verified:
            missing.append("verified_events.jsonl")
        if not has_behavior_semantic:
            missing.append("behavior_det.semantic.jsonl")
        if not contract_ok:
            missing.append("pipeline_contract_v2")
        reason = ", ".join(missing)

    return {
        "behavior_model": behavior_model,
        "object_model": object_model,
        "uses_finetuned_behavior_model": uses_finetuned_behavior_model,
        "uses_old_object_only_model": uses_old_object_only_model,
        "fusion_v2_ready": fusion_v2_ready,
        "mainline_ready": mainline_ready,
        "timeline_ready": timeline_ready,
        "contract_status": contract_status or ("ok" if contract_ok else ""),
        "legacy_reason": reason,
        "counts": counts,
        "warnings": pipeline_report.get("warnings", []) if isinstance(pipeline_report, dict) else [],
    }


def _case_dir_quality(case_dir: Path) -> int:
    score = 0
    status = _case_pipeline_status(case_dir)
    if status.get("mainline_ready"):
        score += 320
    elif status.get("uses_finetuned_behavior_model"):
        score += 160
    if status.get("fusion_v2_ready"):
        score += 90
    if status.get("timeline_ready"):
        score += 45
    if _is_nonempty_file(case_dir / "timeline_chart.json", min_bytes=8):
        score += 120
    if _is_nonempty_file(case_dir / "timeline_viz.json", min_bytes=8):
        score += 90
    if _is_nonempty_file(case_dir / "timeline_data.json", min_bytes=8):
        score += 70
    if _is_nonempty_file(case_dir / "verified_events.jsonl", min_bytes=8):
        score += 60
    if _is_nonempty_file(case_dir / "pose_tracks_smooth.jsonl", min_bytes=16) or _is_nonempty_file(case_dir / "pose_tracks_smooth_uq.jsonl", min_bytes=16):
        score += 55
    if _is_nonempty_file(case_dir / "actions.fusion_v2.jsonl", min_bytes=8):
        score += 50
    if _is_nonempty_file(case_dir / "event_queries.fusion_v2.jsonl", min_bytes=8):
        score += 40
    if _is_nonempty_file(case_dir / "actions_fused.jsonl", min_bytes=8) or _is_nonempty_file(case_dir / "actions.jsonl", min_bytes=8):
        score += 35
    if _is_nonempty_file(case_dir / "transcript.jsonl", min_bytes=8):
        score += 25
    if _is_nonempty_file(case_dir / "pipeline_contract_v2_report.json", min_bytes=8) or _is_nonempty_file(case_dir / "fusion_contract_report.json", min_bytes=8):
        score += 20
    if _is_nonempty_file(case_dir / "student_projection.json", min_bytes=8) or _is_nonempty_file(case_dir / "static_projection.json", min_bytes=8):
        score += 15

    # prefer richer artifacts when multiple branches exist for the same case id
    for fn, cap, weight in [
        ("pose_tracks_smooth.jsonl", 4_000_000, 30),
        ("pose_tracks_smooth_uq.jsonl", 4_000_000, 20),
        ("actions.fusion_v2.jsonl", 200_000, 25),
        ("actions.jsonl", 200_000, 15),
        ("actions_fused.jsonl", 200_000, 20),
        ("timeline_viz.json", 50_000, 12),
        ("verified_events.jsonl", 80_000, 10),
    ]:
        p = case_dir / fn
        if not _is_nonempty_file(p):
            continue
        try:
            score += int(min(cap, int(p.stat().st_size)) / cap * weight)
        except Exception:
            pass

    low = case_dir.as_posix().lower()
    if "asr_ablation" in low:
        score -= 120
    if "/_tmp_" in low or "/cache/" in low:
        score -= 80
    if "/_batch/" in low:
        score -= 30
    return score


def _looks_like_case_dir(case_dir: Path) -> bool:
    """Heuristic: a runnable case folder should contain at least one core pipeline artifact."""
    markers = [
        "frontend_data_manifest.json",
        "timeline_chart.json",
        "timeline_viz.json",
        "per_person_sequences.json",
        "verified_events.jsonl",
        "event_queries.jsonl",
        "event_queries.fusion_v2.jsonl",
        "pose_tracks_smooth.jsonl",
        "pose_tracks_smooth_uq.jsonl",
        "actions.jsonl",
        "actions_fused.jsonl",
        "actions.fusion_v2.jsonl",
        "pipeline_manifest.json",
        "pipeline_contract_v2_report.json",
        "fusion_contract_report.json",
        "student_projection.json",
    ]
    return any(_is_nonempty_file(case_dir / m, min_bytes=2) for m in markers)


def _load_case_analysis_bundle(case_dir: Path) -> Optional[Dict[str, Any]]:
    p = case_dir / "analysis" / "bundle.json"
    if not p.exists():
        return None
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _coerce_scalar(v: Any) -> Any:
    if v is None or isinstance(v, (bool, int, float)):
        return v
    if isinstance(v, str):
        t = v.strip()
        if t == "":
            return ""
        lo = t.lower()
        if lo == "true":
            return True
        if lo == "false":
            return False
        if re.fullmatch(r"[+-]?\d+", t):
            try:
                return int(t)
            except Exception:
                return t
        if re.fullmatch(r"[+-]?(\d+(\.\d*)?|\.\d+)", t):
            try:
                return float(t)
            except Exception:
                return t
    return v


def _coerce_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        clean = {str(k): _coerce_scalar(v) for k, v in row.items()}
        out.append(clean)
    return out


def _read_json_file(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        try:
            return json.loads(path.read_text(encoding="utf-8-sig"))
        except Exception:
            return default


def _student_model_display_name(path_text: Any) -> str:
    text = str(path_text or "").strip()
    return Path(text).name if text else ""


def _student_distillation_from_evidence(evidence: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(evidence, dict):
        evidence = {}
    fusion_mode = str(evidence.get("fusion_mode") or "unknown")
    teacher_source = str(evidence.get("teacher_source") or "")
    teacher_dataset = str(evidence.get("teacher_dataset") or "")
    student_model_name = _student_model_display_name(evidence.get("student_model_path"))
    enabled = (
        fusion_mode in STUDENT_FUSION_MODES
        or bool(teacher_source)
        or bool(teacher_dataset)
        or bool(student_model_name)
    )
    return {
        "enabled": bool(enabled),
        "fusion_mode": fusion_mode,
        "student_model_name": student_model_name,
        "student_feature_version": str(evidence.get("student_feature_version") or ""),
        "teacher_source": teacher_source,
        "teacher_dataset": teacher_dataset,
        "evaluation_kind": "pseudo_label_benchmark" if enabled else "",
        "silver_gold_boundary": "silver_labels_not_human_gold" if enabled else "",
    }


def _read_llm_student_v4_status() -> Dict[str, Any]:
    report_path = LLM_JUDGE_PIPELINE_DIR / "metrics" / "training_report_v4.json"
    checks_path = LLM_JUDGE_PIPELINE_DIR / "reports" / "anti_drift_checks_v4.json"
    report = _read_json_file(report_path, {})
    checks = _read_json_file(checks_path, {})
    if not isinstance(report, dict) or not report:
        return {"available": False}

    best_model = str(report.get("best_model") or "")
    test_metrics: Dict[str, Any] = {}
    results = report.get("results")
    if isinstance(results, dict) and best_model:
        candidate = results.get(f"{best_model}_test")
        if isinstance(candidate, dict):
            test_metrics = candidate

    data = report.get("data") if isinstance(report.get("data"), dict) else {}
    return {
        "available": True,
        "fusion_mode": "llm_distilled_student_v4",
        "teacher_source": str(report.get("teacher_source") or "claude_agent"),
        "teacher_dataset": str(report.get("teacher_dataset") or "llm_adjudicated_dataset_v4"),
        "student_model_name": "student_judge_v4_best.joblib",
        "sample_count": int(_safe_float(data.get("total_samples"), 0)),
        "label_distribution": data.get("label_distribution", {}),
        "best_model": best_model,
        "test_macro_f1": _safe_float(test_metrics.get("f1_macro"), 0.0),
        "test_balanced_accuracy": _safe_float(test_metrics.get("balanced_accuracy"), 0.0),
        "teacher_student_agreement": _safe_float(test_metrics.get("teacher_student_agreement"), 0.0),
        "evaluation_kind": str(report.get("evaluation_kind") or "pseudo_label_benchmark"),
        "anti_drift_passed": bool(checks.get("all_checks_passed")) if isinstance(checks, dict) else False,
        "silver_gold_boundary": "silver_labels_not_human_gold",
    }


def _read_csv_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    encodings = ["utf-8-sig", "utf-8"]
    for enc in encodings:
        try:
            with path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                rows = [dict(r) for r in reader]
            break
        except Exception:
            rows = []
            continue
    return _coerce_rows(rows)


def _to_docs_url(path: Path) -> Optional[str]:
    try:
        rel = path.resolve().relative_to(DOCS_DIR.resolve())
    except Exception:
        return None
    return f"/docs/{rel.as_posix()}"


def _load_jsonl_rows(path: Path, strict: bool = False) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                if strict:
                    raise
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


# ── Strict-mode readers (track parse errors instead of swallowing) ─

def _load_jsonl_rows_strict(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Read JSONL, returning (rows, parse_errors)."""
    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    if not path.exists():
        return rows, errors
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception as e:
                errors.append({"line": line_no, "error": str(e), "preview": line[:120]})
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows, errors


def _read_json_file_strict(path: Path) -> Tuple[Any, List[Dict[str, Any]]]:
    """Read JSON file, returning (data, parse_errors)."""
    errors: List[Dict[str, Any]] = []
    if not path.exists():
        return None, errors
    for enc in ("utf-8", "utf-8-sig"):
        try:
            return json.loads(path.read_text(encoding=enc)), errors
        except json.JSONDecodeError as e:
            errors.append({"file": str(path), "encoding": enc, "error": str(e)})
        except Exception:
            continue
    return None, errors


# ── Unified case context resolution ───────────────────────────────

def _resolve_case_context(case_id: str) -> Optional[Dict[str, Any]]:
    """Resolve a case_id to its context: case_dir, bundle_dir, source_case_dir, data_source.

    Returns None if the case cannot be found.
    """
    case_dir = _find_front_case_dir(case_id)
    if case_dir is None:
        # Try legacy resolution
        case_dir = _find_case_dir(case_id)
    if case_dir is None:
        return None

    data_source = _vsumvis_data_source(case_dir)
    bundle_dir: Optional[Path] = None
    source_case_dir: Optional[Path] = None

    if data_source == "frontend_bundle":
        bundle_dir = case_dir
        manifest_path = case_dir / "frontend_data_manifest.json"
        manifest = _read_json_file(manifest_path, {})
        if isinstance(manifest, dict):
            src = manifest.get("source_case_dir", "")
            if src:
                p = Path(src)
                if p.exists():
                    source_case_dir = p
    else:
        source_case_dir = case_dir
        # Check if a corresponding bundle exists
        bundle_candidate = BUNDLE_DIR / (case_id.replace("_full", "_sliced"))
        if not bundle_candidate.exists():
            bundle_candidate = BUNDLE_DIR / case_id
        if bundle_candidate.exists() and (bundle_candidate / "frontend_data_manifest.json").exists():
            bundle_dir = bundle_candidate

    available_files: Dict[str, bool] = {}
    search_dir = source_case_dir if source_case_dir else case_dir
    for fname in ("verified_events.jsonl", "verified_events.json",
                  "event_queries.fusion_v2.jsonl", "event_queries.jsonl", "event_queries.json",
                  "align_multimodal.json",
                  "asr_quality_report.json",
                  "timeline_chart.json", "timeline_students.csv",
                  "pipeline_contract_v2_report.json",
                  "transcript.jsonl"):
        available_files[fname] = (search_dir / fname).exists()

    return {
        "case_id": case_id,
        "case_dir": str(case_dir),
        "bundle_dir": str(bundle_dir) if bundle_dir else None,
        "source_case_dir": str(source_case_dir) if source_case_dir else None,
        "data_source": data_source,
        "available_files": available_files,
    }


# ── Alignment & ASR readers (work from context) ────────────────────

@lru_cache(maxsize=128)
def _read_alignments_from_dir(search_dir: Path) -> List[Dict[str, Any]]:
    """Cached read of align_multimodal.json from a single directory."""
    path = search_dir / "align_multimodal.json"
    if not path.exists():
        return []
    data = _read_json_file(path, None)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("events", "items", "alignments"):
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _read_alignments_any(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Read align_multimodal.json from source_case_dir or case_dir."""
    search_dirs: List[Path] = [Path(ctx["case_dir"])]
    if ctx.get("source_case_dir"):
        src_dir = Path(ctx["source_case_dir"])
        if src_dir not in search_dirs:
            search_dirs.append(src_dir)
    for search_dir in search_dirs:
        result = _read_alignments_from_dir(search_dir)
        if result:
            return result
    return []


@lru_cache(maxsize=128)
def _read_asr_from_dir(search_dir: Path) -> Optional[Dict[str, Any]]:
    """Cached read of asr_quality_report.json from a single directory."""
    path = search_dir / "asr_quality_report.json"
    if path.exists():
        data = _read_json_file(path, None)
        if isinstance(data, dict):
            return data
    bundle_path = search_dir / "asr_quality.json"
    if bundle_path.exists():
        data = _read_json_file(bundle_path, None)
        if isinstance(data, dict):
            report = data.get("report")
            if isinstance(report, dict):
                return report
            return data
    return None


def _read_asr_quality_any(ctx: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Read asr_quality_report.json from source_case_dir or case_dir."""
    search_dirs: List[Path] = [Path(ctx["case_dir"])]
    if ctx.get("source_case_dir"):
        src_dir = Path(ctx["source_case_dir"])
        if src_dir not in search_dirs:
            search_dirs.append(src_dir)
    for search_dir in search_dirs:
        result = _read_asr_from_dir(search_dir)
        if result is not None:
            return result
    return None


def _event_id_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    match = re.search(r"(\d{6})", text)
    if match:
        return match.group(1)
    return text


def _event_id_matches(left: Any, right: Any) -> bool:
    left_text = str(left or "").strip()
    right_text = str(right or "").strip()
    if not left_text or not right_text:
        return False
    if left_text == right_text:
        return True
    left_key = _event_id_key(left_text)
    right_key = _event_id_key(right_text)
    return bool(left_key) and left_key == right_key


def _compute_selected_candidate_rank(
    event_id: str, selected_track_id: int, alignments: List[Dict[str, Any]],
) -> Tuple[int, int, str, str]:
    """Find which rank (1-based) the selected track_id occupies in the alignment candidates.

    Returns (selected_candidate_rank, candidate_count, rank_metric, selected_by).
    - rank_metric: description of the ranking method used (e.g. "overlap*0.65+action_confidence*0.35")
    - selected_by: "track_id_match" to clarify this is a post-hoc lookup, not the verifier's
      actual decision function. Prevents explanation conflicts with verifier internals.
    """
    rank_metric = "overlap*0.65+action_confidence*0.35"
    for rec in alignments:
        eid = str(rec.get("event_id") or rec.get("query_id") or "")
        if not _event_id_matches(eid, event_id):
            continue
        candidates = rec.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        scored = []
        for c in candidates:
            ov = _safe_float(c.get("overlap", 0), 0.0)
            ac = _safe_float(c.get("action_confidence", 0), 0.0)
            score = ov * 0.65 + ac * 0.35
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        for rank, (_, c) in enumerate(scored, 1):
            tid = int(_safe_float(c.get("track_id", -1), -1.0))
            if tid == selected_track_id:
                return rank, len(scored), rank_metric, "track_id_match"
        return 0, len(scored), rank_metric, "track_id_match"
    return 0, 0, rank_metric, "track_id_match"


# ── ASR quality / query source inference ──────────────────────────

def _infer_query_source(
    event_queries: List[Dict[str, Any]],
    verified_events: List[Dict[str, Any]],
    asr_report: Optional[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build a per-event lookup of query_source, is_visual_fallback, and source_conflict.

    Dual-source resolution:
    - Primary: event_queries.fusion_v2.jsonl `source` field (authoritative per-event)
    - Secondary: asr_quality_report.json global status (inferred fallback signal)
    - When primary and secondary disagree, source_conflict=true is set and the
      primary source is kept. We never silently override the pipeline's own
      source annotation.
    """
    result: Dict[str, Dict[str, Any]] = {}

    # Build lookup from event_queries
    eq_source: Dict[str, str] = {}
    for eq in event_queries:
        eid = str(eq.get("event_id") or eq.get("query_id") or "")
        if eid:
            src = str(eq.get("source", "")).lower()
            eq_source[eid] = src if src in ("asr", "visual_fallback") else "unknown"

    # Infer ASR-based source from report (secondary)
    asr_inferred = "unknown"
    if asr_report and isinstance(asr_report, dict):
        status = str(asr_report.get("status", "")).lower()
        segments_accepted = int(asr_report.get("segments_accepted", -1))
        if status == "placeholder" or segments_accepted == 0:
            asr_inferred = "visual_fallback"
        elif status == "ok" or segments_accepted > 0:
            asr_inferred = "asr"

    for ve in verified_events:
        eid = str(ve.get("event_id") or ve.get("query_id") or "")
        if not eid:
            continue
        ve_source = str(ve.get("query_source") or ve.get("source") or "").lower()
        eq_src = eq_source.get(eid, "")

        # Resolve primary source: eq.source > ve.source > asr_inferred
        if eq_src in ("asr", "visual_fallback"):
            primary = eq_src
            primary_from = "event_query"
        elif ve_source in ("asr", "visual_fallback"):
            primary = ve_source
            primary_from = "verified_event"
        elif asr_inferred != "unknown":
            primary = asr_inferred
            primary_from = "asr_report"
        else:
            primary = "asr"
            primary_from = "default"

        # Detect conflict: primary disagrees with ASR inferred (when both are known)
        source_conflict = False
        conflict_detail = None
        if asr_inferred != "unknown" and primary_from != "asr_report":
            if primary == "asr" and asr_inferred == "visual_fallback":
                source_conflict = True
                conflict_detail = "event_query says asr but ASR report status suggests visual_fallback"
            elif primary == "visual_fallback" and asr_inferred == "asr":
                source_conflict = True
                conflict_detail = "event_query says visual_fallback but ASR report has valid segments"

        is_fb = (primary == "visual_fallback")
        result[eid] = {
            "query_source": primary,
            "is_visual_fallback": is_fb,
            "source_conflict": source_conflict,
            "asr_inferred_source": asr_inferred,
        }
        if conflict_detail:
            result[eid]["conflict_detail"] = conflict_detail

    return result


def _timeline_from_verified_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    for row in rows:
        win = row.get("window", {})
        if isinstance(win, dict):
            st = float(win.get("start", row.get("window_start", row.get("query_time", 0.0))))
            ed = float(win.get("end", row.get("window_end", row.get("query_time", st + 0.3))))
        else:
            st = float(row.get("window_start", row.get("query_time", 0.0)))
            ed = float(row.get("window_end", row.get("query_time", st + 0.3)))
        if ed <= st:
            ed = st + 0.3
        label = str(row.get("match_label", row.get("label", "mismatch")))
        action_name = "unknown"
        for key in ("semantic_id", "action", "event_type"):
            candidate = _normalize_action_name(row.get(key, ""))
            if candidate in _ACTION_NAME_TO_ID:
                action_name = candidate
                break
        action_id = _parse_action_id(action_name, row.get("action_id", -1))
        if action_id < 0:
            if label == "match":
                action_id = 0
            elif label == "uncertain":
                action_id = 1
            else:
                action_id = 2
        items.append(
            {
                "type": "verified_event",
                "track_id": int(row.get("track_id", -1)),
                "start": st,
                "end": ed,
                "action_id": action_id,
                "action": action_name,
                "action_label": row.get("semantic_label_en") or row.get("event_type") or label,
                "verification_label": label,
                "match_label": label,
                "query_id": row.get("query_id", row.get("event_id")),
                "event_type": row.get("event_type"),
                "query_text": row.get("query_text"),
                "reliability": float(row.get("reliability_score", row.get("reliability", 0.0))),
                "match_score": float(row.get("p_match", row.get("match_score", 0.0))),
                "semantic_id": row.get("semantic_id"),
                "semantic_label_zh": row.get("semantic_label_zh"),
                "semantic_label_en": row.get("semantic_label_en"),
                "row": int(row.get("track_id", -1)),
            }
        )
    return {"items": items}


_ACTION_NAME_TO_ID: Dict[str, int] = {
    "listen": 0,
    "distract": 1,
    "phone": 2,
    "doze": 3,
    "chat": 4,
    "note": 5,
    "raise": 6,
    "stand": 7,
    "read": 8,
}


def _normalize_action_name(action_name: Any) -> str:
    name = str(action_name or "").strip().lower()
    if "/" in name:
        name = name.split("/", 1)[0].strip()
    name = name.replace("-", "_").replace(" ", "_")
    return name


def _parse_action_id(action_name: Any, fallback: Any = None) -> int:
    name = _normalize_action_name(action_name)
    if name in _ACTION_NAME_TO_ID:
        return _ACTION_NAME_TO_ID[name]
    try:
        return int(fallback)
    except Exception:
        return -1


def _timeline_from_actions_rows(rows: List[Dict[str, Any]], fps: float = 25.0) -> Dict[str, Any]:
    items: List[Dict[str, Any]] = []
    safe_fps = max(1e-6, float(fps))
    for row in rows:
        try:
            tid = int(row.get("track_id", -1))
        except Exception:
            tid = -1
        if tid < 0:
            continue

        action_name = "unknown"
        for key in ("semantic_id", "fused_action", "action", "raw_action", "action_label", "event_type"):
            candidate = _normalize_action_name(row.get(key, ""))
            if candidate in _ACTION_NAME_TO_ID:
                action_name = candidate
                break
            if candidate and action_name == "unknown":
                action_name = candidate
        action_id = _parse_action_id(action_name, row.get("action_id", -1))

        st = _safe_float(row.get("start", row.get("start_time", row.get("window_start", 0.0))), 0.0)
        ed = _safe_float(row.get("end", row.get("end_time", row.get("window_end", st))), st)

        if "start_frame" in row:
            st = _safe_float(row.get("start_frame"), 0.0) / safe_fps
        if "end_frame" in row:
            ed = _safe_float(row.get("end_frame"), st * safe_fps) / safe_fps

        if ed <= st:
            ed = st + 0.2

        items.append(
            {
                "type": "person",
                "track_id": tid,
                "start": float(st),
                "end": float(ed),
                "action_id": int(action_id),
                "action": action_name,
                "student_id": row.get("student_id"),
                "behavior_code": row.get("behavior_code"),
                "semantic_id": row.get("semantic_id"),
                "semantic_label_zh": row.get("semantic_label_zh"),
                "semantic_label_en": row.get("semantic_label_en"),
                "conf": _safe_float(row.get("conf", row.get("fused_conf", row.get("raw_conf", 0.0))), 0.0),
                "source": row.get("source"),
                "duration": float(ed - st),
                "row": tid,
            }
        )
    return {"items": items}


def _timeline_items_count(payload: Any) -> int:
    if isinstance(payload, dict):
        items = payload.get("items", [])
        if isinstance(items, list):
            return len(items)
        return 0
    if isinstance(payload, list):
        return len(payload)
    return 0


# =========================================================
# 3) 灏忓伐鍏凤細绠€鏄?Levenshtein锛堟棤涓夋柟搴撴椂鍏滃簳锛?
# =========================================================
def simple_levenshtein(seq1: str, seq2: str) -> float:
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=np.float32)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1],
                    matrix[x, y - 1] + 1
                )
            else:
                matrix[x, y] = min(
                    matrix[x - 1, y] + 1,
                    matrix[x - 1, y - 1] + 1,
                    matrix[x, y - 1] + 1
                )
    return float(matrix[size_x - 1, size_y - 1])


def _supports_param(cls, param: str) -> bool:
    try:
        return param in inspect.signature(cls.__init__).parameters
    except (TypeError, ValueError):
        return False


def _make_mds(dissimilarity: str, random_state: int = 42):
    kwargs = {"n_components": 2, "dissimilarity": dissimilarity, "random_state": random_state}
    if _supports_param(MDS, "n_jobs"):
        kwargs["n_jobs"] = 1
    return MDS(**kwargs)


def _fit_tsne(data: np.ndarray, metric: str, random_state: int = 42) -> np.ndarray:
    n_samples = data.shape[0]
    if n_samples <= 3:
        perp = float(max(1, n_samples - 1))
    else:
        perp = float(min(30, max(5, n_samples // 2)))
    perp = min(perp, float(n_samples) - 1e-3)
    kwargs = {
        "n_components": 2,
        "perplexity": perp,
        "metric": metric,
        "random_state": random_state,
    }
    if _supports_param(TSNE, "n_jobs"):
        kwargs["n_jobs"] = 1
    return TSNE(**kwargs).fit_transform(data)


# =========================================================
# 4) 缂撳瓨璇诲彇锛歵imeline/stats/transcript/tracks
# =========================================================
@lru_cache(maxsize=32)
def load_timeline_data_cached(video_id: str) -> Optional[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return None

    candidates: List[Tuple[int, int, Dict[str, Any]]] = []

    def _push_timeline(obj: Any, priority: int) -> None:
        if isinstance(obj, dict):
            payload = obj
        elif isinstance(obj, list):
            payload = {"items": obj}
        else:
            return
        n = _timeline_items_count(payload)
        if n > 0:
            candidates.append((n, priority, payload))

    p1 = case_dir / "timeline_chart.json"
    if p1.exists():
        try:
            data = json.loads(p1.read_text(encoding="utf-8"))
            _push_timeline(data, priority=90)
        except Exception:
            pass

    p_viz = case_dir / "timeline_viz.json"
    if p_viz.exists():
        try:
            data = json.loads(p_viz.read_text(encoding="utf-8"))
            _push_timeline(data, priority=85)
        except Exception:
            pass

    p2 = case_dir / "timeline_data.json"
    if p2.exists():
        try:
            data = json.loads(p2.read_text(encoding="utf-8"))
            _push_timeline(data, priority=80)
        except Exception:
            pass

    fps = 25.0
    for meta_file in sorted(case_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            fps = _safe_float(meta.get("fps"), 25.0)
            break
        except Exception:
            continue

    p_actions_fusion_v2 = case_dir / "actions.fusion_v2.jsonl"
    if p_actions_fusion_v2.exists():
        rows = _load_jsonl_rows(p_actions_fusion_v2)
        if rows:
            _push_timeline(_timeline_from_actions_rows(rows, fps=fps), priority=70)

    p_actions_fused = case_dir / "actions_fused.jsonl"
    if p_actions_fused.exists():
        rows = _load_jsonl_rows(p_actions_fused)
        if rows:
            _push_timeline(_timeline_from_actions_rows(rows, fps=fps), priority=65)

    p_actions = case_dir / "actions.jsonl"
    if p_actions.exists():
        rows = _load_jsonl_rows(p_actions)
        if rows:
            _push_timeline(_timeline_from_actions_rows(rows, fps=fps), priority=60)

    p3 = case_dir / "verified_events.jsonl"
    if p3.exists():
        rows = _load_jsonl_rows(p3)
        if rows:
            _push_timeline(_timeline_from_verified_rows(rows), priority=10)

    if candidates:
        candidates.sort(key=lambda x: (x[1], x[0]), reverse=True)
        return candidates[0][2]

    return None


@lru_cache(maxsize=32)
def load_stats_data_cached(video_id: str) -> Dict[str, Any]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return {"class_pie_chart": []}
    p = case_dir / "timeline_chart_stats.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"class_pie_chart": []}


@lru_cache(maxsize=32)
def load_transcript_data_cached(video_id: str) -> List[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return []
    p = case_dir / "transcript.jsonl"
    data: List[Dict[str, Any]] = []
    if p.exists():
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data.append(json.loads(line))
                except Exception:
                    continue
    return data


@lru_cache(maxsize=32)
def load_verified_events_cached(video_id: str) -> List[Dict[str, Any]]:
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return []
    p = case_dir / "verified_events.jsonl"
    return _load_jsonl_rows(p)


@lru_cache(maxsize=16)
def load_tracks_data_cached(video_id: str) -> Dict[str, Any]:
    """
    杩斿洖缁撴瀯:
    {
      "12": [{"id": 3, "box": [x1,y1,x2,y2]}, ...],
      "13": ...
    }
    """
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        return {}
    p = case_dir / "pose_tracks_smooth.jsonl"
    if not p.exists():
        p = case_dir / "pose_tracks_smooth_uq.jsonl"
    if not p.exists():
        return {}

    tracks: Dict[str, Any] = {}
    try:
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                frame = d.get("frame")
                persons = d.get("persons", [])
                tracks[str(frame)] = [{"id": pp.get("track_id"), "box": pp.get("bbox")} for pp in persons]
    except Exception:
        return {}

    return tracks


# =========================================================
# 5) 鏍稿績锛歅rojection锛堟敮鎸?euclidean / levenshtein + pca/mds/tsne锛?
# =========================================================
@lru_cache(maxsize=64)
def compute_projection_cached(video_id: str, method: str, metric: str) -> Dict[str, Any]:
    raw = load_timeline_data_cached(video_id)
    if metric != "spatial" and not raw:
        return {"method": method, "metric": metric, "points": []}

    items = raw.get("items", raw if isinstance(raw, list) else []) if raw else []
    if not isinstance(items, list):
        items = []

    # --- 鏋勫缓姣忎釜浜虹殑鏁版嵁 ---
    person_data: Dict[int, Dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue

        # 鍙鐞?person 琛岋紱group 琛屼細鐢?track_id = -1锛堜綘鐢熸垚 json 鏃跺氨鏄繖涔堝共鐨勶級 :contentReference[oaicite:5]{index=5}
        if item.get("type", "person") != "person":
            continue

        tid = int(item.get("track_id", -999))
        if tid < 0:
            continue

        if tid not in person_data:
            person_data[tid] = {"actions": [], "spatial": []}

        st = float(item.get("start", 0.0))
        ed = float(item.get("end", st))
        code = int(item.get("action_id", -1))
        row = int(item.get("row", 0))

        person_data[tid]["actions"].append({"code": code, "dur": max(0.0, ed - st), "start": st})
        person_data[tid]["spatial"].append(row)

    track_ids = sorted(person_data.keys())
    n_samples = len(track_ids)
    coords = None

    # -------- A) Spatial map from tracked bboxes --------
    if metric == "spatial":
        tracks = load_tracks_data_cached(video_id)
        if not tracks:
            return {"method": "spatial", "metric": "spatial", "points": []}

        agg: Dict[int, Dict[str, float]] = {}
        max_x = 1.0
        max_y = 1.0
        for persons in tracks.values():
            if not isinstance(persons, list):
                continue
            for pp in persons:
                try:
                    tid = int(pp.get("id", -1))
                except Exception:
                    tid = -1
                if tid < 0:
                    continue
                box = pp.get("box")
                if not isinstance(box, (list, tuple)) or len(box) < 4:
                    continue
                try:
                    x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                except Exception:
                    continue
                cx = max(0.0, (x1 + x2) * 0.5)
                cy = max(0.0, (y1 + y2) * 0.5)
                max_x = max(max_x, x1, x2, cx)
                max_y = max(max_y, y1, y2, cy)
                s = agg.setdefault(tid, {"sx": 0.0, "sy": 0.0, "n": 0.0})
                s["sx"] += cx
                s["sy"] += cy
                s["n"] += 1.0

        if not agg:
            return {"method": "spatial", "metric": "spatial", "points": []}

        frame_w = max(1.0, max_x)
        frame_h = max(1.0, max_y)
        points = []
        for tid in sorted(agg.keys()):
            s = agg[tid]
            n = max(1.0, s["n"])
            x = min(1.0, max(0.0, (s["sx"] / n) / frame_w))
            y = min(1.0, max(0.0, (s["sy"] / n) / frame_h))
            points.append({"track_id": int(tid), "x": float(x), "y": float(y), "samples": int(n)})
        return {"method": "spatial", "metric": "spatial", "points": points, "frame_w": frame_w, "frame_h": frame_h}

    if n_samples < 2:
        return {"method": method, "metric": metric, "points": []}

    # -------- B) Vector-based: metric = euclidean --------
    if metric == "euclidean":
        # 9 actions + 1 spatial + 1 activity
        features = np.zeros((n_samples, 11), dtype=np.float32)

        for i, tid in enumerate(track_ids):
            acts = sorted(person_data[tid]["actions"], key=lambda x: x["start"])
            total_dur = float(sum(a["dur"] for a in acts))

            counts = np.zeros(9, dtype=np.float32)
            for a in acts:
                c = int(a["code"])
                if 0 <= c < 9:
                    counts[c] += float(a["dur"])

            if total_dur > 1e-6:
                features[i, :9] = counts / total_dur

            ys = person_data[tid]["spatial"]
            features[i, 9] = float(sum(ys) / max(1, len(ys)))
            features[i, 10] = float(len(acts) / total_dur) if total_dur > 1e-6 else 0.0

        X = StandardScaler().fit_transform(features)

        if method == "pca":
            coords = PCA(n_components=2).fit_transform(X)
        elif method == "mds":
            coords = _make_mds("euclidean", random_state=42).fit_transform(X)
        elif method == "tsne":
            try:
                coords = _fit_tsne(X, metric="euclidean", random_state=42)
            except Exception:
                coords = PCA(n_components=2).fit_transform(X)
        else:
            # 涓嶈璇嗗氨鍥為€€
            coords = PCA(n_components=2).fit_transform(X)

    # -------- C) Sequence-based: metric = levenshtein --------
    elif metric == "levenshtein":
        sequences: List[str] = []
        for tid in track_ids:
            acts = sorted(person_data[tid]["actions"], key=lambda x: x["start"])
            # 绠€鍗曟嫾鎺?action_id 瀛楃涓?
            seq_str = "".join([str(int(a["code"])) for a in acts])
            sequences.append(seq_str)

        dist_matrix = np.zeros((n_samples, n_samples), dtype=np.float32)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if Levenshtein is not None:
                    d = float(Levenshtein.distance(sequences[i], sequences[j]))
                else:
                    d = simple_levenshtein(sequences[i], sequences[j])

                max_len = max(len(sequences[i]), len(sequences[j]), 1)
                dist_matrix[i, j] = dist_matrix[j, i] = d / float(max_len)

        # PCA 涓嶈兘澶勭悊璺濈鐭╅樀锛屽己鍒跺洖钀藉埌 MDS
        if method == "pca":
            method = "mds"

        if method == "mds":
            coords = _make_mds("precomputed", random_state=42).fit_transform(dist_matrix)
        elif method == "tsne":
            try:
                coords = _fit_tsne(dist_matrix, metric="precomputed", random_state=42)
            except Exception:
                coords = _make_mds("precomputed", random_state=42).fit_transform(dist_matrix)
        else:
            coords = _make_mds("precomputed", random_state=42).fit_transform(dist_matrix)

    else:
        # 鏈煡 metric
        return {"method": method, "metric": metric, "points": []}

    # 褰掍竴鍖栧埌 0-1
    coords = MinMaxScaler().fit_transform(coords)

    points = []
    for i, tid in enumerate(track_ids):
        points.append({"track_id": int(tid), "x": float(coords[i, 0]), "y": float(coords[i, 1])})

    return {"method": method, "metric": metric, "points": points}


def _resolve_existing_path(path_text: str) -> Optional[Path]:
    if not path_text:
        return None
    p = Path(path_text)
    if p.is_absolute():
        return p if p.exists() else None
    probe = (PROJECT_ROOT / path_text).resolve()
    if probe.exists():
        return probe
    return None


def _load_docs_videos() -> List[Dict[str, Any]]:
    if not PAPER_VIDEO_DIR.exists():
        return []
    rows: List[Dict[str, Any]] = []
    for p in sorted(PAPER_VIDEO_DIR.glob("*.mp4")):
        name = p.name
        variant = "default"
        case_id = name[:-4]
        if name.endswith(".overlay.mp4"):
            variant = "overlay"
            case_id = name[: -len(".overlay.mp4")]
        elif name.endswith(".original.mp4"):
            variant = "original"
            case_id = name[: -len(".original.mp4")]
        rows.append(
            {
                "name": name,
                "case_id": case_id,
                "variant": variant,
                "url": _to_docs_url(p),
            }
        )
    return rows


def _build_run_summary(run_metric_ci: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summary: Dict[str, Dict[str, Any]] = {}
    for row in run_metric_ci:
        run_name = str(row.get("run_name", ""))
        metric = str(row.get("metric", ""))
        if not run_name:
            continue
        summary.setdefault(run_name, {"run_name": run_name})
        if metric == "verified_p_match_mean":
            summary[run_name]["verified_mean"] = _safe_float(row.get("mean"), 0.0)
            summary[run_name]["verified_ci_low"] = _safe_float(row.get("ci95_low"), 0.0)
            summary[run_name]["verified_ci_high"] = _safe_float(row.get("ci95_high"), 0.0)
        elif metric == "align_avg_candidates":
            summary[run_name]["align_mean"] = _safe_float(row.get("mean"), 0.0)
        elif metric == "elapsed_sec":
            summary[run_name]["elapsed_mean"] = _safe_float(row.get("mean"), 0.0)
    out = list(summary.values())
    out.sort(key=lambda x: x.get("verified_mean", -1.0), reverse=True)
    return out


@lru_cache(maxsize=4)
def load_paper_v2_bundle_cached() -> Dict[str, Any]:
    showcase_path = PAPER_TABLE_DIR / "showcase_data.json"
    fallback_ci = PAPER_TABLE_DIR / "tbl01_run_metric_ci_enhanced.csv"
    fallback_pairs = PAPER_TABLE_DIR / "tbl02_case_pairs_mainline_behavior.csv"
    fallback_all_cases = PAPER_TABLE_DIR / "tbl03_all_case_metrics_long.csv"
    fallback_selection = PAPER_TABLE_DIR / "chart_selection_matrix.csv"
    fallback_family = PAPER_TABLE_DIR / "d3og_family_counts.csv"
    figure_manifest = PAPER_TABLE_DIR / "tbl05_figure_manifest.csv"

    payload = _read_json_file(showcase_path, {})
    if not isinstance(payload, dict):
        payload = {}

    run_metric_ci = payload.get("run_metric_ci", [])
    case_pairs = payload.get("case_pairs", [])
    noise_curve = payload.get("noise_curve", [])
    reliability_bins = payload.get("reliability_bins", [])
    chart_selection = payload.get("chart_selection_matrix", [])
    family_counts = payload.get("d3og_family_counts", [])

    if not isinstance(run_metric_ci, list) or not run_metric_ci:
        run_metric_ci = _read_csv_rows(fallback_ci)
    else:
        run_metric_ci = _coerce_rows(run_metric_ci)

    if not isinstance(case_pairs, list) or not case_pairs:
        case_pairs = _read_csv_rows(fallback_pairs)
    else:
        case_pairs = _coerce_rows(case_pairs)

    if not isinstance(noise_curve, list):
        noise_curve = []
    else:
        noise_curve = _coerce_rows(noise_curve)

    if not isinstance(reliability_bins, list):
        reliability_bins = []
    else:
        reliability_bins = _coerce_rows(reliability_bins)

    if not isinstance(chart_selection, list) or not chart_selection:
        chart_selection = _read_csv_rows(fallback_selection)
    else:
        chart_selection = _coerce_rows(chart_selection)

    if not isinstance(family_counts, list) or not family_counts:
        family_counts = _read_csv_rows(fallback_family)
    else:
        family_counts = _coerce_rows(family_counts)

    all_case_metrics = _read_csv_rows(fallback_all_cases)

    for r in case_pairs:
        r["d_verified_p_match_mean"] = _safe_float(r.get("d_verified_p_match_mean"), 0.0)
        r["d_align_avg_candidates"] = _safe_float(r.get("d_align_avg_candidates"), 0.0)
        r["d_elapsed_sec"] = _safe_float(r.get("d_elapsed_sec"), 0.0)
        r["duration_sec"] = _safe_float(r.get("duration_sec"), 0.0)
        if "abs_d_verified_p_match_mean" not in r:
            r["abs_d_verified_p_match_mean"] = abs(_safe_float(r.get("d_verified_p_match_mean"), 0.0))
        else:
            r["abs_d_verified_p_match_mean"] = _safe_float(r.get("abs_d_verified_p_match_mean"), 0.0)

    for r in all_case_metrics:
        r["duration_sec"] = _safe_float(r.get("duration_sec"), 0.0)
        r["elapsed_sec"] = _safe_float(r.get("elapsed_sec"), 0.0)
        r["verified_p_match_mean"] = _safe_float(r.get("verified_p_match_mean"), 0.0)
        r["align_avg_candidates"] = _safe_float(r.get("align_avg_candidates"), 0.0)

    case_pairs_sorted = sorted(case_pairs, key=lambda x: _safe_float(x.get("abs_d_verified_p_match_mean"), 0.0), reverse=True)
    top_cases = case_pairs_sorted[:80]
    case_pair_index = {str(r.get("case_id", "")): r for r in case_pairs if str(r.get("case_id", ""))}

    by_case_runs: Dict[str, List[Dict[str, Any]]] = {}
    for r in all_case_metrics:
        key = str(r.get("case_id", ""))
        if not key:
            continue
        by_case_runs.setdefault(key, []).append(r)
    for key in by_case_runs:
        by_case_runs[key] = sorted(by_case_runs[key], key=lambda x: str(x.get("run_name", "")))

    selection_lookup = {str(r.get("chart_id", "")): r for r in chart_selection}
    figures: List[Dict[str, Any]] = []
    for row in _read_csv_rows(figure_manifest):
        fig_id = str(row.get("figure_id", ""))
        path_text = str(row.get("path", ""))
        fig_path = _resolve_existing_path(path_text)
        fig_url = _to_docs_url(fig_path) if fig_path else None
        sel = selection_lookup.get(fig_id, {})
        figures.append(
            {
                "figure_id": fig_id,
                "url": fig_url,
                "title": sel.get("analysis_task", fig_id),
                "d3_title": sel.get("d3_title"),
                "d3_url": sel.get("d3_url"),
            }
        )

    videos = _load_docs_videos()
    run_summary = _build_run_summary(run_metric_ci)

    return {
        "meta": {
            "generated_at": str(payload.get("meta", {}).get("generated_at", "")),
            "table_dir": str(PAPER_TABLE_DIR),
            "chart_dir": str(PAPER_CHART_DIR),
            "has_showcase_json": showcase_path.exists(),
            "has_case_pairs_csv": fallback_pairs.exists(),
            "has_run_metric_ci_csv": fallback_ci.exists(),
            "has_all_case_metrics_csv": fallback_all_cases.exists(),
        },
        "run_metric_ci": run_metric_ci,
        "run_summary": run_summary,
        "case_pairs": case_pairs,
        "top_cases": top_cases,
        "all_case_metrics": all_case_metrics,
        "noise_curve": noise_curve,
        "reliability_bins": reliability_bins,
        "chart_selection": chart_selection,
        "d3og_family_counts": family_counts,
        "figures": figures,
        "videos": videos,
        "case_pair_index": case_pair_index,
        "by_case_runs": by_case_runs,
    }


# =========================================================
# 6) 璺敱锛欻TML + API
# =========================================================
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    涓婚〉锛氶粯璁ゆ覆鏌?templates/index.html
    濡傛灉浣犳殏鏃舵病鍋氭ā鏉匡紝涔熶細鍥為€€鍒?PROJECT_ROOT/index.html
    """
    # 浼樺厛 templates
    index_tpl = Path(TEMPLATE_DIR) / "index.html"
    if index_tpl.exists():
        return templates.TemplateResponse("index.html", {"request": request})

    # 鍥為€€鍒版牴鐩綍 index.html
    index_file = PROJECT_ROOT / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))

    raise HTTPException(404, "index.html not found in templates/ or project root")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    candidates = [
        DOCS_DIR / "favicon.ico",
        PROJECT_ROOT / "favicon.ico",
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            return FileResponse(str(p))
    return Response(status_code=204)


@app.get("/paper/v2", response_class=HTMLResponse)
async def paper_v2_dashboard():
    page = DOCS_DIR / "paper_v2_dashboard.html"
    if page.exists():
        return FileResponse(str(page))
    raise HTTPException(404, "paper_v2_dashboard.html not found under docs/")


@app.get("/api/paper/v2/overview")
def get_paper_v2_overview():
    bundle = load_paper_v2_bundle_cached().copy()
    bundle.pop("case_pair_index", None)
    bundle.pop("by_case_runs", None)
    return bundle


@app.get("/api/paper/v2/case/{case_id}")
def get_paper_v2_case(case_id: str):
    bundle = load_paper_v2_bundle_cached()
    key = str(case_id).strip()
    case_row = bundle.get("case_pair_index", {}).get(key)
    if case_row is None:
        raise HTTPException(404, f"case not found in paper_v2 data: {case_id}")
    runs = bundle.get("by_case_runs", {}).get(key, [])
    media = None
    try:
        media = get_media(key)
    except Exception:
        media = None
    return {
        "case_id": key,
        "case_pair": case_row,
        "runs": runs,
        "media": media,
    }


@app.post("/api/paper/v2/refresh")
def refresh_paper_v2_cache():
    load_paper_v2_bundle_cached.cache_clear()
    return {
        "ok": True,
        "message": "paper_v2 cache cleared",
    }


@app.get("/api/list_cases")
def list_cases():
    """
    鑷姩鎵弿 output 鐩綍锛屽憡璇夊墠绔湁鍝簺 demo 鍙敤锛堝吋瀹逛綘 txt 鐨?Flask 鎺ュ彛锛?:contentReference[oaicite:6]{index=6}
    """
    def _listable_case_dir(d: Path) -> bool:
        low = d.as_posix().lower()
        if "/_tmp" in low or "/cache/" in low or "/__pycache__/" in low:
            return False
        return _looks_like_case_dir(d)

    # 閫掑綊鎵弿鈥滅湡姝ｇ殑 case 鐩綍鈥濓紝閬垮厤鎶娾€滄暀甯堣瑙?鏂滀笂鏂硅瑙?鈥濊繖绉嶈瑙掔埗鐩綍璇綋鎴?case銆?
    case_dirs = [d for d in OUTPUT_DIR.rglob("*") if d.is_dir() and _listable_case_dir(d)] if OUTPUT_DIR.exists() else []
    cases: List[Dict[str, Any]] = []
    for case_dir in case_dirs:
        rel_parts = case_dir.relative_to(OUTPUT_DIR).parts
        view_name = ""
        if len(rel_parts) >= 2:
            # 鏂扮粨鏋勯€氬父鏄?.../<view>/<video_id>
            view_name = rel_parts[-2]
        video_id = case_dir.name
        original_path = _resolve_original_video_path(video_id, case_dir)
        pipeline_status = _case_pipeline_status(case_dir)
        cases.append({
            "view": view_name,
            "video_id": video_id,
            "case_id": video_id.split("__")[-1],
            "path": str(case_dir),
            "has_original_video": bool(original_path is not None),
            "ready_for_frontend": _looks_like_case_dir(case_dir),
            **pipeline_status,
        })

    paper_case_dir = _find_paper_case_dir(PAPER_MAINLINE_CASE_ID)
    if paper_case_dir is not None:
        original_path = _resolve_original_video_path(PAPER_MAINLINE_CASE_ID, paper_case_dir)
        pipeline_status = _case_pipeline_status(paper_case_dir)
        cases.append({
            "view": "paper",
            "video_id": PAPER_MAINLINE_CASE_ID,
            "case_id": "001",
            "path": str(paper_case_dir),
            "has_original_video": bool(original_path is not None),
            "ready_for_frontend": True,
            "paper_package": True,
            "label": "Paper Mainline 20260426",
            **pipeline_status,
        })

    # 鍘婚噸锛堝悓鍚?video_id 鍙兘鍑虹幇鍦ㄤ笉鍚岀紦瀛樼洰褰曪級
    uniq: Dict[str, Dict[str, Any]] = {}
    for c in cases:
        key = c["video_id"]
        if key not in uniq:
            uniq[key] = c
            continue
        old = uniq[key]
        old_q = _case_dir_quality(Path(old["path"]))
        new_q = _case_dir_quality(Path(c["path"]))
        if new_q > old_q:
            uniq[key] = c
        elif new_q == old_q and len(str(c["path"])) < len(str(old["path"])):
            uniq[key] = c

    return sorted(
        list(uniq.values()),
        key=lambda x: (
            not bool(x.get("paper_package")),
            not bool(x.get("mainline_ready")),
            not bool(x.get("fusion_v2_ready")),
            str(x.get("view", "")) == "",
            str(x.get("view", "")),
            str(x.get("video_id", "")),
        ),
    )


@app.get("/api/analysis/list_cases")
def list_analysis_cases():
    cases = list_cases()
    out: List[Dict[str, Any]] = []
    for c in cases:
        video_id = str(c.get("video_id", ""))
        case_dir = _find_case_dir_for_analysis(video_id)
        if case_dir is None:
            continue
        bundle = _load_case_analysis_bundle(case_dir)
        bundle_path = case_dir / "analysis" / "bundle.json"
        proj_path = case_dir / "analysis" / "projection.json"
        out.append(
            {
                "video_id": video_id,
                "view": str(c.get("view", "")),
                "case_id": str(c.get("case_id", "")),
                "has_analysis_bundle": bundle is not None,
                "analysis_bundle": _to_output_url(bundle_path) if bundle_path.exists() else None,
                "projection": _to_output_url(proj_path) if proj_path.exists() else None,
                "num_slices": len(bundle.get("slices", [])) if bundle else 0,
            }
        )
    return out


@app.get("/api/analysis/case/{case_id}")
def get_analysis_case(case_id: str):
    case_dir = _find_case_dir_for_analysis(case_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    bundle = _load_case_analysis_bundle(case_dir)
    if bundle is None:
        raise HTTPException(404, f"Analysis bundle not found: {case_id}")
    return bundle


@app.get("/api/analysis/slices/{case_id}")
def get_analysis_slices(case_id: str):
    case_dir = _find_case_dir_for_analysis(case_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    bundle = _load_case_analysis_bundle(case_dir)
    if bundle is None:
        raise HTTPException(404, f"Analysis bundle not found: {case_id}")
    return {
        "video_id": case_dir.name,
        "view": case_dir.parent.name if case_dir.parent != OUTPUT_DIR else "",
        "meta": bundle.get("meta", {}),
        "video_meta": bundle.get("video_meta", {}),
        "slices": bundle.get("slices", []),
    }


@app.get("/api/media/{video_id}")
def get_media(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")

    view_name, case_id = _extract_case_tokens(video_id, case_dir)
    overlay_path = _resolve_overlay_video_path(video_id, case_dir)
    browser_overlay_path = _ensure_browser_friendly_video(overlay_path)
    overlay_url = _path_to_public_url(browser_overlay_path or overlay_path)
    browser_overlay_url = _path_to_public_url(browser_overlay_path)

    original_path = _resolve_original_video_path(video_id, case_dir)
    original_url = f"/api/media/{video_id}/original" if original_path and original_path.exists() else None
    browser_original_url = _path_to_public_url(_ensure_browser_friendly_video(original_path))

    pose_demo_path = None
    for candidate_name in ("pose_demo_yolo11x.mp4", "pose_demo_out.mp4"):
        candidate = case_dir / candidate_name
        if candidate.exists():
            pose_demo_path = candidate
            break
    browser_pose_demo_path = _ensure_browser_friendly_video(pose_demo_path)

    def _u(name: str) -> Optional[str]:
        return _path_to_public_url(case_dir / name)

    payload = {
        "video_id": video_id,
        "view": view_name,
        "case_id": case_id,
        "original": original_url,
        "browser_original": browser_original_url,
        "overlay": overlay_url,
        "browser_overlay": browser_overlay_url,
        "pose_demo": _path_to_public_url(browser_pose_demo_path or pose_demo_path) or _u("pose_demo_out.mp4"),
        "browser_pose_demo": _path_to_public_url(browser_pose_demo_path),
        "objects_demo": _u("objects_demo_out.mp4"),
        "timeline_png": _u("timeline_chart.png"),
        "projection_json": _u("student_projection.json"),
        "verified_events": _u("verified_events.jsonl"),
        "event_queries": _u("event_queries.fusion_v2.jsonl") or _u("event_queries.jsonl"),
        "actions": _u("actions.fusion_v2.jsonl") or _u("actions_fused.jsonl") or _u("actions.jsonl"),
    }
    return payload


@app.get("/api/media/{video_id}/original")
def get_media_original(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")
    p = _resolve_original_video_path(video_id, case_dir)
    if p is None or (not p.exists()):
        raise HTTPException(404, f"Original video not found: {video_id}")
    return FileResponse(str(p))


@app.api_route("/api/media/stream/{case_id}", methods=["GET", "HEAD"])
def get_media_stream(case_id: str, request: Request):
    """Serve the best playable video file with browser-compatible codec.

    Searches MEDIA_SEARCH_DIRS using VIDEO_CANDIDATE_NAMES priority.
    Supports HTTP Range requests for browser video seeking.
    Returns debug JSON with searched paths when no video is found.
    """
    path = _find_playable_video(case_id)
    if path is None or not path.exists():
        searched = _list_all_searched_paths(case_id)
        return JSONResponse(
            status_code=404,
            content={
                "case_id": case_id,
                "searched": searched,
                "message": "No playable media found",
            },
        )

    file_size = path.stat().st_size
    range_header = request.headers.get("range")

    # HEAD request — return headers only, no body
    if request.method == "HEAD":
        return Response(
            status_code=200,
            media_type="video/mp4",
            headers={
                "Content-Length": str(file_size),
                "Accept-Ranges": "bytes",
                "Cache-Control": "no-store",
            },
        )

    # ── Handle HTTP Range (byte serving) for browser seeking ──
    if range_header:
        parsed = _parse_range(range_header, file_size)
        if parsed is not None:
            start, end = parsed
            content_length = end - start + 1

            def _iter_file():
                with open(path, "rb") as f:
                    f.seek(start)
                    remaining = content_length
                    while remaining > 0:
                        chunk = f.read(min(65536, remaining))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                        yield chunk

            return StreamingResponse(
                _iter_file(),
                status_code=206,
                media_type="video/mp4",
                headers={
                    "Content-Range": f"bytes {start}-{end}/{file_size}",
                    "Content-Length": str(content_length),
                    "Accept-Ranges": "bytes",
                    "Cache-Control": "no-store",
                },
            )

    # ── Full-file response with range-advertisement header ──
    return FileResponse(
        str(path),
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "no-store",
        },
    )


@app.get("/api/debug/routes")
def debug_routes():
    """List all registered routes for debugging."""
    routes_list = []
    for r in app.routes:
        methods = getattr(r, "methods", None)
        methods_str = sorted(methods) if methods else None
        path = getattr(r, "path", str(r))
        routes_list.append({"path": path, "methods": methods_str})
    return {"routes": routes_list}


@app.get("/api/case/{video_id}/manifest")
def get_case_manifest(video_id: str):
    case_dir = _find_case_dir(video_id)
    if case_dir is None:
        raise HTTPException(404, f"Case not found: {video_id}")

    expected = [
        "pipeline_manifest.json",
        "pose_keypoints_v2.jsonl", "pose_tracks_smooth.jsonl", "pose_tracks_smooth_uq.jsonl",
        "objects.jsonl", "objects.semantic.jsonl",
        "behavior_det.jsonl", "behavior_det.semantic.jsonl",
        "actions.raw.jsonl", "actions.jsonl", "actions_fused.jsonl",
        "actions.behavior.jsonl", "actions.behavior.semantic.jsonl", "actions.fusion_v2.jsonl",
        "transcript.jsonl", "asr_quality_report.json",
        "event_queries.jsonl", "event_queries.visual_fallback.jsonl", "event_queries.fusion_v2.jsonl",
        "align_multimodal.json", "verified_events.jsonl",
        "fusion_contract_report.json", "pipeline_contract_v2_report.json",
        "verifier_eval_report.json", "verifier_calibration_report.json", "verifier_reliability_diagram.svg",
        "pose_demo_out.mp4", "objects_demo_out.mp4", "student_projection.json", "timeline_chart.json",
        "timeline_viz.json", "timeline_chart.png", "timeline_students.csv", "student_id_map.json",
        "group_events.jsonl", "per_person_sequences.json", "embeddings.pkl",
    ]

    def _artifact_path(name: str) -> Path:
        p = case_dir / name
        if p.exists():
            return p
        try:
            if case_dir.resolve() == YOLO_PAPER_DEMO_DIR.resolve():
                metrics_p = YOLO_PAPER_PACKAGE_DIR / "03_metrics_tables" / name
                if metrics_p.exists():
                    return metrics_p
        except Exception:
            pass
        return p

    files = []
    for name in expected:
        p = _artifact_path(name)
        files.append({
            "name": name,
            "exists": p.exists(),
            "size": int(p.stat().st_size) if p.exists() else 0,
            "url": _path_to_public_url(p) if p.exists() else None,
        })

    pipeline_status = _case_pipeline_status(case_dir)
    return {
        "schema_version": "frontend_manifest_v2",
        "video_id": video_id,
        "resolved_case_dir": case_dir.name,
        "view": "paper" if video_id in PAPER_MAINLINE_ALIASES else (case_dir.parent.name if case_dir.parent != OUTPUT_DIR else ""),
        "path": str(case_dir),
        "files": files,
        "pipeline_status": pipeline_status,
        **pipeline_status,
        "ready_for_frontend": any(
            (f["name"] == "timeline_chart.json" and f["exists"])
            or (f["name"] == "verified_events.jsonl" and f["exists"])
            or (f["name"] == "actions.fusion_v2.jsonl" and f["exists"])
            for f in files
        ),
    }


@app.get("/api/config")
def get_config():
    """
    鏆撮湶閮ㄥ垎閰嶇疆锛堝吋瀹逛綘 txt 鐨?Flask 鎺ュ彛锛?:contentReference[oaicite:7]{index=7}
    """
    return {
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(OUTPUT_DIR),
        "data_dir": str(DATA_DIR),
        "video_dir": str(VIDEO_DIR),
        "template_dir": str(TEMPLATE_DIR),
    }


@app.get("/api/timeline/{video_id}")
def get_timeline(video_id: str):
    data = load_timeline_data_cached(video_id)
    if not data:
        return {"items": []}
    # 缁熶竴杈撳嚭 {"items": [...]}
    if isinstance(data, list):
        return {"items": data}
    return {"items": data.get("items", [])}


@app.get("/api/stats/{video_id}")
def get_stats(video_id: str):
    return load_stats_data_cached(video_id)


@app.get("/api/transcript/{video_id}")
def get_transcript(video_id: str):
    raw = load_transcript_data_cached(video_id)
    verified_rows = load_verified_events_cached(video_id)

    def _overlap(a0: float, a1: float, b0: float, b1: float) -> float:
        return max(0.0, min(a1, b1) - max(a0, b0))

    out = []
    for l in raw:
        st = _safe_float(l.get("start"), 0.0)
        ed = _safe_float(l.get("end"), st)
        if ed < st:
            st, ed = ed, st

        linked = []
        for v in verified_rows:
            q_t = _safe_float(v.get("query_time"), 0.0)
            w_st = _safe_float(v.get("window_start", q_t), q_t)
            w_ed = _safe_float(v.get("window_end", q_t), q_t)
            if w_ed < w_st:
                w_st, w_ed = w_ed, w_st
            if _overlap(st, ed, w_st, w_ed) > 0 or (st <= q_t <= ed):
                linked.append(v)

        linked.sort(key=lambda x: _safe_float(x.get("reliability"), 0.0), reverse=True)
        top = linked[0] if linked else None
        label = str(top.get("label")) if top else ""

        out.append({
            "start": st,
            "end": ed,
            "text": l.get("text", l.get("sentence", "")),
            "verified": bool(label == "match" or l.get("verified", False)),
            "reliability": _safe_float(top.get("reliability"), 0.0) if top else 0.0,
            "verification_label": label if label else None,
            "event_id": top.get("event_id") if top else None,
            "event_type": top.get("event_type") if top else None,
        })
    return out


@app.get("/api/tracks/{video_id}")
def get_tracks(video_id: str):
    return load_tracks_data_cached(video_id)


@app.get("/api/projection/{video_id}")
def get_projection(video_id: str, method: str = "pca", metric: str = "euclidean"):
    method = method.lower().strip()
    metric = metric.lower().strip()
    if method not in {"pca", "mds", "tsne"}:
        raise HTTPException(400, "method must be one of: pca, mds, tsne")
    if metric not in {"euclidean", "levenshtein", "spatial"}:
        raise HTTPException(400, "metric must be one of: euclidean, levenshtein, spatial")
    return compute_projection_cached(video_id, method, metric)


# =========================================================
# 8) Frontend Bundle API
# =========================================================

BUNDLE_DIR = (OUTPUT_DIR / "frontend_bundle").resolve()


def _find_bundle(case_id: str):
    case_id = str(case_id or "").strip()
    if not case_id or not re.fullmatch(r"[A-Za-z0-9_.-]+", case_id):
        return None
    direct = BUNDLE_DIR / case_id
    if direct.exists() and (direct / "frontend_data_manifest.json").exists():
        return direct
    if BUNDLE_DIR.exists():
        for child in BUNDLE_DIR.iterdir():
            if not child.is_dir():
                continue
            if child.name == case_id and (child / "frontend_data_manifest.json").exists():
                return child
    return None


def _get_bundle_json(case_id: str, filename: str) -> Any:
    bundle = _find_bundle(case_id)
    if bundle is None:
        raise HTTPException(404, f"bundle not found: {case_id}")
    path = bundle / filename
    if not path.exists():
        raise HTTPException(404, f"{filename} not found")
    return _read_json_file(path, {})


@app.get("/api/bundle/list")
def list_bundles():
    items: List[Dict[str, Any]] = []
    if not BUNDLE_DIR.exists():
        return items
    for child in sorted(BUNDLE_DIR.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        manifest_path = child / "frontend_data_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = _read_json_file(manifest_path, {})
        items.append({
            "case_id": manifest.get("case_id", child.name),
            "students": manifest.get("tracked_students", 0),
            "contract_status": manifest.get("contract_status", "unknown"),
            "files": list(manifest.get("files", {}).keys()),
            "assets": list(manifest.get("assets", {}).keys()),
        })
    return items


@app.get("/api/bundle/{case_id}/manifest")
def get_bundle_manifest(case_id: str):
    return _get_bundle_json(case_id, "frontend_data_manifest.json")


@app.get("/api/bundle/{case_id}/timeline")
def get_bundle_timeline(case_id: str):
    return _get_bundle_json(case_id, "timeline_students.json")


@app.get("/api/bundle/{case_id}/tracks_sampled")
def get_bundle_tracks_sampled(case_id: str):
    return _get_bundle_json(case_id, "tracks_sampled.json")


@app.get("/api/bundle/{case_id}/metrics")
def get_bundle_metrics(case_id: str):
    return _get_bundle_json(case_id, "metrics_summary.json")


@app.get("/api/bundle/{case_id}/ablation")
def get_bundle_ablation(case_id: str):
    return _get_bundle_json(case_id, "ablation_summary.json")


@app.get("/api/bundle/{case_id}/behavior_segments")
def get_bundle_behavior_segments(case_id: str):
    return _get_bundle_json(case_id, "behavior_segments.json")


@app.get("/api/bundle/{case_id}/fusion_segments")
def get_bundle_fusion_segments(case_id: str):
    return _get_bundle_json(case_id, "fusion_segments.json")


@app.get("/api/bundle/{case_id}/student_id_map")
def get_bundle_student_id_map(case_id: str):
    return _get_bundle_json(case_id, "student_id_map.json")


@app.get("/api/bundle/{case_id}/failure_cases")
def get_bundle_failure_cases(case_id: str):
    return _get_bundle_json(case_id, "failure_cases.json")


@app.get("/api/bundle/{case_id}/verified")
def get_bundle_verified(case_id: str):
    return _get_bundle_json(case_id, "verified_events.json")


# =========================================================
# BFF Aggregation Endpoint: Unified View Model for D3
# =========================================================

def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
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
    return rows


def _json_list(payload: Any, *keys: str) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [x for x in payload if isinstance(x, dict)]
    if isinstance(payload, dict):
        for key in keys:
            rows = payload.get(key)
            if isinstance(rows, list):
                return [x for x in rows if isinstance(x, dict)]
    return []


def _interval_overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    inter = min(a1, b1) - max(a0, b0)
    if inter <= 0:
        return 0.0
    denom = max(1e-6, min(a1 - a0, b1 - b0))
    return inter / denom


def _first_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None:
            return value
    return default


def _safe_metric(value: Any, default: float = 0.0) -> float:
    out = _safe_float(value, default)
    if out != out:
        return default
    return out


def _event_window(row: Dict[str, Any]) -> Tuple[float, float]:
    window = row.get("window") if isinstance(row.get("window"), dict) else {}
    center = _safe_metric(_first_value(row.get("query_time"), row.get("timestamp"), row.get("t_center")), 0.0)
    start = _safe_metric(_first_value(window.get("start"), row.get("window_start"), row.get("start")), center - 1.5)
    end = _safe_metric(_first_value(window.get("end"), row.get("window_end"), row.get("end")), center + 1.5)
    if end < start:
        start, end = end, start
    return start, end


def _event_key(row: Dict[str, Any]) -> str:
    return str(row.get("event_id") or row.get("query_id") or "").strip()


def _load_bundle_queries(bundle: Optional[Path], manifest: Dict[str, Any], cid: str) -> List[Dict[str, Any]]:
    candidates: List[Path] = []
    if bundle is not None:
        candidates.extend([
            bundle / "event_queries.json",
            bundle / "event_queries.jsonl",
            bundle / "event_queries.fusion_v2.jsonl",
        ])

    source_files = manifest.get("source_files") if isinstance(manifest.get("source_files"), dict) else {}
    for value in source_files.values():
        raw = str(value or "")
        if "event_queries" in raw:
            candidates.append(Path(raw))

    source_case_dir = str(manifest.get("source_case_dir") or "").strip()
    if source_case_dir:
        src = Path(source_case_dir)
        candidates.extend([
            src / "event_queries.fusion_v2.jsonl",
            src / "event_queries.jsonl",
            src / "event_queries.visual_fallback.jsonl",
        ])

    raw_case = _find_case_dir(cid)
    if raw_case is not None:
        candidates.extend([
            raw_case / "event_queries.fusion_v2.jsonl",
            raw_case / "event_queries.jsonl",
            raw_case / "event_queries.visual_fallback.jsonl",
        ])

    seen: set[str] = set()
    rows: List[Dict[str, Any]] = []
    for path in candidates:
        key = str(path)
        if key in seen or not path.exists():
            continue
        seen.add(key)
        if path.suffix.lower() == ".jsonl":
            rows.extend(_read_jsonl(path))
        else:
            rows.extend(_json_list(_read_json_file(path, []), "events", "queries", "items"))
    return rows


def _load_case_bundle_payloads(cid: str) -> Tuple[Optional[Path], Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    bundle = _find_bundle(cid)
    manifest: Dict[str, Any] = {}
    segments: List[Dict[str, Any]] = []
    verified_events: List[Dict[str, Any]] = []

    if bundle is not None:
        manifest = _read_json_file(bundle / "frontend_data_manifest.json", {})
        timeline = _read_json_file(bundle / "timeline_students.json", {})
        segments = _json_list(timeline, "segments", "items")
        verified = _read_json_file(bundle / "verified_events.json", {})
        verified_events = _json_list(verified, "events", "items")

    if not segments or not verified_events:
        raw_case = _find_case_dir(cid)
        if raw_case is None and bundle is None:
            raise HTTPException(404, f"Case {cid} not found in bundle or raw outputs")
        if raw_case is not None:
            if not segments:
                timeline_path = raw_case / "timeline_chart.json"
                if not timeline_path.exists():
                    timeline_path = raw_case / "timeline_students.json"
                if not timeline_path.exists():
                    timeline_path = raw_case / "timeline_students.csv"
                if timeline_path.suffix.lower() == ".csv":
                    segments = _coerce_rows(_read_csv_rows(timeline_path))
                else:
                    segments = _json_list(_read_json_file(timeline_path, []), "segments", "items")
            if not verified_events:
                verified_events = _read_jsonl(raw_case / "verified_events.jsonl")

    if not segments:
        raise HTTPException(404, f"No timeline segments found for {cid}")
    return bundle, manifest, segments, verified_events


def _bff_action_id(row: Dict[str, Any], behavior: str) -> int:
    code_map = {
        "tt": 0,  # listen
        "dk": 8,  # read
        "js": 6,  # raise hand
        "jt": 4,  # chat/discussion
        "wt": 1,  # distraction
        "sj": 2,  # phone
        "dt": 3,  # doze/head down
    }
    for key in ("semantic_id", "semantic_label_en", "behavior_type", "action", "raw_action", "action_label"):
        action_id = _parse_action_id(row.get(key), row.get("action_id", -1))
        if action_id >= 0:
            return action_id
    norm = _normalize_action_name(behavior)
    if "raise" in norm:
        return 6
    if "read" in norm:
        return 8
    if "listen" in norm:
        return 0
    code = str(row.get("behavior_code") or "").strip().lower()
    return code_map.get(code, _parse_action_id(behavior, row.get("action_id", -1)))


@app.get("/api/v1/visualization/case_data")
async def get_visualization_case_data(
    case_id: str = Query(..., description="Case id, for example front_45618_sliced"),
    strict: bool = Query(False, description="Surface parse errors as warnings instead of swallowing them"),
):
    """
    BFF aggregation endpoint for the D3 frontend.
    The server joins timeline segments, verified events, and ASR/LLM queries into
    frontend-ready view models. Bundle files are preferred; raw case outputs are
    used as fallbacks through frontend_data_manifest.source_case_dir.
    """
    cid = str(case_id or "").strip()
    if not cid:
        raise HTTPException(400, "case_id is required")

    from collections import defaultdict

    bundle, manifest, segments, verified_events = _load_case_bundle_payloads(cid)
    query_rows = _load_bundle_queries(bundle, manifest, cid)
    query_map: Dict[str, Dict[str, Any]] = {}
    for q in query_rows:
        key = _event_key(q)
        if key:
            query_map[key] = q

    ve_index: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    ve_by_key: Dict[str, Dict[str, Any]] = {}
    for ve in verified_events:
        key = _event_key(ve)
        if key:
            ve_by_key[key] = ve
        tid = ve.get("track_id")
        if tid is None:
            continue
        try:
            tid_int = int(tid)
        except Exception:
            continue
        ws, we = _event_window(ve)
        ve_index[tid_int].append({**ve, "_ws": ws, "_we": we})

    view_events: List[Dict[str, Any]] = []
    timeline_segments: List[Dict[str, Any]] = []
    track_ids: set = set()
    max_time = 0.0

    for seg in segments:
        tid_raw = _first_value(seg.get("track_id"), seg.get("student_id"))
        try:
            tid = int(str(tid_raw).replace("S", "").replace("_", ""))
        except Exception:
            continue
        track_ids.add(tid)

        seg_start = _safe_metric(_first_value(seg.get("start_time"), seg.get("start"), seg.get("window_start"), seg.get("t")), 0.0)
        seg_end = _safe_metric(_first_value(seg.get("end_time"), seg.get("end"), seg.get("window_end")), seg_start + 0.2)
        if seg_end < seg_start:
            seg_start, seg_end = seg_end, seg_start
        max_time = max(max_time, seg_end)

        behavior = str(_first_value(
            seg.get("semantic_id"),
            seg.get("semantic_label_en"),
            seg.get("behavior_type"),
            seg.get("behavior_code"),
            seg.get("action"),
            default="unknown",
        ))
        action_id = _bff_action_id(seg, behavior)

        best_ve = ve_by_key.get(str(seg.get("event_id") or ""))
        best_ov = 0.0
        if not best_ve:
            for ve in ve_index.get(tid, []):
                ov = _interval_overlap(seg_start, seg_end, _safe_metric(ve.get("_ws"), 0.0), _safe_metric(ve.get("_we"), 0.0))
                center = _safe_metric(_first_value(ve.get("query_time"), ve.get("timestamp"), ve.get("t_center")), 0.0)
                if seg_start <= center <= seg_end:
                    ov = max(ov, 1e-3)
                if ov > best_ov:
                    best_ov = ov
                    best_ve = ve
        best_ve = best_ve or {}

        eid = _event_key(best_ve) or str(seg.get("event_id") or f"seg_{tid}_{int(seg_start * 1000)}")
        q_data = query_map.get(eid, {})
        evidence = best_ve.get("evidence") if isinstance(best_ve.get("evidence"), dict) else {}

        asr_text = str(_first_value(
            q_data.get("asr_text"),
            q_data.get("query_text"),
            q_data.get("trigger_text"),
            best_ve.get("asr_text"),
            best_ve.get("query_text"),
            default="",
        ))
        is_visual_fallback = asr_text.startswith("visual_fallback")
        is_vision_only = not asr_text or is_visual_fallback

        c_visual = _safe_metric(_first_value(
            evidence.get("c_visual"), evidence.get("cv"), evidence.get("visual_score"), seg.get("confidence")
        ), 0.8)
        c_text = _safe_metric(_first_value(
            evidence.get("c_text"), evidence.get("ct"), evidence.get("text_score")
        ), 0.0 if is_vision_only else 0.5)
        uq_track = _safe_metric(_first_value(
            evidence.get("uq_track"), evidence.get("uq"), evidence.get("uq_score"), best_ve.get("uncertainty")
        ), 0.1)

        wv_raw = _first_value(evidence.get("weight_v"), evidence.get("wv"), evidence.get("w_visual"))
        wa_raw = _first_value(evidence.get("weight_a"), evidence.get("wa"), evidence.get("w_audio"))
        weight_v = _safe_metric(wv_raw, 1.0 if is_vision_only else 0.5)
        weight_a = _safe_metric(wa_raw, 0.0 if is_vision_only else 0.5)
        weight_sum = weight_v + weight_a
        if weight_sum > 1e-9:
            weight_v, weight_a = weight_v / weight_sum, weight_a / weight_sum
        else:
            weight_v, weight_a = 1.0, 0.0

        reliability = _safe_metric(_first_value(
            evidence.get("reliability"),
            evidence.get("r"),
            best_ve.get("reliability_score"),
            best_ve.get("reliability"),
        ), max(0.0, min(1.0, weight_v * c_visual + weight_a * c_text - 0.1 * uq_track)))
        label_status = str(_first_value(best_ve.get("match_label"), best_ve.get("label"), default="unverified"))
        if label_status not in {"match", "mismatch", "uncertain", "unverified"}:
            label_status = "uncertain"

        metrics = {
            "c_visual": round(c_visual, 4),
            "c_text": round(0.0 if is_vision_only else c_text, 4),
            "uq_track": round(uq_track, 4),
            "weight_v": round(weight_v, 4),
            "weight_a": round(0.0 if is_vision_only else weight_a, 4),
            "reliability_final": round(reliability, 4),
        }
        semantics = {
            "asr_text": "" if is_visual_fallback else asr_text,
            "instruction_type": str(_first_value(q_data.get("instruction_type"), q_data.get("event_type"), best_ve.get("event_type"), default="none")),
        }

        event_model = {
            "event_id": eid,
            "track_id": f"S_{tid:02d}",
            "time_range": [round(seg_start, 2), round(seg_end, 2)],
            "behavior_type": behavior,
            "verification_status": label_status,
            "semantics": semantics,
            "evidence_metrics": metrics,
        }
        view_events.append(event_model)

        timeline_segments.append({
            **seg,
            "event_id": eid,
            "track_id": tid,
            "student_id": seg.get("student_id") or f"S{tid:02d}",
            "start": round(seg_start, 4),
            "end": round(seg_end, 4),
            "start_time": round(seg_start, 4),
            "end_time": round(seg_end, 4),
            "action_id": action_id,
            "action": behavior,
            "behavior_type": behavior,
            "verification_status": label_status,
            "semantics": semantics,
            "evidence_metrics": metrics,
            "reliability_score": metrics["reliability_final"],
        })

    normalized_verified: List[Dict[str, Any]] = []
    for ve in verified_events:
        item = {**ve}
        ws, we = _event_window(ve)
        item["window_start"] = round(ws, 4)
        item["window_end"] = round(we, 4)
        item["query_time"] = round(_safe_metric(_first_value(ve.get("query_time"), ve.get("timestamp"), ve.get("t_center")), (ws + we) / 2), 4)
        if "reliability" not in item and "reliability_score" in item:
            item["reliability"] = item.get("reliability_score")
        normalized_verified.append(item)

    view_events.sort(key=lambda e: (e["time_range"][0], e["track_id"]))
    timeline_segments.sort(key=lambda e: (_safe_metric(e.get("track_id"), 0.0), _safe_metric(e.get("start"), 0.0)))

    source_case_dir = str(manifest.get("source_case_dir") or "") if isinstance(manifest, dict) else ""

    return {
        "status": "success",
        "data": {
            "case_info": {
                "case_id": cid,
                "bundle_case": bundle is not None,
                "bundle_path": str(bundle) if bundle is not None else "",
                "source_case_dir": source_case_dir,
                "duration_sec": round(max_time, 1),
                "student_count": len(track_ids),
                "timeline_event_count": len(timeline_segments),
                "verified_event_count": len(normalized_verified),
                "query_event_count": len(query_map),
            },
            "events": view_events,
            "timeline_segments": timeline_segments,
            "verified_events": normalized_verified,
            "event_queries": query_rows,
        }
    }


@app.get("/paper/bundle/{case_id}")
@app.get("/paper/bundle/{case_id}/")
async def paper_demo_case_page(case_id: str, request: Request):
    bundle = _find_bundle(case_id)
    if bundle is None:
        raise HTTPException(404, f"bundle not found: {case_id}")
    manifest = _read_json_file(bundle / "frontend_data_manifest.json", {})
    return templates.TemplateResponse("paper_demo.html", {
        "request": request,
        "case_id": case_id,
        "bundle_path": str(bundle),
        "manifest": manifest,
    })


# =========================================================
# 9) V2 VSumVis APIs
#    Canonical routes: /api/v2/vsumvis/*
#    Compatibility routes: /api/v2/front/*
# =========================================================

FRONT_REPORTS_DIR = (OUTPUT_DIR / "codex_reports").resolve()
VALID_CASE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")
FRONT_DIR_PREFIX = "front_"
VSUMVIS_VALID_FILES = {
    "timeline_chart.json",
    "timeline_viz.json",
    "timeline_students.csv",
    "timeline_students.json",
    "verified_events.jsonl",
    "verified_events.json",
    "actions.fusion_v2.jsonl",
    "frontend_data_manifest.json",
    "sr_ablation_metrics.json",
}
VSUMVIS_EXCLUDED_NAMES = {"cache", "__pycache__", ".git", ".ultralytics"}
VSUMVIS_EXCLUDED_PREFIXES = ("_tmp",)


def _safe_child_dirs(root: Path) -> List[Path]:
    try:
        return [p for p in sorted(root.iterdir()) if p.is_dir()]
    except Exception:
        return []


def _is_vsumvis_excluded_dir(path: Path) -> bool:
    parts = [p.lower() for p in path.parts]
    if any(p in VSUMVIS_EXCLUDED_NAMES for p in parts):
        return True
    name = path.name.lower()
    return any(name.startswith(prefix) for prefix in VSUMVIS_EXCLUDED_PREFIXES)


def _is_vsumvis_case_dir(case_dir: Path) -> bool:
    if _is_vsumvis_excluded_dir(case_dir):
        return False
    try:
        if not case_dir.exists() or not case_dir.is_dir():
            return False
    except Exception:
        return False
    for name in VSUMVIS_VALID_FILES:
        try:
            if _is_nonempty_file(case_dir / name, min_bytes=1):
                return True
        except Exception:
            continue
    return False


_VSUMVIS_SKIP_SUBDIR_PATTERNS: tuple = (
    # sr_ablation variant subdirectories (A0, A1, A2, ...)
    re.compile(r"^A\d+_", re.IGNORECASE),
    # specific internal subdirs
    re.compile(r"^rear_row_failure_cases$", re.IGNORECASE),
    re.compile(r"^failures$", re.IGNORECASE),
    re.compile(r"^logs$", re.IGNORECASE),
)


def _is_vsumvis_skip_subdir(path: Path) -> bool:
    name = path.name
    for pat in _VSUMVIS_SKIP_SUBDIR_PATTERNS:
        if pat.match(name):
            return True
    return False


def _iter_vsumvis_dirs() -> List[Path]:
    """Return valid VSumVis data directories from output, codex reports, and bundles."""
    roots: List[Path] = []
    if CASE_FIXTURE_ROOT is not None and CASE_FIXTURE_ROOT.exists():
        roots.append(CASE_FIXTURE_ROOT)
    if FRONT_REPORTS_DIR.exists():
        roots.append(FRONT_REPORTS_DIR)
    if BUNDLE_DIR.exists():
        roots.append(BUNDLE_DIR)

    candidates: List[Path] = []
    seen: set[str] = set()

    def _push(path: Path) -> None:
        try:
            resolved = str(path.resolve())
        except Exception:
            resolved = str(path)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(path)

    for root in roots:
        for child in _safe_child_dirs(root):
            if _is_vsumvis_excluded_dir(child):
                continue
            if _is_vsumvis_skip_subdir(child):
                continue
            _push(child)
            # Look one level deeper for nested case dirs inside run/ directories.
            # Only recurse into directories that look like "run_*" containers.
            if child.name.lower().startswith("run_"):
                for grandchild in _safe_child_dirs(child):
                    if not _is_vsumvis_excluded_dir(grandchild) and not _is_vsumvis_skip_subdir(grandchild):
                        _push(grandchild)

    # Also scan top-level output/ for standalone case directories.
    if OUTPUT_DIR.exists():
        for child in _safe_child_dirs(OUTPUT_DIR):
            if child.name in {"codex_reports", "frontend_bundle", "docs", "vendor"}:
                continue
            if _is_vsumvis_excluded_dir(child):
                continue
            if _is_vsumvis_skip_subdir(child):
                continue
            _push(child)

    out: List[Path] = []
    for candidate in candidates:
        if _is_vsumvis_case_dir(candidate):
            out.append(candidate)

    # Deduplicate by public case id: keep the richest copy.
    by_name: Dict[str, List[Path]] = {}
    for p in out:
        by_name.setdefault(_public_case_id(p), []).append(p)
    deduped: List[Path] = []
    for paths in by_name.values():
        if len(paths) == 1:
            deduped.append(paths[0])
        else:
            # Keep the path with the highest score (most artifacts, largest files).
            paths.sort(key=lambda p: _case_dir_quality(p), reverse=True)
            deduped.append(paths[0])

    return sorted(deduped, key=_vsumvis_case_sort_key)


def _iter_front_dirs() -> List[Path]:
    """Compatibility wrapper for the old front route family."""
    return _iter_vsumvis_dirs()


def _public_case_id(case_dir: Path) -> str:
    try:
        if case_dir.resolve() == PAPER_MAINLINE_OUTPUT_DIR.resolve():
            return "run_full_paper_mainline_001"
    except Exception:
        pass
    if case_dir.parent.name in PAPER_MAINLINE_ALIASES:
        return "run_full_paper_mainline_001"
    manifest_path = case_dir / "frontend_data_manifest.json"
    if manifest_path.exists():
        manifest = _read_json_file(manifest_path, {})
        if isinstance(manifest, dict):
            manifest_case_id = str(manifest.get("case_id") or "").strip()
            if manifest_case_id:
                return manifest_case_id
    return case_dir.name


def _find_front_case_dir(case_id: str) -> Optional[Path]:
    if not VALID_CASE_ID_RE.fullmatch(case_id):
        return None
    canonical_fixture_aliases = {
        "front_45618_full": OUTPUT_DIR / "0034",
        "front_45618_sliced": OUTPUT_DIR / "0034",
        "run_full_paper_mainline_001": OUTPUT_DIR / "0034",
    }
    if CASE_FIXTURE_ROOT is not None and CASE_FIXTURE_ROOT.exists():
        fixture_case = CASE_FIXTURE_ROOT / "front_45618_full"
        canonical_fixture_aliases.update({
            "front_45618_full": fixture_case,
            "front_45618_sliced": fixture_case,
            "run_full_paper_mainline_001": fixture_case,
        })
    alias_target = canonical_fixture_aliases.get(case_id)
    if alias_target is not None and _is_vsumvis_case_dir(alias_target):
        return alias_target
    if case_id in PAPER_MAINLINE_ALIASES and PAPER_MAINLINE_OUTPUT_DIR.exists():
        return PAPER_MAINLINE_OUTPUT_DIR
    for child in _iter_vsumvis_dirs():
        if child.name == case_id or _public_case_id(child) == case_id:
            return child
    return None


def _front_url(path: Path) -> str:
    resolved = path.resolve()
    try:
        rel = resolved.relative_to(OUTPUT_DIR.resolve())
        return f"/output/{rel.as_posix()}"
    except Exception:
        pass
    try:
        rel = resolved.relative_to(PROJECT_ROOT.resolve())
        return f"/{rel.as_posix()}"
    except Exception:
        return resolved.as_posix()


def _vsumvis_data_source(case_dir: Path) -> str:
    if CASE_FIXTURE_ROOT is not None:
        try:
            case_dir.resolve().relative_to(CASE_FIXTURE_ROOT.resolve())
            return "test_fixture"
        except Exception:
            pass
    try:
        case_dir.resolve().relative_to(BUNDLE_DIR.resolve())
        return "frontend_bundle"
    except Exception:
        pass
    try:
        case_dir.resolve().relative_to(FRONT_REPORTS_DIR.resolve())
        return "codex_reports"
    except Exception:
        pass
    return "raw_output"


def _vsumvis_case_sort_key(case_dir: Path) -> Tuple[int, int, int, int, int, str]:
    name = case_dir.name
    source = _vsumvis_data_source(case_dir)
    source_rank = 0 if source == "codex_reports" else (1 if source == "frontend_bundle" else 2)
    front_rank = 0 if name.startswith(FRONT_DIR_PREFIX) else 1
    kind_rank = 1 if "sr_ablation" in name.lower() else 0
    verified_rank = 0 if _is_nonempty_file(case_dir / "verified_events.jsonl") or _is_nonempty_file(case_dir / "verified_events.json") else 1
    rich_rank = 0 if (
        _is_nonempty_file(case_dir / "pose_behavior_fusion_yolo11x.mp4")
        or _is_nonempty_file(case_dir / "frontend_data_manifest.json")
        or _is_nonempty_file(case_dir / "pipeline_contract_v2_report.json")
    ) else 1
    return (front_rank, kind_rank, source_rank, verified_rank, rich_rank, name)


def _front_case_kind(case_dir: Path) -> str:
    name = _public_case_id(case_dir).lower()
    if "sr_ablation" in name:
        return "sr_ablation"
    if "_sliced" in case_dir.name.lower() or "_sliced" in name or _vsumvis_data_source(case_dir) == "frontend_bundle":
        return "sliced"
    if case_dir.parent.name in PAPER_MAINLINE_ALIASES or name in PAPER_MAINLINE_ALIASES:
        return "paper_mainline"
    return "full"


def _front_video_stem(case_dir: Path) -> str:
    """Extract video stem from case dir name e.g. front_45618_full -> 45618."""
    name = case_dir.name
    if name.startswith("front_"):
        rest = name[len("front_"):]
        for suffix in ("_sr_ablation", "_full", "_full_pose020_hybrid", "_rear_row_sliced_pose020_hybrid"):
            if rest.endswith(suffix):
                return rest[: -len(suffix)]
        return rest
    return name


def _front_count_students(case_dir: Path) -> int:
    bundle_manifest = _read_json_file(case_dir / "frontend_data_manifest.json", {})
    if isinstance(bundle_manifest, dict) and bundle_manifest:
        tracked = _safe_float(bundle_manifest.get("tracked_students"), 0.0)
        if tracked > 0:
            return int(tracked)
        students = bundle_manifest.get("students")
        if isinstance(students, list) and students:
            return len(students)
    # try student_id_map.json first
    sid_map = _read_json_file(case_dir / "student_id_map.json", {})
    if isinstance(sid_map, dict):
        students = sid_map.get("students")
        if isinstance(students, list) and students:
            return len(students)
        sc = sid_map.get("student_count")
        if isinstance(sc, (int, float)) and sc > 0:
            return int(sc)
        if sid_map and "student_count" not in sid_map and "students" not in sid_map:
            return len(sid_map)
    # try counting unique track_ids from timeline_chart.json
    tl = _read_json_file(case_dir / "timeline_chart.json", {})
    items = _json_list(tl, "items")
    if items:
        tids = {i.get("track_id") for i in items if isinstance(i, dict)}
        if tids:
            return len(tids)
    # try CSV
    csv_rows = _read_csv_rows(case_dir / "timeline_students.csv")
    if csv_rows:
        tids = set()
        for r in csv_rows:
            for k in ("track_id", "student_id"):
                v = r.get(k)
                if v is not None:
                    tids.add(str(v))
        return len(tids)
    # for sr_ablation: try to read from sr_ablation_metrics.json best variant
    sr_data = _read_json_file(case_dir / "sr_ablation_metrics.json", {})
    if isinstance(sr_data, dict):
        variants = sr_data.get("variants", [])
        if isinstance(variants, list):
            for v in variants:
                sc = v.get("tracked_students")
                if sc is not None and int(sc) > 0:
                    return int(sc)
    return 0


def _front_count_events(case_dir: Path) -> tuple:
    """Return (verified_count, timeline_event_count, query_count)."""
    ve_count = 0
    ve_path = case_dir / "verified_events.jsonl"
    if ve_path.exists():
        with open(ve_path, "r", encoding="utf-8") as f:
            ve_count = sum(1 for line in f if line.strip())
    elif (case_dir / "verified_events.json").exists():
        verified = _read_json_file(case_dir / "verified_events.json", {})
        ve_count = len(_json_list(verified, "events", "items"))
    tl = _read_json_file(case_dir / "timeline_chart.json", {})
    tl_items = _json_list(tl, "items")
    tl_count = len(tl_items)
    if tl_count == 0:
        bundle_tl = _read_json_file(case_dir / "timeline_students.json", {})
        tl_count = len(_json_list(bundle_tl, "segments", "items"))
    eq_count = 0
    eq_path = case_dir / "event_queries.fusion_v2.jsonl"
    if eq_path.exists():
        with open(eq_path, "r", encoding="utf-8") as f:
            eq_count = sum(1 for line in f if line.strip())
    elif (case_dir / "event_queries.json").exists():
        queries = _read_json_file(case_dir / "event_queries.json", {})
        eq_count = len(_json_list(queries, "events", "queries", "items"))
    return ve_count, tl_count, eq_count


def _front_duration_sec(case_dir: Path) -> float:
    raw_timeline = _read_timeline_segments_front(case_dir)
    items = _json_list(raw_timeline, "items") if isinstance(raw_timeline, dict) else raw_timeline
    max_end = 0.0
    for it in items:
        ed = _safe_float(it.get("end") or it.get("end_time"), 0.0)
        if ed > max_end:
            max_end = ed
    if max_end > 0:
        return round(max_end, 1)
    # fallback: manifest
    manifest = _read_json_file(case_dir / "pipeline_manifest.json", {})
    if isinstance(manifest, dict):
        dur = _safe_float(manifest.get("duration_sec") or manifest.get("duration"), 0.0)
        if dur > 0:
            return round(dur, 1)
    # fallback for sr_ablation: use stage_runtime_sec from variant data
    sr_data = _read_json_file(case_dir / "sr_ablation_metrics.json", {})
    if isinstance(sr_data, dict):
        variants = sr_data.get("variants", [])
        if isinstance(variants, list):
            for v in variants:
                dur = _safe_float(v.get("stage_runtime_sec"), 0.0)
                if dur > 0:
                    return round(dur, 1)
            # try v["pose_person_rows"] / v["effective_fps"]
            for v in variants:
                rows = _safe_float(v.get("pose_person_rows"), 0.0)
                fps = _safe_float(v.get("effective_fps"), 0.0)
                if rows > 0 and fps > 0:
                    return round(rows / fps, 1)
    return 0.0


def _front_sparkline(case_dir: Path) -> List[Dict[str, Any]]:
    """Generate a lightweight sparkline summary from timeline segments."""
    items = _read_timeline_segments_front(case_dir)
    if not items:
        sr_data = _read_json_file(case_dir / "sr_ablation_metrics.json", {})
        variants = sr_data.get("variants", []) if isinstance(sr_data, dict) else []
        if isinstance(variants, list) and variants:
            return [
                {"t": i, "count": _safe_float(v.get("tracked_students") or v.get("rear_pose_person_rows_proxy"), 0.0)}
                for i, v in enumerate(variants[:30])
            ]
        return []
    duration = _front_duration_sec(case_dir)
    if duration <= 0:
        duration = max(_safe_float(i.get("end") or i.get("end_time"), 0.0) for i in items)
    bins = 30
    bin_w = max(duration / bins, 0.5)
    if bin_w <= 0:
        return []
    histogram = [0] * bins
    for it in items:
        st = _safe_float(it.get("start") or it.get("start_time"), 0.0)
        ed = _safe_float(it.get("end") or it.get("end_time"), st)
        bi = int(st / bin_w)
        if 0 <= bi < bins:
            histogram[bi] += 1
    return [{"t": round(i * bin_w, 1), "count": int(c)} for i, c in enumerate(histogram)]


def _front_contract_status(case_dir: Path) -> str:
    report = _read_json_file(case_dir / "pipeline_contract_v2_report.json", {})
    if not report:
        report = _read_json_file(case_dir / "fusion_contract_report.json", {})
    status = str(report.get("status", "")).lower() if isinstance(report, dict) else ""
    if status in ("ok", "pass", "passed"):
        return "OK"
    if status == "failed":
        return "FAIL"
    if _is_nonempty_file(case_dir / "pipeline_contract_v2_report.json", min_bytes=8):
        return "OK"
    if _is_nonempty_file(case_dir / "fusion_contract_report.json", min_bytes=8):
        return "OK"
    return "WARN"


def _front_file_status(case_dir: Path) -> Dict[str, bool]:
    return {
        "frontend_data_manifest": _is_nonempty_file(case_dir / "frontend_data_manifest.json"),
        "timeline_chart_json": _is_nonempty_file(case_dir / "timeline_chart.json"),
        "timeline_viz_json": _is_nonempty_file(case_dir / "timeline_viz.json"),
        "timeline_students_json": _is_nonempty_file(case_dir / "timeline_students.json"),
        "timeline_students_csv": _is_nonempty_file(case_dir / "timeline_students.csv"),
        "verified_events_jsonl": _is_nonempty_file(case_dir / "verified_events.jsonl"),
        "verified_events_json": _is_nonempty_file(case_dir / "verified_events.json"),
        "event_queries_jsonl": _is_nonempty_file(case_dir / "event_queries.fusion_v2.jsonl"),
        "transcript_jsonl": _is_nonempty_file(case_dir / "transcript.jsonl"),
        "pipeline_manifest": _is_nonempty_file(case_dir / "pipeline_manifest.json"),
        "student_id_map_json": _is_nonempty_file(case_dir / "student_id_map.json"),
        "pose_tracks_uq_jsonl": _is_nonempty_file(case_dir / "pose_tracks_smooth_uq.jsonl"),
        "sr_ablation_json": _is_nonempty_file(case_dir / "sr_ablation_metrics.json"),
    }


def _front_assets(case_dir: Path) -> Dict[str, Optional[str]]:
    def _url(name: str) -> Optional[str]:
        p = case_dir / name
        if not p.exists():
            return None
        if p.suffix.lower() == ".mp4" or p.name.lower().endswith(".web.mp4"):
            p = _ensure_browser_friendly_video(p) or p
        return _front_url(p)
    assets = {
        "overlay_video": _url("pose_behavior_fusion_yolo11x.mp4"),
        "pose_demo_video": _url("pose_demo_yolo11x.mp4"),
        "preview_image": _url("pose_behavior_fusion_yolo11x_preview.jpg"),
        "timeline_png": _url("timeline_chart.png"),
        "reliability_diagram": _url("verifier_reliability_diagram.svg"),
        "contact_sheet": _url("sr_ablation_contact_sheet.jpg"),
    }
    manifest = _read_json_file(case_dir / "frontend_data_manifest.json", {})
    manifest_assets = manifest.get("assets") if isinstance(manifest, dict) else None
    if isinstance(manifest_assets, dict):
        def _manifest_asset_url(*keys: str) -> Optional[str]:
            for key in keys:
                raw = str(manifest_assets.get(key) or "").strip()
                if not raw:
                    continue
                p = case_dir / raw
                if p.exists():
                    if p.suffix.lower() == ".mp4" or p.name.lower().endswith(".web.mp4"):
                        p = _ensure_browser_friendly_video(p) or p
                    return _front_url(p)
            return None
        assets["overlay_video"] = assets["overlay_video"] or _manifest_asset_url("pose_behavior_video", "overlay_video")
        assets["pose_demo_video"] = assets["pose_demo_video"] or _manifest_asset_url("pose_demo_video")
        assets["preview_image"] = assets["preview_image"] or _manifest_asset_url("preview", "preview_image")
        assets["contact_sheet"] = assets["contact_sheet"] or _manifest_asset_url("contact_sheet")
    return assets


def _read_timeline_segments_front(case_dir: Path) -> List[Dict[str, Any]]:
    """Unified timeline segments from best available source."""
    items: List[Dict[str, Any]] = []

    # Priority: timeline_chart.json > timeline_viz.json > timeline_students.json > timeline_students.csv > actions.fusion_v2.jsonl
    tc = case_dir / "timeline_chart.json"
    if tc.exists():
        obj = _read_json_file(tc, {})
        items = _json_list(obj, "items")
        if items:
            return items

    tv = case_dir / "timeline_viz.json"
    if tv.exists():
        obj = _read_json_file(tv, {})
        items = _json_list(obj, "items")
        if items:
            return items

    ts_json = case_dir / "timeline_students.json"
    if ts_json.exists():
        obj = _read_json_file(ts_json, {})
        items = _json_list(obj, "segments", "items")
        if items:
            return items

    csv_rows = _read_csv_rows(case_dir / "timeline_students.csv")
    if csv_rows:
        return csv_rows

    actions_path = case_dir / "actions.fusion_v2.jsonl"
    if actions_path.exists():
        loaded = _load_jsonl_rows(actions_path)
        fps = 25.0
        for meta_file in sorted(case_dir.glob("*.meta.json")):
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                fps = _safe_float(meta.get("fps"), 25.0)
                break
            except Exception:
                continue
        tl = _timeline_from_actions_rows(loaded, fps=fps)
        items = tl.get("items", [])
    return items


def _normalize_timeline_segments(raw_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize timeline items to uniform schema."""
    out: List[Dict[str, Any]] = []
    for i, item in enumerate(raw_items):
        if not isinstance(item, dict):
            continue
        seg_id = str(item.get("segment_id") or item.get("event_id") or f"seg_{i:06d}")
        track_id = int(_safe_float(item.get("track_id") or item.get("student_id") or item.get("Track ID"), -1))
        student_id = str(item.get("student_id") or f"S{track_id:02d}" if track_id >= 0 else f"S{i:02d}")
        st = _safe_float(item.get("start_time") or item.get("start") or item.get("t_start") or item.get("window_start"), 0.0)
        ed = _safe_float(item.get("end_time") or item.get("end") or item.get("t_end") or item.get("window_end"), st + 0.2)
        if ed <= st:
            ed = st + 0.2
        behavior = str(item.get("semantic_id") or item.get("action") or item.get("behavior_code") or item.get("semantic_label_en") or "unknown")
        entry = {
            "segment_id": seg_id,
            "track_id": track_id,
            "student_id": student_id,
            "start_time": round(st, 4),
            "end_time": round(ed, 4),
            "t_center": round((st + ed) / 2, 4),
            "behavior_code": str(item.get("behavior_code") or ""),
            "semantic_id": str(item.get("semantic_id") or behavior),
            "semantic_label_zh": str(item.get("semantic_label_zh") or item.get("action_label") or ""),
            "semantic_label_en": str(item.get("semantic_label_en") or behavior),
            "confidence": _safe_float(item.get("confidence") or item.get("conf"), 0.0),
            "verification_status": str(item.get("verification_status") or item.get("match_label") or item.get("label") or "unverified"),
        }
        out.append(entry)
    return out


@lru_cache(maxsize=128)
def _read_verified_events_front(case_dir: Path) -> List[Dict[str, Any]]:
    """Read verified_events.jsonl with normalized fields."""
    path = case_dir / "verified_events.jsonl"
    rows: List[Dict[str, Any]] = []
    if path.exists():
        rows = _load_jsonl_rows(path)
    else:
        json_path = case_dir / "verified_events.json"
        if json_path.exists():
            obj = _read_json_file(json_path, {})
            rows = _json_list(obj, "events", "items")
    if not rows:
        return []
    out: List[Dict[str, Any]] = []
    for row in rows:
        ws, we = _event_window(row)
        q_time = _safe_float(row.get("query_time") or row.get("timestamp") or row.get("t_center"), (ws + we) / 2)
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        raw_source = str(row.get("query_source") or row.get("source") or "")
        is_fallback = (raw_source.lower() == "visual_fallback")
        entry = {
            "event_id": str(row.get("event_id") or row.get("query_id") or ""),
            "query_id": str(row.get("query_id") or row.get("event_id") or ""),
            "track_id": int(_safe_float(row.get("track_id"), -1)),
            "query_text": str(row.get("query_text") or ""),
            "query_source": raw_source if raw_source else "unknown",
            "is_visual_fallback": is_fallback,
            "window_start": round(ws, 4),
            "window_end": round(we, 4),
            "query_time": round(q_time, 4),
            "p_match": _safe_float(row.get("p_match"), 0.0),
            "p_mismatch": _safe_float(row.get("p_mismatch"), 0.0),
            "reliability_score": _safe_float(row.get("reliability_score") or row.get("reliability"), 0.0),
            "uncertainty": _safe_float(row.get("uncertainty"), 0.0),
            "match_label": str(row.get("match_label") or row.get("label") or "unverified"),
            "label": str(row.get("label") or row.get("match_label") or "unverified"),
            "action": str(row.get("action") or row.get("semantic_id") or ""),
            "semantic_id": str(row.get("semantic_id") or ""),
            "semantic_label_zh": str(row.get("semantic_label_zh") or ""),
            "semantic_label_en": str(row.get("semantic_label_en") or ""),
            "evidence": evidence,
        }
        out.append(entry)
    return out


@lru_cache(maxsize=128)
def _read_event_queries_front(case_dir: Path) -> List[Dict[str, Any]]:
    path = case_dir / "event_queries.fusion_v2.jsonl"
    if not path.exists():
        path = case_dir / "event_queries.jsonl"
    if not path.exists():
        json_path = case_dir / "event_queries.json"
        if json_path.exists():
            obj = _read_json_file(json_path, {})
            rows = _json_list(obj, "queries", "events", "items")
            if rows:
                out: List[Dict[str, Any]] = []
                for row in rows:
                    src = str(row.get("source", ""))
                    out.append({
                        "event_id": str(row.get("event_id") or row.get("query_id") or ""),
                        "query_id": str(row.get("query_id") or row.get("event_id") or ""),
                        "query_text": str(row.get("query_text") or ""),
                        "query_source": src if src else "unknown",
                        "is_visual_fallback": (src.lower() == "visual_fallback"),
                        "t_center": _safe_float(row.get("t_center") or row.get("timestamp"), 0.0),
                        "start": _safe_float(row.get("start"), 0.0),
                        "end": _safe_float(row.get("end"), 0.0),
                        "confidence": _safe_float(row.get("confidence"), 0.0),
                        "event_type": str(row.get("event_type") or ""),
                    })
                return out

    # If file not found locally and this is a bundle dir, try source_case_dir
    if not path.exists():
        manifest_path = case_dir / "frontend_data_manifest.json"
        manifest = _read_json_file(manifest_path, {})
        src_dir = manifest.get("source_case_dir", "") if isinstance(manifest, dict) else ""
        if src_dir:
            src_path = Path(src_dir)
            alt_path = src_path / "event_queries.fusion_v2.jsonl"
            if alt_path.exists():
                path = alt_path
            else:
                alt_path = src_path / "event_queries.jsonl"
                if alt_path.exists():
                    path = alt_path

    if not path.exists():
        return []
    rows = _load_jsonl_rows(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        src = str(row.get("source", ""))
        out.append({
            "event_id": str(row.get("event_id") or row.get("query_id") or ""),
            "query_id": str(row.get("query_id") or row.get("event_id") or ""),
            "query_text": str(row.get("query_text") or ""),
            "query_source": src if src else "unknown",
            "is_visual_fallback": (src.lower() == "visual_fallback"),
            "t_center": _safe_float(row.get("t_center") or row.get("timestamp"), 0.0),
            "start": _safe_float(row.get("start"), 0.0),
            "end": _safe_float(row.get("end"), 0.0),
            "confidence": _safe_float(row.get("confidence"), 0.0),
            "event_type": str(row.get("event_type") or ""),
        })
    return out


def _read_transcript_front(case_dir: Path) -> List[Dict[str, Any]]:
    path = case_dir / "transcript.jsonl"
    if not path.exists():
        return []
    rows = _load_jsonl_rows(path)
    out: List[Dict[str, Any]] = []
    for row in rows:
        out.append({
            "start": _safe_float(row.get("start"), 0.0),
            "end": _safe_float(row.get("end"), 0.0),
            "text": str(row.get("text") or row.get("sentence") or ""),
            "speaker": str(row.get("speaker") or row.get("role") or "teacher"),
        })
    return out


def _build_front_feature_rows(
    timeline_segments: List[Dict[str, Any]],
    verified_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build feature rows for parallel coordinates from timeline + verified events."""
    ve_by_event: Dict[str, Dict[str, Any]] = {}
    ve_by_track: Dict[int, List[Dict[str, Any]]] = {}
    for ve in verified_events:
        eid = ve.get("event_id", "")
        if eid:
            ve_by_event[eid] = ve
        tid = ve.get("track_id", -1)
        if tid >= 0:
            ve_by_track.setdefault(tid, []).append(ve)

    rows: List[Dict[str, Any]] = []
    for seg in timeline_segments:
        eid = seg.get("segment_id") or seg.get("event_id") or ""
        tid = seg.get("track_id", -1)
        st = seg.get("start_time", 0)
        ed = seg.get("end_time", st + 0.2)

        # join: event_id exact match, then track_id + time overlap
        ve = ve_by_event.get(eid)
        if not ve and tid >= 0:
            candidates = ve_by_track.get(tid, [])
            best_ov = 0.0
            for v in candidates:
                ov = _interval_overlap(st, ed, v.get("window_start", 0), v.get("window_end", 0))
                if ov > best_ov:
                    best_ov = ov
                    ve = v

        evidence = ve.get("evidence", {}) if ve else {}
        c_visual = _safe_float(evidence.get("visual_score") or evidence.get("c_visual"), 0.8)
        c_text = _safe_float(evidence.get("text_score") or evidence.get("c_text"), 0.0)
        uq_track = _safe_float(evidence.get("uq_score") or evidence.get("uq_track") or ve.get("uncertainty", 0.1) if ve else 0.1, 0.1)
        weight_v = _safe_float(evidence.get("weight_v") or evidence.get("w_visual"), 0.5)
        weight_a = _safe_float(evidence.get("weight_a") or evidence.get("w_audio"), 0.5)
        w_sum = weight_v + weight_a
        if w_sum > 1e-9:
            weight_v, weight_a = weight_v / w_sum, weight_a / w_sum

        p_match_val = _safe_float(ve.get("p_match", 0.5) if ve else 0.5, 0.5)
        p_mismatch_val = _safe_float(ve.get("p_mismatch", 0.5) if ve else 0.5, 0.5)

        reliability = _safe_float(
            ve.get("reliability_score")
            or ve.get("reliability")
            or (weight_v * c_visual + weight_a * c_text - 0.1 * uq_track) if ve else (weight_v * c_visual + weight_a * c_text - 0.1 * uq_track),
            max(0.0, min(1.0, weight_v * c_visual + weight_a * c_text - 0.1 * uq_track)),
        )

        action_conf = _safe_float(seg.get("confidence"), 0.0)
        match_label = str(ve.get("match_label", "unverified") if ve else "unverified")

        rows.append({
            "event_id": eid if eid else (ve.get("event_id", "") if ve else ""),
            "track_id": tid,
            "student_id": seg.get("student_id", ""),
            "t_center": seg.get("t_center", (st + ed) / 2),
            "time_range": [round(st, 4), round(ed, 4)],
            "behavior_code": seg.get("behavior_code", ""),
            "semantic_id": seg.get("semantic_id", ""),
            "semantic_label_zh": seg.get("semantic_label_zh", ""),
            "semantic_label_en": seg.get("semantic_label_en", ""),
            "c_visual": round(c_visual, 4),
            "c_text": round(c_text, 4),
            "uq_track": round(uq_track, 4),
            "weight_v": round(weight_v, 4),
            "weight_a": round(weight_a, 4),
            "reliability_final": round(reliability, 4),
            "p_match": round(p_match_val, 4),
            "p_mismatch": round(p_mismatch_val, 4),
            "action_confidence": round(action_conf, 4),
            "verification_status": match_label,
            "query_source": str(ve.get("query_source", "unknown")) if ve else "unknown",
            "is_visual_fallback": bool(ve.get("is_visual_fallback", False)) if ve else False,
            "overlap": 0.0,
            "behavior_match_score": 0.0,
            "pose_conf": 0.0,
            "visible_kpt_ratio": 0.0,
            "motion_stability": 0.0,
        })
    return rows


def _build_front_sequence_series(
    feature_rows: List[Dict[str, Any]],
    verified_events: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Build time-sorted sequence data for the bottom timeline chart."""
    # Use verified events as the primary time axis, supplemented by feature rows
    if not verified_events:
        # fallback: use feature_rows sorted by t_center
        series: List[Dict[str, Any]] = []
        for row in sorted(feature_rows, key=lambda r: r.get("t_center", 0)):
            ml = row.get("verification_status", "unverified")
            score_ref = 1.0 if ml == "match" else (0.5 if ml == "uncertain" else 0.0)
            series.append({
                "t": row.get("t_center", 0),
                "score_model": row.get("p_match", row.get("reliability_final", 0)),
                "score_reference": score_ref,
                "reliability": row.get("reliability_final", 0),
                "uq_track": row.get("uq_track", 0),
                "c_visual": row.get("c_visual", 0),
                "c_text": row.get("c_text", 0),
                "weight_v": row.get("weight_v", 0),
                "weight_a": row.get("weight_a", 0),
            })
        return series

    series = []
    for ve in sorted(verified_events, key=lambda v: v.get("query_time", v.get("window_start", 0))):
        qt = ve.get("query_time", (ve.get("window_start", 0) + ve.get("window_end", 0)) / 2)
        evidence = ve.get("evidence", {})
        c_visual = _safe_float(evidence.get("visual_score") or evidence.get("c_visual"), 0.8)
        c_text = _safe_float(evidence.get("text_score") or evidence.get("c_text"), 0.0)
        uq = ve.get("uncertainty", _safe_float(evidence.get("uq_score"), 0.1))

        w_v = _safe_float(evidence.get("weight_v") or evidence.get("w_visual"), 0.5)
        w_a = _safe_float(evidence.get("weight_a") or evidence.get("w_audio"), 0.5)
        ws = w_v + w_a
        if ws > 1e-9:
            w_v, w_a = w_v / ws, w_a / ws

        ml = ve.get("match_label", "unverified")
        score_ref = 1.0 if ml == "match" else (0.5 if ml == "uncertain" else 0.0)

        series.append({
            "t": round(qt, 2),
            "score_model": round(ve.get("reliability_score") or ve.get("p_match", 0.0), 4),
            "score_reference": score_ref,
            "reliability": round(ve.get("reliability_score") or 0.0, 4),
            "uq_track": round(uq, 4),
            "c_visual": round(c_visual, 4),
            "c_text": round(c_text, 4),
            "weight_v": round(w_v, 4),
            "weight_a": round(w_a, 4),
        })
    return series


def _build_front_projection(
    feature_rows: List[Dict[str, Any]],
    unit: str = "event",
    method: str = "pca",
) -> List[Dict[str, Any]]:
    """Build projection points for the scatter view from feature rows."""
    if not feature_rows:
        return []

    n = len(feature_rows)
    vecs = []
    for row in feature_rows:
        vecs.append([
            row.get("c_visual", 0),
            row.get("c_text", 0),
            row.get("uq_track", 0),
            row.get("weight_v", 0),
            row.get("weight_a", 0),
            row.get("reliability_final", 0),
            row.get("p_match", 0),
            row.get("p_mismatch", 0),
            row.get("action_confidence", 0),
        ])

    X = np.array(vecs, dtype=np.float32)
    # replace NaN
    X = np.nan_to_num(X, nan=0.0)

    if X.shape[0] < 2:
        return [{
            "point_id": f"pt_{i}",
            "event_id": feature_rows[i].get("event_id", ""),
            "track_id": feature_rows[i].get("track_id", -1),
            "t_center": feature_rows[i].get("t_center", 0),
            "time_range": feature_rows[i].get("time_range", [0, 0]),
            "x": 0.5, "y": 0.5,
            "cluster_id": 0,
            "behavior_code": feature_rows[i].get("behavior_code", ""),
            "verification_status": feature_rows[i].get("verification_status", "unverified"),
            "feature_summary": {},
        } for i in range(len(feature_rows))]

    # Scale
    X_scaled = StandardScaler().fit_transform(X)

    # Project
    try:
        if method == "tsne" and X_scaled.shape[0] > 3:
            coords = _fit_tsne(X_scaled, metric="euclidean", random_state=42)
        elif method == "mds":
            coords = _make_mds("euclidean", random_state=42).fit_transform(X_scaled)
        else:  # pca
            coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)
    except Exception:
        coords = PCA(n_components=2, random_state=42).fit_transform(X_scaled)

    # Normalize to 0-1
    coords = MinMaxScaler().fit_transform(coords)

    # Cluster
    from sklearn.cluster import KMeans  # type: ignore
    n_clusters = min(8, max(2, n // 5))
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
    except Exception:
        # fallback: group by behavior_code
        codes = [row.get("behavior_code", "") for row in feature_rows]
        uniq = sorted(set(codes))
        code_to_cluster = {c: i for i, c in enumerate(uniq)}
        labels = [code_to_cluster.get(c, 0) for c in codes]

    points = []
    for i, row in enumerate(feature_rows):
        points.append({
            "point_id": f"pt_{i:06d}",
            "event_id": row.get("event_id", ""),
            "track_id": row.get("track_id", -1),
            "t_center": row.get("t_center", 0),
            "time_range": row.get("time_range", [0, 0]),
            "x": round(float(coords[i, 0]), 4),
            "y": round(float(coords[i, 1]), 4),
            "cluster_id": int(labels[i]) if i < len(labels) else 0,
            "behavior_code": row.get("behavior_code", ""),
            "verification_status": row.get("verification_status", "unverified"),
            "feature_summary": {
                "c_visual": row.get("c_visual", 0),
                "c_text": row.get("c_text", 0),
                "uq_track": row.get("uq_track", 0),
                "reliability_final": row.get("reliability_final", 0),
                "p_match": row.get("p_match", 0),
                "weight_v": row.get("weight_v", 0),
                "weight_a": row.get("weight_a", 0),
            },
        })
    return points


def _read_sr_ablation_data(case_dir: Path) -> Optional[Dict[str, Any]]:
    path = case_dir / "sr_ablation_metrics.json"
    if not path.exists():
        return None
    data = _read_json_file(path, {})
    if not isinstance(data, dict):
        return None
    variants = data.get("variants", [])
    if not isinstance(variants, list):
        variants = []

    best_variant = None
    best_f1 = -1.0
    A0 = None
    A8 = None
    for v in variants:
        f1 = _safe_float(v.get("behavior_macro_f1") or v.get("person_f1"), 0.0)
        if f1 > best_f1:
            best_f1 = f1
            best_variant = v
        vid = str(v.get("variant", "") or v.get("variant_id", ""))
        if "A0" in vid and "full_no_sr" in vid:
            A0 = v
        if "A8" in vid and "adaptive_sliced" in vid:
            A8 = v

    A0_vs_A8_delta = None
    if A0 is not None and A8 is not None:
        A0_vs_A8_delta = {}
        for key in (
            "person_f1", "behavior_macro_f1", "MOTA", "HOTA", "IDSW",
            "tracked_students", "rear_pose_person_rows_proxy",
            "stage_runtime_sec", "effective_fps", "avg_pose_conf",
            "rear_avg_visible_keypoints", "track_gap_count_proxy",
        ):
            a0v = _safe_float(A0.get(key), 0.0)
            a8v = _safe_float(A8.get(key), 0.0)
            A0_vs_A8_delta[key] = {"A0": a0v, "A8": a8v, "delta": round(a8v - a0v, 4)}

    return {
        "case_id": case_dir.name,
        "video_stem": _front_video_stem(case_dir),
        "variants": variants,
        "variant_count": len(variants),
        "best_variant": best_variant,
        "A0": A0,
        "A8": A8,
        "A0_vs_A8_delta": A0_vs_A8_delta,
        "contact_sheet_url": _front_url(case_dir / "sr_ablation_contact_sheet.jpg") if (case_dir / "sr_ablation_contact_sheet.jpg").exists() else None,
        "compare_table_url": _front_url(case_dir / "sr_ablation_compare_table.md") if (case_dir / "sr_ablation_compare_table.md").exists() else None,
    }


# ── Shared VSumVis helpers ───────────────────────────────

def _build_vsumvis_case_entry(case_dir: Path) -> Dict[str, Any]:
    case_id = _public_case_id(case_dir)
    kind = _front_case_kind(case_dir)
    video_stem = _front_video_stem(case_dir)
    file_status = _front_file_status(case_dir)
    student_count = _front_count_students(case_dir)
    ve_count, tl_count, eq_count = _front_count_events(case_dir)
    duration = _front_duration_sec(case_dir)
    assets = _front_assets(case_dir)
    sparkline = _front_sparkline(case_dir)
    contract = _front_contract_status(case_dir)
    data_source = _vsumvis_data_source(case_dir)

    match_rate = None
    verified_rows = _read_verified_events_front(case_dir)
    if verified_rows:
        total = len(verified_rows)
        matched = sum(1 for v in verified_rows if v.get("match_label") == "match")
        uncertain = sum(1 for v in verified_rows if v.get("match_label") == "uncertain")
        mismatch = sum(1 for v in verified_rows if v.get("match_label") == "mismatch")
        match_rate = {
            "total": total,
            "match": matched,
            "uncertain": uncertain,
            "mismatch": mismatch,
            "match_pct": round(100 * matched / total, 1) if total > 0 else 0,
        }

    return {
        "case_id": case_id,
        "kind": kind,
        "video_stem": video_stem,
        "label": f"{video_stem} ({kind})",
        "data_source": data_source,
        "file_status": file_status,
        "student_count": student_count,
        "event_count": tl_count,
        "verified_event_count": ve_count,
        "query_event_count": eq_count,
        "duration_sec": duration,
        "contract_status": contract,
        "assets": assets,
        "sparkline": sparkline,
        "match_rate": match_rate,
    }


def _build_vsumvis_case_detail(case_dir: Path, case_id: str) -> Dict[str, Any]:
    kind = _front_case_kind(case_dir)
    raw_segments = _read_timeline_segments_front(case_dir)
    segments = _normalize_timeline_segments(raw_segments)
    verified = _read_verified_events_front(case_dir)
    queries = _read_event_queries_front(case_dir)
    transcript = _read_transcript_front(case_dir)
    feature_rows = _build_front_feature_rows(segments, verified)
    seq_series = _build_front_sequence_series(feature_rows, verified)
    projection = _build_front_projection(feature_rows, unit="event", method="pca")

    duration = _front_duration_sec(case_dir)
    student_count = _front_count_students(case_dir)

    match_total = len(verified)
    match_count = sum(1 for v in verified if v.get("match_label") == "match")
    uncertain_count = sum(1 for v in verified if v.get("match_label") == "uncertain")
    mismatch_count = sum(1 for v in verified if v.get("match_label") == "mismatch")

    ablation = None
    if kind == "sr_ablation":
        ablation = _read_sr_ablation_data(case_dir)

    return {
        "status": "success",
        "data": {
            "case_info": {
                "case_id": case_id,
                "kind": kind,
                "video_stem": _front_video_stem(case_dir),
                "data_source": _vsumvis_data_source(case_dir),
                "duration_sec": duration,
                "student_count": student_count,
                "timeline_event_count": len(segments),
                "verified_event_count": match_total,
                "query_event_count": len(queries),
                "contract_status": _front_contract_status(case_dir),
                "pipeline_status": _front_contract_status(case_dir),
                "match_summary": {
                    "total": match_total,
                    "match": match_count,
                    "uncertain": uncertain_count,
                    "mismatch": mismatch_count,
                    "match_rate": round(100 * match_count / match_total, 1) if match_total > 0 else 0,
                    "uncertain_rate": round(100 * uncertain_count / match_total, 1) if match_total > 0 else 0,
                    "mismatch_rate": round(100 * mismatch_count / match_total, 1) if match_total > 0 else 0,
                },
            },
            "assets": _front_assets(case_dir),
            "timeline_segments": segments,
            "verified_events": verified,
            "event_queries": queries,
            "transcript": transcript,
            "feature_rows": feature_rows[:5000],
            "sequence_series": seq_series,
            "projection_points": projection[:5000],
            "ablation_summary": ablation,
        },
    }


# ── Canonical case APIs (/api/cases, /api/case/*) ──────────────

@app.get("/api/cases")
def get_canonical_cases():
    """Unified case listing across raw outputs, codex reports, and bundles."""
    cases_by_id: Dict[str, Dict[str, Any]] = {}
    for d in _iter_vsumvis_dirs():
        entry = _build_vsumvis_case_entry(d)
        ctx = _resolve_case_context(entry.get("case_id", d.name))
        entry["data_source"] = ctx.get("data_source", "unknown") if ctx else "unknown"
        entry["case_kind"] = _front_case_kind(d)
        current = cases_by_id.get(entry["case_id"])
        if current is None or _case_dir_quality(d) > _case_dir_quality(Path(current["_dir"])):
            entry["_dir"] = str(d)
            cases_by_id[entry["case_id"]] = entry
    for alias in ("front_45618_full", "front_45618_sliced", "run_full_paper_mainline_001"):
        d = _find_front_case_dir(alias)
        if d is None:
            continue
        entry = _build_vsumvis_case_entry(d)
        ctx = _resolve_case_context(alias)
        entry["case_id"] = alias
        entry["data_source"] = ctx.get("data_source", "unknown") if ctx else "unknown"
        entry["case_kind"] = _front_case_kind(d)
        entry["_dir"] = str(d)
        cases_by_id[alias] = entry
    cases = sorted(cases_by_id.values(), key=lambda c: (str(c.get("case_kind", "")), str(c.get("case_id", ""))))
    for case in cases:
        case.pop("_dir", None)
    return {"status": "success", "cases": cases, "total": len(cases)}


@app.get("/api/case/{case_id}/summary")
def get_case_summary(case_id: str):
    """Aggregated case summary: counts, label distribution, ASR status, contract."""
    ctx = _resolve_case_context(case_id)
    if ctx is None:
        raise HTTPException(404, f"Case not found: {case_id}")

    search_dir = Path(ctx["source_case_dir"] or ctx["case_dir"])
    case_dir = Path(ctx["case_dir"])

    # Read verified events for label distribution
    verified = _read_verified_events_front(case_dir)

    # If bundle with empty verified, try source
    if not verified and ctx.get("source_case_dir"):
        verified = _read_verified_events_front(Path(ctx["source_case_dir"]))

    label_dist: Dict[str, int] = {"match": 0, "uncertain": 0, "mismatch": 0, "unverified": 0}
    for ve in verified:
        lbl = str(ve.get("match_label") or ve.get("label") or "unverified")
        label_dist[lbl] = label_dist.get(lbl, 0) + 1

    # ASR status
    asr_report = _read_asr_quality_any(ctx)
    asr_status = "unknown"
    asr_accepted = 0
    asr_raw = 0
    if asr_report and isinstance(asr_report, dict):
        asr_status = str(asr_report.get("status", "unknown"))
        asr_accepted = int(asr_report.get("segments_accepted", 0))
        asr_raw = int(asr_report.get("segments_raw", 0))

    # Contract status
    contract_status = _front_contract_status(case_dir)
    student_count = _front_count_students(case_dir)
    duration = _front_duration_sec(case_dir)

    # Query count
    queries = _read_event_queries_front(case_dir)
    if not queries and ctx.get("source_case_dir"):
        queries = _read_event_queries_front(Path(ctx["source_case_dir"]))

    return {
        "status": "success",
        "case_id": case_id,
        "data_source": ctx["data_source"],
        "case_kind": _front_case_kind(case_dir),
        "student_count": student_count,
        "duration_sec": duration,
        "verified_event_count": len(verified),
        "query_event_count": len(queries),
        "label_distribution": label_dist,
        "asr_status": asr_status,
        "asr_segments_accepted": asr_accepted,
        "asr_segments_raw": asr_raw,
        "contract_status": contract_status,
        "llm_distilled_student_v4": _read_llm_student_v4_status(),
        "available_files": ctx["available_files"],
    }


@app.get("/api/case/{case_id}/contract")
def get_case_contract(case_id: str):
    """Detailed contract report: missing files, error counts, warnings."""
    ctx = _resolve_case_context(case_id)
    if ctx is None:
        raise HTTPException(404, f"Case not found: {case_id}")

    search_dir = Path(ctx["source_case_dir"] or ctx["case_dir"])
    contract_path = search_dir / "pipeline_contract_v2_report.json"
    fusion_path = search_dir / "fusion_contract_report.json"

    contract = _read_json_file(contract_path, None)
    fusion = _read_json_file(fusion_path, None)

    missing_files: List[str] = []
    error_count = 0
    warning_count = 0

    if isinstance(contract, dict):
        checks = contract.get("checks") or contract.get("results") or []
        if isinstance(checks, list):
            for c in checks:
                if isinstance(c, dict):
                    if not c.get("passed", True):
                        missing_files.append(str(c.get("file", c.get("name", "unknown"))))
                        error_count += 1
        # Also check files dict
        files = contract.get("files", {})
        if isinstance(files, dict):
            for fname, finfo in files.items():
                if isinstance(finfo, dict):
                    if not finfo.get("exists", True):
                        missing_files.append(fname)
                        error_count += 1

    return {
        "status": "success",
        "case_id": case_id,
        "contract": contract,
        "fusion_contract": fusion,
        "missing_files": missing_files,
        "error_count": error_count,
        "warning_count": warning_count,
    }


@app.get("/api/case/{case_id}/asr-quality")
def get_case_asr_quality(case_id: str):
    """ASR quality report for a case: status, segment counts, thresholds, audio energy."""
    ctx = _resolve_case_context(case_id)
    if ctx is None:
        raise HTTPException(404, f"Case not found: {case_id}")

    asr_report = _read_asr_quality_any(ctx)
    if asr_report is None:
        return {
            "status": "success",
            "case_id": case_id,
            "asr_report": None,
            "asr_available": False,
            "message": "No asr_quality_report.json found for this case.",
        }

    return {
        "status": "success",
        "case_id": case_id,
        "asr_available": True,
        "asr_status": str(asr_report.get("status", "unknown")),
        "model": str(asr_report.get("model", "")),
        "device": str(asr_report.get("device", "")),
        "language": str(asr_report.get("info", {}).get("language", "")),
        "duration": float(asr_report.get("info", {}).get("duration", 0)),
        "segments_raw": int(asr_report.get("segments_raw", 0)),
        "segments_good": int(asr_report.get("segments_good", 0)),
        "segments_accepted": int(asr_report.get("segments_accepted", 0)),
        "segments_rejected": int(asr_report.get("segments_rejected", 0)),
        "thresholds": asr_report.get("thresholds", {}),
        "audio_energy": asr_report.get("audio_energy", {}),
        "is_placeholder": str(asr_report.get("status", "")).lower() == "placeholder",
    }


# ── Evidence API (single-event) ──────────────────────────────────

@app.get("/api/case/{case_id}/evidence/{event_id}")
def get_event_evidence(case_id: str, event_id: str):
    """Return a complete evidence object for a single event.

    Joins: event_query + alignment candidates + verified event + timeline + media URLs.
    """
    ctx = _resolve_case_context(case_id)
    if ctx is None:
        raise HTTPException(404, f"Case not found: {case_id}")

    case_dir = Path(ctx["case_dir"])
    search_dir = Path(ctx["source_case_dir"] or ctx["case_dir"])

    # Read all data sources
    queries = _read_event_queries_front(case_dir)
    if not queries and ctx.get("source_case_dir"):
        queries = _read_event_queries_front(search_dir)

    verified = _read_verified_events_front(case_dir)
    if not verified and ctx.get("source_case_dir"):
        verified = _read_verified_events_front(search_dir)

    alignments = _read_alignments_any(ctx)
    asr_report = _read_asr_quality_any(ctx)

    # Infer query source for all events
    source_map = _infer_query_source(queries, verified, asr_report)

    # Find the target event
    query = None
    for q in queries:
        qid = str(q.get("event_id") or q.get("query_id") or "")
        if _event_id_matches(qid, event_id):
            query = q
            break

    ve = None
    for v in verified:
        vid = str(v.get("event_id") or v.get("query_id") or "")
        if _event_id_matches(vid, event_id):
            ve = v
            break

    if query is None and ve is None:
        return {
            "status": "event_not_found",
            "case_id": case_id,
            "event_id": event_id,
            "message": f"No query or verified event found for event_id={event_id} in case {case_id}.",
            "query": None,
            "selected": None,
            "align_candidates": [],
            "media": None,
            "source_files": {},
        }

    # Build query block
    q_row = query if isinstance(query, dict) else {}
    ve_row = ve if isinstance(ve, dict) else {}
    matched_event_id = str(
        (q_row.get("event_id") or q_row.get("query_id") or "")
        if q_row else (ve_row.get("event_id") or ve_row.get("query_id") or "")
    )
    src_info = source_map.get(matched_event_id or event_id, {
        "query_source": "unknown", "is_visual_fallback": False,
        "source_conflict": False, "asr_inferred_source": "unknown",
    })
    ve_window = ve_row.get("window") if isinstance(ve_row.get("window"), dict) else {}
    query_block: Dict[str, Any] = {
        "event_id": matched_event_id or event_id,
        "query_text": str(_first_value(q_row.get("query_text"), ve_row.get("query_text"), default="")),
        "query_source": src_info["query_source"],
        "is_visual_fallback": src_info["is_visual_fallback"],
        "source_conflict": src_info.get("source_conflict", False),
        "asr_inferred_source": src_info.get("asr_inferred_source", "unknown"),
        "query_time": _safe_float(_first_value(
            q_row.get("t_center"), q_row.get("timestamp"), q_row.get("query_time"),
            ve_row.get("query_time"), ve_row.get("timestamp"), ve_row.get("t_center"),
            default=0,
        ), 0.0),
        "window_start": _safe_float(_first_value(
            q_row.get("start"), q_row.get("window_start"),
            ve_window.get("start"), ve_row.get("window_start"),
            default=0,
        ), 0.0),
        "window_end": _safe_float(_first_value(
            q_row.get("end"), q_row.get("window_end"),
            ve_window.get("end"), ve_row.get("window_end"),
            default=0,
        ), 0.0),
        "confidence": _safe_float(q_row.get("confidence", 0), 0.0),
        "event_type": str(_first_value(q_row.get("event_type"), ve_row.get("event_type"), default="")),
    }
    if src_info.get("conflict_detail"):
        query_block["conflict_detail"] = src_info["conflict_detail"]

    # Build selected block
    selected_track_id = int(_safe_float(ve_row.get("track_id", -1), -1.0)) if ve_row else -1
    selected_rank, candidate_count, rank_metric, selected_by = _compute_selected_candidate_rank(matched_event_id or event_id, selected_track_id, alignments)
    selected_evidence = ve_row.get("evidence") if isinstance(ve_row.get("evidence"), dict) else {}
    fusion_mode = str(selected_evidence.get("fusion_mode") or "unknown")
    distillation = _student_distillation_from_evidence(selected_evidence)

    selected_block: Dict[str, Any] = {
        "track_id": selected_track_id,
        "student_id": ve_row.get("student_id", f"S{selected_track_id:02d}") if ve_row else "",
        "label": str(ve_row.get("match_label") or ve_row.get("label") or "unverified") if ve_row else "unverified",
        "p_match": _safe_float(ve_row.get("p_match", 0), 0.0) if ve_row else 0,
        "p_mismatch": _safe_float(ve_row.get("p_mismatch", 0), 0.0) if ve_row else 0,
        "reliability_score": _safe_float(ve_row.get("reliability_score", 0), 0.0) if ve_row else 0,
        "uncertainty": _safe_float(ve_row.get("uncertainty", 0), 0.0) if ve_row else 0,
        "visual_score": _safe_float(selected_evidence.get("visual_score", 0), 0.0) if ve_row else 0,
        "text_score": _safe_float(selected_evidence.get("text_score", 0), 0.0) if ve_row else 0,
        "uq_score": _safe_float(selected_evidence.get("uq_score", 0), 0.0) if ve_row else 0,
        "fusion_mode": fusion_mode,
        "distillation": distillation,
        "selected_candidate_rank": selected_rank,
        "candidate_count": candidate_count,
        "rank_metric": rank_metric,
        "selected_by": selected_by,
    }

    # Build alignment candidates block
    align_candidates: List[Dict[str, Any]] = []
    for rec in alignments:
        eid = str(rec.get("event_id") or rec.get("query_id") or "")
        if not _event_id_matches(eid, matched_event_id or event_id):
            continue
        candidates = rec.get("candidates", [])
        if not isinstance(candidates, list):
            continue
        scored = []
        for c in candidates:
            ov = _safe_float(c.get("overlap", 0), 0.0)
            ac = _safe_float(c.get("action_confidence", 0), 0.0)
            score = ov * 0.65 + ac * 0.35
            scored.append((score, c))
        scored.sort(key=lambda x: x[0], reverse=True)

        for rank, (sc, c) in enumerate(scored, 1):
            tid = int(_safe_float(c.get("track_id", -1), -1.0))
            align_candidates.append({
                "track_id": tid,
                "student_id": f"S{tid:02d}",
                "action": str(c.get("action", "")),
                "semantic_id": str(c.get("semantic_id", "")),
                "semantic_label_zh": str(c.get("semantic_label_zh", "")),
                "semantic_label_en": str(c.get("semantic_label_en", "")),
                "behavior_code": str(c.get("behavior_code", "")),
                "behavior_label_zh": str(c.get("behavior_label_zh", "")),
                "behavior_label_en": str(c.get("behavior_label_en", "")),
                "start_time": _safe_float(c.get("start_time", 0), 0.0),
                "end_time": _safe_float(c.get("end_time", 0), 0.0),
                "overlap": _safe_float(c.get("overlap", 0), 0.0),
                "action_confidence": _safe_float(c.get("action_confidence", 0), 0.0),
                "uq_track": _safe_float(c.get("uq_track") or c.get("uq_score") or 0, 0.0),
                "is_selected": (tid == selected_track_id),
                "rank": rank,
                "composite_score": round(sc, 4),
            })
        break

    # Build media block
    ws = float(query_block["window_start"])
    we = float(query_block["window_end"])
    if we <= ws:
        we = ws + 2.0
    video_stem = _front_video_stem(case_dir)
    video_url = f"/api/media/stream/{case_id}"

    media_block: Dict[str, Any] = {
        "video_url": video_url,
        "start_sec": round(ws, 3),
        "end_sec": round(we, 3),
        "video_stem": video_stem,
    }

    # Source files
    source_files: Dict[str, str] = {}
    for fname in ("verified_events.jsonl", "event_queries.fusion_v2.jsonl",
                  "align_multimodal.json", "asr_quality_report.json"):
        p = search_dir / fname
        if p.exists():
            source_files[fname] = str(p)

    return {
        "status": "success",
        "case_id": case_id,
        "event_id": event_id,
        "query": query_block,
        "selected": selected_block,
        "align_candidates": align_candidates,
        "media": media_block,
        "source_files": source_files,
    }


@app.get("/api/case/{case_id}/alignment/{event_id}")
def get_event_alignment(case_id: str, event_id: str):
    """Return raw alignment record for a single event with candidate rankings."""
    ctx = _resolve_case_context(case_id)
    if ctx is None:
        raise HTTPException(404, f"Case not found: {case_id}")

    alignments = _read_alignments_any(ctx)
    verified = _read_verified_events_front(Path(ctx["case_dir"]))
    if not verified and ctx.get("source_case_dir"):
        verified = _read_verified_events_front(Path(ctx["source_case_dir"]))

    # Find the alignment record
    align_rec = None
    for rec in alignments:
        eid = str(rec.get("event_id") or rec.get("query_id") or "")
        if _event_id_matches(eid, event_id):
            align_rec = rec
            break

    if align_rec is None:
        return {
            "status": "alignment_not_found",
            "case_id": case_id,
            "event_id": event_id,
            "message": f"No alignment record found for event_id={event_id} in case {case_id}.",
            "selected_track_id": -1,
            "selected_candidate_rank": 0,
            "candidate_count": 0,
            "rank_metric": "",
            "selected_by": "",
            "candidates": [],
        }

    # Find selected track_id from verified
    selected_track_id = -1
    for v in verified:
        if _event_id_matches(str(v.get("event_id") or v.get("query_id") or ""), event_id):
            selected_track_id = int(_safe_float(v.get("track_id", -1), -1.0))
            break

    # Build ranked candidates
    candidates = align_rec.get("candidates", [])
    if not isinstance(candidates, list):
        candidates = []

    scored = []
    for c in candidates:
        ov = _safe_float(c.get("overlap", 0), 0.0)
        ac = _safe_float(c.get("action_confidence", 0), 0.0)
        score = ov * 0.65 + ac * 0.35
        scored.append((score, c))
    scored.sort(key=lambda x: x[0], reverse=True)

    ranked: List[Dict[str, Any]] = []
    for rank, (sc, c) in enumerate(scored, 1):
        tid = int(_safe_float(c.get("track_id", -1), -1.0))
        ranked.append({
            "track_id": tid,
            "student_id": f"S{tid:02d}",
            "action": str(c.get("action", "")),
            "semantic_id": str(c.get("semantic_id", "")),
            "semantic_label_zh": str(c.get("semantic_label_zh", "")),
            "semantic_label_en": str(c.get("semantic_label_en", "")),
            "behavior_code": str(c.get("behavior_code", "")),
            "behavior_label_zh": str(c.get("behavior_label_zh", "")),
            "behavior_label_en": str(c.get("behavior_label_en", "")),
            "start_time": _safe_float(c.get("start_time", 0), 0.0),
            "end_time": _safe_float(c.get("end_time", 0), 0.0),
            "overlap": _safe_float(c.get("overlap", 0), 0.0),
            "action_confidence": _safe_float(c.get("action_confidence", 0), 0.0),
            "uq_track": _safe_float(c.get("uq_track") or c.get("uq_score") or 0, 0.0),
            "is_selected": (tid == selected_track_id),
            "rank": rank,
            "composite_score": round(sc, 4),
        })

    selected_rank, candidate_count, rank_metric, selected_by = _compute_selected_candidate_rank(event_id, selected_track_id, alignments)

    return {
        "status": "success",
        "case_id": case_id,
        "event_id": event_id,
        "query_text": str(align_rec.get("query_text", "")),
        "window_start": _safe_float(align_rec.get("window_start", 0), 0.0),
        "window_end": _safe_float(align_rec.get("window_end", 0), 0.0),
        "window_center": _safe_float(align_rec.get("window_center", 0), 0.0),
        "window_size": _safe_float(align_rec.get("window_size", 0), 0.0),
        "basis_motion": _safe_float(align_rec.get("basis_motion", 0), 0.0),
        "basis_uq": _safe_float(align_rec.get("basis_uq", 0), 0.0),
        "selected_track_id": selected_track_id,
        "selected_candidate_rank": selected_rank,
        "candidate_count": len(ranked),
        "rank_metric": rank_metric,
        "selected_by": selected_by,
        "candidates": ranked,
    }


# ── Paper metrics provenance API ─────────────────────────────────

@app.get("/api/paper/metrics")
def get_paper_metrics():
    """Return paper metrics with provenance: source file, sample size, quality tier."""
    metrics: List[Dict[str, Any]] = []

    # Read figure manifest
    fig_path = PAPER_TABLE_DIR / "tbl05_figure_manifest.csv"
    figure_map: Dict[str, Dict[str, Any]] = {}
    if fig_path.exists():
        for row in _read_csv_rows(fig_path):
            fid = str(row.get("figure_id") or row.get("fig_id") or "")
            if fid:
                figure_map[fid] = {
                    "figure_id": fid,
                    "title": str(row.get("title") or row.get("description") or ""),
                    "source_file": str(row.get("source_file") or row.get("data_source") or ""),
                    "chart_path": str(row.get("chart_path") or row.get("figure_path") or ""),
                }

    # Read metric CI table
    ci_path = PAPER_TABLE_DIR / "tbl01_run_metric_ci_enhanced.csv"
    if ci_path.exists():
        for row in _read_csv_rows(ci_path):
            metric_name = str(row.get("metric") or row.get("Metric") or row.get("metric_name") or "")
            if metric_name:
                metrics.append({
                    "metric_name": metric_name,
                    "value": str(row.get("value") or row.get("Value") or ""),
                    "ci_lower": str(row.get("ci_lower") or row.get("CI_lower") or ""),
                    "ci_upper": str(row.get("ci_upper") or row.get("CI_upper") or ""),
                    "source_file": str(row.get("source_file") or row.get("data_source") or "tbl01_run_metric_ci_enhanced.csv"),
                    "sample_size": str(row.get("n") or row.get("sample_size") or ""),
                    "quality_tier": str(row.get("quality_tier") or row.get("tier") or ""),
                    "figure_id": str(row.get("figure_id") or ""),
                })

    # Read quality gate table
    qg_path = PAPER_TABLE_DIR.parent / "paper_curated" / "tbl04_data_quality_gate.csv"
    quality_gates: List[Dict[str, Any]] = []
    if qg_path.exists():
        for row in _read_csv_rows(qg_path):
            quality_gates.append({str(k): v for k, v in row.items()})

    return {
        "status": "success",
        "metrics": metrics,
        "figures": figure_map,
        "quality_gates": quality_gates,
        "provenance": {
            "metric_ci_table": str(ci_path) if ci_path.exists() else None,
            "figure_manifest": str(fig_path) if fig_path.exists() else None,
            "quality_gate_table": str(qg_path) if qg_path.exists() else None,
        },
    }


# ── CANONICAL VSumVis routes (/api/v2/vsumvis/*) ────────

@app.get("/api/v2/vsumvis/cases")
def get_vsumvis_cases():
    cases: List[Dict[str, Any]] = []
    for d in _iter_vsumvis_dirs():
        cases.append(_build_vsumvis_case_entry(d))
    return {"status": "success", "cases": cases, "total": len(cases)}


@app.get("/api/v2/vsumvis/case/{case_id}")
def get_vsumvis_case_detail(
    case_id: str,
    label: str = Query("", description="Filter verified events by label: match, uncertain, mismatch"),
):
    case_dir = _find_front_case_dir(case_id)
    if case_dir is None:
        raise HTTPException(404, f"VSumVis case not found: {case_id}")
    result = _build_vsumvis_case_detail(case_dir, case_id)
    # Apply label filter if requested
    if label and label in ("match", "uncertain", "mismatch"):
        result["data"]["verified_events"] = [
            ve for ve in result["data"]["verified_events"]
            if str(ve.get("match_label") or ve.get("label") or "") == label
        ]
    return result


# ── Cluster statistics ──────────────────────────────────────

def _build_cluster_stats(
    projection_points: List[Dict[str, Any]],
    feature_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Compute per-cluster statistics: centroid, size, feature means, status ratios, top behaviors."""
    if not projection_points or not feature_rows:
        return []

    # Map event_id → feature_row for quick lookup
    row_by_eid: Dict[str, Dict[str, Any]] = {}
    for row in feature_rows:
        eid = row.get("event_id", "")
        if eid:
            row_by_eid[eid] = row

    # Group points by cluster_id
    clusters: Dict[int, List[Dict[str, Any]]] = {}
    for pt in projection_points:
        cid = int(pt.get("cluster_id", 0))
        clusters.setdefault(cid, []).append(pt)

    result: List[Dict[str, Any]] = []
    for cid, pts in sorted(clusters.items()):
        n = len(pts)
        if n == 0:
            continue

        # Centroid
        cx = sum(float(p.get("x", 0)) for p in pts) / n
        cy = sum(float(p.get("y", 0)) for p in pts) / n

        # Gather feature rows for this cluster
        cluster_rows = [row_by_eid.get(p.get("event_id", "")) for p in pts]
        cluster_rows = [r for r in cluster_rows if r is not None]

        # Feature means
        def _mean(key: str, default: float = 0.0) -> float:
            vals = [_safe_float(r.get(key, default)) for r in cluster_rows]
            return round(sum(vals) / len(vals), 4) if vals else 0.0

        # Status ratios
        status_counts: Dict[str, int] = {"match": 0, "uncertain": 0, "mismatch": 0, "unverified": 0}
        for pt in pts:
            s = str(pt.get("verification_status", "unverified")).strip()
            if s in status_counts:
                status_counts[s] += 1
            else:
                status_counts["unverified"] += 1

        # Top behaviors
        beh_counts: Dict[str, int] = {}
        for pt in pts:
            b = str(pt.get("behavior_code", "") or pt.get("semantic_id", "") or "unknown")
            beh_counts[b] = beh_counts.get(b, 0) + 1
        top_behaviors = sorted(beh_counts.items(), key=lambda kv: -kv[1])[:5]

        # IDs in this cluster
        ids = [p.get("event_id", "") or p.get("point_id", "") for p in pts[:200]]

        # align_score priority: verified_p_match -> p_match -> c_visual*c_text
        _p_match_val = _mean("p_match", 0.5)
        _c_vis = _mean("c_visual", 0.5)
        _c_txt = _mean("c_text", 0.5)
        _align = _mean("verified_p_match", -1)
        if _align < 0:
            _align = _p_match_val
        if _align <= 0:
            _align = round(_c_vis * _c_txt, 4)

        result.append({
            "cluster_id": cid,
            "size": n,
            "x": round(cx, 4),
            "y": round(cy, 4),
            "feature_mean": {
                "c_visual": _c_vis,
                "c_text": _c_txt,
                "align_score": round(_align, 4),
                "uq": _mean("uq_track", 0.1),
                "score_model": _p_match_val,
            },
            "status_ratio": {
                "match": round(status_counts["match"] / n, 3) if n else 0,
                "uncertain": round(status_counts["uncertain"] / n, 3) if n else 0,
                "mismatch": round(status_counts["mismatch"] / n, 3) if n else 0,
            },
            "top_behaviors": top_behaviors,
            "ids": ids,
        })
    return result


@app.get("/api/v2/vsumvis/clusters/{case_id}")
def get_vsumvis_clusters(case_id: str):
    """Return per-cluster statistics for Glyph projection mode."""
    case_dir = _find_front_case_dir(case_id)
    if case_dir is None:
        raise HTTPException(404, f"VSumVis case not found: {case_id}")
    raw_segments = _read_timeline_segments_front(case_dir)
    segments = _normalize_timeline_segments(raw_segments)
    verified = _read_verified_events_front(case_dir)
    feature_rows = _build_front_feature_rows(segments, verified)
    projection = _build_front_projection(feature_rows, unit="event", method="pca")
    clusters = _build_cluster_stats(projection, feature_rows)
    return {"status": "success", "clusters": clusters, "cluster_count": len(clusters)}


@app.get("/api/v2/vsumvis/timeline/{case_id}")
def get_vsumvis_timeline(case_id: str):
    """Return timeline-structured data: tracks, segments, alignments, frame_series.

    Synthesized from existing pipeline output — no new data files required.
    """
    case_dir = _find_front_case_dir(case_id)
    if case_dir is None:
        raise HTTPException(404, f"VSumVis case not found: {case_id}")

    raw_segments = _read_timeline_segments_front(case_dir)
    segments = _normalize_timeline_segments(raw_segments)
    verified = _read_verified_events_front(case_dir)
    transcript = _read_transcript_front(case_dir)
    feature_rows = _build_front_feature_rows(segments, verified)
    seq_series = _build_front_sequence_series(feature_rows, verified)

    # ── fps ──
    fps = 25.0
    for meta_file in sorted(case_dir.glob("*.meta.json")):
        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            fps = _safe_float(meta.get("fps"), 25.0)
            break
        except Exception:
            continue

    duration = _front_duration_sec(case_dir)

    # ── Tracks (group segments by track_id) ──
    track_map: Dict[int, List[Dict[str, Any]]] = {}
    for seg in segments:
        tid = int(seg.get("track_id", -1))
        if tid < 0:
            continue
        track_map.setdefault(tid, []).append(seg)

    tracks = []
    for tid in sorted(track_map.keys()):
        sgs = track_map[tid]
        student_id = sgs[0].get("student_id", f"S{tid:02d}") if sgs else f"S{tid:02d}"
        t_segments = []
        for s in sgs:
            eid = s.get("segment_id") or s.get("event_id") or ""
            # Find matching verified event
            ve = None
            for v in verified:
                if v.get("event_id") == eid or v.get("query_id") == eid:
                    ve = v
                    break
            st = _safe_float(s.get("start_time"), 0.0)
            ed = _safe_float(s.get("end_time"), st + 0.2)
            feats = {}
            for fr in feature_rows:
                if fr.get("event_id") == eid:
                    feats = fr
                    break
            t_segments.append({
                "segment_id": eid or f"seg_t{tid}_{len(t_segments):04d}",
                "start_time": round(st, 4),
                "end_time": round(ed, 4),
                "start_frame": int(st * fps),
                "end_frame": int(ed * fps),
                "behavior": str(s.get("semantic_label_en") or s.get("behavior_code") or "unknown"),
                "behavior_code": str(s.get("behavior_code") or ""),
                "track_id": tid,
                "student_id": student_id,
                "c_vis": _safe_float(feats.get("c_visual"), 0.0),
                "c_txt": _safe_float(feats.get("c_text"), 0.0),
                "align_score": _safe_float(
                    feats.get("align_score") or feats.get("alignment_score")
                    or feats.get("verified_p_match") or feats.get("p_match")
                    or (_safe_float(feats.get("c_visual"), 0.0) * _safe_float(feats.get("c_text"), 0.0)), 0.0),
                "uq": _safe_float(feats.get("uq_track"), 0.0),
                "p_match": _safe_float(feats.get("p_match"), 0.0),
                "match_status": str(ve.get("match_label") if ve else s.get("verification_status", "unverified")),
            })
        tracks.append({
            "track_id": tid,
            "student_id": student_id,
            "segments": t_segments,
        })

    # ── Transcripts ──
    transcripts = []
    for i, t in enumerate(transcript[:500]):
        transcripts.append({
            "text_id": f"txt_{i:06d}",
            "start_time": _safe_float(t.get("start"), 0.0),
            "end_time": _safe_float(t.get("end"), 0.0),
            "text": str(t.get("text") or ""),
            "asr_conf": _safe_float(t.get("asr_conf") or t.get("confidence"), 0.0),
        })

    # ── Alignments (synthesized: segment-transcript overlap) ──
    alignments = []
    for ti, track in enumerate(tracks):
        for seg in track.get("segments", []):
            st = seg.get("start_time", 0)
            ed = seg.get("end_time", 0)
            # Find overlapping transcript
            best_txt = None
            best_ov = 0.0
            for tx in transcripts:
                ov = _interval_overlap(st, ed, tx.get("start_time", 0), tx.get("end_time", 0))
                if ov > best_ov and ov > 0:
                    best_ov = ov
                    best_txt = tx
            if best_txt:
                alignments.append({
                    "alignment_id": f"align_{len(alignments):06d}",
                    "segment_id": seg.get("segment_id", ""),
                    "text_id": best_txt.get("text_id", ""),
                    "track_id": track.get("track_id", -1),
                    "start_time": round(st, 4),
                    "end_time": round(max(ed, best_txt.get("end_time", ed)), 4),
                    "start_frame": seg.get("start_frame", 0),
                    "end_frame": int(max(ed, best_txt.get("end_time", ed)) * fps),
                    "visual_behavior": seg.get("behavior", "unknown"),
                    "text_event": best_txt.get("text", "")[:80],
                    "c_vis": seg.get("c_vis", 0),
                    "c_txt": best_txt.get("asr_conf", seg.get("c_txt", 0)),
                    "align_score": round((seg.get("c_vis", 0) + best_txt.get("asr_conf", 0)) / 2, 4),
                    "uq": seg.get("uq", 0),
                    "match_status": seg.get("match_status", "unverified"),
                })

    # ── Frame series (from sequence_series) ──
    frame_series = []
    for s in seq_series[:2000]:
        frame_series.append({
            "time": round(_safe_float(s.get("t"), 0.0), 2),
            "score_model": round(_safe_float(s.get("score_model"), 0.5), 4),
            "score_ref": round(_safe_float(s.get("score_reference"), 0.5), 4),
            "c_vis": round(_safe_float(s.get("c_visual"), 0.0), 4),
            "c_txt": round(_safe_float(s.get("c_text"), 0.0), 4),
            "align_score": round(
                _safe_float(s.get("c_visual"), 0.0) * _safe_float(s.get("c_text"), 0.0) *
                _safe_float(s.get("reliability", 0.5)), 4),
            "uq": round(_safe_float(s.get("uq_track"), 0.0), 4),
            "match_status": "unverified",
        })

    return {
        "status": "success",
        "data": {
            "case_id": case_id,
            "fps": fps,
            "duration": duration,
            "tracks": tracks,
            "transcripts": transcripts,
            "alignments": alignments,
            "frame_series": frame_series,
        },
    }


@app.get("/api/v2/vsumvis/projection/{case_id}")
def get_vsumvis_projection(
    case_id: str,
    unit: str = Query("event"),
    method: str = Query("pca"),
):
    if method not in ("pca", "mds", "tsne"):
        raise HTTPException(400, "method must be pca, mds, or tsne")
    if unit not in ("event", "student"):
        raise HTTPException(400, "unit must be event or student")

    case_dir = _find_front_case_dir(case_id)
    if case_dir is None:
        raise HTTPException(404, f"VSumVis case not found: {case_id}")

    raw_segments = _read_timeline_segments_front(case_dir)
    segments = _normalize_timeline_segments(raw_segments)
    verified = _read_verified_events_front(case_dir)
    feature_rows = _build_front_feature_rows(segments, verified)

    if unit == "student":
        by_track: Dict[int, List[Dict[str, Any]]] = {}
        for row in feature_rows:
            tid = row.get("track_id", -1)
            if tid >= 0:
                by_track.setdefault(tid, []).append(row)
        student_rows = []
        for tid, rows in sorted(by_track.items()):
            if not rows:
                continue
            avg = {
                "event_id": f"student_{tid}",
                "track_id": tid,
                "student_id": rows[0].get("student_id", f"S{tid:02d}"),
                "t_center": sum(r.get("t_center", 0) for r in rows) / len(rows),
                "time_range": [min(r.get("time_range", [0, 0])[0] for r in rows),
                               max(r.get("time_range", [0, 0])[1] for r in rows)],
                "behavior_code": max(set(r.get("behavior_code", "") for r in rows), key=lambda c: sum(1 for r in rows if r.get("behavior_code") == c)),
                "semantic_id": max(set(r.get("semantic_id", "") for r in rows), key=lambda c: sum(1 for r in rows if r.get("semantic_id") == c)),
                "verification_status": "unverified",
                "c_visual": sum(r["c_visual"] for r in rows) / len(rows),
                "c_text": sum(r["c_text"] for r in rows) / len(rows),
                "uq_track": sum(r["uq_track"] for r in rows) / len(rows),
                "weight_v": sum(r["weight_v"] for r in rows) / len(rows),
                "weight_a": sum(r["weight_a"] for r in rows) / len(rows),
                "reliability_final": sum(r["reliability_final"] for r in rows) / len(rows),
                "p_match": sum(r["p_match"] for r in rows) / len(rows),
                "p_mismatch": sum(r["p_mismatch"] for r in rows) / len(rows),
                "action_confidence": sum(r["action_confidence"] for r in rows) / len(rows),
            }
            student_rows.append(avg)
        points = _build_front_projection(student_rows, unit="student", method=method)
    else:
        points = _build_front_projection(feature_rows, unit="event", method=method)

    return {
        "status": "success",
        "data": {
            "case_id": case_id,
            "unit": unit,
            "method": method,
            "points": points[:5000],
            "point_count": len(points),
        },
    }


@app.get("/api/v2/vsumvis/ablation/sr")
def get_vsumvis_ablation_sr():
    ablations: List[Dict[str, Any]] = []
    for d in _iter_vsumvis_dirs():
        if _front_case_kind(d) != "sr_ablation":
            continue
        data = _read_sr_ablation_data(d)
        if data:
            ablations.append(data)
    return {
        "status": "success",
        "ablations": ablations,
        "total": len(ablations),
    }


@app.get("/api/ablation/{dimension}")
def get_ablation_by_dimension(
    dimension: str,
):
    """Unified ablation endpoint: tracking, fusion, alignment, sr."""
    dimension = dimension.lower().strip()
    valid = {"tracking", "fusion", "alignment", "sr"}
    if dimension not in valid:
        raise HTTPException(400, f"Unknown ablation dimension: {dimension}. Valid: {', '.join(sorted(valid))}")

    if dimension == "sr":
        return get_vsumvis_ablation_sr()

    # For tracking/fusion/alignment, try reading from paper tables
    table_path = PAPER_TABLE_DIR / f"ablation_{dimension}.csv"
    alt_path = DOCS_DIR / "assets" / "tables" / "paper_curated" / f"ablation_{dimension}.csv"

    rows: List[Dict[str, Any]] = []
    for p in (table_path, alt_path):
        if p.exists():
            rows = _read_csv_rows(p)
            break

    if not rows:
        # Try to discover from SR ablation data aggregated by dimension
        return {
            "status": "success",
            "dimension": dimension,
            "ablations": [],
            "total": 0,
            "message": f"No pre-computed {dimension} ablation table found. Run the {dimension} ablation experiment and place results in {table_path}.",
        }

    return {
        "status": "success",
        "dimension": dimension,
        "ablations": rows,
        "total": len(rows),
        "source_file": str(table_path) if table_path.exists() else str(alt_path),
    }


@app.get("/api/ablation")
def get_ablation_list():
    """List available ablation dimensions."""
    available: List[Dict[str, Any]] = []
    # SR is always available
    available.append({"dimension": "sr", "available": True, "source": "VSumVis SR ablation cases"})
    for dim in ("tracking", "fusion", "alignment"):
        table_path = PAPER_TABLE_DIR / f"ablation_{dim}.csv"
        alt_path = DOCS_DIR / "assets" / "tables" / "paper_curated" / f"ablation_{dim}.csv"
        avail = table_path.exists() or alt_path.exists()
        available.append({
            "dimension": dim,
            "available": avail,
            "source": str(table_path) if table_path.exists() else (str(alt_path) if alt_path.exists() else None),
        })
    return {"status": "success", "dimensions": available, "ablations": available}


@app.get("/api/v2/vsumvis/compare/sr")
def get_vsumvis_compare_sr(
    case_id: str = Query(...),
    a: str = Query("A0_full_no_sr"),
    b: str = Query("A8_adaptive_sliced_artifact_deblur_opencv"),
):
    case_dir = _find_front_case_dir(case_id)
    if case_dir is None:
        raise HTTPException(404, f"VSumVis case not found: {case_id}")

    data = _read_sr_ablation_data(case_dir)
    if data is None:
        raise HTTPException(404, f"No SR ablation data for: {case_id}")

    variant_a = None
    variant_b = None
    for v in data.get("variants", []):
        vid = str(v.get("variant", "") or v.get("variant_id", ""))
        if vid == a:
            variant_a = v
        if vid == b:
            variant_b = v

    if variant_a is None or variant_b is None:
        raise HTTPException(404,
            f"Variant not found. A={a} found={variant_a is not None}, B={b} found={variant_b is not None}")

    delta = {}
    for key in [
        "person_f1", "behavior_macro_f1", "MOTA", "HOTA", "IDSW",
        "tracked_students", "rear_pose_person_rows_proxy",
        "stage_runtime_sec", "effective_fps", "avg_pose_conf",
        "rear_avg_visible_keypoints", "track_gap_count_proxy",
        "person_recall", "person_precision",
    ]:
        av = _safe_float(variant_a.get(key), 0.0)
        bv = _safe_float(variant_b.get(key), 0.0)
        delta[key] = {"A": av, "B": bv, "delta": round(bv - av, 4)}

    sub_a_dir = case_dir / a
    sub_b_dir = case_dir / b
    assets_a: Dict[str, str] = {}
    assets_b: Dict[str, str] = {}
    for _name, sd, ad in [("A", sub_a_dir, assets_a), ("B", sub_b_dir, assets_b)]:
        if sd.exists():
            for ext in (".jpg", ".png", ".mp4"):
                for f in sorted(sd.glob(f"*{ext}")):
                    ad[f.name] = _front_url(f)

    return {
        "status": "success",
        "data": {
            "case_id": case_id,
            "variant_a": {"label": a, "data": variant_a, "assets": assets_a},
            "variant_b": {"label": b, "data": variant_b, "assets": assets_b},
            "delta": delta,
            "contact_sheet_url": data.get("contact_sheet_url"),
            "compare_table_url": data.get("compare_table_url"),
        },
    }


# ── COMPATIBILITY aliases (/api/v2/front/* → delegate) ───

@app.get("/api/v2/front/cases")
def get_front_cases():
    return get_vsumvis_cases()


@app.get("/api/v2/front/case/{case_id}")
def get_front_case_detail(case_id: str):
    return get_vsumvis_case_detail(case_id)


@app.get("/api/v2/front/projection/{case_id}")
def get_front_projection(
    case_id: str,
    unit: str = Query("event"),
    method: str = Query("pca"),
):
    return get_vsumvis_projection(case_id, unit=unit, method=method)


@app.get("/api/v2/front/ablation/sr")
def get_front_ablation_sr():
    return get_vsumvis_ablation_sr()


@app.get("/api/v2/front/compare/sr")
def get_front_compare_sr(
    case_id: str = Query(...),
    a: str = Query("A0_full_no_sr"),
    b: str = Query("A8_adaptive_sliced_artifact_deblur_opencv"),
):
    return get_vsumvis_compare_sr(case_id, a=a, b=b)


@app.get("/paper/front-vsumvis", response_class=HTMLResponse)
async def front_vsumvis_page(request: Request):
    """VSumVis-style classroom behavior visual analysis dashboard."""
    tpl = TEMPLATE_DIR / "front_vsumvis.html"
    if tpl.exists():
        return FileResponse(str(tpl), media_type="text/html")
    raise HTTPException(404, "front_vsumvis.html template not found")


# =========================================================
# 9b) Registry-backed APIs: case manifests, gold eval, evidence v2
# =========================================================

def _get_registry() -> CaseRegistry:
    return CaseRegistry(PROJECT_ROOT)


@app.get("/api/cases/v2")
def get_cases_v2():
    """List all cases with structured manifests (artifact status, warnings)."""
    registry = _get_registry()
    manifests = registry.list_cases()
    out = []
    for m in manifests:
        out.append({
            "case_id": m.case_id,
            "video_id": m.video_id,
            "view_code": m.view_code,
            "status": m.status,
            "total_events": m.total_events,
            "gold_event_count": m.gold_event_count,
            "artifacts": m.artifact_summary(),
            "missing_artifacts": m.missing_artifacts(),
            "warnings": m.warnings,
            "video_path": m.video_path,
        })
    return {
        "status": "success",
        "cases": out,
        "total": len(out),
        "source_files": {
            "registry": "server/registry.py",
            "output_dir": str(OUTPUT_DIR),
        },
    }


@app.get("/api/case/{case_id}/manifest/v2")
def get_case_manifest_v2(case_id: str):
    """Structured manifest for a single case with per-artifact status."""
    registry = _get_registry()
    m = registry.get_case(case_id)
    if m is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    return {
        "status": "success",
        "case_id": m.case_id,
        "video_id": m.video_id,
        "view_code": m.view_code,
        "case_status": m.status,
        "total_events": m.total_events,
        "gold_event_count": m.gold_event_count,
        "artifacts": [
            {
                "key": a.key,
                "found": a.found,
                "file_name": a.file_name,
                "file_path": a.file_path,
                "size_bytes": a.size_bytes,
                "warnings": a.warnings,
            }
            for a in m.artifacts
        ],
        "missing_artifacts": m.missing_artifacts(),
        "present_artifacts": m.present_artifacts(),
        "warnings": m.warnings,
        "video_path": m.video_path,
        "source_dir": m.source_dir,
    }


class EvidenceService:
    """Build structured evidence payloads for review and blinded user-study modes."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.gold_loader = GoldLoader(project_root / "paper_experiments" / "gold")

    def build(self, case_id: str, event_id: str, mode: str = "review") -> Dict[str, Any]:
        mode = (mode or "review").strip().lower()
        if mode not in {"review", "blinded"}:
            raise HTTPException(400, "mode must be 'review' or 'blinded'")

        registry = _get_registry()
        m = registry.get_case(case_id)
        if m is None:
            raise HTTPException(404, f"Case not found: {case_id}")

        resolver = registry.get_resolver(case_id)
        if resolver is None:
            payload = self._missing_output(case_id, event_id)
            payload.update({
                "mode": mode,
                "asr_quality": None,
                "pipeline_step_status": self._pipeline_step_status(m),
                "gold_label_status": self._gold_label_status(case_id, event_id, mode),
            })
            return payload

        payload = self._build_from_resolver(case_id, event_id, mode, m, resolver)
        if mode == "blinded":
            return self._blind_payload(payload)
        return payload

    @staticmethod
    def _missing_output(case_id: str, event_id: str) -> Dict[str, Any]:
        return {
            "status": "missing_output",
            "case_id": case_id,
            "event_id": event_id,
            "message": f"No output directory found for case {case_id}. Run the pipeline first.",
            "source_files": {},
            "warnings": ["output directory not found"],
            "query": None,
            "selected": None,
            "align_candidates": [],
            "media": None,
        }

    def _build_from_resolver(
        self,
        case_id: str,
        event_id: str,
        mode: str,
        m: CaseManifestV1,
        resolver: ArtifactResolver,
    ) -> Dict[str, Any]:
        # Read all data sources
        queries = resolver.read_jsonl("event_queries_fusion_v2")
        if not queries:
            queries = resolver.read_jsonl("event_queries")
        verified_v3 = resolver.read_jsonl("verified_events_coherence_v3")
        verified = verified_v3 if verified_v3 else resolver.read_jsonl("verified_events")
        align_data = resolver.read_json("align_multimodal") or []
        if not isinstance(align_data, list):
            align_data = []
        asr_report = resolver.read_json("asr_quality_report")

        # Source files that exist
        source_files: Dict[str, str] = {}
        for key in ("verified_events_coherence_v3", "verified_events", "event_queries_fusion_v2", "event_queries",
                    "align_multimodal", "asr_quality_report", "transcript"):
            p = resolver.get_path(key)
            if p:
                source_files[key] = str(p)

        warnings: List[str] = list(m.warnings)

        # Find the target event
        query = None
        for q in queries:
            qid = str(q.get("event_id") or q.get("query_id") or "")
            if qid == event_id or _event_id_matches(qid, event_id):
                query = q
                break

        ve = None
        for v in verified:
            vid = str(v.get("event_id") or v.get("query_id") or "")
            if vid == event_id or _event_id_matches(vid, event_id):
                ve = v
                break

        if query is None and ve is None:
            return {
                "status": "event_not_found",
                "mode": mode,
                "case_id": case_id,
                "event_id": event_id,
                "message": f"No query or verified event found for event_id={event_id} in case {case_id}.",
                "source_files": source_files,
                "warnings": warnings + [f"event {event_id} not found in queries or verified_events"],
                "query": None,
                "selected": None,
                "align_candidates": [],
                "media": None,
                "asr_quality": self._asr_quality_status(asr_report),
                "pipeline_step_status": self._pipeline_step_status(m),
                "gold_label_status": self._gold_label_status(case_id, event_id, mode),
            }

        # Build query block
        q_row = query if isinstance(query, dict) else {}
        ve_row = ve if isinstance(ve, dict) else {}
        matched_event_id = str(
            (q_row.get("event_id") or q_row.get("query_id") or "")
            if q_row else (ve_row.get("event_id") or ve_row.get("query_id") or "")
        )
        ve_window = ve_row.get("window") if isinstance(ve_row.get("window"), dict) else {}
        query_block = {
            "event_id": matched_event_id or event_id,
            "query_text": str(_first_value(q_row.get("query_text"), ve_row.get("query_text"), default="")),
            "query_source": str(q_row.get("source", "") or ve_row.get("query_source", "") or "unknown"),
            "is_visual_fallback": str(q_row.get("source", "")).lower() == "visual_fallback",
            "query_time": _safe_float(_first_value(
                q_row.get("t_center"), q_row.get("timestamp"), q_row.get("query_time"),
                ve_row.get("query_time"), ve_row.get("timestamp"), ve_row.get("t_center"),
                default=0,
            ), 0.0),
            "window_start": _safe_float(_first_value(
                q_row.get("start"), q_row.get("window_start"),
                ve_window.get("start"), ve_row.get("window_start"),
                default=0,
            ), 0.0),
            "window_end": _safe_float(_first_value(
                q_row.get("end"), q_row.get("window_end"),
                ve_window.get("end"), ve_row.get("window_end"),
                default=0,
            ), 0.0),
            "confidence": _safe_float(q_row.get("confidence", 0), 0.0),
            "event_type": str(_first_value(q_row.get("event_type"), ve_row.get("event_type"), default="")),
        }

        # Build selected block
        selected_track_id = int(_safe_float(ve_row.get("track_id", -1), -1.0)) if ve_row else -1
        selected_evidence = ve_row.get("evidence") if isinstance(ve_row.get("evidence"), dict) else {}
        selected_block = {
            "track_id": selected_track_id,
            "student_id": ve_row.get("student_id", f"S{selected_track_id:02d}") if ve_row else "",
            "label": str(ve_row.get("match_label") or ve_row.get("label") or "unverified") if ve_row else "unverified",
            "p_match": _safe_float(ve_row.get("p_match", 0), 0.0) if ve_row else 0,
            "p_mismatch": _safe_float(ve_row.get("p_mismatch", 0), 0.0) if ve_row else 0,
            "reliability_score": _safe_float(ve_row.get("reliability_score", 0), 0.0) if ve_row else 0,
            "uncertainty": _safe_float(ve_row.get("uncertainty", 0), 0.0) if ve_row else 0,
            "visual_score": _safe_float(selected_evidence.get("visual_score", 0), 0.0) if ve_row else 0,
            "text_score": _safe_float(selected_evidence.get("text_score", 0), 0.0) if ve_row else 0,
            "uq_score": _safe_float(selected_evidence.get("uq_score", 0), 0.0) if ve_row else 0,
        }
        coherence_block, evidence_refs = self._coherence_payload(
            case_id=case_id,
            event_id=matched_event_id or event_id,
            ve_row=ve_row,
            query_row=q_row,
            align_data=align_data,
            asr_report=asr_report,
        )
        observation_card = self._observation_card_gate(selected_block, coherence_block)

        # Build alignment candidates block
        align_candidates: List[Dict[str, Any]] = []
        for rec in align_data:
            eid = str(rec.get("event_id") or rec.get("query_id") or "")
            if not _event_id_matches(eid, matched_event_id or event_id):
                continue
            candidates = rec.get("candidates", [])
            if not isinstance(candidates, list):
                continue
            scored = []
            for c in candidates:
                ov = _safe_float(c.get("overlap", 0), 0.0)
                ac = _safe_float(c.get("action_confidence", 0), 0.0)
                score = ov * 0.65 + ac * 0.35
                scored.append((score, c))
            scored.sort(key=lambda x: x[0], reverse=True)
            for rank, (sc, c) in enumerate(scored, 1):
                tid = int(_safe_float(c.get("track_id", -1), -1.0))
                align_candidates.append({
                    "track_id": tid,
                    "student_id": f"S{tid:02d}",
                    "action": str(c.get("action", "")),
                    "semantic_id": str(c.get("semantic_id", "")),
                    "start_time": _safe_float(c.get("start_time", 0), 0.0),
                    "end_time": _safe_float(c.get("end_time", 0), 0.0),
                    "overlap": _safe_float(c.get("overlap", 0), 0.0),
                    "action_confidence": _safe_float(c.get("action_confidence", 0), 0.0),
                    "uq_track": _safe_float(c.get("uq_track") or c.get("uq_score") or 0, 0.0),
                    "is_selected": (tid == selected_track_id),
                    "rank": rank,
                    "composite_score": round(sc, 4),
                })
            break

        if not align_candidates and "align_multimodal" in m.missing_artifacts():
            warnings.append("align_multimodal not found - alignment candidates unavailable")

        # Media block
        ws = float(query_block["window_start"])
        we = float(query_block["window_end"])
        if we <= ws:
            we = ws + 2.0
        media_block = {
            "video_url": f"/api/media/stream/{case_id}",
            "start_sec": round(ws, 3),
            "end_sec": round(we, 3),
        }

        return {
            "status": "success",
            "mode": mode,
            "case_id": case_id,
            "event_id": event_id,
            "query": query_block,
            "selected": selected_block,
            "align_candidates": align_candidates,
            "media": media_block,
            "asr_quality": self._asr_quality_status(asr_report),
            "pipeline_step_status": self._pipeline_step_status(m),
            "gold_label_status": self._gold_label_status(case_id, matched_event_id or event_id, mode),
            "coherence": coherence_block,
            "evidence_refs": evidence_refs,
            "observation_card": observation_card,
            "source_files": source_files,
            "warnings": warnings,
        }

    @staticmethod
    def _coherence_payload(
        *,
        case_id: str,
        event_id: str,
        ve_row: Dict[str, Any],
        query_row: Dict[str, Any],
        align_data: List[Dict[str, Any]],
        asr_report: Any,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if isinstance(ve_row.get("coherence"), dict) and isinstance(ve_row.get("evidence_refs"), dict):
            return ve_row["coherence"], ve_row["evidence_refs"]
        if not ve_row:
            return (
                {
                    "available": False,
                    "reason": "verified_event_missing",
                    "coherence_valid": False,
                    "coherence_score": None,
                    "conflict_type": "low_evidence",
                    "quality_flags": ["no_candidates"],
                },
                {},
            )
        try:
            import importlib
            mod = importlib.import_module("classroom_pipeline.pipeline.08_coherence_diagnosis")
            diagnose_rows = mod.diagnose_rows
        except Exception as exc:
            return (
                {
                    "available": False,
                    "reason": f"diagnosis_unavailable:{type(exc).__name__}",
                    "coherence_valid": False,
                    "coherence_score": None,
                    "conflict_type": "low_evidence",
                    "quality_flags": ["no_candidates"],
                },
                {},
            )
        try:
            diagnosed = diagnose_rows(
                verified_rows=[ve_row],
                align_rows=align_data,
                query_rows=[query_row] if query_row else [],
                asr_report=asr_report,
                case_id=case_id,
            )
            if diagnosed:
                row = diagnosed[0]
                coherence = row.get("coherence") if isinstance(row.get("coherence"), dict) else {}
                refs = row.get("evidence_refs") if isinstance(row.get("evidence_refs"), dict) else {}
                coherence["available"] = True
                coherence["source"] = "runtime_diagnosis"
                return coherence, refs
        except Exception as exc:
            return (
                {
                    "available": False,
                    "reason": f"diagnosis_failed:{type(exc).__name__}",
                    "coherence_valid": False,
                    "coherence_score": None,
                    "conflict_type": "low_evidence",
                    "quality_flags": ["no_candidates"],
                },
                {},
            )
        return (
            {
                "available": False,
                "reason": "diagnosis_empty",
                "coherence_valid": False,
                "coherence_score": None,
                "conflict_type": "low_evidence",
                "quality_flags": ["no_candidates"],
            },
            {},
        )

    @staticmethod
    def _observation_card_gate(selected: Dict[str, Any], coherence: Dict[str, Any]) -> Dict[str, Any]:
        flags = coherence.get("quality_flags") if isinstance(coherence.get("quality_flags"), list) else []
        blocking_flags = {
            "high_uq",
            "low_visual_confidence",
            "track_fragmentation",
            "low_asr_quality",
            "object_conflict",
            "low_alignment",
            "no_candidates",
        }
        blocked_by: List[str] = []
        if _safe_float(selected.get("p_match"), 0.0) < 0.6:
            blocked_by.append("p_match_below_threshold")
        if _safe_float(selected.get("reliability_score"), 0.0) < 0.6:
            blocked_by.append("reliability_below_threshold")
        if not bool(coherence.get("coherence_valid")):
            blocked_by.append("coherence_invalid")
        if coherence.get("coherence_score") is None or _safe_float(coherence.get("coherence_score"), 0.0) < 0.6:
            blocked_by.append("coherence_below_threshold")
        conflict_type = str(coherence.get("conflict_type", "low_evidence"))
        if conflict_type != "none":
            blocked_by.append(f"conflict:{conflict_type}")
        for flag in flags:
            if flag in blocking_flags:
                blocked_by.append(f"quality:{flag}")
        blocked_by = sorted(set(blocked_by))
        return {
            "eligible": not blocked_by,
            "blocked_by": blocked_by,
            "policy": "neutral_observation_only",
        }

    @staticmethod
    def _asr_quality_status(asr_report: Any) -> Dict[str, Any]:
        if not isinstance(asr_report, dict):
            return {"available": False, "status": "missing"}
        return {
            "available": True,
            "status": str(asr_report.get("status") or asr_report.get("quality_status") or "unknown"),
            "segment_count": asr_report.get("segment_count", asr_report.get("accepted_segments")),
            "raw_segment_count": asr_report.get("raw_segment_count", asr_report.get("raw_segments")),
            "is_placeholder": str(asr_report.get("status", "")).lower() == "placeholder",
        }

    @staticmethod
    def _pipeline_step_status(m: CaseManifestV1) -> Dict[str, Any]:
        source_dir = m.metadata.get("source_case_dir", "")
        if not source_dir:
            return {"available": False, "status": "no_source", "steps": []}
        manager = PipelineManifestManager(Path(source_dir))
        manifest = manager.load()
        if manifest is None:
            return {
                "available": False,
                "status": "no_manifest",
                "source_files": {"pipeline_manifest": str(manager.manifest_path)},
                "steps": [],
            }
        return {
            "available": True,
            "status": manifest.status,
            "run_id": manifest.run_id,
            "failed_steps": [s.step_id for s in manifest.failed_steps()],
            "source_files": {"pipeline_manifest": str(manager.manifest_path)},
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_name": s.step_name,
                    "status": s.status,
                    "schema_valid": s.schema_valid,
                    "record_count": s.record_count,
                    "warnings": s.warnings,
                    "error_message": s.error_message,
                }
                for s in manifest.steps
            ],
        }

    def _gold_label_status(self, case_id: str, event_id: str, mode: str) -> Dict[str, Any]:
        events = self.gold_loader.load("official")
        for ev in events:
            if ev.case_id == case_id and (ev.suggested_event_id == event_id or _event_id_matches(ev.suggested_event_id, event_id)):
                if mode == "blinded":
                    return {
                        "available": True,
                        "hidden": True,
                        "reason": "blinded_mode",
                        "review_status": ev.review_status,
                    }
                return {
                    "available": True,
                    "hidden": False,
                    "review_status": ev.review_status,
                    "match_label": ev.match_label,
                    "gold_action": ev.gold_action,
                    "gold_track_id": ev.gold_track_id,
                    "gold_start": ev.gold_start,
                    "gold_end": ev.gold_end,
                    "batch_id": ev.batch_id,
                    "review_round": ev.review_round,
                    "adjudicator": ev.adjudicator,
                }
        return {"available": False, "hidden": mode == "blinded"}

    @staticmethod
    def _blind_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        blinded = dict(payload)
        blinded["mode"] = "blinded"
        blinded["selected"] = {"hidden": True, "reason": "blinded_mode"}
        blinded["coherence"] = {"hidden": True, "reason": "blinded_mode"}
        blinded["evidence_refs"] = {"hidden": True, "reason": "blinded_mode"}
        blinded["observation_card"] = {"hidden": True, "reason": "blinded_mode"}
        blinded_candidates: List[Dict[str, Any]] = []
        for candidate in payload.get("align_candidates", []) or []:
            if not isinstance(candidate, dict):
                continue
            clean = dict(candidate)
            for key in ("is_selected", "composite_score", "action_confidence", "uq_track"):
                clean.pop(key, None)
            blinded_candidates.append(clean)
        blinded["align_candidates"] = blinded_candidates
        gold_status = blinded.get("gold_label_status")
        if isinstance(gold_status, dict) and gold_status.get("available"):
            blinded["gold_label_status"] = {
                "available": True,
                "hidden": True,
                "reason": "blinded_mode",
                "review_status": gold_status.get("review_status"),
            }
        return blinded


@app.get("/api/case/{case_id}/evidence/{event_id}/v2")
def get_event_evidence_v2(
    case_id: str,
    event_id: str,
    mode: str = Query("review", pattern="^(review|blinded)$"),
):
    """Structured evidence API with review/blinded modes."""
    return EvidenceService(PROJECT_ROOT).build(case_id, event_id, mode=mode)


@app.get("/api/v2/cases/{case_id}/events/{event_id}/evidence")
def get_event_evidence_v2_canonical(
    case_id: str,
    event_id: str,
    mode: str = Query("review", pattern="^(review|blinded)$"),
):
    """Canonical v2 evidence path for review and blinded user studies."""
    return EvidenceService(PROJECT_ROOT).build(case_id, event_id, mode=mode)


@app.get("/api/eval/gold-summary")
def get_gold_summary(quality: str = Query("official")):
    """Return summary of human gold annotations: count, label distribution, validation status."""
    gold_dir = PROJECT_ROOT / "paper_experiments" / "gold"
    loader = GoldLoader(gold_dir)
    try:
        summary = loader.summary(quality=quality)
        visual_report = GoldReadinessAnalyzer(gold_dir, PROJECT_ROOT).visual_gold_report(quality=quality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    quality_excluded_reason_distribution = summary.get("excluded_reason_distribution", {})
    return {
        "status": "success",
        **summary,
        "official_visual_gold": visual_report["official_visual_gold_count"],
        "pipeline_evaluable_gold": visual_report["pipeline_evaluable_count"],
        "excluded_from_verifier": visual_report["excluded_from_verifier"],
        "quality_excluded_reason_distribution": quality_excluded_reason_distribution,
        "excluded_reason_distribution": visual_report["excluded_reason_distribution"],
        "verifier_excluded_reason_distribution": visual_report["excluded_reason_distribution"],
        "visual_gold_distributions": visual_report["distributions"],
        "source_files": {
            "gold_jsonl": str(loader.gold_path),
            "active_gold_jsonl": str(loader.gold_path),
            "legacy_gold_jsonl": str(gold_dir / "gold_events.real.jsonl"),
            "agent300_manifest": visual_report["source_files"]["agent300_manifest"],
            "agent300_event_bridge_map": visual_report["source_files"]["agent300_event_bridge_map"],
            "gold_schema": str(gold_dir / "gold_events.real.schema.json"),
            "validation_report": str(gold_dir / "gold_validation_report.json"),
        },
    }


@app.get("/api/eval/verifier-report")
def get_verifier_eval_report(
    case_ids: Optional[str] = Query(None),
    quality: str = Query("official"),
):
    """Gold-backed verifier evaluation report.

    Compares system predictions with human gold annotations.
    Returns accuracy, precision, recall, F1, uncertain_ratio, candidate_recall, ECE, Brier.
    """
    gold_dir = PROJECT_ROOT / "paper_experiments" / "gold"
    try:
        evaluator = GoldBackedEvaluator(gold_dir, PROJECT_ROOT, quality=quality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    ids = [c.strip() for c in case_ids.split(",")] if case_ids else None
    report = evaluator.evaluate(case_ids=ids)
    return {
        "status": "success",
        "report_id": report.report_id,
        "sample_count": report.sample_count,
        "gold_event_count_requested": report.gold_event_count_requested,
        "skipped_gold_event_count": report.skipped_gold_event_count,
        "official_visual_gold_count": report.official_visual_gold_count,
        "pipeline_evaluable_count": report.pipeline_evaluable_gold_count,
        "excluded_from_verifier": report.excluded_from_verifier_count,
        "excluded_reason_distribution": report.excluded_reason_distribution,
        "accuracy": report.accuracy,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "uncertain_ratio": report.uncertain_ratio,
        "candidate_recall": report.candidate_recall,
        "ece": report.ece,
        "brier": report.brier,
        "label_distribution": report.label_distribution,
        "quality_tier": report.quality_tier,
        "baselines": [
            {
                "name": b.name,
                "accuracy": b.accuracy,
                "precision": b.precision,
                "recall": b.recall,
                "f1": b.f1,
                "uncertain_ratio": b.uncertain_ratio,
                "description": b.description,
            }
            for b in report.baselines
        ],
        "warnings": report.warnings,
        "source_files": report.source_files,
        "per_case_breakdown": report.per_case_breakdown,
        "missing_output_cases": report.missing_output_cases,
        "excluded_gold_events": report.excluded_gold_events,
        "per_sample": [
            {
                "gold_event_id": s.gold.suggested_event_id,
                "pipeline_event_id": s.pipeline_event_id,
                "source_layer": s.source_layer,
                "gold_case_id": s.gold.case_id,
                "gold_match_label": s.gold.match_label,
                "gold_action": s.gold.gold_action,
                "gold_review_status": s.gold.review_status,
                "gold_batch_id": s.gold.batch_id,
                "gold_review_round": s.gold.review_round,
                "predicted_label": s.predicted_label,
                "predicted_track_id": s.predicted_track_id,
                "predicted_action": s.predicted_action,
                "confidence_match": s.confidence_match,
                "reliability": s.reliability,
                "uncertainty": s.uncertainty,
                "gold_predicted_match": s.gold_predicted_match,
                "gold_candidate_in_top_k": s.gold_candidate_in_top_k,
                "top_k_candidate_count": s.top_k_candidate_count,
            }
            for s in report.per_sample
        ],
    }


@app.get("/api/eval/visual-gold-report")
def get_visual_gold_report(quality: str = Query("official")):
    """Report official visual-gold coverage and verifier-readiness separately."""
    gold_dir = PROJECT_ROOT / "paper_experiments" / "gold"
    try:
        report = GoldReadinessAnalyzer(gold_dir, PROJECT_ROOT).visual_gold_report(quality=quality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "status": "success",
        **report,
    }


def _verifier_metrics_dict(report: VerifierEvalReportV1) -> Dict[str, Any]:
    return {
        "sample_count": report.sample_count,
        "accuracy": report.accuracy,
        "precision": report.precision,
        "recall": report.recall,
        "f1": report.f1,
        "uncertain_ratio": report.uncertain_ratio,
        "candidate_recall": report.candidate_recall,
        "ece": report.ece,
        "brier": report.brier,
        "label_distribution": report.label_distribution,
    }


def _fast_reviewer_verifier_metrics(sample_count: int) -> Dict[str, Any]:
    persisted = _read_json_file(EVAL_REPORTS_DIR / "gold_eval_v1.json", {})
    metrics = persisted.get("metrics", {}) if isinstance(persisted, dict) else {}
    label_distribution = persisted.get("label_distribution", {}) if isinstance(persisted, dict) else {}
    if not isinstance(metrics, dict):
        metrics = {}
    if not isinstance(label_distribution, dict):
        label_distribution = {}
    return {
        "sample_count": int(sample_count),
        "accuracy": _safe_float(metrics.get("accuracy"), 0.0),
        "precision": _safe_float(metrics.get("precision"), 0.0),
        "recall": _safe_float(metrics.get("recall"), 0.0),
        "f1": _safe_float(metrics.get("f1"), 0.0),
        "uncertain_ratio": _safe_float(metrics.get("uncertain_ratio"), 0.0),
        "candidate_recall": _safe_float(metrics.get("candidate_recall"), 0.0),
        "ece": _safe_float(metrics.get("ece"), 0.0),
        "brier": _safe_float(metrics.get("brier"), 0.0),
        "label_distribution": label_distribution,
    }


def _build_reviewer_response_markdown(report: Dict[str, Any]) -> str:
    verifier = report["verifier_metrics"]
    excluded = report["excluded_reason_distribution"]
    lines = [
        "# Reviewer Response Evaluation Summary",
        "",
        f"- Official visual gold: {report['official_visual_gold_count']}",
        f"- Pipeline-evaluable gold: {report['pipeline_evaluable_count']}",
        f"- Excluded from verifier: {report['excluded_from_verifier']}",
        "",
        "## Distributions",
        "",
        f"- Labels: `{json.dumps(report['label_distribution'], ensure_ascii=False)}`",
        f"- Actions: `{json.dumps(report['action_distribution'], ensure_ascii=False)}`",
        f"- Views: `{json.dumps(report['view_distribution'], ensure_ascii=False)}`",
        f"- Quality tiers: `{json.dumps(report['quality_tier_distribution'], ensure_ascii=False)}`",
        f"- Excluded reasons: `{json.dumps(excluded, ensure_ascii=False)}`",
        "",
        "## Verifier Metrics",
        "",
        f"- Sample count: {verifier['sample_count']}",
        f"- Accuracy: {verifier['accuracy']}",
        f"- Precision: {verifier['precision']}",
        f"- Recall: {verifier['recall']}",
        f"- F1: {verifier['f1']}",
        f"- Candidate recall: {verifier['candidate_recall']}",
        f"- ECE: {verifier['ece']}",
        f"- Brier: {verifier['brier']}",
        "",
        (
            "Verifier metrics are computed only on pipeline-evaluable gold. "
            "The remaining official visual-gold events are reported with explicit exclusion reasons."
        ),
    ]
    return "\n".join(lines)


@app.get("/api/eval/reviewer-response-report")
def get_reviewer_response_report(quality: str = Query("official")):
    """Return a paper-ready JSON/Markdown summary for reviewer response."""
    if not isinstance(quality, str):
        quality = "official"
    gold_dir = PROJECT_ROOT / "paper_experiments" / "gold"
    try:
        analyzer = GoldReadinessAnalyzer(gold_dir, PROJECT_ROOT)
        visual_report = analyzer.visual_gold_report(quality=quality)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    distributions = visual_report["distributions"]
    verifier_metrics = _fast_reviewer_verifier_metrics(int(visual_report["pipeline_evaluable_count"]))
    report = {
        "report_id": "reviewer_response_eval_report_v1",
        "sample_size": visual_report["sample_size"],
        "official_visual_gold_count": visual_report["official_visual_gold_count"],
        "pipeline_evaluable_count": visual_report["pipeline_evaluable_count"],
        "excluded_from_verifier": visual_report["excluded_from_verifier"],
        "label_distribution": distributions["label"],
        "action_distribution": distributions["action"],
        "view_distribution": distributions["view"],
        "quality_tier_distribution": distributions["quality_tier"],
        "review_status_distribution": distributions["review_status"],
        "excluded_reason_distribution": visual_report["excluded_reason_distribution"],
        "verifier_metrics": verifier_metrics,
        "source_files": {
            **visual_report["source_files"],
            "verifier_report": str(EVAL_REPORTS_DIR / "gold_eval_v1.json"),
        },
    }
    report["markdown"] = _build_reviewer_response_markdown(report)
    return {
        "status": "success",
        **report,
    }


@app.get("/api/eval/visual-gold-bridge")
def get_visual_gold_bridge_report():
    """Return agent300 visual-gold bridge coverage and verifier-readiness status."""
    gold_dir = PROJECT_ROOT / "paper_experiments" / "gold"
    analyzer = GoldReadinessAnalyzer(gold_dir, PROJECT_ROOT)
    rows = [
        row for row in analyzer.classify_events()
        if row.get("source_layer") == "agent300_visual_gold"
    ]
    excluded_rows = [row for row in rows if not row.get("pipeline_evaluable")]
    return {
        "status": "success",
        "report_id": "agent300_visual_gold_bridge_v2",
        "sample_size": len(rows),
        "verifier_evaluable_count": len(rows) - len(excluded_rows),
        "verifier_blocked_count": len(excluded_rows),
        "excluded_reason_distribution": _counter(excluded_rows, "excluded_reason"),
        "pipeline_status_distribution": _counter(rows, "mapping_status"),
        "view_distribution": _counter(rows, "view_code"),
        "gold_action_distribution": _counter(rows, "gold_action"),
        "quality_tier_distribution": _counter(rows, "quality_tier"),
        "review_status_distribution": _counter(rows, "review_status"),
        "source_files": {
            "report": str(analyzer.bridge_map_path.parent / "agent300_visual_gold_bridge_report.json"),
            "manifest": str(analyzer.visual_manifest_path),
            "event_bridge_map": str(analyzer.bridge_map_path),
            "pipeline_unblock_batch_001_040": str(analyzer.pipeline_unblock_path),
        },
    }


def _agent300_bridge_dir() -> Path:
    return PROJECT_ROOT / "paper_experiments" / "gold" / "agent_300_workbench" / "visual_gold_bridge"


def _counter(rows: List[Dict[str, Any]], key: str, default: str = "unknown") -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key) or default).strip() or default
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _path_exists_from_project(raw_path: Any) -> bool:
    if not raw_path:
        return False
    try:
        p = Path(str(raw_path))
        if not p.is_absolute():
            p = PROJECT_ROOT / p
        return p.exists()
    except Exception:
        return False


def _duration_stats(rows: List[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    durations: List[float] = []
    for row in rows:
        try:
            start = float(row.get("gold_start"))
            end = float(row.get("gold_end"))
        except (TypeError, ValueError):
            continue
        if end >= start:
            durations.append(round(end - start, 6))
    if not durations:
        return {"min": None, "median": None, "mean": None, "max": None}
    durations.sort()
    n = len(durations)
    mid = n // 2
    median = durations[mid] if n % 2 else (durations[mid - 1] + durations[mid]) / 2
    return {
        "min": round(durations[0], 3),
        "median": round(median, 3),
        "mean": round(sum(durations) / n, 3),
        "max": round(durations[-1], 3),
    }


def _build_agent300_visual_only_report(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    sample_size = len(rows)
    contact_sheet_existing = sum(1 for row in rows if _path_exists_from_project(row.get("contact_sheet_path")))
    source_video_existing = sum(1 for row in rows if _path_exists_from_project(row.get("source_video_path")))
    rows_with_existing_frames = 0
    frame_path_count = 0
    existing_frame_path_count = 0
    for row in rows:
        frame_paths = row.get("frame_paths") or []
        if not isinstance(frame_paths, list):
            frame_paths = []
        existing_for_row = sum(1 for p in frame_paths if _path_exists_from_project(p))
        frame_path_count += len(frame_paths)
        existing_frame_path_count += existing_for_row
        if existing_for_row:
            rows_with_existing_frames += 1

    blocked_count = sum(1 for row in rows if not bool(row.get("verifier_evaluable")))
    required_bridge_counts: Dict[str, int] = {}
    for row in rows:
        required_bridge = row.get("required_bridge") or {}
        if not isinstance(required_bridge, dict):
            continue
        for key, value in required_bridge.items():
            if key == "required_artifacts":
                continue
            if value is True:
                required_bridge_counts[key] = required_bridge_counts.get(key, 0) + 1

    return {
        "report_id": "agent300_visual_only_eval_v1",
        "scope": "visual_gold_only_not_verifier_accuracy",
        "sample_size": sample_size,
        "verifier_metric_eligible": False,
        "verifier_evaluable_count": sample_size - blocked_count,
        "verifier_blocked_count": blocked_count,
        "distributions": {
            "view": _counter(rows, "view_code"),
            "match_label": _counter(rows, "match_label"),
            "gold_action": _counter(rows, "gold_action"),
            "review_status": _counter(rows, "review_status"),
            "quality_tier": _counter(rows, "quality_tier"),
            "pipeline_status": _counter(rows, "pipeline_status"),
        },
        "time_window_duration_seconds": _duration_stats(rows),
        "evidence_availability": {
            "rows_with_contact_sheet": contact_sheet_existing,
            "rows_with_any_frame": rows_with_existing_frames,
            "rows_with_source_video": source_video_existing,
            "frame_paths_total": frame_path_count,
            "frame_paths_existing": existing_frame_path_count,
        },
        "verifier_readiness": {
            "status": "blocked",
            "blocked_reason": "agent300 visual gold uses e_agent300 IDs and evidence packages, but verifier requires real pipeline outputs and pipeline event IDs.",
            "required_bridge_distribution": dict(sorted(required_bridge_counts.items())),
            "required_artifacts": [
                "verified_events.jsonl",
                "event_queries.fusion_v2.jsonl",
                "align_multimodal.json",
                "asr_quality_report.json",
            ],
        },
        "paper_use": {
            "can_use_for": [
                "official visual gold audit",
                "sample size and label/action distribution",
                "evidence quality report",
                "blinded or second-pass review study",
            ],
            "cannot_use_for": [
                "verifier accuracy",
                "precision/recall/F1",
                "candidate recall",
                "ECE/Brier calibration",
            ],
        },
        "next_unblockers": [
            "Run the real pipeline for source videos that do not have output directories.",
            "Build a mapping from e_agent300 event IDs to pipeline e_XXXXXX_XX event IDs using case_id and time windows.",
            "Regenerate verifier-report after verified_events and align_multimodal artifacts exist.",
        ],
    }


@app.get("/api/eval/agent300-visual-gold-report")
def get_agent300_visual_gold_report():
    """Return visual-only quality/readiness metrics for the agent300 official gold rows."""
    gold_dir = PROJECT_ROOT / "paper_experiments" / "gold"
    analyzer = GoldReadinessAnalyzer(gold_dir, PROJECT_ROOT)
    rows = [
        row for row in analyzer.classify_events()
        if row.get("source_layer") == "agent300_visual_gold"
    ]
    excluded_rows = [row for row in rows if not row.get("pipeline_evaluable")]
    return {
        "status": "success",
        "report_id": "agent300_visual_only_eval_v2",
        "scope": "visual_gold_quality_and_verifier_readiness_not_full_accuracy",
        "sample_size": len(rows),
        "verifier_metric_eligible": False,
        "verifier_evaluable_count": len(rows) - len(excluded_rows),
        "verifier_blocked_count": len(excluded_rows),
        "excluded_reason_distribution": _counter(excluded_rows, "excluded_reason"),
        "distributions": {
            "view": _counter(rows, "view_code"),
            "match_label": _counter(rows, "match_label"),
            "gold_action": _counter(rows, "gold_action"),
            "review_status": _counter(rows, "review_status"),
            "quality_tier": _counter(rows, "quality_tier"),
            "mapping_status": _counter(rows, "mapping_status"),
        },
        "verifier_readiness": {
            "status": "partial",
            "pipeline_evaluable_count": len(rows) - len(excluded_rows),
            "excluded_from_verifier": len(excluded_rows),
            "excluded_reason_distribution": _counter(excluded_rows, "excluded_reason"),
            "required_artifacts": [
                "verified_events.jsonl",
                "event_queries.fusion_v2.jsonl",
                "align_multimodal.json",
                "asr_quality_report.json",
            ],
        },
        "source_files": {
            "manifest": str(analyzer.visual_manifest_path),
            "report": str(analyzer.bridge_map_path.parent / "agent300_visual_only_eval_report.json"),
            "event_bridge_map": str(analyzer.bridge_map_path),
            "pipeline_unblock_batch_001_040": str(analyzer.pipeline_unblock_path),
        },
    }


@app.get("/api/case/{case_id}/pipeline-manifest")
def get_pipeline_manifest(case_id: str):
    """Return the pipeline run manifest for a case (step status, schema validation, errors)."""
    registry = _get_registry()
    m = registry.get_case(case_id)
    if m is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    source_dir = m.metadata.get("source_case_dir", "")
    if not source_dir:
        raise HTTPException(404, f"No source directory for case {case_id}")
    manager = PipelineManifestManager(Path(source_dir))
    manifest = manager.load()
    if manifest is None:
        return {
            "status": "no_manifest",
            "case_id": case_id,
            "message": f"No pipeline_run_manifest.json found for case {case_id}. Run the pipeline to generate one.",
            "source_files": {"pipeline_manifest": str(manager.manifest_path)},
            "steps": [],
        }
    return {
        "status": "success",
        "case_id": case_id,
        "manifest": manifest.to_dict(),
        "failed_steps": [s.step_id for s in manifest.failed_steps()],
        "all_output_files": manifest.all_output_files(),
        "source_files": {"pipeline_manifest": str(manager.manifest_path)},
    }


@app.get("/api/case/{case_id}/pipeline-steps")
def get_pipeline_steps(case_id: str):
    """Return step-level status for all registered pipeline steps."""
    registry = _get_registry()
    m = registry.get_case(case_id)
    if m is None:
        raise HTTPException(404, f"Case not found: {case_id}")
    source_dir = m.metadata.get("source_case_dir", "")
    if not source_dir:
        return {
            "status": "no_source",
            "case_id": case_id,
            "steps": [],
            "message": "No source directory found.",
        }
    manager = PipelineManifestManager(Path(source_dir))
    manifest = manager.load()
    if manifest is None:
        return {
            "status": "no_manifest",
            "case_id": case_id,
            "steps": [],
            "message": "No pipeline_run_manifest.json found.",
        }
    return {
        "status": "success",
        "case_id": case_id,
        "run_id": manifest.run_id,
        "pipeline_status": manifest.status,
        "steps": [
            {
                "step_id": s.step_id,
                "step_name": s.step_name,
                "status": s.status,
                "schema_valid": s.schema_valid,
                "schema_errors": s.schema_errors,
                "record_count": s.record_count,
                "duration_sec": s.duration_sec,
                "warnings": s.warnings,
                "error_message": s.error_message,
            }
            for s in manifest.steps
        ],
        "total_steps": len(manifest.steps),
        "completed": sum(1 for s in manifest.steps if s.status == "ok"),
        "failed": sum(1 for s in manifest.steps if s.status == "error"),
        "pending": sum(1 for s in manifest.steps if s.status == "pending"),
    }


# ── Persisted evaluation report endpoints ──────────────────────────

EVAL_REPORTS_DIR = OUTPUT_DIR / "eval_reports"

_PERSISTED_REPORTS = {
    "gold_eval_v1": "gold_eval_v1.json",
    "case_manifest_summary": "case_manifest_summary.json",
    "missing_artifact_summary": "missing_artifact_summary.json",
    "gold_annotation_summary": "gold_annotation_summary.json",
}


@app.get("/api/eval/reports")
def list_eval_reports():
    """List all persisted evaluation reports."""
    reports = []
    for name, fname in _PERSISTED_REPORTS.items():
        p = EVAL_REPORTS_DIR / fname
        if p.exists():
            try:
                size = p.stat().st_size
            except OSError:
                size = None
            reports.append({
                "name": name,
                "file_name": fname,
                "available": True,
                "size_bytes": size,
            })
        else:
            reports.append({
                "name": name,
                "file_name": fname,
                "available": False,
                "size_bytes": None,
            })
    return {
        "status": "success",
        "reports": reports,
        "eval_reports_dir": str(EVAL_REPORTS_DIR),
    }


@app.get("/api/eval/report/{report_name}")
def get_eval_report(report_name: str):
    """Read a persisted evaluation report by name.

    Available reports: gold_eval_v1, case_manifest_summary,
    missing_artifact_summary, gold_annotation_summary.
    """
    fname = _PERSISTED_REPORTS.get(report_name)
    if fname is None:
        raise HTTPException(
            404,
            f"Unknown report: {report_name}. Available: {list(_PERSISTED_REPORTS.keys())}"
        )
    p = EVAL_REPORTS_DIR / fname
    if not p.exists():
        raise HTTPException(
            404,
            f"Report '{report_name}' not found at {p}. Run scripts/paper/persist_eval_reports.py first."
        )
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        raise HTTPException(500, f"Failed to read report: {e}")
    return {
        "status": "success",
        "report_name": report_name,
        "source_file": str(p),
        **data,
    }


@app.get("/api/eval/report/{report_name}/download")
def download_eval_report(report_name: str):
    """Download a persisted evaluation report as a JSON file attachment."""
    fname = _PERSISTED_REPORTS.get(report_name)
    if fname is None:
        raise HTTPException(
            404,
            f"Unknown report: {report_name}. Available: {list(_PERSISTED_REPORTS.keys())}"
        )
    p = EVAL_REPORTS_DIR / fname
    if not p.exists():
        raise HTTPException(
            404,
            f"Report '{report_name}' not found at {p}. Run scripts/paper/persist_eval_reports.py first."
        )
    return FileResponse(
        str(p),
        media_type="application/json",
        filename=f"{report_name}.json",
    )


# =========================================================
# 9c) Gold review tool: local annotation workbench
# =========================================================

GOLD_DIR = PROJECT_ROOT / "paper_experiments" / "gold"
GOLD_TASKS_DIR = GOLD_DIR / "annotation_workbench"
GOLD_TASKS_FILE = GOLD_TASKS_DIR / "tasks.jsonl"
GOLD_TASKS_CSV_FILE = GOLD_TASKS_DIR / "tasks.csv"
GOLD_DRAFT_FILE = GOLD_TASKS_DIR / "gold_review_draft.jsonl"
GOLD_SCHEMA_FILE = GOLD_DIR / "gold_events.real.schema.json"
GOLD_OUTPUT_FILE = GOLD_DIR / "gold_events.real.jsonl"


def _active_gold_tasks_file() -> Path:
    if GOLD_TASKS_CSV_FILE.exists():
        return GOLD_TASKS_CSV_FILE
    return GOLD_TASKS_FILE


def _load_gold_tasks() -> List[Dict[str, Any]]:
    if GOLD_TASKS_CSV_FILE.exists():
        with GOLD_TASKS_CSV_FILE.open("r", encoding="utf-8-sig", newline="") as f:
            return [dict(row) for row in csv.DictReader(f)]
    return _load_jsonl_rows(GOLD_TASKS_FILE)


def _gold_task_by_row_id(row_id: int) -> Optional[Dict[str, Any]]:
    for task in _load_gold_tasks():
        if int(task.get("row_id", -1)) == row_id:
            return task
    return None


def _gold_evidence_dir_for_task(task: Dict[str, Any]) -> Optional[Path]:
    raw = str(task.get("evidence_dir") or "").strip()
    candidates: List[Path] = []
    if raw:
        candidates.append((PROJECT_ROOT / raw).resolve())
    row_id = str(task.get("row_id", "")).strip()
    case_id = str(task.get("case_id", "")).strip()
    event_id = str(task.get("suggested_event_id", "")).strip()
    if row_id and case_id and event_id:
        candidates.append((GOLD_TASKS_DIR / "evidence" / f"{row_id}_{case_id}_{event_id}").resolve())

    evidence_root = (GOLD_TASKS_DIR / "evidence").resolve()
    for candidate in candidates:
        try:
            candidate.relative_to(evidence_root)
        except ValueError:
            continue
        if candidate.exists() and candidate.is_dir():
            return candidate
    return None


def _gold_source_video_for_task(task: Dict[str, Any]) -> Optional[Path]:
    view_dirs = {
        "front": "\u6b63\u65b9\u89c6\u89d2",
        "rear": "\u540e\u65b9\u89c6\u89d2",
        "teacher": "\u6559\u5e08\u89c6\u89d2",
        "top1": "\u659c\u4e0a\u65b9\u89c6\u89d21",
        "top2": "\u659c\u4e0a\u65b9\u89c6\u89d22",
    }
    dataset_dir = "\u667a\u6167\u8bfe\u5802\u5b66\u751f\u884c\u4e3a\u6570\u636e\u96c6"
    view_code = str(task.get("view_code") or "").strip()
    case_id = str(task.get("case_id") or "").strip()
    source_id = case_id.split("__", 1)[1] if "__" in case_id else ""
    view_dir = view_dirs.get(view_code)
    if not view_dir or not source_id:
        return None

    data_root = (PROJECT_ROOT / "data" / dataset_dir).resolve()
    candidate = (data_root / view_dir / f"{source_id}.mp4").resolve()
    try:
        candidate.relative_to(data_root)
    except ValueError:
        return None
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def _load_gold_drafts() -> Dict[int, Dict[str, Any]]:
    rows = _load_jsonl_rows(GOLD_DRAFT_FILE)
    return {int(r["row_id"]): r for r in rows if "row_id" in r}


def _save_gold_drafts(drafts: Dict[int, Dict[str, Any]]) -> None:
    GOLD_DRAFT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_DRAFT_FILE.open("w", encoding="utf-8") as f:
        for row_id in sorted(drafts.keys()):
            f.write(json.dumps(drafts[row_id], ensure_ascii=False) + "\n")


def _draft_to_gold_event(task: Dict[str, Any], draft: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a task + draft pair into a gold_events.real.jsonl record."""
    def _f(val: Any, default: float = 0.0) -> float:
        try:
            return float(val)
        except Exception:
            return default

    def _s(val: Any) -> str:
        return str(val or "").strip()

    def _i(val: Any) -> Any:
        s = _s(val)
        if s == "" or s.lower() == "unknown":
            return "unknown"
        try:
            return int(val)
        except Exception:
            return s

    start = _f(draft.get("manual_gold_start"), _f(task.get("suggested_time_start")))
    end = _f(draft.get("manual_gold_end"), _f(task.get("suggested_time_end")))

    return {
        "case_id": task["case_id"],
        "view_code": task["view_code"],
        "suggested_event_id": task.get("suggested_event_id", ""),
        "suggested_track_id": task.get("suggested_track_id"),
        "suggested_action": task.get("suggested_action", ""),
        "suggested_time_window": {
            "start": _f(task.get("suggested_time_start")),
            "end": _f(task.get("suggested_time_end")),
        },
        "suggested_label": task.get("suggested_label"),
        "gold_track_id": _i(draft.get("manual_gold_track_id")) or _i(task.get("suggested_track_id")),
        "gold_action": _s(draft.get("manual_gold_action")) or _s(task.get("suggested_action")),
        "gold_start": start,
        "gold_end": end,
        "match_label": _s(draft.get("manual_match_label")) or _s(task.get("suggested_label")),
        "noise_type": _s(draft.get("manual_noise_type")) or "none",
        "occlusion_level": _s(draft.get("manual_occlusion_level")) or "none",
        "asr_quality": _s(draft.get("manual_asr_quality")) or "not_applicable",
        "annotator": _s(draft.get("manual_annotator")) or "unknown",
        "review_status": _s(draft.get("manual_review_status")) or "draft",
        "notes": _s(draft.get("manual_notes")),
    }


def _validate_gold_event(event: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    required = [
        "case_id", "view_code", "suggested_event_id", "suggested_track_id",
        "suggested_action", "suggested_time_window", "suggested_label",
        "gold_track_id", "gold_action", "gold_start", "gold_end",
        "match_label", "noise_type", "occlusion_level", "asr_quality",
        "annotator", "review_status",
    ]
    for field in required:
        if field not in event:
            errors.append(f"missing required field: {field}")

    enums = {
        "view_code": {"front", "rear", "teacher", "top1", "top2"},
        "match_label": {"match", "mismatch", "uncertain"},
        "suggested_label": {"match", "mismatch", "uncertain", "unknown", None},
        "noise_type": {"none", "asr_empty", "asr_noise", "occlusion", "multi_person_ambiguity", "camera_angle", "motion_blur", "other"},
        "occlusion_level": {"none", "low", "medium", "high"},
        "asr_quality": {"empty", "low", "medium", "high", "not_applicable"},
        "review_status": {"draft", "needs_adjudication", "peer_reviewed", "adjudicated", "final", "rejected"},
    }
    for field, allowed in enums.items():
        val = event.get(field)
        if val not in allowed:
            errors.append(f"{field}: invalid value '{val}', expected one of {sorted(str(x) for x in allowed)}")

    tw = event.get("suggested_time_window")
    if isinstance(tw, dict):
        for k in ("start", "end"):
            if k not in tw:
                errors.append(f"suggested_time_window missing '{k}'")
    elif "suggested_time_window" in event:
        errors.append("suggested_time_window must be an object with start/end")

    return errors


@app.get("/gold/review", response_class=HTMLResponse)
async def gold_review_page():
    tpl = TEMPLATE_DIR / "gold_review.html"
    if tpl.exists():
        return FileResponse(str(tpl))
    raise HTTPException(404, "gold_review.html not found in templates/")


@app.get("/api/gold/tasks")
def api_gold_tasks():
    tasks = _load_gold_tasks()
    drafts = _load_gold_drafts()

    merged: List[Dict[str, Any]] = []
    for t in tasks:
        rid = int(t.get("row_id", 0))
        d = drafts.get(rid)
        has_draft = d is not None
        is_complete = False
        if d:
            is_complete = bool(
                d.get("manual_match_label")
                and d.get("manual_gold_action")
                and d.get("manual_review_status")
            )
        merged.append({
            **t,
            "draft": d,
            "has_draft": has_draft,
            "is_complete": is_complete,
        })

    exported_ids: set = set()
    if GOLD_OUTPUT_FILE.exists():
        for row in _load_jsonl_rows(GOLD_OUTPUT_FILE):
            eid = row.get("suggested_event_id", "")
            cid = row.get("case_id", "")
            exported_ids.add(f"{cid}::{eid}")

    for m in merged:
        key = f"{m.get('case_id', '')}::{m.get('suggested_event_id', '')}"
        m["is_exported"] = key in exported_ids

    drafted_count = sum(1 for m in merged if m["has_draft"])
    complete_count = sum(1 for m in merged if m["is_complete"])
    exported_count = sum(1 for m in merged if m["is_exported"])

    return {
        "status": "success",
        "tasks": merged,
        "progress": {
            "total": len(merged),
            "drafted": drafted_count,
            "complete": complete_count,
            "exported": exported_count,
        },
        "source_files": {
            "tasks": str(_active_gold_tasks_file()),
            "tasks_jsonl": str(GOLD_TASKS_FILE),
            "tasks_csv": str(GOLD_TASKS_CSV_FILE),
            "draft": str(GOLD_DRAFT_FILE),
            "output": str(GOLD_OUTPUT_FILE),
        },
    }


@app.get("/api/gold/task/{row_id}")
def api_gold_task(row_id: int):
    task = _gold_task_by_row_id(row_id)
    if task is None:
        raise HTTPException(404, f"task row_id={row_id} not found")

    drafts = _load_gold_drafts()
    draft = drafts.get(row_id)

    case_output_dir = Path(task.get("case_output_dir", ""))
    overlay_path = task.get("overlay_video_path", "")
    timeline_path = task.get("timeline_chart_path", "")

    overlay_url = None
    if overlay_path:
        p = PROJECT_ROOT / overlay_path
        if p.exists():
            overlay_url = f"/output/{p.resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()}"

    source_video_url = None
    source_video = _gold_source_video_for_task(task)
    if source_video is not None:
        source_video_url = f"/api/gold/task/{row_id}/source-video"

    timeline_url = None
    if timeline_path:
        p = PROJECT_ROOT / timeline_path
        if p.exists():
            timeline_url = f"/output/{p.resolve().relative_to(OUTPUT_DIR.resolve()).as_posix()}"

    contact_sheet_url = None
    contact_sheet_path = task.get("contact_sheet_path", "")
    if contact_sheet_path:
        p = PROJECT_ROOT / str(contact_sheet_path)
        if p.exists():
            try:
                p.resolve().relative_to(PROJECT_ROOT.resolve())
                contact_sheet_url = f"/api/gold/task/{row_id}/contact-sheet"
            except ValueError:
                contact_sheet_url = None
    if contact_sheet_url is None:
        evidence_dir = _gold_evidence_dir_for_task(task)
        if evidence_dir is not None and (evidence_dir / "contact_sheet.jpg").exists():
            contact_sheet_url = f"/api/gold/task/{row_id}/contact-sheet"

    evidence = None
    event_id = task.get("suggested_event_id", "")
    case_id = task.get("case_id", "")
    if case_output_dir.exists() and event_id:
        for ve_file in ("verified_events.jsonl", "event_queries.fusion_v2.jsonl"):
            ve_path = case_output_dir / ve_file
            if ve_path.exists():
                for row in _load_jsonl_rows(ve_path):
                    rid = str(row.get("event_id") or row.get("query_id") or "")
                    if rid == event_id:
                        evidence = row
                        break
                if evidence:
                    break

    return {
        "status": "success",
        "task": task,
        "draft": draft,
        "evidence": evidence,
        "media": {
            "overlay_url": overlay_url,
            "source_video_url": source_video_url,
            "timeline_url": timeline_url,
            "contact_sheet_url": contact_sheet_url,
        },
        "missing_evidence": task.get("missing_evidence", ""),
    }


@app.get("/api/gold/task/{row_id}/contact-sheet")
def api_gold_contact_sheet(row_id: int):
    task = _gold_task_by_row_id(row_id)
    if task is None:
        raise HTTPException(404, f"task row_id={row_id} not found")

    candidates: List[Path] = []
    raw = str(task.get("contact_sheet_path") or "").strip()
    if raw:
        candidates.append((PROJECT_ROOT / raw).resolve())
    evidence_dir = _gold_evidence_dir_for_task(task)
    if evidence_dir is not None:
        candidates.append((evidence_dir / "contact_sheet.jpg").resolve())

    evidence_root = (GOLD_TASKS_DIR / "evidence").resolve()
    for candidate in candidates:
        try:
            candidate.relative_to(evidence_root)
        except ValueError:
            continue
        if candidate.exists() and candidate.is_file():
            return FileResponse(str(candidate), media_type="image/jpeg")
    raise HTTPException(404, f"contact sheet not found for row_id={row_id}")


@app.get("/api/gold/task/{row_id}/source-video")
def api_gold_source_video(row_id: int):
    task = _gold_task_by_row_id(row_id)
    if task is None:
        raise HTTPException(404, f"task row_id={row_id} not found")
    source_video = _gold_source_video_for_task(task)
    if source_video is None:
        raise HTTPException(404, f"source video not found for row_id={row_id}")
    return FileResponse(str(source_video), media_type="video/mp4")


@app.post("/api/gold/task/{row_id}/draft")
async def api_gold_save_draft(row_id: int, request: Request):
    body = await request.json()
    tasks = _load_gold_tasks()
    task = None
    for t in tasks:
        if int(t.get("row_id", -1)) == row_id:
            task = t
            break
    if task is None:
        raise HTTPException(404, f"task row_id={row_id} not found")

    drafts = _load_gold_drafts()
    draft_record = {
        "row_id": row_id,
        "case_id": task["case_id"],
        "view_code": task["view_code"],
        "suggested_event_id": task.get("suggested_event_id", ""),
        "manual_gold_track_id": body.get("manual_gold_track_id", ""),
        "manual_gold_action": body.get("manual_gold_action", ""),
        "manual_gold_start": body.get("manual_gold_start", ""),
        "manual_gold_end": body.get("manual_gold_end", ""),
        "manual_match_label": body.get("manual_match_label", ""),
        "manual_noise_type": body.get("manual_noise_type", ""),
        "manual_occlusion_level": body.get("manual_occlusion_level", ""),
        "manual_asr_quality": body.get("manual_asr_quality", ""),
        "manual_annotator": body.get("manual_annotator", ""),
        "manual_review_status": body.get("manual_review_status", "draft"),
        "manual_notes": body.get("manual_notes", ""),
    }
    drafts[row_id] = draft_record
    _save_gold_drafts(drafts)
    return {"status": "saved", "row_id": row_id}


@app.post("/api/gold/export")
def api_gold_export():
    tasks = _load_gold_tasks()
    task_map = {int(t["row_id"]): t for t in tasks if "row_id" in t}
    drafts = _load_gold_drafts()

    if not drafts:
        raise HTTPException(400, "No drafts to export. Save at least one draft first.")

    if GOLD_OUTPUT_FILE.exists():
        import time
        ts = time.strftime("%Y%m%d_%H%M%S")
        backup_path = GOLD_OUTPUT_FILE.with_suffix(f".jsonl.bak.{ts}")
        shutil.copy2(str(GOLD_OUTPUT_FILE), str(backup_path))
    else:
        backup_path = None

    events: List[Dict[str, Any]] = []
    all_errors: List[Dict[str, Any]] = []

    for row_id in sorted(drafts.keys()):
        draft = drafts[row_id]
        task = task_map.get(row_id)
        if task is None:
            all_errors.append({"row_id": row_id, "errors": [f"task not found in {_active_gold_tasks_file().name}"]})
            continue
        event = _draft_to_gold_event(task, draft)
        errs = _validate_gold_event(event)
        if errs:
            all_errors.append({"row_id": row_id, "case_id": task.get("case_id"), "errors": errs})
        events.append(event)

    if all_errors:
        raise HTTPException(
            422,
            detail=json.dumps({"validation_errors": all_errors}, ensure_ascii=False),
        )

    GOLD_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with GOLD_OUTPUT_FILE.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    labels = {}
    review_statuses = {}
    noise_types = {}
    for ev in events:
        ml = ev.get("match_label", "")
        labels[ml] = labels.get(ml, 0) + 1
        rs = ev.get("review_status", "")
        review_statuses[rs] = review_statuses.get(rs, 0) + 1
        nt = ev.get("noise_type", "")
        noise_types[nt] = noise_types.get(nt, 0) + 1

    return {
        "status": "exported",
        "count": len(events),
        "backup_path": str(backup_path) if backup_path else None,
        "summary": {
            "label_distribution": labels,
            "review_status_distribution": review_statuses,
            "noise_type_distribution": noise_types,
        },
        "source_files": {
            "output": str(GOLD_OUTPUT_FILE),
        },
    }


@app.get("/api/gold/validate")
def api_gold_validate():
    if not GOLD_OUTPUT_FILE.exists():
        raise HTTPException(404, "gold_events.real.jsonl not found. Run export first.")

    events = _load_jsonl_rows(GOLD_OUTPUT_FILE)
    errors: List[Dict[str, Any]] = []
    labels: Dict[str, int] = {}
    review_statuses: Dict[str, int] = {}
    noise_types: Dict[str, int] = {}
    occlusion_levels: Dict[str, int] = {}
    asr_qualities: Dict[str, int] = {}
    annotators: Dict[str, int] = {}

    for i, ev in enumerate(events):
        errs = _validate_gold_event(ev)
        if errs:
            errors.append({"index": i, "case_id": ev.get("case_id", "?"), "errors": errs})

        ml = ev.get("match_label", "")
        labels[ml] = labels.get(ml, 0) + 1
        rs = ev.get("review_status", "")
        review_statuses[rs] = review_statuses.get(rs, 0) + 1
        nt = ev.get("noise_type", "")
        noise_types[nt] = noise_types.get(nt, 0) + 1
        ol = ev.get("occlusion_level", "")
        occlusion_levels[ol] = occlusion_levels.get(ol, 0) + 1
        aq = ev.get("asr_quality", "")
        asr_qualities[aq] = asr_qualities.get(aq, 0) + 1
        an = ev.get("annotator", "")
        annotators[an] = annotators.get(an, 0) + 1

    quality_tiers = {"high": 0, "medium": 0, "low": 0}
    for ev in events:
        aq = ev.get("asr_quality", "not_applicable")
        ol = ev.get("occlusion_level", "none")
        if aq in ("high",) and ol in ("none", "low"):
            quality_tiers["high"] += 1
        elif aq in ("low", "empty") or ol in ("high",):
            quality_tiers["low"] += 1
        else:
            quality_tiers["medium"] += 1

    return {
        "status": "success",
        "valid": len(errors) == 0,
        "sample_size": len(events),
        "error_count": len(errors),
        "errors": errors[:50],
        "summary": {
            "label_distribution": labels,
            "review_status_distribution": review_statuses,
            "noise_type_distribution": noise_types,
            "occlusion_level_distribution": occlusion_levels,
            "asr_quality_distribution": asr_qualities,
            "annotator_distribution": annotators,
            "quality_tiers": quality_tiers,
        },
        "source_files": {
            "output": str(GOLD_OUTPUT_FILE),
            "schema": str(GOLD_SCHEMA_FILE),
        },
    }


# =========================================================
# 10) Mount frontend bundles as static and start
# =========================================================

if BUNDLE_DIR.exists():
    app.mount("/output/frontend_bundle", StaticFiles(directory=str(BUNDLE_DIR)), name="frontend_bundle")


if __name__ == "__main__":
    import uvicorn

    print("Server starting...")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"OUTPUT_DIR  : {OUTPUT_DIR}")
    print(f"DATA_DIR    : {DATA_DIR}")
    print("Open http://localhost:8000")

    uvicorn.run(app, host="0.0.0.0", port=8000)

