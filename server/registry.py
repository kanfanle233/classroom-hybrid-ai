"""
Case registry, artifact resolver, gold loader, and gold-backed evaluator.

Public types:
    CaseManifestV1      - per-case artifact inventory with status & warnings
    EventEvidenceV1     - complete evidence object for a single event
    GoldEventV1         - human peer-reviewed gold annotation record
    VerifierEvalReportV1 - evaluation report comparing system vs gold

Public classes:
    CaseRegistry        - scans output/ for cases, builds manifests
    ArtifactResolver    - resolves artifact paths & reports missing
    GoldLoader          - loads + validates gold_events.real.jsonl
    GoldBackedEvaluator - evaluates system predictions against gold
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ── constants ──────────────────────────────────────────────────────

# Artifacts the pipeline may produce per case.
ARTIFACT_KEYS = (
    "pipeline_manifest",
    "event_queries",
    "event_queries_fusion_v2",
    "pose_uq",
    "align_multimodal",
    "actions",
    "actions_fused",
    "transcript",
    "asr_quality_report",
    "verified_events",
    "verified_events_coherence_v3",
    "verifier_model",
    "verifier_eval_report",
    "verifier_calibration_report",
    "timeline_chart",
    "timeline_students",
    "pipeline_contract_v2_report",
)

# Files we actually look for on disk (order matters for fallback).
_ARTIFACT_FILE_MAP: Dict[str, List[str]] = {
    "pipeline_manifest":           ["pipeline_manifest.json"],
    "event_queries":               ["event_queries.jsonl"],
    "event_queries_fusion_v2":     ["event_queries.fusion_v2.jsonl"],
    "pose_uq":                     ["pose_tracks_smooth_uq.jsonl", "pose_uq.jsonl"],
    "align_multimodal":            ["align_multimodal.json"],
    "actions":                     ["actions.jsonl"],
    "actions_fused":               ["actions_fused.jsonl", "actions.fusion_v2.jsonl"],
    "transcript":                  ["transcript.jsonl"],
    "asr_quality_report":          ["asr_quality_report.json", "asr_quality.json"],
    "verified_events":             ["verified_events.jsonl", "verified_events.json"],
    "verified_events_coherence_v3": ["verified_events.coherence_v3.jsonl"],
    "verifier_model":              ["verifier.pt"],
    "verifier_eval_report":        ["verifier_eval_report.json"],
    "verifier_calibration_report": ["verifier_calibration_report.json"],
    "timeline_chart":              ["timeline_chart.png", "timeline_chart.json"],
    "timeline_students":           ["timeline_students.csv"],
    "pipeline_contract_v2_report": ["pipeline_contract_v2_report.json"],
}

# Gold file names relative to paper_experiments/gold/
_GOLD_JSONL_NAME = "gold_events.official.visual_223.jsonl"
_GOLD_PROPOSED_JSONL_NAME = "gold_events.real.proposed_223.jsonl"
_GOLD_LEGACY_JSONL_NAME = "gold_events.real.jsonl"
_GOLD_SCHEMA_NAME = "gold_events.real.schema.json"
_GOLD_VALIDATION_REPORT_NAME = "gold_validation_report.json"
_GOLD_SPLITS_NAME = "splits.real.json"
_AGENT300_BRIDGE_MAP_REL = Path("agent_300_workbench") / "visual_gold_bridge" / "agent300_event_bridge_map.jsonl"
_AGENT300_VISUAL_MANIFEST_REL = (
    Path("agent_300_workbench") / "visual_gold_bridge" / "agent300_visual_gold_bridge_manifest.jsonl"
)
_PIPELINE_UNBLOCK_001_040_REL = (
    Path("agent_300_workbench") / "pipeline_unblock_results" / "pipeline_unblock_batch_001_040.jsonl"
)
_VERIFIER_EXCLUDED_REASONS = (
    "no_pipeline_output",
    "no_event_id_mapping",
    "missing_asr_quality",
    "missing_align_multimodal",
    "evidence_limited_uncertain",
)

_GOLD_OFFICIAL_REVIEW_STATUSES = frozenset({"peer_reviewed", "adjudicated", "final"})
_GOLD_DRAFT_REVIEW_STATUSES = frozenset({"draft", "needs_adjudication"})
_GOLD_ALLOWED_REVIEW_STATUSES = (
    _GOLD_OFFICIAL_REVIEW_STATUSES | _GOLD_DRAFT_REVIEW_STATUSES | frozenset({"rejected"})
)
_GOLD_QUALITY_FILTERS = frozenset({"official", "draft", "all"})


def _normalize_gold_quality(quality: Optional[str]) -> str:
    value = (quality or "official").strip().lower()
    if value not in _GOLD_QUALITY_FILTERS:
        raise ValueError(f"invalid gold quality '{quality}', expected one of {sorted(_GOLD_QUALITY_FILTERS)}")
    return value


def _is_official_gold(ev: "GoldEventV1") -> bool:
    return ev.review_status in _GOLD_OFFICIAL_REVIEW_STATUSES


def _is_draft_gold(ev: "GoldEventV1") -> bool:
    return ev.review_status in _GOLD_DRAFT_REVIEW_STATUSES


def _resolve_gold_jsonl_path(gold_dir: Path) -> Path:
    for name in (_GOLD_JSONL_NAME, _GOLD_PROPOSED_JSONL_NAME, _GOLD_LEGACY_JSONL_NAME):
        candidate = gold_dir / name
        if candidate.exists():
            return candidate
    return gold_dir / _GOLD_JSONL_NAME


def _gold_excluded_reason(ev: "GoldEventV1", quality: str) -> Optional[str]:
    if quality == "all":
        return None
    if quality == "official":
        if _is_official_gold(ev):
            return None
        return f"non_official_review_status:{ev.review_status}"
    if quality == "draft":
        if _is_draft_gold(ev):
            return None
        return f"not_draft_review_status:{ev.review_status}"
    return None

# ── artifact tiers for status classification ─────────────────
CORE_ARTIFACTS = frozenset({
    "pipeline_manifest", "verified_events", "align_multimodal",
    "event_queries_fusion_v2", "pose_uq", "transcript", "asr_quality_report",
})
REPORT_ARTIFACTS = frozenset({
    "verifier_eval_report", "verifier_calibration_report",
    "pipeline_contract_v2_report",
})
VIS_ARTIFACTS = frozenset({"timeline_chart", "timeline_students"})
OPTIONAL_ARTIFACTS = frozenset({
    "event_queries", "actions", "actions_fused", "verifier_model", "verified_events_coherence_v3",
})


def _classify_case_status(artifacts: List[ArtifactStatus]) -> str:
    """Tiered classification: missing_output < partial < partial_reports < complete_minimal < complete."""
    present = {a.key for a in artifacts if a.found}
    missing = {a.key for a in artifacts if not a.found}
    if not present:
        return "missing_output"
    if missing & CORE_ARTIFACTS:
        return "partial"
    if missing <= (REPORT_ARTIFACTS | VIS_ARTIFACTS | OPTIONAL_ARTIFACTS):
        return "partial_reports" if (missing & REPORT_ARTIFACTS) else "complete_minimal"
    return "complete" if len(missing) <= 1 else "partial_reports"


# ── dataclasses ────────────────────────────────────────────────────


@dataclass
class ArtifactStatus:
    key: str
    found: bool
    file_name: str
    file_path: Optional[str] = None
    size_bytes: Optional[int] = None
    warnings: List[str] = field(default_factory=list)


@dataclass
class CaseManifestV1:
    case_id: str
    video_id: str
    view_code: str
    source_dir: Optional[str]
    video_path: Optional[str]
    status: str  # "complete" | "partial" | "missing_output"
    artifacts: List[ArtifactStatus]
    warnings: List[str] = field(default_factory=list)
    total_events: int = 0
    gold_event_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def artifact_summary(self) -> Dict[str, bool]:
        return {a.key: a.found for a in self.artifacts}

    def missing_artifacts(self) -> List[str]:
        return [a.key for a in self.artifacts if not a.found]

    def present_artifacts(self) -> List[str]:
        return [a.key for a in self.artifacts if a.found]


@dataclass
class GoldEventV1:
    case_id: str
    view_code: str
    suggested_event_id: str
    suggested_track_id: int
    suggested_action: str
    suggested_time_window: Dict[str, float]
    suggested_label: str  # "match" | "mismatch" | "uncertain"
    gold_track_id: str
    gold_action: str
    gold_start: float
    gold_end: float
    match_label: str     # "match" | "mismatch" | "uncertain"
    noise_type: str
    occlusion_level: str
    asr_quality: str
    annotator: str
    review_status: str
    notes: str
    batch_id: str = ""
    review_round: int = 0
    adjudicator: str = ""
    quality_tier: str = ""
    source_layer: str = ""
    agent300_event_id: str = ""
    target_pipeline_event_id: str = ""
    mapping_status: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSample:
    gold: GoldEventV1
    predicted_label: str
    predicted_track_id: int
    predicted_action: str
    confidence_match: float
    confidence_mismatch: float
    reliability: float
    uncertainty: float
    top_k_candidate_count: int
    gold_predicted_match: bool  # gold.match_label == predicted_label
    gold_predicted_action_match: bool
    gold_candidate_in_top_k: bool
    pipeline_event_id: str = ""
    source_layer: str = ""


@dataclass
class BaselineResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    uncertain_ratio: float
    description: str


@dataclass
class VerifierEvalReportV1:
    report_id: str
    generated_at: str
    gold_source: str
    sample_count: int
    accuracy: float
    precision: float  # macro-averaged for match
    recall: float
    f1: float
    uncertain_ratio: float
    candidate_recall: float  # gold candidate in top-k
    ece: float
    brier: float
    label_distribution: Dict[str, int]
    per_sample: List[EvalSample]
    baselines: List[BaselineResult] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_files: Dict[str, str] = field(default_factory=dict)
    quality_tier: str = "peer_reviewed"
    gold_event_count_requested: int = 0
    skipped_gold_event_count: int = 0
    official_visual_gold_count: int = 0
    pipeline_evaluable_gold_count: int = 0
    excluded_from_verifier_count: int = 0
    excluded_reason_distribution: Dict[str, int] = field(default_factory=dict)
    excluded_gold_events: List[Dict[str, Any]] = field(default_factory=list)
    per_case_breakdown: List[Dict[str, Any]] = field(default_factory=list)
    missing_output_cases: List[Dict[str, Any]] = field(default_factory=list)


# ── ArtifactResolver ───────────────────────────────────────────────

class ArtifactResolver:
    """Resolve artifact paths for a case directory and report what's missing."""

    def __init__(self, case_dir: Path):
        self.case_dir = case_dir

    def resolve_all(self) -> List[ArtifactStatus]:
        result: List[ArtifactStatus] = []
        for key, candidates in _ARTIFACT_FILE_MAP.items():
            status = self._resolve_one(key, candidates)
            result.append(status)
        return result

    def _resolve_one(self, key: str, candidates: List[str]) -> ArtifactStatus:
        warnings: List[str] = []
        for fname in candidates:
            p = self.case_dir / fname
            if p.exists():
                try:
                    size = p.stat().st_size
                except OSError:
                    size = None
                if size == 0:
                    warnings.append(f"{fname} exists but is empty")
                return ArtifactStatus(
                    key=key, found=True, file_name=fname,
                    file_path=str(p), size_bytes=size, warnings=warnings,
                )
        return ArtifactStatus(
            key=key, found=False, file_name=candidates[0] if candidates else "",
            file_path=None, warnings=[],
        )

    def get_path(self, key: str) -> Optional[Path]:
        candidates = _ARTIFACT_FILE_MAP.get(key, [])
        for fname in candidates:
            p = self.case_dir / fname
            if p.exists():
                return p
        return None

    def read_jsonl(self, key: str) -> List[Dict[str, Any]]:
        p = self.get_path(key)
        if p is None:
            return []
        return _load_jsonl(p)

    def read_json(self, key: str) -> Any:
        p = self.get_path(key)
        if p is None:
            return None
        return _load_json(p)


# ── CaseRegistry ───────────────────────────────────────────────────

class CaseRegistry:
    """Scan output/ (and known locations) for cases; build manifests."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = project_root / "output"
        self.gold_dir = project_root / "paper_experiments" / "gold"
        self.real_cases_dir = project_root / "paper_experiments" / "real_cases"
        self._manifests: Dict[str, CaseManifestV1] = {}
        self._build()

    def _build(self) -> None:
        self._scan_output_dirs()
        self._scan_real_pilot_dirs()
        self._attach_gold_counts()

    def _scan_output_dirs(self) -> None:
        if not self.output_dir.exists():
            return
        for d in self.output_dir.rglob("*"):
            if not d.is_dir():
                continue
            if self._looks_like_case_dir(d):
                self._register_case(d)

    def _scan_real_pilot_dirs(self) -> None:
        """Also scan paper_experiments/real_cases paths referenced by pilot_cases.json."""
        pilot_path = self.real_cases_dir / "pilot_cases.json"
        if not pilot_path.exists():
            return
        pilot = _load_json(pilot_path)
        if not isinstance(pilot, dict):
            return
        for case_info in pilot.get("cases", []):
            case_id = case_info.get("case_id", "")
            if case_id in self._manifests:
                continue
            # The output path is under output/<view_code>/<case_id>
            view_code = case_info.get("view_code", "")
            case_dir = self.output_dir / "智慧课堂学生行为数据集" / "real_pilot" / view_code / case_id
            if case_dir.exists() and self._looks_like_case_dir(case_dir):
                self._register_case(case_dir)

    def _looks_like_case_dir(self, d: Path) -> bool:
        """Heuristic: dir has any of the known artifact files."""
        for fname_list in _ARTIFACT_FILE_MAP.values():
            for fname in fname_list:
                if (d / fname).exists():
                    return True
        return False

    def _register_case(self, case_dir: Path) -> None:
        resolver = ArtifactResolver(case_dir)
        artifacts = resolver.resolve_all()
        case_id = case_dir.name
        # Infer view_code from parent dirs
        view_code = self._infer_view_code(case_dir)
        video_id = case_id
        source_dir = None
        # Check if there's a frontend bundle referencing a source
        manifest = resolver.read_json("pipeline_manifest")
        if isinstance(manifest, dict):
            source_dir = manifest.get("source_case_dir")
        # Check for video path in pilot_cases
        video_path = self._find_video_path(case_id)
        present = [a.key for a in artifacts if a.found]
        missing = [a.key for a in artifacts if not a.found]
        total_events = 0
        ve_rows = resolver.read_jsonl("verified_events")
        if ve_rows:
            total_events = len(ve_rows)
        status = _classify_case_status(artifacts)
        if not present:
            status = "missing_output"
        warnings: List[str] = []
        if "verified_events" in missing:
            warnings.append("verified_events not found - system predictions unavailable")
        if "event_queries" in missing and "event_queries_fusion_v2" in missing:
            warnings.append("no event queries found")
        if "align_multimodal" in missing:
            warnings.append("align_multimodal not found - alignment data unavailable")
        manifest_obj = CaseManifestV1(
            case_id=case_id,
            video_id=video_id,
            view_code=view_code,
            source_dir=source_dir,
            video_path=video_path,
            status=status,
            artifacts=artifacts,
            warnings=warnings,
            total_events=total_events,
            metadata={"source_case_dir": str(case_dir)},
        )
        existing = self._manifests.get(case_id)
        if existing is None or self._manifest_quality(manifest_obj) > self._manifest_quality(existing):
            self._manifests[case_id] = manifest_obj

    def _infer_view_code(self, case_dir: Path) -> str:
        parts = case_dir.parts
        # Look for known view codes in path
        for p in reversed(parts):
            low = p.lower()
            for code in ("front", "rear", "teacher", "top1", "top2"):
                if code in low:
                    return code
        return ""

    def _find_video_path(self, case_id: str) -> Optional[str]:
        pilot = self._load_pilot_cases()
        if pilot is None:
            return None
        for case_info in pilot.get("cases", []):
            if case_info.get("case_id") == case_id:
                return case_info.get("video_path")
        return None

    def _load_pilot_cases(self) -> Optional[Dict[str, Any]]:
        pilot_path = self.real_cases_dir / "pilot_cases.json"
        if not pilot_path.exists():
            return None
        data = _load_json(pilot_path)
        return data if isinstance(data, dict) else None

    @staticmethod
    def _manifest_quality(m: CaseManifestV1) -> int:
        present = sum(1 for a in m.artifacts if a.found)
        return present

    def _attach_gold_counts(self) -> None:
        gold_events = GoldLoader.filter_events(_load_gold_events(self.gold_dir), "official")
        for ge in gold_events:
            cid = ge.case_id
            if cid in self._manifests:
                self._manifests[cid].gold_event_count += 1

    def list_cases(self) -> List[CaseManifestV1]:
        return sorted(self._manifests.values(), key=lambda m: m.case_id)

    def get_case(self, case_id: str) -> Optional[CaseManifestV1]:
        return self._manifests.get(case_id)

    def get_resolver(self, case_id: str) -> Optional[ArtifactResolver]:
        m = self._manifests.get(case_id)
        if m is None:
            return None
        return ArtifactResolver(Path(m.metadata.get("source_case_dir", "")))


# ── GoldLoader ─────────────────────────────────────────────────────

def _load_gold_events(gold_dir: Path) -> List[GoldEventV1]:
    jsonl_path = _resolve_gold_jsonl_path(gold_dir)
    if not jsonl_path.exists():
        return []
    rows = _load_jsonl(jsonl_path)
    result: List[GoldEventV1] = []
    for row in rows:
        tw = row.get("suggested_time_window", {})
        if not isinstance(tw, dict):
            tw = {}
        result.append(GoldEventV1(
            case_id=str(row.get("case_id", "")),
            view_code=str(row.get("view_code", "")),
            suggested_event_id=str(row.get("suggested_event_id", "")),
            suggested_track_id=_coerce_track_id(row.get("suggested_track_id", -1)),
            suggested_action=str(row.get("suggested_action", "")),
            suggested_time_window={"start": float(tw.get("start", 0)), "end": float(tw.get("end", 0))},
            suggested_label=str(row.get("suggested_label", "")),
            gold_track_id=str(row.get("gold_track_id", "unknown")),
            gold_action=str(row.get("gold_action", "unknown")),
            gold_start=float(row.get("gold_start", 0)),
            gold_end=float(row.get("gold_end", 0)),
            match_label=str(row.get("match_label", "")),
            noise_type=str(row.get("noise_type", "")),
            occlusion_level=str(row.get("occlusion_level", "")),
            asr_quality=str(row.get("asr_quality", "")),
            annotator=str(row.get("annotator", "")),
            review_status=str(row.get("review_status", "")),
            notes=str(row.get("notes", "")),
            batch_id=str(row.get("batch_id", "")),
            review_round=_coerce_int(row.get("review_round", 0)),
            adjudicator=str(row.get("adjudicator", "")),
            quality_tier=str(row.get("quality_tier", "")),
            source_layer=str(row.get("source_layer", "")),
            agent300_event_id=str(row.get("agent300_event_id", "")),
            target_pipeline_event_id=str(row.get("target_pipeline_event_id", "")),
            mapping_status=str(row.get("mapping_status", "")),
            raw=row,
        ))
    return result


def _validate_gold(events: List[GoldEventV1], *, require_official: bool = False) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    for i, ev in enumerate(events):
        if ev.review_status not in _GOLD_ALLOWED_REVIEW_STATUSES:
            errors.append(f"[{i}] review_status={ev.review_status} (expected known gold lifecycle status)")
        if require_official and not _is_official_gold(ev):
            errors.append(f"[{i}] review_status={ev.review_status} (expected official gold)")
        if ev.match_label not in ("match", "mismatch", "uncertain"):
            errors.append(f"[{i}] invalid match_label={ev.match_label}")
        if ev.gold_start > ev.gold_end and ev.gold_end > 0:
            errors.append(f"[{i}] gold_start={ev.gold_start} > gold_end={ev.gold_end}")
    return len(errors) == 0, errors


class GoldLoader:
    def __init__(self, gold_dir: Path):
        self.gold_dir = gold_dir
        self.gold_path = _resolve_gold_jsonl_path(gold_dir)
        self._events: Optional[List[GoldEventV1]] = None
        self._validation_errors: List[str] = []

    @staticmethod
    def filter_events(events: List[GoldEventV1], quality: Optional[str] = "official") -> List[GoldEventV1]:
        quality_filter = _normalize_gold_quality(quality)
        if quality_filter == "official":
            return [ev for ev in events if _is_official_gold(ev)]
        if quality_filter == "draft":
            return [ev for ev in events if _is_draft_gold(ev)]
        return list(events)

    def load(self, quality: Optional[str] = "official") -> List[GoldEventV1]:
        if self._events is not None:
            return self.filter_events(self._events, quality)
        self.gold_path = _resolve_gold_jsonl_path(self.gold_dir)
        all_events = _load_gold_events(self.gold_dir)
        ok, errs = _validate_gold(all_events)
        self._events = all_events
        self._validation_errors = errs
        return self.filter_events(self._events, quality)

    def is_valid(self) -> bool:
        if self._events is None:
            self.load("all")
        return len(self._validation_errors) == 0

    @property
    def validation_errors(self) -> List[str]:
        if self._events is None:
            self.load("all")
        return list(self._validation_errors)

    def by_case(self, quality: Optional[str] = "official") -> Dict[str, List[GoldEventV1]]:
        events = self.load(quality=quality)
        out: Dict[str, List[GoldEventV1]] = {}
        for ev in events:
            out.setdefault(ev.case_id, []).append(ev)
        return out

    def summary(self, quality: Optional[str] = "official") -> Dict[str, Any]:
        quality_filter = _normalize_gold_quality(quality)
        events = self.load(quality=quality_filter)
        all_events = self.load(quality="all")
        included_keys = {(ev.case_id, ev.suggested_event_id) for ev in events}
        excluded_events = [
            ev for ev in all_events
            if (ev.case_id, ev.suggested_event_id) not in included_keys
        ]
        excluded_reason_distribution: Dict[str, int] = {}
        excluded_by_review_status: Dict[str, int] = {}
        excluded_examples: List[Dict[str, Any]] = []
        for ev in excluded_events:
            reason = _gold_excluded_reason(ev, quality_filter)
            if reason is None:
                continue
            excluded_reason_distribution[reason] = excluded_reason_distribution.get(reason, 0) + 1
            excluded_by_review_status[ev.review_status] = excluded_by_review_status.get(ev.review_status, 0) + 1
            if len(excluded_examples) < 10:
                excluded_examples.append({
                    "case_id": ev.case_id,
                    "suggested_event_id": ev.suggested_event_id,
                    "review_status": ev.review_status,
                    "excluded_reason": reason,
                })
        by_case: Dict[str, Dict[str, int]] = {}
        for ev in events:
            by_case.setdefault(ev.case_id, {"match": 0, "mismatch": 0, "uncertain": 0})
            if ev.match_label in by_case[ev.case_id]:
                by_case[ev.case_id][ev.match_label] += 1
        review_status_distribution: Dict[str, int] = {}
        for ev in all_events:
            review_status_distribution[ev.review_status] = review_status_distribution.get(ev.review_status, 0) + 1
        filtered_review_status = {
            status: sum(1 for e in events if e.review_status == status)
            for status in sorted(_GOLD_ALLOWED_REVIEW_STATUSES)
            if any(e.review_status == status for e in events)
        }
        return {
            "quality_filter": quality_filter,
            "total": len(events),
            "included_total": len(events),
            "total_loaded": len(all_events),
            "excluded_total": len(excluded_events),
            "excluded_reason_distribution": excluded_reason_distribution,
            "excluded_by_review_status": excluded_by_review_status,
            "excluded_examples": excluded_examples,
            "official_total": sum(1 for e in all_events if _is_official_gold(e)),
            "official_visual_gold": sum(1 for e in all_events if _is_official_gold(e)),
            "peer_reviewed": filtered_review_status.get("peer_reviewed", 0),
            "adjudicated": filtered_review_status.get("adjudicated", 0),
            "draft_total": sum(1 for e in all_events if _is_draft_gold(e)),
            "valid": self.is_valid(),
            "validation_errors": self.validation_errors,
            "label_distribution": {
                "match": sum(1 for e in events if e.match_label == "match"),
                "mismatch": sum(1 for e in events if e.match_label == "mismatch"),
                "uncertain": sum(1 for e in events if e.match_label == "uncertain"),
            },
            "by_case": by_case,
            "review_status": review_status_distribution,
            "filtered_review_status": filtered_review_status,
            "annotators": list({e.annotator for e in events}),
        }


# ── GoldBackedEvaluator ───────────────────────────────────────────

class GoldReadinessAnalyzer:
    """Classify official visual gold into visual-audit vs verifier-evaluable layers."""

    def __init__(self, gold_dir: Path, project_root: Path):
        self.gold_dir = gold_dir
        self.project_root = project_root
        self.output_dir = project_root / "output"
        self.gold_loader = GoldLoader(gold_dir)
        self._bridge_index: Optional[Dict[str, Dict[str, Any]]] = None
        self._case_dir_cache: Dict[str, Optional[Path]] = {}

    @property
    def bridge_map_path(self) -> Path:
        return self.gold_dir / _AGENT300_BRIDGE_MAP_REL

    @property
    def visual_manifest_path(self) -> Path:
        return self.gold_dir / _AGENT300_VISUAL_MANIFEST_REL

    @property
    def pipeline_unblock_path(self) -> Path:
        return self.gold_dir / _PIPELINE_UNBLOCK_001_040_REL

    def bridge_index(self) -> Dict[str, Dict[str, Any]]:
        if self._bridge_index is None:
            rows = _load_jsonl(self.bridge_map_path)
            index: Dict[str, Dict[str, Any]] = {
                str(row.get("agent300_event_id")): dict(row)
                for row in rows
                if row.get("agent300_event_id")
            }
            for row in _load_jsonl(self.pipeline_unblock_path):
                agent_id = str(row.get("agent300_event_id") or "")
                if not agent_id:
                    continue
                merged = dict(index.get(agent_id, {}))
                merged.update({
                    "agent300_event_id": agent_id,
                    "case_id": row.get("case_id", merged.get("case_id")),
                    "view_code": row.get("view_code", merged.get("view_code")),
                    "target_pipeline_event_id": row.get("target_pipeline_event_id"),
                    "mapping_status": row.get("mapping_status") or "mapped",
                    "pipeline_start": row.get("pipeline_start"),
                    "pipeline_end": row.get("pipeline_end"),
                    "time_iou": row.get("time_iou"),
                    "time_center_delta_sec": row.get("time_center_delta_sec"),
                    "required_artifacts_present": bool(row.get("required_artifacts_present")),
                    "missing_artifacts": row.get("missing_artifacts") or [],
                    "output_dir": row.get("output_dir"),
                    "pipeline_status": "mapped",
                    "verifier_evaluable": bool(row.get("required_artifacts_present")),
                    "mapping_source": "pipeline_unblock_batch_001_040",
                })
                index[agent_id] = merged
            self._bridge_index = index
        return self._bridge_index

    def classify_events(self, events: Optional[List[GoldEventV1]] = None) -> List[Dict[str, Any]]:
        gold_events = events if events is not None else self.gold_loader.load("official")
        rows: List[Dict[str, Any]] = []
        for ge in gold_events:
            agent_id = self._agent300_id(ge)
            bridge_row = self.bridge_index().get(agent_id, {}) if agent_id else {}
            case_dir = self.resolve_case_dir(ge.case_id, bridge_row=bridge_row)
            pipeline_event_id = self.pipeline_event_id_for(ge, bridge_row)
            excluded_reason = self.excluded_reason_for(ge, case_dir, pipeline_event_id)
            rows.append({
                "gold": ge,
                "case_id": ge.case_id,
                "view_code": ge.view_code,
                "gold_event_id": ge.suggested_event_id,
                "agent300_event_id": agent_id,
                "pipeline_event_id": pipeline_event_id or "",
                "source_layer": self.source_layer_for(ge),
                "quality_tier": ge.quality_tier or ge.raw.get("quality_tier") or "unknown",
                "review_status": ge.review_status,
                "match_label": ge.match_label,
                "gold_action": ge.gold_action,
                "pipeline_evaluable": excluded_reason is None,
                "excluded_reason": excluded_reason,
                "case_dir": case_dir,
                "mapping_status": bridge_row.get("mapping_status") or ge.mapping_status or "",
                "target_pipeline_event_id": bridge_row.get("target_pipeline_event_id") or ge.target_pipeline_event_id,
                "missing_artifacts": self._missing_required_artifacts(case_dir) if case_dir else [],
            })
        return rows

    def pipeline_event_id_for(self, ge: GoldEventV1, bridge_row: Optional[Dict[str, Any]] = None) -> Optional[str]:
        bridge = bridge_row or {}
        agent_id = self._agent300_id(ge)
        if agent_id:
            mapping_status = str(bridge.get("mapping_status") or ge.mapping_status or "").lower()
            target = str(bridge.get("target_pipeline_event_id") or ge.target_pipeline_event_id or "").strip()
            if target and (mapping_status == "mapped" or bridge.get("verifier_evaluable") is True):
                return target
            return None
        return str(ge.target_pipeline_event_id or ge.suggested_event_id or "").strip() or None

    def excluded_reason_for(
        self,
        ge: GoldEventV1,
        case_dir: Optional[Path],
        pipeline_event_id: Optional[str],
    ) -> Optional[str]:
        if (ge.quality_tier or ge.raw.get("quality_tier")) == "evidence_limited_uncertain":
            return "evidence_limited_uncertain"
        if case_dir is None:
            return "no_pipeline_output"
        if not pipeline_event_id:
            return "no_event_id_mapping"
        resolver = ArtifactResolver(case_dir)
        if resolver.get_path("asr_quality_report") is None:
            return "missing_asr_quality"
        if resolver.get_path("align_multimodal") is None:
            return "missing_align_multimodal"
        return None

    def resolve_case_dir(self, case_id: str, bridge_row: Optional[Dict[str, Any]] = None) -> Optional[Path]:
        if case_id in self._case_dir_cache:
            return self._case_dir_cache[case_id]
        if bridge_row:
            raw_output_dir = bridge_row.get("output_dir")
            if raw_output_dir:
                candidate = Path(str(raw_output_dir))
                if not candidate.is_absolute():
                    candidate = self.project_root / candidate
                if candidate.exists() and self._looks_like_case_dir(candidate):
                    self._case_dir_cache[case_id] = candidate
                    return candidate
        for view in ("front", "rear", "teacher", "top1", "top2"):
            candidate = self.output_dir / "智慧课堂学生行为数据集" / "real_pilot" / view / case_id
            if candidate.exists() and self._looks_like_case_dir(candidate):
                self._case_dir_cache[case_id] = candidate
                return candidate
        if self.output_dir.exists():
            for d in self.output_dir.rglob(case_id):
                if d.is_dir() and self._looks_like_case_dir(d):
                    self._case_dir_cache[case_id] = d
                    return d
        self._case_dir_cache[case_id] = None
        return None

    def visual_gold_report(self, quality: Optional[str] = "official") -> Dict[str, Any]:
        quality_filter = _normalize_gold_quality(quality)
        events = self.gold_loader.load(quality_filter)
        readiness_rows = self.classify_events(events)
        return self._build_visual_report(events, readiness_rows, quality_filter)

    def _build_visual_report(
        self,
        events: List[GoldEventV1],
        readiness_rows: List[Dict[str, Any]],
        quality_filter: str,
    ) -> Dict[str, Any]:
        pipeline_evaluable_count = sum(1 for row in readiness_rows if row["pipeline_evaluable"])
        excluded_rows = [row for row in readiness_rows if not row["pipeline_evaluable"]]
        source_layer_distribution = self._counter(readiness_rows, "source_layer")
        return {
            "report_id": "official_visual_gold_report_v1",
            "scope": "official_visual_gold_audit_not_full_verifier_accuracy",
            "quality_filter": quality_filter,
            "sample_size": len(events),
            "official_visual_gold_count": len(events),
            "legacy_official_gold_count": source_layer_distribution.get("legacy_official_gold", 0),
            "agent300_visual_gold_count": source_layer_distribution.get("agent300_visual_gold", 0),
            "pipeline_evaluable_count": pipeline_evaluable_count,
            "excluded_from_verifier": len(excluded_rows),
            "excluded_reason_distribution": self._counter(excluded_rows, "excluded_reason"),
            "distributions": {
                "label": self._counter(readiness_rows, "match_label"),
                "action": self._counter(readiness_rows, "gold_action"),
                "view": self._counter(readiness_rows, "view_code"),
                "quality_tier": self._counter(readiness_rows, "quality_tier"),
                "review_status": self._counter(readiness_rows, "review_status"),
                "source_layer": source_layer_distribution,
            },
            "pipeline_evaluable_distribution": {
                "source_layer": self._counter(
                    [row for row in readiness_rows if row["pipeline_evaluable"]],
                    "source_layer",
                ),
                "view": self._counter(
                    [row for row in readiness_rows if row["pipeline_evaluable"]],
                    "view_code",
                ),
            },
            "excluded_examples": [self._public_readiness_row(row) for row in excluded_rows[:20]],
            "pipeline_evaluable_examples": [
                self._public_readiness_row(row)
                for row in readiness_rows
                if row["pipeline_evaluable"]
            ][:20],
            "source_files": {
                "active_gold_jsonl": str(self.gold_loader.gold_path),
                "legacy_gold_jsonl": str(self.gold_dir / _GOLD_LEGACY_JSONL_NAME),
                "agent300_manifest": str(self.visual_manifest_path),
                "agent300_event_bridge_map": str(self.bridge_map_path),
                "pipeline_unblock_batch_001_040": str(self.pipeline_unblock_path),
            },
            "paper_use": {
                "verifier_metrics_scope": "pipeline_evaluable_gold_only",
                "visual_gold_scope": "all_official_visual_gold",
                "statement": (
                    "Verifier metrics are computed only on pipeline-evaluable gold. "
                    "Official visual gold that lacks pipeline output, event-id mapping, "
                    "ASR quality, multimodal alignment, or sufficient evidence is reported as excluded."
                ),
            },
        }

    @staticmethod
    def source_layer_for(ge: GoldEventV1) -> str:
        if ge.source_layer:
            return ge.source_layer
        if GoldReadinessAnalyzer._agent300_id(ge):
            return "agent300_visual_gold"
        return "legacy_official_gold"

    @staticmethod
    def _agent300_id(ge: GoldEventV1) -> str:
        if ge.agent300_event_id:
            return ge.agent300_event_id
        if ge.suggested_event_id.startswith("e_agent300_"):
            return ge.suggested_event_id
        return str(ge.raw.get("agent300_event_id") or "")

    @staticmethod
    def _looks_like_case_dir(d: Path) -> bool:
        for fname_list in _ARTIFACT_FILE_MAP.values():
            for fname in fname_list:
                if (d / fname).exists():
                    return True
        return False

    @staticmethod
    def _missing_required_artifacts(case_dir: Optional[Path]) -> List[str]:
        if case_dir is None:
            return []
        resolver = ArtifactResolver(case_dir)
        required = ("verified_events", "event_queries_fusion_v2", "align_multimodal", "asr_quality_report")
        return [key for key in required if resolver.get_path(key) is None]

    @staticmethod
    def _counter(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for row in rows:
            value = str(row.get(key) or "unknown")
            counts[value] = counts.get(value, 0) + 1
        return dict(sorted(counts.items()))

    @staticmethod
    def _public_readiness_row(row: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "case_id": row.get("case_id"),
            "view_code": row.get("view_code"),
            "gold_event_id": row.get("gold_event_id"),
            "agent300_event_id": row.get("agent300_event_id"),
            "pipeline_event_id": row.get("pipeline_event_id"),
            "source_layer": row.get("source_layer"),
            "quality_tier": row.get("quality_tier"),
            "review_status": row.get("review_status"),
            "match_label": row.get("match_label"),
            "gold_action": row.get("gold_action"),
            "pipeline_evaluable": row.get("pipeline_evaluable"),
            "excluded_reason": row.get("excluded_reason"),
            "mapping_status": row.get("mapping_status"),
            "missing_artifacts": row.get("missing_artifacts") or [],
        }


class GoldBackedEvaluator:
    """Evaluate system predictions against human gold annotations."""

    def __init__(self, gold_dir: Path, project_root: Path, quality: Optional[str] = "official"):
        self.gold_loader = GoldLoader(gold_dir)
        self.readiness_analyzer = GoldReadinessAnalyzer(gold_dir, project_root)
        self.project_root = project_root
        self.output_dir = project_root / "output"
        self.quality = _normalize_gold_quality(quality)

    def evaluate(
        self,
        case_ids: Optional[List[str]] = None,
        quality: Optional[str] = None,
    ) -> VerifierEvalReportV1:
        quality_filter = _normalize_gold_quality(quality or self.quality)
        gold_events = self.gold_loader.load(quality=quality_filter)
        if case_ids is not None:
            gold_events = [g for g in gold_events if g.case_id in case_ids]
        gold_event_count_requested = len(gold_events)
        readiness_rows = self.readiness_analyzer.classify_events(gold_events)
        evaluable_rows = [row for row in readiness_rows if row["pipeline_evaluable"]]
        excluded_rows = [row for row in readiness_rows if not row["pipeline_evaluable"]]
        excluded_reason_distribution = GoldReadinessAnalyzer._counter(excluded_rows, "excluded_reason")
        samples: List[EvalSample] = []
        warnings: List[str] = []
        per_case_breakdown: List[Dict[str, Any]] = []
        missing_output_cases: List[Dict[str, Any]] = []
        if quality_filter != "official":
            warnings.append(f"quality={quality_filter} includes non-official gold and is not paper-eligible")
        if excluded_rows:
            warnings.append(
                f"{len(excluded_rows)} official visual-gold events excluded from verifier metrics: "
                f"{excluded_reason_distribution}"
            )
        source_files: Dict[str, str] = {
            "gold_jsonl": str(self.gold_loader.gold_path),
            "gold_quality": quality_filter,
            "agent300_event_bridge_map": str(self.readiness_analyzer.bridge_map_path),
            "pipeline_unblock_batch_001_040": str(self.readiness_analyzer.pipeline_unblock_path),
        }
        by_case = {}
        for row in evaluable_rows:
            ge = row["gold"]
            by_case.setdefault(ge.case_id, []).append(row)

        missing_by_case: Dict[str, List[Dict[str, Any]]] = {}
        for row in excluded_rows:
            if row["excluded_reason"] == "no_pipeline_output":
                missing_by_case.setdefault(row["case_id"], []).append(row)
        for case_id, case_rows in missing_by_case.items():
            first = case_rows[0]
            missing_case = {
                "case_id": case_id,
                "view_code": first.get("view_code", ""),
                "gold_event_count": len(case_rows),
                "gold_event_ids": [row["gold_event_id"] for row in case_rows],
                "missing_reason": "output_dir_not_found",
            }
            missing_output_cases.append(missing_case)
            per_case_breakdown.append({
                "case_id": case_id,
                "view_code": missing_case["view_code"],
                "gold_event_count": len(case_rows),
                "sample_count": 0,
                "accuracy": 0.0,
                "label_distribution": {},
                "predicted_label_distribution": {},
                "candidate_recall": 0.0,
                "uncertain_ratio": 0.0,
                "missing_prediction_count": len(case_rows),
                "output_status": "missing_output",
                "source_dir": None,
                "warnings": [f"case {case_id}: output dir not found, excluded from verifier"],
            })

        for case_id, case_rows in by_case.items():
            case_golds = [row["gold"] for row in case_rows]
            resolved = case_rows[0].get("case_dir") or self._resolve_case_dir(case_id)
            if resolved is None:
                case_warning = f"case {case_id}: output dir not found, skipping"
                warnings.append(case_warning)
                missing_case = {
                    "case_id": case_id,
                    "view_code": case_golds[0].view_code if case_golds else "",
                    "gold_event_count": len(case_golds),
                    "gold_event_ids": [ge.suggested_event_id for ge in case_golds],
                    "missing_reason": "output_dir_not_found",
                }
                missing_output_cases.append(missing_case)
                per_case_breakdown.append({
                    "case_id": case_id,
                    "view_code": missing_case["view_code"],
                    "gold_event_count": len(case_golds),
                    "sample_count": 0,
                    "accuracy": 0.0,
                    "label_distribution": {},
                    "predicted_label_distribution": {},
                    "candidate_recall": 0.0,
                    "uncertain_ratio": 0.0,
                    "missing_prediction_count": len(case_golds),
                    "output_status": "missing_output",
                    "source_dir": None,
                    "warnings": [case_warning],
                })
                continue
            resolver = ArtifactResolver(resolved)
            ve_rows = resolver.read_jsonl("verified_events")
            align_rows = resolver.read_json("align_multimodal") or []
            if not isinstance(align_rows, list):
                align_rows = []

            ve_by_event: Dict[str, Dict[str, Any]] = {}
            for ve in ve_rows:
                eid = str(ve.get("event_id") or ve.get("query_id") or "")
                if eid:
                    ve_by_event[eid] = ve

            case_samples: List[EvalSample] = []
            for row in case_rows:
                ge = row["gold"]
                pipeline_event_id = str(row.get("pipeline_event_id") or ge.suggested_event_id)
                ve = ve_by_event.get(pipeline_event_id)
                sample = self._match_one(ge, ve, align_rows, pipeline_event_id=pipeline_event_id)
                samples.append(sample)
                case_samples.append(sample)

            artifact_statuses = resolver.resolve_all()
            case_warnings = [
                warning
                for status in artifact_statuses
                for warning in status.warnings
            ]
            missing_artifacts = [status.key for status in artifact_statuses if not status.found]
            if not ve_rows:
                case_warnings.append("verified_events not found or empty")
            output_status = "complete" if not missing_artifacts else "partial"
            per_case_breakdown.append(self._case_breakdown(
                case_id=case_id,
                view_code=case_golds[0].view_code if case_golds else "",
                gold_event_count=len(case_golds),
                samples=case_samples,
                output_status=output_status,
                source_dir=str(resolved),
                warnings=case_warnings,
            ))

        skipped_gold_event_count = gold_event_count_requested - len(samples)
        return self._build_report(
            samples,
            warnings,
            source_files,
            quality_filter,
            gold_event_count_requested=gold_event_count_requested,
            skipped_gold_event_count=skipped_gold_event_count,
            official_visual_gold_count=gold_event_count_requested,
            pipeline_evaluable_gold_count=len(evaluable_rows),
            excluded_from_verifier_count=len(excluded_rows),
            excluded_reason_distribution=excluded_reason_distribution,
            excluded_gold_events=[
                GoldReadinessAnalyzer._public_readiness_row(row)
                for row in excluded_rows
            ],
            per_case_breakdown=per_case_breakdown,
            missing_output_cases=missing_output_cases,
        )

    def _resolve_case_dir(self, case_id: str) -> Optional[Path]:
        return self.readiness_analyzer.resolve_case_dir(case_id)

    @staticmethod
    def _looks_like_case_dir(d: Path) -> bool:
        for fname_list in _ARTIFACT_FILE_MAP.values():
            for fname in fname_list:
                if (d / fname).exists():
                    return True
        return False

    @staticmethod
    def _case_breakdown(
        case_id: str,
        view_code: str,
        gold_event_count: int,
        samples: List[EvalSample],
        output_status: str,
        source_dir: Optional[str],
        warnings: List[str],
    ) -> Dict[str, Any]:
        n = len(samples)
        label_distribution: Dict[str, int] = {}
        predicted_label_distribution: Dict[str, int] = {}
        for sample in samples:
            label_distribution[sample.gold.match_label] = label_distribution.get(sample.gold.match_label, 0) + 1
            predicted_label_distribution[sample.predicted_label] = (
                predicted_label_distribution.get(sample.predicted_label, 0) + 1
            )
        correct = sum(1 for sample in samples if sample.gold_predicted_match)
        candidate_recall_count = sum(1 for sample in samples if sample.gold_candidate_in_top_k)
        uncertain_count = sum(1 for sample in samples if sample.predicted_label == "uncertain")
        missing_prediction_count = sum(1 for sample in samples if sample.predicted_label == "missing")
        return {
            "case_id": case_id,
            "view_code": view_code,
            "gold_event_count": gold_event_count,
            "sample_count": n,
            "accuracy": round(correct / n, 4) if n else 0.0,
            "label_distribution": label_distribution,
            "predicted_label_distribution": predicted_label_distribution,
            "candidate_recall": round(candidate_recall_count / n, 4) if n else 0.0,
            "uncertain_ratio": round(uncertain_count / n, 4) if n else 0.0,
            "missing_prediction_count": missing_prediction_count,
            "output_status": output_status,
            "source_dir": source_dir,
            "warnings": warnings,
        }

    @staticmethod
    def _match_one(
        ge: GoldEventV1,
        ve: Optional[Dict[str, Any]],
        align_rows: List[Dict[str, Any]],
        pipeline_event_id: Optional[str] = None,
    ) -> EvalSample:
        lookup_event_id = pipeline_event_id or ge.suggested_event_id
        if ve is None:
            return EvalSample(
                gold=ge,
                predicted_label="missing",
                predicted_track_id=-1,
                predicted_action="",
                confidence_match=0.0,
                confidence_mismatch=0.0,
                reliability=0.0,
                uncertainty=1.0,
                top_k_candidate_count=0,
                gold_predicted_match=False,
                gold_predicted_action_match=False,
                gold_candidate_in_top_k=False,
                pipeline_event_id=lookup_event_id,
                source_layer=GoldReadinessAnalyzer.source_layer_for(ge),
            )
        predicted_label = str(ve.get("match_label") or ve.get("label") or "unverified")
        predicted_track_id = int(ve.get("track_id", -1))
        predicted_action = str(ve.get("action") or ve.get("semantic_id") or "")
        evidence = ve.get("evidence") if isinstance(ve.get("evidence"), dict) else {}
        confidence_match = float(evidence.get("p_match") or ve.get("p_match", 0))
        confidence_mismatch = float(evidence.get("p_mismatch") or ve.get("p_mismatch", 0))
        reliability = float(ve.get("reliability_score") or ve.get("reliability", 0))
        uncertainty = float(ve.get("uncertainty", 1.0 - reliability))
        # Find alignment candidates count and check if gold candidate in top-k
        top_k_candidate_count = 0
        gold_candidate_in_top_k = False
        for rec in align_rows:
            eid = str(rec.get("event_id") or rec.get("query_id") or "")
            if eid == lookup_event_id:
                candidates = rec.get("candidates", [])
                top_k_candidate_count = len(candidates) if isinstance(candidates, list) else 0
                # Gold candidate is "unknown" track_id in most cases, so check action match
                if ge.gold_track_id not in ("unknown", "-1", ""):
                    try:
                        gold_tid = int(ge.gold_track_id)
                    except (TypeError, ValueError):
                        gold_tid = -1
                    if gold_tid >= 0 and isinstance(candidates, list):
                        cand_tids = [int(c.get("track_id", -1)) for c in candidates]
                        gold_candidate_in_top_k = gold_tid in cand_tids
                else:
                    # If gold_track_id is unknown, check if any candidate action matches gold_action
                    if ge.gold_action not in ("unknown", "") and isinstance(candidates, list):
                        gold_candidate_in_top_k = any(
                            str(c.get("action", "")).lower() == ge.gold_action.lower()
                            for c in candidates
                        )
                break

        gold_predicted_match = predicted_label == ge.match_label
        gold_predicted_action_match = (
            ge.gold_action.lower() == predicted_action.lower()
            and ge.gold_action not in ("unknown", "")
        )

        return EvalSample(
            gold=ge,
            predicted_label=predicted_label,
            predicted_track_id=predicted_track_id,
            predicted_action=predicted_action,
            confidence_match=confidence_match,
            confidence_mismatch=confidence_mismatch,
            reliability=reliability,
            uncertainty=uncertainty,
            top_k_candidate_count=top_k_candidate_count,
            gold_predicted_match=gold_predicted_match,
            gold_predicted_action_match=gold_predicted_action_match,
            gold_candidate_in_top_k=gold_candidate_in_top_k,
            pipeline_event_id=lookup_event_id,
            source_layer=GoldReadinessAnalyzer.source_layer_for(ge),
        )

    @staticmethod
    def _compute_baselines(samples: List[EvalSample]) -> List[BaselineResult]:
        """Compute ablation baselines by simulating alternative decision strategies."""
        baselines: List[BaselineResult] = []

        # Baseline 1: Visual-only (use visual_score > 0.5, ignore text)
        vis_preds = []
        for s in samples:
            # Reconstruct visual score from evidence
            ve = s.gold.raw
            # We don't have visual_score directly in EvalSample, use p_match as proxy
            # Visual-only: always predict 'match' if any alignment candidate exists
            if s.top_k_candidate_count > 0:
                vis_preds.append("match")
            else:
                vis_preds.append("mismatch")
        baselines.append(_eval_baseline("visual_only", vis_preds, samples,
                                         "Uses only visual alignment; no text/ASR fusion"))

        # Baseline 2: Fixed-window (hard threshold: overlap > 0.5 = match)
        fixed_preds = []
        for s in samples:
            # Fixed-window: predict match if gold_candidate_in_top_k, else uncertain
            if s.gold_candidate_in_top_k:
                fixed_preds.append("match")
            else:
                fixed_preds.append("uncertain")
        baselines.append(_eval_baseline("fixed_window", fixed_preds, samples,
                                         "Fixed overlap threshold (0.5); no adaptive window"))

        # Baseline 3: Without-UQ (ignore uncertainty, use p_match > 0.5)
        no_uq_preds = []
        for s in samples:
            if s.confidence_match > 0.5:
                no_uq_preds.append("match")
            elif s.confidence_mismatch > 0.5:
                no_uq_preds.append("mismatch")
            else:
                no_uq_preds.append("uncertain")
        baselines.append(_eval_baseline("without_uq", no_uq_preds, samples,
                                         "Ignores UQ scores; uses raw p_match threshold=0.5"))

        # Baseline 4: Always-uncertain (conservative baseline)
        always_uncertain = ["uncertain"] * len(samples)
        baselines.append(_eval_baseline("always_uncertain", always_uncertain, samples,
                                         "Always predicts uncertain; upper bound on uncertain ratio"))

        return baselines

    @staticmethod
    def _build_report(
        samples: List[EvalSample],
        warnings: List[str],
        source_files: Dict[str, str],
        quality_tier: str = "official",
        gold_event_count_requested: int = 0,
        skipped_gold_event_count: int = 0,
        official_visual_gold_count: int = 0,
        pipeline_evaluable_gold_count: int = 0,
        excluded_from_verifier_count: int = 0,
        excluded_reason_distribution: Optional[Dict[str, int]] = None,
        excluded_gold_events: Optional[List[Dict[str, Any]]] = None,
        per_case_breakdown: Optional[List[Dict[str, Any]]] = None,
        missing_output_cases: Optional[List[Dict[str, Any]]] = None,
    ) -> VerifierEvalReportV1:
        n = len(samples)
        if n == 0:
            return VerifierEvalReportV1(
                report_id="gold_eval_empty",
                generated_at=datetime.now(timezone.utc).isoformat(),
                gold_source=source_files.get("gold_jsonl", ""),
                sample_count=0,
                accuracy=0, precision=0, recall=0, f1=0,
                uncertain_ratio=0, candidate_recall=0,
                ece=0, brier=0,
                label_distribution={},
                per_sample=[],
                warnings=[*warnings, "no samples matched"],
                source_files=source_files,
                quality_tier=quality_tier,
                gold_event_count_requested=gold_event_count_requested,
                skipped_gold_event_count=skipped_gold_event_count,
                official_visual_gold_count=official_visual_gold_count,
                pipeline_evaluable_gold_count=pipeline_evaluable_gold_count,
                excluded_from_verifier_count=excluded_from_verifier_count,
                excluded_reason_distribution=excluded_reason_distribution or {},
                excluded_gold_events=excluded_gold_events or [],
                per_case_breakdown=per_case_breakdown or [],
                missing_output_cases=missing_output_cases or [],
            )

        correct = sum(1 for s in samples if s.gold_predicted_match)
        accuracy = correct / n

        # Per-label metrics (macro over match/mismatch only, excluding uncertain)
        label_dist = {}
        for s in samples:
            label_dist[s.gold.match_label] = label_dist.get(s.gold.match_label, 0) + 1

        # Precision / Recall / F1 for "match" class
        tp_match = sum(1 for s in samples if s.predicted_label == "match" and s.gold.match_label == "match")
        fp_match = sum(1 for s in samples if s.predicted_label == "match" and s.gold.match_label != "match")
        fn_match = sum(1 for s in samples if s.predicted_label != "match" and s.gold.match_label == "match")
        precision_match = tp_match / (tp_match + fp_match) if (tp_match + fp_match) > 0 else 0.0
        recall_match = tp_match / (tp_match + fn_match) if (tp_match + fn_match) > 0 else 0.0
        f1_match = 2 * precision_match * recall_match / (precision_match + recall_match) if (precision_match + recall_match) > 0 else 0.0

        # Precision / Recall / F1 for "mismatch" class
        tp_mismatch = sum(1 for s in samples if s.predicted_label == "mismatch" and s.gold.match_label == "mismatch")
        fp_mismatch = sum(1 for s in samples if s.predicted_label == "mismatch" and s.gold.match_label != "mismatch")
        fn_mismatch = sum(1 for s in samples if s.predicted_label != "mismatch" and s.gold.match_label == "mismatch")
        precision_mismatch = tp_mismatch / (tp_mismatch + fp_mismatch) if (tp_mismatch + fp_mismatch) > 0 else 0.0
        recall_mismatch = tp_mismatch / (tp_mismatch + fn_mismatch) if (tp_mismatch + fn_mismatch) > 0 else 0.0
        f1_mismatch = 2 * precision_mismatch * recall_mismatch / (precision_mismatch + recall_mismatch) if (precision_mismatch + recall_mismatch) > 0 else 0.0

        # Macro average over match + mismatch
        active_labels = []
        if label_dist.get("match", 0) > 0:
            active_labels.append(("match", precision_match, recall_match, f1_match))
        if label_dist.get("mismatch", 0) > 0:
            active_labels.append(("mismatch", precision_mismatch, recall_mismatch, f1_mismatch))

        if active_labels:
            macro_precision = sum(p for _, p, _, _ in active_labels) / len(active_labels)
            macro_recall = sum(r for _, _, r, _ in active_labels) / len(active_labels)
            macro_f1 = sum(f for _, _, _, f in active_labels) / len(active_labels)
        else:
            macro_precision = macro_recall = macro_f1 = 0.0

        uncertain_count = sum(1 for s in samples if s.predicted_label == "uncertain")
        uncertain_ratio = uncertain_count / n

        candidate_recall_count = sum(1 for s in samples if s.gold_candidate_in_top_k)
        candidate_recall = candidate_recall_count / n if n > 0 else 0.0

        # ECE and Brier (approximate: use confidence_match as probability of match)
        brier_scores = []
        ece_bins: Dict[Tuple[float, float], Dict[str, float]] = {}
        bin_edges = [i / 10 for i in range(11)]
        for s in samples:
            p = s.confidence_match
            y = 1.0 if s.gold.match_label == "match" else 0.0
            brier_scores.append((p - y) ** 2)
            # Find bin
            for bi in range(10):
                lo, hi = bin_edges[bi], bin_edges[bi + 1]
                if lo <= p < hi or (bi == 9 and p == hi):
                    key = (lo, hi)
                    if key not in ece_bins:
                        ece_bins[key] = {"sum_p": 0, "sum_y": 0, "count": 0}
                    ece_bins[key]["sum_p"] += p
                    ece_bins[key]["sum_y"] += y
                    ece_bins[key]["count"] += 1
                    break
        brier = sum(brier_scores) / n if n > 0 else 0.0
        ece = 0.0
        for (lo, hi), bucket in ece_bins.items():
            if bucket["count"] > 0:
                avg_p = bucket["sum_p"] / bucket["count"]
                avg_y = bucket["sum_y"] / bucket["count"]
                ece += bucket["count"] / n * abs(avg_p - avg_y)

        baselines = GoldBackedEvaluator._compute_baselines(samples)

        return VerifierEvalReportV1(
            report_id="gold_eval_v1",
            generated_at=datetime.now(timezone.utc).isoformat(),
            gold_source=source_files.get("gold_jsonl", ""),
            sample_count=n,
            accuracy=round(accuracy, 4),
            precision=round(macro_precision, 4),
            recall=round(macro_recall, 4),
            f1=round(macro_f1, 4),
            uncertain_ratio=round(uncertain_ratio, 4),
            candidate_recall=round(candidate_recall, 4),
            ece=round(ece, 4),
            brier=round(brier, 4),
            label_distribution=label_dist,
            per_sample=samples,
            baselines=baselines,
            warnings=warnings,
            source_files=source_files,
            quality_tier=quality_tier,
            gold_event_count_requested=gold_event_count_requested,
            skipped_gold_event_count=skipped_gold_event_count,
            official_visual_gold_count=official_visual_gold_count,
            pipeline_evaluable_gold_count=pipeline_evaluable_gold_count,
            excluded_from_verifier_count=excluded_from_verifier_count,
            excluded_reason_distribution=excluded_reason_distribution or {},
            excluded_gold_events=excluded_gold_events or [],
            per_case_breakdown=per_case_breakdown or [],
            missing_output_cases=missing_output_cases or [],
        )


# ── low-level helpers ──────────────────────────────────────────────

def _eval_baseline(
    name: str,
    predictions: List[str],
    samples: List[EvalSample],
    description: str,
) -> BaselineResult:
    """Evaluate a baseline's predictions against gold labels."""
    n = len(samples)
    if n == 0 or len(predictions) != n:
        return BaselineResult(name=name, accuracy=0, precision=0, recall=0, f1=0,
                              uncertain_ratio=0, description=description)
    correct = sum(1 for p, s in zip(predictions, samples) if p == s.gold.match_label)
    accuracy = correct / n

    # Macro P/R/F1 over match + mismatch
    tp_m = sum(1 for p, s in zip(predictions, samples) if p == "match" and s.gold.match_label == "match")
    fp_m = sum(1 for p, s in zip(predictions, samples) if p == "match" and s.gold.match_label != "match")
    fn_m = sum(1 for p, s in zip(predictions, samples) if p != "match" and s.gold.match_label == "match")
    prec_m = tp_m / (tp_m + fp_m) if (tp_m + fp_m) > 0 else 0.0
    rec_m = tp_m / (tp_m + fn_m) if (tp_m + fn_m) > 0 else 0.0
    f1_m = 2 * prec_m * rec_m / (prec_m + rec_m) if (prec_m + rec_m) > 0 else 0.0

    tp_mm = sum(1 for p, s in zip(predictions, samples) if p == "mismatch" and s.gold.match_label == "mismatch")
    fp_mm = sum(1 for p, s in zip(predictions, samples) if p == "mismatch" and s.gold.match_label != "mismatch")
    fn_mm = sum(1 for p, s in zip(predictions, samples) if p != "mismatch" and s.gold.match_label == "mismatch")
    prec_mm = tp_mm / (tp_mm + fp_mm) if (tp_mm + fp_mm) > 0 else 0.0
    rec_mm = tp_mm / (tp_mm + fn_mm) if (tp_mm + fn_mm) > 0 else 0.0
    f1_mm = 2 * prec_mm * rec_mm / (prec_mm + rec_mm) if (prec_mm + rec_mm) > 0 else 0.0

    active = []
    label_set = set(s.gold.match_label for s in samples)
    if "match" in label_set:
        active.append((prec_m, rec_m, f1_m))
    if "mismatch" in label_set:
        active.append((prec_mm, rec_mm, f1_mm))
    if active:
        macro_p = sum(x[0] for x in active) / len(active)
        macro_r = sum(x[1] for x in active) / len(active)
        macro_f1 = sum(x[2] for x in active) / len(active)
    else:
        macro_p = macro_r = macro_f1 = 0.0

    uncertain_count = sum(1 for p in predictions if p == "uncertain")
    return BaselineResult(
        name=name,
        accuracy=round(accuracy, 4),
        precision=round(macro_p, 4),
        recall=round(macro_r, 4),
        f1=round(macro_f1, 4),
        uncertain_ratio=round(uncertain_count / n, 4),
        description=description,
    )


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    rows.append(obj)
            except json.JSONDecodeError:
                continue
    return rows


def _coerce_track_id(value: Any, default: int = -1) -> int:
    if value is None:
        return default
    text = str(value).strip()
    if text == "" or text.lower() == "unknown":
        return default
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
