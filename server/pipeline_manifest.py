"""
Pipeline manifest tracking: per-step results, schema validation, error reports.

Provides structured output for each pipeline step so that runs are
reproducible, comparable, and auditable.

Public types:
    PipelineStepResult   - result of a single pipeline step
    PipelineRunManifest  - full manifest for one case pipeline run
    StepValidator        - validates step output against expected schema
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


# ── Step result ────────────────────────────────────────────────────

@dataclass
class PipelineStepResult:
    step_id: str
    step_name: str
    status: str  # "ok" | "error" | "skipped" | "partial"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_sec: Optional[float] = None
    input_files: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    schema_valid: Optional[bool] = None
    schema_errors: List[str] = field(default_factory=list)
    record_count: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ── Run manifest ───────────────────────────────────────────────────

@dataclass
class PipelineRunManifest:
    case_id: str
    video_id: str
    run_id: str
    schema_version: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    status: str = "running"  # "running" | "complete" | "failed" | "partial"
    steps: List[PipelineStepResult] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    overall_warnings: List[str] = field(default_factory=list)

    def step_by_id(self, step_id: str) -> Optional[PipelineStepResult]:
        for s in self.steps:
            if s.step_id == step_id:
                return s
        return None

    def failed_steps(self) -> List[PipelineStepResult]:
        return [s for s in self.steps if s.status == "error"]

    def step_summary(self) -> Dict[str, str]:
        return {s.step_id: s.status for s in self.steps}

    def all_output_files(self) -> List[str]:
        files: List[str] = []
        for s in self.steps:
            files.extend(s.output_files)
        return files

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "video_id": self.video_id,
            "run_id": self.run_id,
            "schema_version": self.schema_version,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "status": self.status,
            "steps": [
                {
                    "step_id": s.step_id,
                    "step_name": s.step_name,
                    "status": s.status,
                    "started_at": s.started_at,
                    "finished_at": s.finished_at,
                    "duration_sec": s.duration_sec,
                    "input_files": s.input_files,
                    "output_files": s.output_files,
                    "schema_valid": s.schema_valid,
                    "schema_errors": s.schema_errors,
                    "record_count": s.record_count,
                    "warnings": s.warnings,
                    "error_message": s.error_message,
                    "metadata": s.metadata,
                }
                for s in self.steps
            ],
            "config_snapshot": self.config_snapshot,
            "overall_warnings": self.overall_warnings,
        }


# ── Step definitions ──────────────────────────────────────────────

STEP_DEFINITIONS = [
    {"step_id": "01_pose_video",     "step_name": "Pose video demo"},
    {"step_id": "02_keypoints",      "step_name": "Export keypoints JSONL"},
    {"step_id": "02b_objects",       "step_name": "Export objects JSONL"},
    {"step_id": "02c_rear_sr",       "step_name": "Rear ROI SR cache"},
    {"step_id": "02d_behavior_det",  "step_name": "Behavior detection export"},
    {"step_id": "03_tracking",       "step_name": "Tracking and smoothing"},
    {"step_id": "03c_track_uq",      "step_name": "Track uncertainty estimation"},
    {"step_id": "03e_track_beh",     "step_name": "Behavior tracking"},
    {"step_id": "04_action_rules",   "step_name": "Rule-based action classification"},
    {"step_id": "04_complex",        "step_name": "Complex action logic"},
    {"step_id": "05_slowfast",       "step_name": "SlowFast action recognition"},
    {"step_id": "05b_fuse",          "step_name": "Action-object fusion"},
    {"step_id": "06_asr",            "step_name": "ASR transcription"},
    {"step_id": "06b_event_query",   "step_name": "Event query extraction"},
    {"step_id": "06f_llm_fusion",    "step_name": "LLM semantic fusion"},
    {"step_id": "06h_llm_teacher",   "step_name": "LLM teacher labeling"},
    {"step_id": "06i_student_train", "step_name": "Student judge training"},
    {"step_id": "06j_eval_ts",       "step_name": "Teacher-student evaluation"},
    {"step_id": "07_align",          "step_name": "Multimodal alignment"},
    {"step_id": "08_verify",         "step_name": "Verifier inference"},
    {"step_id": "09_calibrate",      "step_name": "Temperature calibration"},
    {"step_id": "10_timeline",       "step_name": "Timeline visualization"},
]


# ── StepValidator ──────────────────────────────────────────────────

class StepValidator:
    """Validate step output files against expected schemas."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self._validators: Dict[str, Callable] = {}
        self._register_validators()

    def _register_validators(self) -> None:
        self._validators["jsonl"] = self._validate_jsonl
        self._validators["json"] = self._validate_json

    def validate_step_output(
        self,
        step_id: str,
        output_files: List[str],
    ) -> Tuple[bool, List[str], Optional[int]]:
        """Validate all output files for a step.

        Returns (all_valid, errors, total_record_count).
        """
        all_valid = True
        all_errors: List[str] = []
        total_records = 0

        for fpath in output_files:
            p = Path(fpath)
            if not p.exists():
                all_valid = False
                all_errors.append(f"{step_id}: output file not found: {p.name}")
                continue

            if p.suffix == ".jsonl":
                valid, errors, count = self._validate_jsonl(p)
                if not valid:
                    all_valid = False
                    all_errors.extend(errors)
                total_records += count
            elif p.suffix == ".json":
                valid, errors = self._validate_json(p)
                if not valid:
                    all_valid = False
                    all_errors.extend(errors)
                total_records += 1

        return all_valid, all_errors, total_records if total_records > 0 else None

    def _validate_jsonl(self, path: Path) -> Tuple[bool, List[str], int]:
        errors: List[str] = []
        count = 0
        try:
            with path.open("r", encoding="utf-8") as f:
                for line_no, raw in enumerate(f, 1):
                    line = raw.strip()
                    if not line:
                        continue
                    count += 1
                    try:
                        obj = json.loads(line)
                        if not isinstance(obj, dict):
                            errors.append(f"{path.name}:{line_no}: not a JSON object")
                    except json.JSONDecodeError as e:
                        errors.append(f"{path.name}:{line_no}: JSON decode error: {e}")
        except Exception as e:
            errors.append(f"{path.name}: read error: {e}")
        return len(errors) == 0, errors, count

    def _validate_json(self, path: Path) -> Tuple[bool, List[str]]:
        errors: List[str] = []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, (dict, list)):
                errors.append(f"{path.name}: expected dict or list, got {type(data).__name__}")
        except json.JSONDecodeError as e:
            errors.append(f"{path.name}: JSON decode error: {e}")
        except Exception as e:
            errors.append(f"{path.name}: read error: {e}")
        return len(errors) == 0, errors


# ── Manifest manager ───────────────────────────────────────────────

class PipelineManifestManager:
    """Create, load, and persist pipeline run manifests."""

    def __init__(self, case_dir: Path):
        self.case_dir = case_dir
        self.manifest_path = case_dir / "pipeline_run_manifest.json"

    def create(
        self,
        case_id: str,
        video_id: str,
        run_id: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> PipelineRunManifest:
        now = datetime.now(timezone.utc).isoformat()
        manifest = PipelineRunManifest(
            case_id=case_id,
            video_id=video_id,
            run_id=run_id,
            schema_version="2026-04-01",
            started_at=now,
            status="running",
            config_snapshot=config or {},
        )
        # Register all known steps
        for step_def in STEP_DEFINITIONS:
            manifest.steps.append(PipelineStepResult(
                step_id=step_def["step_id"],
                step_name=step_def["step_name"],
                status="pending",
            ))
        return manifest

    def save(self, manifest: PipelineRunManifest) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest.to_dict(), f, ensure_ascii=False, indent=2)

    def load(self) -> Optional[PipelineRunManifest]:
        if not self.manifest_path.exists():
            return None
        try:
            with self.manifest_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return self._from_dict(data)
        except (json.JSONDecodeError, KeyError):
            return None

    @staticmethod
    def _from_dict(data: Dict[str, Any]) -> PipelineRunManifest:
        steps = []
        for s in data.get("steps", []):
            steps.append(PipelineStepResult(
                step_id=s["step_id"],
                step_name=s["step_name"],
                status=s["status"],
                started_at=s.get("started_at"),
                finished_at=s.get("finished_at"),
                duration_sec=s.get("duration_sec"),
                input_files=s.get("input_files", []),
                output_files=s.get("output_files", []),
                schema_valid=s.get("schema_valid"),
                schema_errors=s.get("schema_errors", []),
                record_count=s.get("record_count"),
                warnings=s.get("warnings", []),
                error_message=s.get("error_message"),
                metadata=s.get("metadata", {}),
            ))
        return PipelineRunManifest(
            case_id=data["case_id"],
            video_id=data["video_id"],
            run_id=data["run_id"],
            schema_version=data["schema_version"],
            started_at=data.get("started_at"),
            finished_at=data.get("finished_at"),
            status=data.get("status", "unknown"),
            steps=steps,
            config_snapshot=data.get("config_snapshot", {}),
            overall_warnings=data.get("overall_warnings", []),
        )


def complete_step(
    manifest: PipelineRunManifest,
    step_id: str,
    output_files: List[str],
    input_files: Optional[List[str]] = None,
    record_count: Optional[int] = None,
    metadata: Optional[Dict[str, Any]] = None,
    validator: Optional[StepValidator] = None,
) -> PipelineStepResult:
    """Complete a step in the manifest: validate, record results, update status."""
    step = manifest.step_by_id(step_id)
    if step is None:
        raise ValueError(f"Unknown step_id: {step_id}")

    now = datetime.now(timezone.utc).isoformat()
    step.output_files = output_files
    step.input_files = input_files or []
    step.record_count = record_count
    step.metadata = metadata or {}
    step.finished_at = now

    # Calculate duration
    if step.started_at:
        try:
            from datetime import datetime as dt
            start = dt.fromisoformat(step.started_at)
            end = dt.fromisoformat(now)
            step.duration_sec = round((end - start).total_seconds(), 2)
        except (ValueError, TypeError):
            pass

    # Validate output files
    if validator and output_files:
        valid, errors, count = validator.validate_step_output(step_id, output_files)
        step.schema_valid = valid
        step.schema_errors = errors
        if count is not None and step.record_count is None:
            step.record_count = count
        if not valid:
            step.status = "error"
            step.error_message = f"Schema validation failed: {'; '.join(errors[:3])}"
        else:
            step.status = "ok"
    else:
        step.status = "ok"
        step.schema_valid = None

    return step
