#!/usr/bin/env python3
"""Step 3: Teacher Label Verification — schema, simulate marking, evidence format."""
import json, sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    out = []
    if not path.exists(): return out
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try: out.append(json.loads(line))
                except: continue
    return out


def main() -> None:
    td = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "teacher_labels"
    REPORT_DIR = _PROJECT_ROOT / "output" / "llm_judge_pipeline" / "reports"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    label_files = sorted(td.glob("llm_teacher_output.front_*.jsonl"))

    all_records = []
    for f in label_files:
        all_records.extend(load_jsonl(f))

    # ── Schema compliance check ─────────────────────────────────
    REQUIRED_FIELDS = [
        "schema_version", "case_id", "event_id", "track_id", "model_name",
        "prompt_version", "generated_at", "input_signature", "llm_label",
        "llm_match_score", "llm_mismatch_score", "llm_uncertain_score",
        "llm_confidence", "llm_rationale", "llm_raw_response", "provider_mode",
    ]
    VALID_LABELS = {"match", "mismatch", "uncertain"}
    VALID_PROVIDERS = {"real", "simulate"}

    schema_errors = []
    provider_issues = []
    score_sum_issues = []

    for i, r in enumerate(all_records):
        # Missing fields
        missing = [k for k in REQUIRED_FIELDS if k not in r]
        if missing:
            schema_errors.append({"record_idx": i, "issue": "missing_fields", "fields": missing})
            continue

        # Invalid label
        if r["llm_label"] not in VALID_LABELS:
            schema_errors.append({"record_idx": i, "issue": "invalid_label", "value": r["llm_label"]})

        # Score sum
        total = r["llm_match_score"] + r["llm_mismatch_score"] + r["llm_uncertain_score"]
        if abs(total - 1.0) > 0.05:
            score_sum_issues.append({"record_idx": i, "sum": round(total, 6)})

        # Provider mode
        if r.get("provider_mode") not in VALID_PROVIDERS:
            provider_issues.append({"record_idx": i, "value": r.get("provider_mode")})

        # provider_mode must be "simulate" if model_name == "simulate"
        if r.get("provider_mode") == "real" and r.get("model_name") == "simulate":
            provider_issues.append({"record_idx": i, "issue": "provider=real but model=simulate — fraud detected"})

    # ── Provider audit ──────────────────────────────────────────
    providers = Counter(r.get("provider_mode", "?") for r in all_records)
    models = Counter(r.get("model_name", "?") for r in all_records)

    # ── Evidence format check ───────────────────────────────────
    # Check input_signature is consistently formatted
    sig_formats = Counter()
    for r in all_records[:10]:
        sig = r.get("input_signature", "")
        sig_formats[len(sig)] += 1

    # ── Build result ────────────────────────────────────────────
    result = {
        "step": "3_teacher_labels",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "total_records": len(all_records),
        "total_files": len(label_files),
        "cases_present": sorted(set(r.get("case_id", "?") for r in all_records)),

        "schema_compliance": {
            "required_fields": REQUIRED_FIELDS,
            "records_checked": len(all_records),
            "errors": schema_errors,
            "passed": len(schema_errors) == 0,
        },

        "score_sum_check": {
            "issues": score_sum_issues[:5],
            "n_issues": len(score_sum_issues),
            "passed": len(score_sum_issues) == 0,
        },

        "provider_audit": {
            "distribution": dict(providers),
            "model_distribution": dict(models),
            "all_simulate": providers.get("simulate", 0) == len(all_records),
            "issues": provider_issues,
            "passed": len(provider_issues) == 0,
            "note": "ALL records are simulate mode. Must not be described as real LLM outputs.",
        },

        "simulate_marking_explicit": all(
            r.get("provider_mode") == "simulate" for r in all_records
        ),

        "output_files": [str(f) for f in label_files],

        "status": "PASS" if (len(schema_errors) == 0 and len(provider_issues) == 0) else "FAIL",
    }

    out_path = REPORT_DIR / "step3_teacher_label_verification.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"=== Step 3: Teacher Label Verification ===")
    print(f"Records: {len(all_records)}")
    print(f"Schema errors: {len(schema_errors)}")
    print(f"Score sum issues: {len(score_sum_issues)}")
    print(f"Provider issues: {len(provider_issues)}")
    print(f"Provider distribution: {dict(providers)}")
    print(f"All simulate marked explicitly: {result['simulate_marking_explicit']}")
    print(f"[DONE] {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
