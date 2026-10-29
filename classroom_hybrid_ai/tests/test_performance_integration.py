from copy import deepcopy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from classroom_hybrid_ai.config import ProjectConfig, load_config
from classroom_hybrid_ai.pipeline import PipelineError, _classification_prompt, _locate_pass
from classroom_hybrid_ai.profiles import load_profile
from classroom_hybrid_ai.storage import RunStore, iter_jsonl

ROOT = Path(__file__).resolve().parents[1]


def _config():
    base = load_config(ROOT / "configs/default.yaml")
    payload = deepcopy(base.payload)
    payload["locate_anything"].update(backend="local", required=True)
    return ProjectConfig(base.source_path, base.project_root, base.repository_root, payload)


def test_locate_commits_each_success_before_failure_and_reuses_only_valid_response(tmp_path, monkeypatch):
    config = _config()
    store = RunStore(tmp_path, "sample", "config")
    monkeypatch.setattr("classroom_hybrid_ai.pipeline.release_model", lambda adapter: {"close_error": None})
    scenes = []
    for index in range(3):
        path = tmp_path / f"{index}.jpg"
        cv2.imwrite(str(path), np.full((32, 32, 3), index * 60, np.uint8))
        scenes.append({"path": str(path), "frame_index": index, "pts_seconds": index / 10})
    calls = []
    fail = [True]

    class Adapter:
        def __init__(self, **kwargs):
            assert kwargs["max_new_tokens"] == 256
            assert kwargs["maximum_image_side"] is None

        def detect(self, path, labels, **kwargs):
            calls.append(Path(path).stem)
            if Path(path).stem == "1" and fail[0]:
                fail[0] = False
                raise RuntimeError("interrupted")
            return {"detections": [{"bbox": [1, 1, 20, 30], "label": "student"}], "raw_answer": "student"}

    with pytest.raises(PipelineError, match="interrupted"):
        _locate_pass(config, scenes, store, skip=False, adapter_factory=Adapter, request_fingerprint="stage-v1")
    assert len(list((store.run_dir / "responses/locate").glob("*.json"))) == 1
    result = _locate_pass(config, scenes, store, skip=False, adapter_factory=Adapter, request_fingerprint="stage-v1", reuse_responses=True)
    assert result["reused_requests"] == 1
    assert result["requests"] == 2
    assert calls == ["0", "1", "1", "2"]
    assert [row["frame_index"] for row in iter_jsonl(store.run_dir / "artifacts/locate_results.jsonl")] == [0, 1, 2]
    cached = next((store.run_dir / "responses/locate").glob("*.json"))
    value = json.loads(cached.read_text(encoding="utf-8"))
    value["response"]["detections"] = []
    cached.write_text(json.dumps(value), encoding="utf-8")
    result = _locate_pass(config, scenes, store, skip=False, adapter_factory=Adapter, request_fingerprint="stage-v1", reuse_responses=True)
    assert result["requests"] == 1
    result = _locate_pass(config, scenes, store, skip=False, adapter_factory=Adapter, request_fingerprint="changed-code-or-model", reuse_responses=True)
    assert result["requests"] == 3


def test_compact_prompt_keeps_one_definition_and_visual_rules():
    config = _config()
    profile = load_profile(config.action_profiles_path, "core8_v1")
    legacy = _classification_prompt([], profile.sha256, action_profile=profile)
    compact = _classification_prompt([], profile.sha256, action_profile=profile, prompt_version="compact_v1")
    # The duplicated profile contract was removed from every version: the
    # immutable id/digest stays, the exact definitions live once in the frozen
    # vocabulary, and no colour hints reach the model.
    assert "Profile contract=" not in legacy
    assert "Profile contract=" not in compact
    assert "Action profile=" in legacy
    assert legacy.count("Frozen vocabulary=") == 1
    assert compact.count("Frozen vocabulary=") == 1
    for action in profile.actions:
        assert str(action["definition"]) in compact
        assert compact.count(str(action["definition"])) == 1
        assert str(action["color"]) not in compact
    assert "UNKNOWN" in compact and "observability" in compact
    assert len(compact) < len(legacy)
    with pytest.raises(ValueError, match="unsupported_prompt_version"):
        _classification_prompt([], "sha", prompt_version="future")
