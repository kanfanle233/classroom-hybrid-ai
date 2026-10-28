from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import random
import pytest

from classroom_hybrid_ai.config import load_config, ProjectConfig
from classroom_hybrid_ai.research_backend import _effective_config, ResearchBackend
from classroom_hybrid_ai.research_policy import localization_route, qwen_selection_budget, effective_configuration_report
from classroom_hybrid_ai.localization_evidence import TargetLocalizationIndex
from classroom_hybrid_ai.pipeline import _target_locate_evidence


def config():
    return load_config(Path(__file__).resolve().parents[1] / "configs/default.yaml")


@pytest.mark.parametrize('filename', ['default.yaml','localization_yolo_only.yaml',
    'localization_hybrid.yaml','localization_yolo26m_only.yaml','localization_yolo26m_hybrid.yaml'])
def test_public_loader_accepts_delivered_localization_configs(filename):
    cfg=load_config(Path(__file__).resolve().parents[1] / 'configs' / filename)
    assert not cfg.validate_static()


@pytest.mark.parametrize("route", ["locate_only", "yolo_only", "yolo_locate_hybrid"])
def test_explicit_route_is_not_overwritten(route):
    original = config()
    payload = deepcopy(original.payload)
    payload["pipeline"]["localization_route"] = route
    requested = ProjectConfig(original.source_path, original.project_root, original.repository_root, payload)
    actual = _effective_config(requested, profile=ResearchBackend(requested).profile(), mode="production_adaptive")
    assert localization_route(actual) == route
    assert actual.get("locate_anything.required") is (route != "yolo_only")
    assert localization_route(config()) == "locate_only"


@pytest.mark.parametrize("seconds,students,expected", [(0,0,48),(60,30,60),(600,50,100),(10800,10,240)])
def test_default_budget_matches_historical_effective_formula(seconds, students, expected):
    assert qwen_selection_budget(config(),seconds,students,1000) == expected


def test_all_candidates_requires_explicit_selection_policy():
    cfg = config()
    cfg.payload['locate_first']['qwen_every_screenshot'] = True
    assert qwen_selection_budget(cfg,60,2,1000) == 48
    cfg.payload['research_budget']['selection_mode'] = 'all_candidates'
    assert qwen_selection_budget(cfg,60,2,1000) == 1000
    assert effective_configuration_report(cfg)['all_candidates_experiment'] is True


@pytest.mark.parametrize("key,value", [('minimum',True),('minimum',241),('per_student',float('nan')),('per_minute',-1),('selection_mode','implicit')])
def test_invalid_budget_rejected(key,value):
    cfg=config(); cfg.payload['research_budget'][key]=value
    with pytest.raises(ValueError):
        qwen_selection_budget(cfg,60,2,10)


@pytest.mark.parametrize("seed", range(10))
def test_index_has_identical_metadata_and_stable_tie_order(seed):
    rng=random.Random(seed)
    rows=[{'frame_index':i//2,'pts_seconds':i//2*.25,'detections':[
        {'student_id':f's{j%3}','bbox':[j,0,j+1,2],'detection_index':j+1}
        for j in range(6)]} for i in range(160)]
    rng.shuffle(rows)
    index=TargetLocalizationIndex(rows)
    for student in range(3):
        for pts in (0.,2.,5.,19.75,100.):
            candidate=SimpleNamespace(track_id=f's{student}',trigger_pts_seconds=pts)
            assert index.query(candidate) == _target_locate_evidence(candidate,rows)
            assert index.query(candidate,limit=0) == []


def test_generic_index_never_attributes_yolo_to_locate():
    row={'frame_index':0,'pts_seconds':0.,'source':'yolo','detections':[
        {'student_id':'s1','source':'yolo','bbox':[0,0,1,1]}]}
    result=TargetLocalizationIndex([row]).query(SimpleNamespace(track_id='s1',trigger_pts_seconds=0.),generic=True)[0]
    assert result['detection_source']=='yolo'
    assert 'locate_detection_index' not in result


def test_study_cli_pause_has_interrupt_exit_code(tmp_path, monkeypatch):
    from classroom_hybrid_ai.cli import main
    import classroom_hybrid_ai.performance_study as study
    monkeypatch.setattr(study, 'run_study', lambda *a, **k: {'status':'paused_by_user'})
    assert main(['benchmark-study',str(tmp_path/'manifest.json'),'--output',str(tmp_path/'out')]) == 130


def test_yolo_only_plan_exposes_zero_confirmed_student_workload(monkeypatch):
    import classroom_hybrid_ai.research_backend as backend
    monkeypatch.setattr(backend, 'probe_video', lambda _: SimpleNamespace(fps=30.,
        frame_count=300,duration_seconds=10.,to_dict=lambda:{}))
    cfg=config(); cfg.payload['pipeline']['localization_route']='yolo_only'
    plan=ResearchBackend(cfg).plan('unused.mp4')
    assert plan['estimated_qwen_requests_upper_bound']==0
    assert plan['estimated_workload_assumptions']['student_confirmation']=='unavailable_in_yolo_only'


def test_qwen_pass_builds_profile_once_and_preserves_request_metadata(tmp_path, monkeypatch):
    import json
    import cv2
    import numpy as np
    from classroom_hybrid_ai.contracts import CandidateEvent
    from classroom_hybrid_ai.storage import RunStore
    from classroom_hybrid_ai.pipeline import _qwen_pass
    cfg = config()
    profile = ResearchBackend(cfg).profile()
    original = type(profile).prompt_payload
    expected = original(profile)
    calls = []
    def counted(self):
        calls.append(self.profile_id)
        return original(self)
    monkeypatch.setattr(type(profile), 'prompt_payload', counted)
    picture = tmp_path / 'composite.jpg'
    assert cv2.imwrite(str(picture), np.zeros((16, 16, 3), np.uint8))
    candidates = [CandidateEvent(event_id=f'e{i}', track_id='s1', event_type='review',
        trigger_frame_index=i, trigger_pts_seconds=float(i)) for i in range(2)]
    packets = [{'event_id':c.event_id, 'qwen_composite':{'path':str(picture),
        'frame_indices':[0,1,2]}} for c in candidates]
    rows = [{'frame_index':0, 'pts_seconds':0., 'detections':[
        {'student_id':'s1','bbox':[0,0,10,10],'detection_index':1}]}]
    captured = []
    class Adapter:
        def review(self, event_id, evidence, **kwargs):
            captured.append((evidence, json.dumps(evidence, ensure_ascii=False, separators=(',',':'))))
            return {'decision':'uncertain','behavior':'UNKNOWN','rationale':'证据不足',
                'evidence_frame_indices':[0,1,2],'observability':'none',
                'reviewer':'qwen3.5','schema_valid':True}
        def close(self):
            pass
    reviews, stage = _qwen_pass(cfg, candidates, packets,
        RunStore(tmp_path / 'runs', 'profile', 'same-input'), skip=False,
        adapter=Adapter(), response_contract='classification',
        expected_vocabulary_sha256='digest', action_profile=profile,
        locate_rows=rows, resolve_conflicts=False)
    assert len(reviews) == 2 and len(captured) == 2
    assert calls == [profile.profile_id]
    assert captured[0][0]['action_profile'] is captured[1][0]['action_profile']
    for candidate, (metadata, serialized) in zip(candidates, captured):
        legacy = dict(metadata, action_profile=expected,
            target_locate_evidence=_target_locate_evidence(candidate, rows))
        assert serialized.encode('utf-8') == json.dumps(legacy, ensure_ascii=False,
            separators=(',',':')).encode('utf-8')
