import json

from classroom_hybrid_ai.open_vocab import (
    ActionDescriptor,
    ActionInterval,
    ActionObservation,
    ActionRegistrySnapshot,
    TimelineDocument,
    UNOBSERVED_ACTION_ID,
    build_timeline,
    colour_for_index,
    normalize_phrase,
    session_vocabulary,
    session_vocabulary_digest,
)
from classroom_hybrid_ai.tracking import TrackletSummary, fair_time_stratified_selection, stitch_score, stitch_tracklets
from classroom_hybrid_ai.registry import append_candidates, audit_registry, descriptors_from_discovery, load_latest
from classroom_hybrid_ai.semantic import SemanticCandidate, SemanticRetrieval, canonicalize_discoveries


def _action(identifier: str, name: str, *, status: str = "shared_candidate") -> ActionDescriptor:
    return ActionDescriptor(
        action_id=identifier,
        name_zh=name,
        definition=f"可见的{name}动作",
        color=colour_for_index(int(identifier[-1])),
        status=status,
        support_windows=2,
        support_students=2,
        evidence_quality=0.8,
    )


def test_registry_snapshot_is_verified_and_phrase_normalization_is_narrow() -> None:
    action = _action("a_000000000001", "学生正在写字")
    snapshot = ActionRegistrySnapshot.create([action], created_at="2026-01-01T00:00:00+00:00")
    restored = ActionRegistrySnapshot.from_dict(snapshot.to_dict())
    assert restored.sha256 == snapshot.sha256
    assert normalize_phrase(" 学生正在 写字。 ") == "写字"


def test_session_vocab_preserves_existing_action_and_caps_count() -> None:
    shared = [_action("a_000000000001", "写字")]
    session = [_action("a_000000000002", "阅读", status="session_provisional")]
    frozen = session_vocabulary(shared, session, maximum_actions=1)
    assert frozen[0].action_id == shared[0].action_id


def test_session_vocabulary_digest_matches_frozen_artifact_payload() -> None:
    action = _action("a_000000000001", "写字")
    frozen = session_vocabulary([action], [])
    digest = session_vocabulary_digest(frozen, registry_sha256="registry")
    payload = {
        "schema_version": "classroom_hybrid_ai.session_vocabulary.v1",
        "registry_sha256": "registry",
        "actions": [item.to_dict() for item in frozen],
    }
    from classroom_hybrid_ai.open_vocab import canonical_sha256

    assert digest == canonical_sha256(payload)


def test_timeline_is_single_primary_row_with_explicit_unobserved_gaps() -> None:
    action = _action("a_000000000001", "写字")
    observations = [
        ActionObservation(
            "student-001", 2.0, "写字", action.action_id,
            observability="clear", evidence_frame_indices=(1, 2),
            metadata={"candidate_event_id": "qwen-event-1"},
        ),
        ActionObservation("student-001", 8.0, None, None, observability="none"),
    ]
    intervals = build_timeline(observations, [action], student_ids=["student-001"], duration_seconds=10.0, max_hold_seconds=4.0)
    assert [(row.start_pts_seconds, row.end_pts_seconds, row.action_id) for row in intervals] == [
        (0.0, 2.0, UNOBSERVED_ACTION_ID),
        (2.0, 3.0, action.action_id),
        (3.0, 10.0, UNOBSERVED_ACTION_ID),
    ]
    assert [row.evidence_event_ids for row in intervals if row.status == "confirmed"] == [
        ("qwen-event-1",),
    ]


def test_merged_confirmed_interval_retains_all_qwen_candidate_ids() -> None:
    action = _action("a_000000000001", "写字")
    intervals = build_timeline(
        [
            ActionObservation(
                "student-001", 2.0, "写字", action.action_id,
                observability="clear", evidence_frame_indices=(1, 2),
                metadata={"candidate_event_id": "qwen-event-a"},
            ),
            ActionObservation(
                "student-001", 2.5, "写字", action.action_id,
                observability="clear", evidence_frame_indices=(3, 4),
                metadata={"candidate_event_id": "qwen-event-b"},
            ),
        ],
        [action], student_ids=["student-001"], duration_seconds=5.0,
        max_hold_seconds=2.0, max_gap_seconds=1.0,
    )
    confirmed = [row for row in intervals if row.status == "confirmed"]
    assert len(confirmed) == 1
    assert confirmed[0].evidence_event_ids == ("qwen-event-a", "qwen-event-b")


def test_timeline_preserves_spatial_student_order_supplied_by_catalog() -> None:
    intervals = build_timeline(
        [], [], student_ids=["student-10", "student-2"], duration_seconds=2.0,
    )
    assert [row.student_id for row in intervals if row.start_pts_seconds == 0.0] == [
        "student-10", "student-2",
    ]


def test_timeline_document_round_trips_video_and_interval_index() -> None:
    document = TimelineDocument(
        video={"path": "lesson.mp4", "duration_seconds": 2.0},
        student_rows=("student-001",),
        session_vocabulary_sha256="vocab",
        registry_sha256="registry",
        end_pts_seconds=2.0,
        intervals=(
            ActionInterval(
                "student-001", 0.0, 2.0, UNOBSERVED_ACTION_ID, "未观测", "#B8B8B8", "unobserved", "timeline", "unreviewed"
            ),
        ),
    )
    restored = TimelineDocument.from_json(document.to_json())
    assert restored.to_dict() == document.to_dict()


def test_partial_observation_needs_an_adjacent_same_action_anchor() -> None:
    action = _action("a_000000000001", "鍐欏瓧")
    observations = [
        ActionObservation("student-001", 2.0, "鍐欏瓧", action.action_id, observability="partial", evidence_frame_indices=(1, 2)),
        ActionObservation("student-001", 3.0, "鍐欏瓧", action.action_id, observability="partial", evidence_frame_indices=(3, 4)),
        ActionObservation("student-001", 7.0, "鍐欏瓧", action.action_id, observability="partial", evidence_frame_indices=(5, 6)),
    ]
    intervals = build_timeline(observations, [action], student_ids=["student-001"], duration_seconds=9.0, max_hold_seconds=1.0)
    confirmed = [row for row in intervals if row.status == "confirmed"]
    assert [(row.start_pts_seconds, row.end_pts_seconds) for row in confirmed] == [(2.0, 4.0)]


def test_short_unobserved_gap_can_bridge_same_action() -> None:
    action = _action("a_000000000001", "写字")
    intervals = build_timeline([
        ActionObservation("student-001", 1.0, "写字", action.action_id, observability="clear", evidence_frame_indices=(1, 2)),
        ActionObservation("student-001", 1.5, None, None, observability="none", unobserved_reason="insufficient_evidence"),
        ActionObservation("student-001", 2.0, "写字", action.action_id, observability="clear", evidence_frame_indices=(3, 4)),
    ], [action], student_ids=["student-001"], duration_seconds=4.0, max_hold_seconds=2.0, max_gap_seconds=1.0)
    assert not any(row.unobserved_reason == "insufficient_evidence" for row in intervals)
    assert any(row.status == "confirmed" and row.start_pts_seconds == 1.0 for row in intervals)


def test_long_anchor_gap_becomes_gray_instead_of_held_action() -> None:
    action = _action("a_000000000001", "鍐欏瓧")
    intervals = build_timeline([
        ActionObservation("student-001", 1.0, "鍐欏瓧", action.action_id, observability="clear", evidence_frame_indices=(1, 2)),
        ActionObservation("student-001", 4.0, "鍐欏瓧", action.action_id, observability="clear", evidence_frame_indices=(3, 4)),
    ], [action], student_ids=["student-001"], duration_seconds=6.0, max_hold_seconds=10.0, max_gap_seconds=1.0)
    assert any(row.status == "confirmed" and row.start_pts_seconds == 1.0 and row.end_pts_seconds == 2.0 for row in intervals)
    assert any(row.status == "unobserved" and row.unobserved_reason == "unreviewed" and row.start_pts_seconds == 2.0 and row.end_pts_seconds == 4.0 for row in intervals)


def test_unobserved_marker_does_not_hide_a_later_confirmed_action() -> None:
    action = _action("a_000000000001", "写字")
    intervals = build_timeline([
        ActionObservation("student-001", 0.0, None, None, observability="none", unobserved_reason="out_of_vocab"),
        ActionObservation("student-001", 2.0, "写字", action.action_id, observability="clear", evidence_frame_indices=(1, 2)),
    ], [action], student_ids=["student-001"], duration_seconds=5.0, max_hold_seconds=4.0)
    assert any(row.status == "confirmed" and row.start_pts_seconds == 2.0 for row in intervals)
    assert any(row.unobserved_reason == "out_of_vocab" and row.end_pts_seconds == 2.0 for row in intervals)


def test_conflicting_same_anchor_actions_are_explicitly_unobserved() -> None:
    write = _action("a_000000000001", "写字")
    read = _action("a_000000000002", "阅读")
    intervals = build_timeline([
        ActionObservation("student-001", 2.0, "写字", write.action_id, observability="clear", evidence_frame_indices=(1, 2)),
        ActionObservation("student-001", 2.0, "阅读", read.action_id, observability="clear", evidence_frame_indices=(1, 2)),
    ], [write, read], student_ids=["student-001"], duration_seconds=5.0, max_hold_seconds=2.0)
    assert any(row.status == "unobserved" and row.unobserved_reason == "conflict" for row in intervals)
    assert not any(row.status == "confirmed" for row in intervals)


def test_out_of_vocabulary_observation_keeps_explicit_gray_reason() -> None:
    action = _action("a_000000000001", "写字")
    intervals = build_timeline([
        ActionObservation(
            "student-001", 1.0, "未知动作", None, observability="none",
            unobserved_reason="out_of_vocab",
        ),
    ], [action], student_ids=["student-001"], duration_seconds=3.0, max_hold_seconds=1.0)
    assert any(row.unobserved_reason == "out_of_vocab" for row in intervals)


def test_conservative_stitching_rejects_ambiguous_link() -> None:
    first = TrackletSummary("a", 0, 1, 10, 10, 20)
    second = TrackletSummary("b", 0, 1, 10.1, 10, 20)
    third = TrackletSummary("c", 1.2, 2, 10.05, 10, 20)
    assert stitch_score(first, third) is not None
    mapping = stitch_tracklets([first, second, third])
    assert mapping["c"] != mapping["a"]
    assert mapping["c"] != mapping["b"]


def test_fair_candidate_selection_includes_late_bucket_before_high_score_duplicates() -> None:
    chosen = fair_time_stratified_selection([
        {"event_id": "early-a", "track_id": "a", "pts_seconds": 1, "score": 0.99},
        {"event_id": "early-b", "track_id": "a", "pts_seconds": 2, "score": 0.98},
        {"event_id": "late", "track_id": "z", "pts_seconds": 305, "score": 0.01},
    ], limit=2)
    assert {row["event_id"] for row in chosen} == {"early-a", "late"}


def test_fair_candidate_selection_reserves_capacity_for_late_custom_anomaly() -> None:
    chosen = fair_time_stratified_selection([
        {"event_id": "periodic-early-1", "track_id": "a", "pts_seconds": 1, "score": None, "source": "periodic_audit"},
        {"event_id": "periodic-early-2", "track_id": "b", "pts_seconds": 2, "score": None, "source": "periodic_audit"},
        {"event_id": "periodic-early-3", "track_id": "c", "pts_seconds": 3, "score": None, "source": "periodic_audit"},
        {"event_id": "anomaly-late", "track_id": "z", "pts_seconds": 3600, "score": 0.8, "source": "yolo26_custom_anomaly", "metadata": {"priority": 100}},
    ], limit=2)
    assert "anomaly-late" in {row["event_id"] for row in chosen}


def test_semantic_merge_requires_two_order_invariant_batch_votes() -> None:
    candidate = SemanticCandidate(
        action_id="a_000000000001", name_zh="写字", definition="手在纸面书写",
        attributes={}, source="registry",
    )

    class FakeRetriever:
        status = "ready"

        def retrieve(self, phrase, candidates, *, threshold, top_k):
            return SemanticRetrieval(phrase, (candidate,), (0.91,), "ready")

    calls = []

    def batch(proposals):
        calls.append(proposals)
        return {phrase: "a_000000000001" for phrase, _ in proposals}

    rows, report = canonicalize_discoveries(
        [{"raw_phrase": "书写"}], registry_actions=[], retriever=FakeRetriever(),
        threshold=0.72, top_k=5, batch_adjudicator=batch,
    )
    assert len(calls) == 2
    assert rows[0]["canonical_phrase"] == "写字"
    assert rows[0]["canonical_action_id"] == "a_000000000001"
    assert report["merged"] == 1


def test_semantic_retrieval_without_adjudication_never_merges() -> None:
    candidate = SemanticCandidate("a_000000000001", "写字", "手部动作", {}, "registry")

    class FakeRetriever:
        status = "ready"

        def retrieve(self, phrase, candidates, *, threshold, top_k):
            return SemanticRetrieval(phrase, (candidate,), (0.99,), "ready")

    rows, report = canonicalize_discoveries(
        [{"raw_phrase": "书写"}], registry_actions=[], retriever=FakeRetriever(),
        threshold=0.72, top_k=5,
    )
    assert rows[0]["canonical_phrase"] == "书写"
    assert "canonical_action_id" not in rows[0]
    assert report["merged"] == 0


def test_shared_candidate_registry_is_append_only(tmp_path) -> None:
    empty = load_latest(tmp_path)
    action = _action("a_000000000001", "写字")
    first = append_candidates(tmp_path, empty, [action])
    second = append_candidates(tmp_path, first, [action])
    assert first.registry_version == 1
    assert second.registry_version == 1
    assert (tmp_path / "registry.v0001.json").is_file()


def test_registry_keeps_existing_action_id_and_colour_across_versions(tmp_path) -> None:
    first_action = _action("a_000000000001", "鍐欏瓧")
    initial = append_candidates(tmp_path, load_latest(tmp_path), [first_action])
    second_action = _action("a_000000000002", "闃呰")
    updated = append_candidates(tmp_path, initial, [second_action])
    old = next(item for item in updated.actions if item.action_id == first_action.action_id)
    assert updated.registry_version == 2
    assert old.color == first_action.color
    assert (tmp_path / "registry.v0001.json").is_file()
    assert (tmp_path / "registry.v0002.json").is_file()


def test_registry_audit_accepts_empty_and_append_only_history(tmp_path) -> None:
    assert audit_registry(tmp_path)["status"] == "empty"
    first_action = _action("a_000000000001", "写字")
    first = append_candidates(tmp_path, load_latest(tmp_path), [first_action])
    append_candidates(tmp_path, first, [_action("a_000000000002", "阅读")])
    report = audit_registry(tmp_path)
    assert report["status"] == "ok"
    assert report["versions"] == 2
    assert report["actions"] == 2
    assert report["errors"] == []


def test_registry_audit_rejects_modified_action_and_broken_parent(tmp_path) -> None:
    first_action = _action("a_000000000001", "写字")
    first = append_candidates(tmp_path, load_latest(tmp_path), [first_action])
    append_candidates(tmp_path, first, [_action("a_000000000002", "阅读")])
    second_path = tmp_path / "registry.v0002.json"
    payload = json.loads(second_path.read_text(encoding="utf-8"))
    payload["parent_sha256"] = "tampered-parent"
    payload["actions"][0]["color"] = "#000000"
    from classroom_hybrid_ai.open_vocab import canonical_sha256
    payload["sha256"] = canonical_sha256({
        "schema_version": payload["schema_version"],
        "registry_version": payload["registry_version"],
        "parent_sha256": payload["parent_sha256"],
        "actions": payload["actions"],
    })
    second_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    report = audit_registry(tmp_path)
    assert report["status"] == "failed"
    assert any("parent_digest_mismatch" in error for error in report["errors"])
    assert any("immutable_action_changed" in error for error in report["errors"])


def test_discovery_requires_repeated_independent_evidence_for_shared_memory() -> None:
    shared, session = descriptors_from_discovery([
        {"raw_phrase": "写字", "student_id": "s1", "pts_seconds": 0, "evidence_quality": 0.8},
        {"raw_phrase": "写字", "student_id": "s2", "pts_seconds": 1, "evidence_quality": 0.8},
        {"raw_phrase": "趴桌", "student_id": "s1", "pts_seconds": 1, "evidence_quality": 0.8},
    ], minimum_windows=2, same_student_gap_seconds=10, minimum_quality=0.7, initial_colour_index=0)
    assert [item.name_zh for item in shared] == ["写字"]
    assert [item.name_zh for item in session] == ["趴桌"]
