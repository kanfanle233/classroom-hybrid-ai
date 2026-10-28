"""Boundary tests for the model adapters; no model download is needed."""

from __future__ import annotations

import pytest

from classroom_hybrid_ai.adapters import (
    LocateAnythingAdapter,
    LocateAnythingHealthError,
    LocateAnythingResponseError,
    Qwen35Adapter,
    QwenResponseError,
    parse_locate_anything_response,
)


def test_locate_http_health_uses_documented_endpoint() -> None:
    calls = []

    class Client:
        def get(self, url, timeout):
            calls.append((url, timeout))
            return {"status": "ok", "loaded": False}

    adapter = LocateAnythingAdapter(
        backend="http",
        base_url="http://127.0.0.1:8011/locate",
        http_client=Client(),
        timeout=3.0,
    )
    assert adapter.health()["status"] == "ok"
    assert calls == [("http://127.0.0.1:8011/health", 3.0)]


def test_locate_http_health_wraps_missing_service() -> None:
    class Client:
        def get(self, _url, **_kwargs):
            raise ConnectionRefusedError(10061, "No connection could be made")

    adapter = LocateAnythingAdapter(backend="http", http_client=Client())
    with pytest.raises(LocateAnythingHealthError, match="health check failed"):
        adapter.health()


def test_locate_parser_scales_official_box_coordinates() -> None:
    rows = parse_locate_anything_response(
        "<ref>book</ref><box>100,200,500,600</box>", image_size=(1000, 500)
    )
    assert rows[0]["label"] == "book"
    assert rows[0]["bbox"] == [100.0, 100.0, 500.0, 300.0]


def test_locate_parser_inherits_ref_across_multiple_boxes() -> None:
    rows = parse_locate_anything_response(
        "<ref>student</ref><box><10><20><30><40></box>"
        "<box><50><60><70><80></box>"
    )
    assert [row["label"] for row in rows] == ["student", "student"]


def test_locate_parser_rejects_out_of_range_box() -> None:
    with pytest.raises(LocateAnythingResponseError):
        parse_locate_anything_response("<box>0,0,1001,10</box>")


def test_locate_parser_normalizes_reversed_box_corners_with_audit_trace() -> None:
    rows = parse_locate_anything_response(
        "<ref>student</ref><box>900,700,100,200</box>", image_size=(1000, 500)
    )

    assert rows[0]["bbox"] == [100.0, 100.0, 900.0, 350.0]
    assert rows[0]["attributes"]["coordinates_1000"] == [100.0, 200.0, 900.0, 700.0]
    assert rows[0]["attributes"]["coordinates_reordered"] is True
    assert rows[0]["attributes"]["raw_coordinates_1000"] == [900.0, 700.0, 100.0, 200.0]


def test_qwen_http_review_has_no_history_and_validates_current_id() -> None:
    captured = []

    def client(payload):
        captured.append(payload)
        return {"choices": [{"message": {"content": (
            '{"candidate_event_id":"event-1","decision":"uncertain",'
            '"reviewed_event_type":"review","behavior":null,"rationale":"occluded",'
            '"evidence_frame_indices":[3],"self_reported_confidence":null}'
        )}}]}

    adapter = Qwen35Adapter("local-test", backend="http", http_client=client)
    result = adapter.review("event-1", {"images": []}, event_type="review", evidence_frame_indices=(3,))
    assert result["schema_valid"] is True
    assert len(captured[0]["messages"]) == 1
    assert captured[0]["messages"][0]["role"] == "user"


def test_each_qwen_http_call_starts_a_fresh_single_user_message() -> None:
    captured = []

    def client(payload):
        captured.append(payload)
        event_id = ("event-a", "event-b")[len(captured) - 1]
        return {"choices": [{"message": {"content": (
            '{"candidate_event_id":"' + event_id + '","decision":"uncertain",'
            '"reviewed_event_type":"review","behavior":null,"rationale":"no",'
            '"evidence_frame_indices":[],"self_reported_confidence":null}'
        )}}]}

    adapter = Qwen35Adapter("local-test", backend="http", http_client=client)
    adapter.review("event-a", {"images": []}, event_type="review", evidence_frame_indices=())
    adapter.review("event-b", {"images": []}, event_type="review", evidence_frame_indices=())
    assert all(len(payload["messages"]) == 1 for payload in captured)
    assert all(payload["messages"][0]["role"] == "user" for payload in captured)
    assert all(payload.get("do_sample") is False for payload in captured)


def test_qwen_rejects_non_json_without_salvaging_partial_output() -> None:
    adapter = Qwen35Adapter(
        "local-test", backend="http",
        http_client=lambda _payload: {"choices": [{"message": {"content": "```json {} ```"}}]},
    )
    with pytest.raises(QwenResponseError, match="strict JSON"):
        adapter.complete("return JSON")


def test_qwen_local_prefills_opening_brace_without_accepting_markdown(tmp_path) -> None:
    captured = {}

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return {"input_ids": [[10, 11]]}

        def decode(self, _tokens, **_kwargs):
            return '"mappings":[]}'

    class Model:
        device = None

        def generate(self, **_kwargs):
            return [[10, 11, 12]]

    model_dir = tmp_path / "qwen"
    model_dir.mkdir()
    adapter = Qwen35Adapter(
        model_dir,
        backend="local",
        local_loader=lambda _path: (Model(), Processor()),
        max_new_tokens=8,
    )
    assert adapter.complete("return mappings") == {"mappings": []}
    assert [message["role"] for message in captured["messages"]] == ["user", "assistant"]
    assert captured["messages"][-1]["content"] == "{"
    assert captured["kwargs"]["continue_final_message"] is True
    assert captured["kwargs"]["add_generation_prompt"] is False


def test_qwen_compact_classification_response_is_bound_to_request() -> None:
    def client(_payload):
        return {"choices": [{"message": {"content": (
            '{"decision":"confirmed","behavior":"a_known"," rationale":"可见",'
            '"evidence_frame_indices":[3],"observability":"clear"}'
        )}}]}

    adapter = Qwen35Adapter("local-test", backend="http", http_client=client)
    result = adapter.review(
        "event-1",
        {"images": []},
        event_type="raise_hand",
        evidence_frame_indices=(3,),
        compact_response=True,
    )
    assert result["candidate_event_id"] == "event-1"
    assert result["reviewed_event_type"] == "raise_hand"
    assert result["self_reported_confidence"] is None
    assert result["schema_valid"] is True


def test_qwen_compact_response_rejects_duplicate_keys_after_whitespace_normalization() -> None:
    def client(_payload):
        return {"choices": [{"message": {"content": (
            '{"decision":"confirmed","behavior":"a_known","rationale":"one",'
            '" rationale":"two","evidence_frame_indices":[3],"observability":"clear"}'
        )}}]}

    adapter = Qwen35Adapter("local-test", backend="http", http_client=client)
    with pytest.raises(QwenResponseError, match="duplicate keys after whitespace normalization"):
        adapter.review(
            "event-1",
            {"images": []},
            event_type="raise_hand",
            evidence_frame_indices=(3,),
            compact_response=True,
        )
