from __future__ import annotations

import time
from pathlib import Path

import pytest

from classroom_hybrid_ai import smoke
from classroom_hybrid_ai.cli import build_parser
from classroom_hybrid_ai.smoke import (
    _PhysicalRequestPeakSampler,
    _growth_metrics,
    _physical_gpu_budget_gib,
    _physical_gpu_limit_mib,
    _physical_gpu_measurement_scope,
    _physical_peak_gate,
    _request_peak_mib_fields,
    _smoke_media_list,
    _smoke_response_contract,
)


class _StubConfig:
    def __init__(self, values: dict[str, object]) -> None:
        self._values = values

    def get(self, key: str, default: object = None) -> object:
        return self._values.get(key, default)


def _wait_for(condition, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_v3_growth_metrics_ignore_first_ten_warmup_requests() -> None:
    values = [100.0] * 10 + [200.0] * 90
    result = _growth_metrics(values)
    assert result["warmup_excluded_first_window_mib"] == 200.0
    assert result["steady_last_window_mib"] == 200.0
    assert result["growth_11_30_to_81_100_mib"] == 0.0
    assert result["robust_slope_mib_per_request"] == 0.0
    assert result["three_consecutive_windows_rising"] is False


def test_v3_growth_metrics_flags_three_rising_windows() -> None:
    values = [100.0] * 10 + [200.0] * 20 + [210.0] * 20 + [220.0] * 20 + [230.0] * 30
    result = _growth_metrics(values)
    assert result["three_consecutive_windows_rising"] is True
    assert result["robust_slope_mib_per_request"] > 0.0


def test_windows_whole_device_memory_scope_is_warning_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("classroom_hybrid_ai.smoke.platform.system", lambda: "Windows")
    assert _physical_gpu_measurement_scope() == "device_total_wddm_warning_only"


def test_non_windows_whole_device_memory_scope_is_reliable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("classroom_hybrid_ai.smoke.platform.system", lambda: "Linux")
    assert _physical_gpu_measurement_scope() == "device_total_nvml_reliable"


def test_smoke_media_list_accepts_one_or_many_inputs() -> None:
    assert _smoke_media_list("one.jpg") == ("one.jpg",)
    assert _smoke_media_list(Path("one.jpg")) == (Path("one.jpg"),)
    assert _smoke_media_list([Path("one.jpg"), Path("two.jpg")]) == (
        Path("one.jpg"), Path("two.jpg")
    )


def test_smoke_media_list_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="smoke_media_list_must_not_be_empty"):
        _smoke_media_list([])


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ({"decision": "confirmed", "behavior": "action_raise_hand", "observability": "clear"}, True),
        ({"decision": "rejected", "behavior": None, "observability": "clear"}, True),
        ({"decision": "uncertain", "behavior": None, "observability": "partial"}, True),
        ({"decision": "confirmed", "behavior": None, "observability": "clear"}, False),
        ({"decision": "rejected", "behavior": "action_raise_hand", "observability": "clear"}, False),
    ],
)
def test_smoke_response_contract_binds_candidate(response: dict[str, object], expected: bool) -> None:
    assert _smoke_response_contract(response)[0] is expected


def test_physical_budget_defaults_to_7_4_gib_without_runtime_value() -> None:
    assert _physical_gpu_budget_gib(_StubConfig({})) == 7.4
    assert _physical_gpu_budget_gib(_StubConfig({"runtime.gpu_budget_gib": None})) == 7.4
    assert _physical_gpu_budget_gib(_StubConfig({"runtime.gpu_budget_gib": 7.4})) == 7.4
    assert _physical_gpu_budget_gib(_StubConfig({"runtime.gpu_budget_gib": "7.0"})) == 7.0


def test_physical_limit_is_minimum_of_device_headroom_and_budget() -> None:
    assert _physical_gpu_limit_mib(8188.0, 7.4) == pytest.approx(7.4 * 1024)
    assert _physical_gpu_limit_mib(4096.0, 7.4) == pytest.approx(4096.0 - 512.0)
    assert _physical_gpu_limit_mib(None, 7.4) == pytest.approx(7.4 * 1024)


@pytest.mark.parametrize("gate", ["development", "delivery"])
def test_graded_gate_blocks_physical_peaks_over_limit(gate: str) -> None:
    blockers, warnings = _physical_peak_gate(
        gate=gate,
        physical_peak_mib=7600.0,
        request_peak_mib=7000.0,
        limit_mib=7577.6,
    )
    assert blockers == ["physical_gpu_peak_exceeds_physical_budget_mib"]
    assert warnings == []


@pytest.mark.parametrize("gate", ["development", "delivery"])
def test_graded_gate_blocks_request_peak_over_limit(gate: str) -> None:
    blockers, warnings = _physical_peak_gate(
        gate=gate,
        physical_peak_mib=7000.0,
        request_peak_mib=7600.0,
        limit_mib=7577.6,
    )
    assert blockers == ["physical_gpu_request_peak_exceeds_physical_budget_mib"]
    assert warnings == []


def test_graded_gate_blocks_missing_physical_measurements() -> None:
    blockers, warnings = _physical_peak_gate(
        gate="delivery",
        physical_peak_mib=None,
        request_peak_mib=None,
        limit_mib=7577.6,
    )
    assert blockers == [
        "physical_gpu_peak_measurement_missing",
        "physical_gpu_request_peak_measurement_missing",
    ]
    assert warnings == []


def test_graded_gate_accepts_peaks_within_limit() -> None:
    blockers, warnings = _physical_peak_gate(
        gate="delivery",
        physical_peak_mib=7000.0,
        request_peak_mib=7000.0,
        limit_mib=7577.6,
    )
    assert blockers == []
    assert warnings == []


def test_exploratory_gate_reports_the_same_findings_as_warnings() -> None:
    blockers, warnings = _physical_peak_gate(
        gate="none",
        physical_peak_mib=8000.0,
        request_peak_mib=None,
        limit_mib=7577.6,
    )
    assert blockers == []
    assert warnings == [
        "physical_gpu_peak_exceeds_physical_budget_mib",
        "physical_gpu_request_peak_measurement_missing",
    ]


def test_request_peak_fields_convert_worker_bytes_to_mib() -> None:
    fields = _request_peak_mib_fields({
        "request_peak_allocated_bytes": 3 * 2**20,
        "request_peak_reserved_bytes": 5 * 2**20,
    })
    assert fields["request_peak_allocated_bytes"] == 3 * 2**20
    assert fields["request_peak_allocated_mib"] == 3.0
    assert fields["request_peak_reserved_bytes"] == 5 * 2**20
    assert fields["request_peak_reserved_mib"] == 5.0


def test_request_peak_fields_preserve_missing_worker_metrics() -> None:
    assert _request_peak_mib_fields({}) == {
        "request_peak_allocated_bytes": None,
        "request_peak_allocated_mib": None,
        "request_peak_reserved_bytes": None,
        "request_peak_reserved_mib": None,
    }


def test_request_peak_sampler_clamps_poll_interval() -> None:
    assert _PhysicalRequestPeakSampler(0, interval_seconds=0.01).interval_seconds == 0.2
    assert _PhysicalRequestPeakSampler(0, interval_seconds=5.0).interval_seconds == 0.5


def test_request_peak_sampler_records_whole_device_peak_during_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"count": 0}

    def fake_used(_gpu_index: int) -> float:
        calls["count"] += 1
        return (1200.0, 5600.0, 4000.0)[calls["count"] % 3]

    monkeypatch.setattr(smoke, "_physical_gpu_used_mib", fake_used)
    sampler = _PhysicalRequestPeakSampler(0, interval_seconds=0.2).start()
    assert _wait_for(lambda: sampler.samples >= 2)
    assert sampler.stop() == 5600.0
    assert sampler.missing_reason is None


def test_request_peak_sampler_swallows_probe_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken(_gpu_index: int) -> float:
        raise RuntimeError("nvidia-smi crashed")

    monkeypatch.setattr(smoke, "_physical_gpu_used_mib", broken)
    sampler = _PhysicalRequestPeakSampler(0, interval_seconds=0.2).start()
    assert _wait_for(lambda: sampler.failures >= 1)
    assert sampler.stop() is None
    assert sampler.missing_reason == "physical_sample_failed:RuntimeError"


def test_request_peak_sampler_reports_unavailable_nvidia_smi(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke, "_physical_gpu_used_mib", lambda _gpu_index: None)
    sampler = _PhysicalRequestPeakSampler(0, interval_seconds=0.2).start()
    assert _wait_for(lambda: sampler.failures >= 1)
    assert sampler.stop() is None
    assert sampler.missing_reason == "nvidia_smi_memory_used_unavailable"


def test_request_peak_sampler_reports_no_samples(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(smoke, "_physical_gpu_used_mib", lambda _gpu_index: 1234.0)
    sampler = _PhysicalRequestPeakSampler(0)
    assert sampler.stop() is None
    assert sampler.missing_reason == "no_physical_samples_collected"


def test_smoke_cli_rotates_multiple_inputs_and_uses_gate_default_count() -> None:
    args = build_parser().parse_args([
        "smoke-qwen", "one.jpg", "two.jpg", "--gate", "delivery",
    ])
    assert args.image == [Path("one.jpg"), Path("two.jpg")]
    assert args.repeats is None
    assert args.gate == "delivery"

def test_production_response_contract_accepts_vocabulary_behavior() -> None:
    from classroom_hybrid_ai.smoke import _production_response_contract

    ok, reason = _production_response_contract({
        "decision": "rejected",
        "behavior": "a_js",
        "rationale": "未举手",
        "evidence_frame_indices": [],
        "observability": "none",
    })
    assert ok and reason is None
    ok, reason = _production_response_contract({
        "decision": "uncertain",
        "behavior": None,
        "evidence_frame_indices": [],
        "observability": "partial",
    })
    assert ok and reason is None
    assert _production_response_contract({"decision": "maybe"})[0] is False
    assert _production_response_contract("{}")[0] is False
