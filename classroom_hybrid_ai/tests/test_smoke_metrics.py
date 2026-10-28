from __future__ import annotations

from pathlib import Path

import pytest

from classroom_hybrid_ai.cli import build_parser
from classroom_hybrid_ai.smoke import (
    _growth_metrics,
    _physical_gpu_measurement_scope,
    _smoke_media_list,
    _smoke_response_contract,
)


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


def test_smoke_cli_rotates_multiple_inputs_and_uses_gate_default_count() -> None:
    args = build_parser().parse_args([
        "smoke-qwen", "one.jpg", "two.jpg", "--gate", "delivery",
    ])
    assert args.image == [Path("one.jpg"), Path("two.jpg")]
    assert args.repeats is None
    assert args.gate == "delivery"
