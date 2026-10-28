"""Indexed student-local metadata with the legacy stable selection order."""
from __future__ import annotations

from bisect import bisect_left, bisect_right
import math
from collections import defaultdict
from typing import Any, Mapping, Sequence


class TargetLocalizationIndex:
    def __init__(self, rows: Sequence[Mapping[str, Any]]) -> None:
        groups: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        self.irregular: list[tuple[Any, ...]] = []
        ordinal = 0
        for row in rows:
            try:
                pts, frame = float(row.get("pts_seconds", -1.0)), int(row.get("frame_index", -1))
            except (TypeError, ValueError):
                continue
            for fallback, detection in enumerate(row.get("detections", ()), 1):
                if not isinstance(detection, Mapping):
                    continue
                entry = (pts, ordinal, frame, detection, fallback, row.get("detection_source", row.get("source", "locate_anything")))
                ordinal += 1
                if math.isfinite(pts):
                    groups[str(detection.get("student_id", ""))].append(entry)
                else:
                    self.irregular.append(entry)
        self.groups = {key: sorted(values, key=lambda item: (item[0], item[1])) for key, values in groups.items()}
        self.times = {key: [item[0] for item in values] for key, values in self.groups.items()}

    def query(self, candidate: Any, *, maximum_delta_seconds: float = 2.0, limit: int = 3,
              generic: bool = False) -> list[dict[str, Any]]:
        student = str(candidate.track_id)
        pts = candidate.trigger_pts_seconds
        rows, times = self.groups.get(student, []), self.times.get(student, [])
        start, end = bisect_left(times, pts - maximum_delta_seconds), bisect_right(times, pts + maximum_delta_seconds)
        entries = rows[start:end] + [entry for entry in self.irregular if str(entry[3].get("student_id", "")) == student]
        # Restore original iteration order before sorting; ties and NaNs retain
        # precisely the stable ordering of the previous full scan.
        entries.sort(key=lambda entry: entry[1])
        matches = []
        for timestamp, ordinal, frame, detection, fallback, source in entries:
            delta = abs(timestamp - pts)
            if delta > maximum_delta_seconds:
                continue
            bbox = detection.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            index = int(detection.get("locate_detection_index", detection.get("detection_index", fallback)))
            compact = {"frame_index": frame, "pts_seconds": timestamp, "student_id": student,
                       "locate_detection_index": index, "bbox": [float(value) for value in bbox]}
            if generic:
                compact["detection_index"] = compact.pop("locate_detection_index")
                compact["detection_source"] = detection.get("detection_source", detection.get("source", source))
            matches.append((delta, frame, index, compact))
        matches.sort(key=lambda item: (item[0], item[1], item[2]))
        return [item[3] for item in matches[:max(0, int(limit))]]
