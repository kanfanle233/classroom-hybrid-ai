import argparse
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _video_duration(video_path: Path) -> float:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.5
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    if fps <= 0:
        return 0.5
    return max(0.5, float(n) / float(fps))


def _write_placeholder(path: Path, video_path: Path, reason: str) -> None:
    dur = _video_duration(video_path)
    row = {
        "start": 0.0,
        "end": round(dur, 3),
        "text": f"[ASR_EMPTY:{reason}]",
        "source": "asr_placeholder",
        "is_placeholder": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _ensure_non_empty(path: Path, video_path: Path, reason: str) -> None:
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    return
    _write_placeholder(path, video_path, reason)


def _extract_wav(video_path: Path, wav_path: Path) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v > 1000.0:
        return v / 1000.0
    return v


def _normalize_sentence(sentence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    text = str(sentence.get("text", "")).strip()
    if not text:
        return None
    st = _to_seconds(sentence.get("begin_time", sentence.get("start_time", sentence.get("start"))))
    ed = _to_seconds(sentence.get("end_time", sentence.get("finish_time", sentence.get("end"))))
    if st is None or ed is None:
        return None
    if ed < st:
        st, ed = ed, st
    if ed <= st:
        ed = st + 0.2
    return {"start": round(st, 3), "end": round(ed, 3), "text": text, "source": "dashscope"}


def _deduplicate(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not segments:
        return []
    segments.sort(key=lambda s: (s["start"], s["end"]))
    out = [segments[0]]
    for seg in segments[1:]:
        prev = out[-1]
        if seg["text"] == prev["text"] and min(seg["end"], prev["end"]) > max(seg["start"], prev["start"]):
            prev["end"] = max(prev["end"], seg["end"])
        else:
            out.append(seg)
    return out


def _resolve_api_key() -> Optional[str]:
    return os.environ.get("DASHSCOPE_API_KEY")


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Realtime ASR (DashScope) with non-empty transcript guarantee.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--asr_model", type=str, default="paraformer-realtime-v1")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    video_path = Path(args.video)
    out_dir = Path(args.out_dir)
    if not video_path.is_absolute():
        video_path = (base_dir / video_path).resolve()
    if not out_dir.is_absolute():
        out_dir = (base_dir / out_dir).resolve()

    if not video_path.exists():
        raise FileNotFoundError(f"video not found: {video_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = out_dir / "transcript.jsonl"
    wav_path = out_dir / "asr_audio_16k.wav"

    api_key = _resolve_api_key()
    if not api_key:
        _write_placeholder(transcript_path, video_path, "missing_dashscope_api_key")
        print(f"[WARN] missing DASHSCOPE_API_KEY, wrote placeholder: {transcript_path}")
        return

    if importlib.util.find_spec("dashscope") is None:
        _write_placeholder(transcript_path, video_path, "dashscope_not_installed")
        print(f"[WARN] dashscope package missing, wrote placeholder: {transcript_path}")
        return

    import dashscope
    from dashscope.audio.asr import Recognition, RecognitionCallback, RecognitionResult

    if args.force:
        if wav_path.exists():
            wav_path.unlink(missing_ok=True)
        transcript_path.unlink(missing_ok=True)

    if not wav_path.exists():
        _extract_wav(video_path, wav_path)

    dashscope.api_key = api_key
    raw_sentences: List[Dict[str, Any]] = []

    class _Callback(RecognitionCallback):
        def on_complete(self) -> None:
            return None

        def on_error(self, result) -> None:
            print(f"[WARN] ASR error: {getattr(result, 'message', 'unknown')}")

        def on_event(self, result) -> None:
            sentence = result.get_sentence()
            try:
                is_end = RecognitionResult.is_sentence_end(sentence)
            except Exception:
                is_end = bool(sentence and sentence.get("text"))
            if is_end and isinstance(sentence, dict):
                raw_sentences.append(sentence)

    rec = Recognition(
        model=args.asr_model,
        format="wav",
        sample_rate=16000,
        callback=_Callback(),
    )

    rec.start()
    try:
        with wav_path.open("rb") as f:
            while True:
                chunk = f.read(3200)
                if not chunk:
                    break
                rec.send_audio_frame(chunk)
                time.sleep(0.05)
    finally:
        rec.stop()

    normalized = []
    for s in raw_sentences:
        seg = _normalize_sentence(s)
        if seg:
            normalized.append(seg)
    normalized = _deduplicate(normalized)

    if normalized:
        with transcript_path.open("w", encoding="utf-8") as f:
            for seg in normalized:
                f.write(json.dumps(seg, ensure_ascii=False) + "\n")
    _ensure_non_empty(transcript_path, video_path, "empty_dashscope_result")
    print(f"[DONE] transcript: {transcript_path}")


if __name__ == "__main__":
    main()
