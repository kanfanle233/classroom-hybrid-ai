import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable, Optional

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
    row = {
        "start": 0.0,
        "end": round(_video_duration(video_path), 3),
        "text": f"[ASR_EMPTY:{reason}]",
        "source": "asr_placeholder",
        "is_placeholder": True,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_wav(video_path: Path, wav_path: Path, ffmpeg: str, sr: int) -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sr),
        "-c:a",
        "pcm_s16le",
        str(wav_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _segments_from_response(resp: Any) -> list[dict]:
    out: list[dict] = []
    segments: Optional[Iterable[Any]] = None

    if isinstance(resp, dict):
        segments = resp.get("segments")
    else:
        segments = getattr(resp, "segments", None)

    if segments:
        for seg in segments:
            if isinstance(seg, dict):
                start = _safe_float(seg.get("start"), 0.0)
                end = _safe_float(seg.get("end"), start + 0.2)
                text = str(seg.get("text", "")).strip()
            else:
                start = _safe_float(getattr(seg, "start", 0.0), 0.0)
                end = _safe_float(getattr(seg, "end", start + 0.2), start + 0.2)
                text = str(getattr(seg, "text", "")).strip()
            if not text:
                continue
            if end <= start:
                end = start + 0.2
            out.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    return out


def _text_from_response(resp: Any) -> str:
    if isinstance(resp, dict):
        return str(resp.get("text", "")).strip()
    return str(getattr(resp, "text", "")).strip()


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="OpenAI ASR to transcript.jsonl with non-empty guarantee.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--model", type=str, default="gpt-4o-mini-transcribe")
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--api_key_env", type=str, default="OPENAI_API_KEY")
    parser.add_argument("--max_file_mb", type=float, default=24.5)
    parser.add_argument("--timeout_sec", type=float, default=180.0)
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
    wav_path = out_dir / "asr_audio_16k.wav"
    transcript_path = out_dir / "transcript.jsonl"

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        _write_placeholder(transcript_path, video_path, f"missing_{args.api_key_env.lower()}")
        print(f"[WARN] missing {args.api_key_env}, wrote placeholder: {transcript_path}")
        return

    try:
        from openai import OpenAI
    except Exception:
        _write_placeholder(transcript_path, video_path, "openai_sdk_not_installed")
        print(f"[WARN] openai sdk missing, wrote placeholder: {transcript_path}")
        return

    try:
        _extract_wav(video_path, wav_path, ffmpeg=args.ffmpeg, sr=int(args.sr))
    except Exception:
        _write_placeholder(transcript_path, video_path, "ffmpeg_extract_failed")
        print(f"[WARN] ffmpeg failed, wrote placeholder: {transcript_path}")
        return

    size_mb = wav_path.stat().st_size / (1024.0 * 1024.0)
    if size_mb > float(args.max_file_mb):
        _write_placeholder(transcript_path, video_path, "openai_audio_too_large")
        print(f"[WARN] wav too large ({size_mb:.2f} MB), wrote placeholder: {transcript_path}")
        return

    client = OpenAI(api_key=api_key, timeout=float(args.timeout_sec))
    response: Any
    try:
        with wav_path.open("rb") as f:
            response = client.audio.transcriptions.create(
                model=str(args.model),
                file=f,
                language=str(args.lang) if args.lang else None,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
    except Exception:
        try:
            with wav_path.open("rb") as f:
                response = client.audio.transcriptions.create(
                    model=str(args.model),
                    file=f,
                    language=str(args.lang) if args.lang else None,
                )
        except Exception:
            _write_placeholder(transcript_path, video_path, "openai_transcribe_failed")
            print(f"[WARN] openai transcribe failed, wrote placeholder: {transcript_path}")
            return

    rows = _segments_from_response(response)
    if not rows:
        text = _text_from_response(response)
        if text:
            rows = [{"start": 0.0, "end": round(_video_duration(video_path), 3), "text": text}]

    if not rows:
        _write_placeholder(transcript_path, video_path, "empty_openai_result")
        print(f"[WARN] empty openai result, wrote placeholder: {transcript_path}")
        return

    with transcript_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(
                json.dumps(
                    {
                        "start": round(_safe_float(row["start"], 0.0), 3),
                        "end": round(max(_safe_float(row["end"], 0.2), _safe_float(row["start"], 0.0) + 0.2), 3),
                        "text": str(row["text"]).strip(),
                        "source": f"openai:{args.model}",
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    print(f"[DONE] transcript: {transcript_path}")
    print(f"[INFO] segments: {len(rows)}")
    print(f"[INFO] openai asr config: model={args.model} lang={args.lang} size_mb={size_mb:.2f}")


if __name__ == "__main__":
    main()
