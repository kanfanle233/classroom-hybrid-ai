import argparse
import json
import os
import re
import shutil
import site
import struct
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_WHISPER_MODEL = "Systran/faster-whisper-small"


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _check_audio_energy(wav_path: Path, min_rms: float = 50.0) -> dict:
    """Quick audio energy pre-check. Returns dict with rms, peak, has_signal."""
    try:
        with wave.open(str(wav_path), 'rb') as wf:
            n_frames = wf.getnframes()
            if n_frames < 100:
                return {"rms": 0.0, "peak": 0.0, "has_signal": False, "reason": "too_short"}
            frames = wf.readframes(min(n_frames, 16000 * 30))
            n_samples = len(frames) // 2
            audio = np.array(struct.unpack(f'<{n_samples}h', frames), dtype=np.float32)
            rms = float(np.sqrt(np.mean(audio ** 2)))
            peak = float(np.max(np.abs(audio)))
            has_signal = rms >= float(min_rms)
            return {
                "rms": round(rms, 1),
                "peak": round(peak, 1),
                "has_signal": has_signal,
                "reason": "" if has_signal else "low_energy",
            }
    except Exception as e:
        return {"rms": 0.0, "peak": 0.0, "has_signal": False, "reason": str(e)}


def _simplify_chinese(text: str) -> str:
    """Simplistic traditional-to-simplified Chinese mapping for common chars."""
    MAP = str.maketrans({
        '鈕': '钮', '開': '开', '關': '关', '鍵': '键', '書': '书',
        '畫': '画', '頭': '头', '體': '体', '門': '门', '個': '个',
        '麼': '么', '說': '说', '時': '时', '會': '会', '過': '过',
        '對': '对', '學': '学', '點': '点', '為': '为', '裡': '里',
        '後': '后', '來': '来', '樣': '样', '著': '着', '讓': '让',
        '問': '问', '聽': '听', '進': '进', '話': '话', '寫': '写',
        '課': '课', '師': '师', '嗎': '吗', '長': '长', '當': '当',
        '麼': '么', '見': '见', '覺': '觉', '認': '认', '邊': '边',
        '還': '还', '沒': '没', '從': '从', '間': '间', '動': '动',
        '應': '应', '實': '实', '臺': '台', '灣': '湾', '國': '国',
        '張': '张', '紙': '纸', '筆': '笔', '線': '线', '剛': '刚',
        '纔': '才', '帶': '带', '將': '将', '們': '们', '妳': '你',
        '給': '给', '與': '与', '親': '亲', '愛': '爱', '龍': '龙',
    })
    return text.translate(MAP)


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


def _extract_wav(video_path: Path, wav_path: Path, ffmpeg: str, sr: int, audio_filter: str = "") -> None:
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_error: Exception | None = None
    ffmpeg_bin = str(ffmpeg or "").strip()
    ffmpeg_exists = bool(ffmpeg_bin) and (
        Path(ffmpeg_bin).exists() if any(sep in ffmpeg_bin for sep in ("\\", "/")) else shutil.which(ffmpeg_bin) is not None
    )

    if ffmpeg_exists:
        cmd = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            "1",
            "-ar",
            str(sr),
        ]
        if str(audio_filter).strip():
            cmd.extend(["-af", str(audio_filter).strip()])
        cmd.extend([
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ])
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return
        except Exception as exc:
            ffmpeg_error = exc

    if str(audio_filter).strip():
        reason = f"ffmpeg unavailable for requested audio_filter={audio_filter!r}"
        if ffmpeg_error is not None:
            reason += f"; ffmpeg_error={ffmpeg_error}"
        raise RuntimeError(reason)

    try:
        _extract_wav_pyav(video_path, wav_path, sr)
    except Exception as exc:
        if ffmpeg_error is not None:
            raise RuntimeError(f"ffmpeg extraction failed ({ffmpeg_error}); pyav fallback failed ({exc})") from exc
        raise


def _extract_wav_pyav(video_path: Path, wav_path: Path, sr: int) -> None:
    try:
        import av
        from av.audio.resampler import AudioResampler
    except Exception as exc:
        raise RuntimeError(f"pyav_unavailable: {exc}") from exc

    chunks: list[np.ndarray] = []
    with av.open(str(video_path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise RuntimeError("no_audio_stream")

        resampler = AudioResampler(format="s16", layout="mono", rate=int(sr))
        for packet in container.demux(stream):
            for frame in packet.decode():
                for out_frame in resampler.resample(frame):
                    arr = out_frame.to_ndarray()
                    if arr.ndim > 1:
                        arr = arr.reshape(-1)
                    chunks.append(np.ascontiguousarray(arr.astype(np.int16, copy=False)))

    if not chunks:
        raise RuntimeError("no_audio_samples")

    audio = np.concatenate(chunks)
    with wave.open(str(wav_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sr))
        wf.writeframes(audio.tobytes())


def _optional_float(raw: str) -> float | None:
    text = str(raw).strip()
    if text == "":
        return None
    try:
        return float(text)
    except Exception:
        return None


def _segment_metrics(seg: Any) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
        value = getattr(seg, key, None)
        try:
            out[key] = None if value is None else float(value)
        except Exception:
            out[key] = None
    return out


def _segment_quality_tier(text: str, metrics: dict[str, float | None], args: argparse.Namespace) -> str:
    """Return quality tier: 'good', 'low_conf', or 'reject'."""
    min_chars = max(0, int(args.min_text_chars))
    if min_chars and len(text.strip()) < min_chars:
        return "reject"

    avg_logprob = metrics.get("avg_logprob")
    no_speech_prob = metrics.get("no_speech_prob")
    compression_ratio = metrics.get("compression_ratio")

    # Hard reject: definitely noise
    min_avg_logprob = _optional_float(getattr(args, 'min_avg_logprob', ''))
    if min_avg_logprob is not None and avg_logprob is not None and avg_logprob < -2.0:
        return "reject"
    max_no_speech_prob = _optional_float(getattr(args, 'max_no_speech_prob', ''))
    if max_no_speech_prob is not None and no_speech_prob is not None and no_speech_prob > 0.8:
        return "reject"
    max_compression_ratio = _optional_float(getattr(args, 'max_compression_ratio', ''))
    if max_compression_ratio is not None and compression_ratio is not None and compression_ratio > 5.0:
        return "reject"

    # Low confidence: accept but flag
    if min_avg_logprob is not None and avg_logprob is not None and avg_logprob < min_avg_logprob:
        return "low_conf"
    if max_no_speech_prob is not None and no_speech_prob is not None and no_speech_prob > max_no_speech_prob:
        return "low_conf"

    return "good"


def _rejection_reason(text: str, metrics: dict[str, float | None], args: argparse.Namespace) -> str:
    """Legacy: kept for backward compat. Returns empty string = accept."""
    return ""


def _write_quality_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _write_placeholder_quality_report(
    *,
    path: Path,
    video_path: Path,
    wav_path: Path,
    transcript_path: Path,
    model: str,
    device: str,
    compute_type: str,
    reason: str,
    audio_check: dict | None = None,
    error_detail: str = "",
) -> None:
    _write_quality_report(
        path,
        {
            "video": str(video_path),
            "wav": str(wav_path),
            "transcript": str(transcript_path),
            "model": str(model),
            "device": str(device),
            "compute_type": str(compute_type),
            "beam_size": 0,
            "vad_filter": 0,
            "condition_on_previous_text": 0,
            "audio_filter": "",
            "thresholds": {},
            "audio_energy": audio_check or {"rms": 0.0, "peak": 0.0, "has_signal": False, "reason": reason},
            "info": {},
            "segments_raw": 0,
            "segments_good": 0,
            "segments_low_conf": 0,
            "segments_accepted": 0,
            "segments_rejected": 0,
            "rejected": [],
            "status": "placeholder",
            "error_reason": reason,
            "error_detail": error_detail,
        },
    )


def _configure_windows_cuda_dll_paths() -> list[str]:
    if os.name != "nt":
        return []
    added: list[str] = []
    roots: list[Path] = []
    try:
        roots.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass
    try:
        user_site = site.getusersitepackages()
        if user_site:
            roots.append(Path(user_site))
    except Exception:
        pass
    roots.append(Path(sys.prefix) / "Lib" / "site-packages")

    seen: set[str] = set()
    for root in roots:
        nvidia_root = root / "nvidia"
        if not nvidia_root.exists():
            continue
        for bin_dir in nvidia_root.glob("*/bin"):
            if not bin_dir.is_dir():
                continue
            key = str(bin_dir.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                os.add_dll_directory(key)
            except Exception:
                continue
            os.environ["PATH"] = key + os.pathsep + os.environ.get("PATH", "")
            added.append(key)
    return added


def main() -> None:
    base_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description="Whisper ASR to transcript.jsonl with non-empty guarantee.")
    parser.add_argument("--video", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--lang", type=str, default="zh")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_WHISPER_MODEL,
        help="faster-whisper / CTranslate2 model id or local path",
    )
    parser.add_argument("--device", type=str, default="cpu", help="cpu/cuda/auto")
    parser.add_argument("--compute_type", type=str, default="int8", help="int8/float16/int8_float16")
    parser.add_argument("--beam_size", type=int, default=5)
    parser.add_argument("--vad_filter", type=int, default=0, help="1=enable VAD filter")
    parser.add_argument("--condition_on_previous_text", type=int, default=0, help="1=enable context carry-over")
    parser.add_argument("--audio_filter", type=str, default="", help="optional ffmpeg -af filter chain")
    parser.add_argument("--min_avg_logprob", type=str, default="", help="reject segments below this avg_logprob")
    parser.add_argument("--max_no_speech_prob", type=str, default="", help="reject segments above this no_speech_prob")
    parser.add_argument("--max_compression_ratio", type=str, default="", help="reject segments above this compression_ratio")
    parser.add_argument("--min_text_chars", type=int, default=1)
    parser.add_argument("--min_audio_rms", type=float, default=50.0, help="Skip ASR if audio RMS < this")
    parser.add_argument("--accept_low_conf", type=int, default=1, help="1=accept low-confidence segments with flag")
    parser.add_argument("--simplify_chinese", type=int, default=1, help="1=convert traditional to simplified Chinese")
    parser.add_argument("--quality_report", type=str, default="")
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
    quality_report_path = Path(args.quality_report) if str(args.quality_report).strip() else out_dir / "asr_quality_report.json"
    if not quality_report_path.is_absolute():
        quality_report_path = (out_dir / quality_report_path).resolve() if quality_report_path.parent == Path(".") else (base_dir / quality_report_path).resolve()
    added_dll_paths = _configure_windows_cuda_dll_paths()
    device = str(args.device).strip().lower()
    if device == "auto":
        device = "cuda"

    try:
        from faster_whisper import WhisperModel
    except Exception as exc:
        _write_placeholder(transcript_path, video_path, "faster_whisper_not_installed")
        _write_placeholder_quality_report(
            path=quality_report_path,
            video_path=video_path,
            wav_path=wav_path,
            transcript_path=transcript_path,
            model=str(args.model),
            device=device,
            compute_type=str(args.compute_type),
            reason="faster_whisper_not_installed",
            error_detail=str(exc),
        )
        print(f"[WARN] faster-whisper missing, wrote placeholder: {transcript_path}")
        return

    try:
        _extract_wav(video_path, wav_path, ffmpeg=args.ffmpeg, sr=int(args.sr), audio_filter=str(args.audio_filter))
    except Exception as exc:
        _write_placeholder(transcript_path, video_path, "ffmpeg_extract_failed")
        _write_placeholder_quality_report(
            path=quality_report_path,
            video_path=video_path,
            wav_path=wav_path,
            transcript_path=transcript_path,
            model=str(args.model),
            device=device,
            compute_type=str(args.compute_type),
            reason="ffmpeg_extract_failed",
            error_detail=str(exc),
        )
        print(f"[WARN] ffmpeg failed, wrote placeholder: {transcript_path}")
        return

    # ====== Audio energy pre-check ======
    audio_check = _check_audio_energy(wav_path, min_rms=float(args.min_audio_rms))
    if not audio_check["has_signal"]:
        _write_placeholder(transcript_path, video_path, f"audio_{audio_check.get('reason', 'low_energy')}")
        quality_report = {
            "video": str(video_path), "wav": str(wav_path), "transcript": str(transcript_path),
            "model": str(args.model), "device": device, "compute_type": str(args.compute_type),
            "beam_size": int(args.beam_size), "vad_filter": int(args.vad_filter),
            "condition_on_previous_text": int(args.condition_on_previous_text),
            "audio_filter": str(args.audio_filter),
            "thresholds": {
                "min_avg_logprob": _optional_float(args.min_avg_logprob),
                "max_no_speech_prob": _optional_float(args.max_no_speech_prob),
                "max_compression_ratio": _optional_float(args.max_compression_ratio),
                "min_text_chars": int(args.min_text_chars),
                "min_audio_rms": float(args.min_audio_rms),
            },
            "audio_energy": audio_check,
            "info": {}, "segments_raw": 0, "segments_accepted": 0,
            "segments_low_conf": 0, "segments_rejected": 0, "rejected": [],
            "status": "placeholder",
        }
        _write_quality_report(quality_report_path, quality_report)
        print(f"[SKIP] audio energy too low: rms={audio_check['rms']} < {args.min_audio_rms}")
        return

    # ====== Whisper transcription ======
    info_summary: dict[str, Any] = {}
    try:
        model = WhisperModel(args.model, device=device, compute_type=str(args.compute_type))
        segments, info = model.transcribe(
            str(wav_path),
            language=args.lang,
            vad_filter=bool(int(args.vad_filter)),
            beam_size=max(1, int(args.beam_size)),
            condition_on_previous_text=bool(int(args.condition_on_previous_text)),
        )
        info_summary = {
            "language": getattr(info, "language", None),
            "language_probability": _safe_float(getattr(info, "language_probability", None), 0.0),
            "duration": _safe_float(getattr(info, "duration", None), 0.0),
        }
    except Exception as exc:
        _write_placeholder(transcript_path, video_path, "whisper_inference_failed")
        _write_placeholder_quality_report(
            path=quality_report_path,
            video_path=video_path,
            wav_path=wav_path,
            transcript_path=transcript_path,
            model=str(args.model),
            device=device,
            compute_type=str(args.compute_type),
            reason="whisper_inference_failed",
            audio_check=audio_check,
            error_detail=str(exc),
        )
        print(f"[WARN] whisper inference failed ({exc}), wrote placeholder: {transcript_path}")
        return

    accept_low_conf = bool(int(args.accept_low_conf))
    do_simplify = bool(int(args.simplify_chinese))

    try:
        count_good = 0
        count_low = 0
        count_rejected = 0
        raw_count = 0
        rejected: list[dict[str, Any]] = []
        with transcript_path.open("w", encoding="utf-8") as f:
            for seg in segments:
                text = str(seg.text).strip()
                if not text:
                    continue
                raw_count += 1
                if do_simplify:
                    text = _simplify_chinese(text)
                metrics = _segment_metrics(seg)
                tier = _segment_quality_tier(text, metrics, args)

                if tier == "reject":
                    count_rejected += 1
                    rejected.append({
                        "start": round(_safe_float(seg.start, 0.0), 3),
                        "end": round(_safe_float(seg.end, 0.0), 3),
                        "text": text, "tier": tier, **metrics,
                    })
                    continue

                if tier == "low_conf" and not accept_low_conf:
                    count_rejected += 1
                    rejected.append({
                        "start": round(_safe_float(seg.start, 0.0), 3),
                        "end": round(_safe_float(seg.end, 0.0), 3),
                        "text": text, "tier": tier, **metrics,
                    })
                    continue

                start = _safe_float(seg.start, 0.0)
                end = _safe_float(seg.end, start + 0.2)
                if end <= start:
                    end = start + 0.2

                quality_status = "accepted" if tier == "good" else "low_conf"
                row = {
                    "start": round(start, 3),
                    "end": round(end, 3),
                    "text": text,
                    "source": f"whisper:{args.model}",
                    "asr_quality_status": quality_status,
                    "asr_quality_tier": tier,
                    **metrics,
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                if tier == "good":
                    count_good += 1
                else:
                    count_low += 1
    except Exception as exc:
        _write_placeholder(transcript_path, video_path, "whisper_decode_failed")
        _write_placeholder_quality_report(
            path=quality_report_path,
            video_path=video_path,
            wav_path=wav_path,
            transcript_path=transcript_path,
            model=str(args.model),
            device=device,
            compute_type=str(args.compute_type),
            reason="whisper_decode_failed",
            audio_check=audio_check,
            error_detail=str(exc),
        )
        print(f"[WARN] whisper decode failed ({exc}), wrote placeholder: {transcript_path}")
        return

    total_accepted = count_good + count_low
    if total_accepted == 0:
        reason = "all_rejected" if raw_count > 0 else "empty_whisper_result"
        _write_placeholder(transcript_path, video_path, reason)

    quality_report = {
        "video": str(video_path), "wav": str(wav_path), "transcript": str(transcript_path),
        "model": str(args.model), "device": device, "compute_type": str(args.compute_type),
        "beam_size": int(args.beam_size), "vad_filter": int(args.vad_filter),
        "condition_on_previous_text": int(args.condition_on_previous_text),
        "audio_filter": str(args.audio_filter),
        "thresholds": {
            "min_avg_logprob": _optional_float(args.min_avg_logprob),
            "max_no_speech_prob": _optional_float(args.max_no_speech_prob),
            "max_compression_ratio": _optional_float(args.max_compression_ratio),
            "min_text_chars": int(args.min_text_chars),
            "min_audio_rms": float(args.min_audio_rms),
        },
        "audio_energy": audio_check,
        "info": info_summary,
        "segments_raw": raw_count,
        "segments_good": count_good,
        "segments_low_conf": count_low,
        "segments_accepted": total_accepted,
        "segments_rejected": count_rejected,
        "rejected": rejected[:20],
        "status": "ok" if total_accepted > 0 else "placeholder",
    }
    _write_quality_report(quality_report_path, quality_report)

    print(f"[DONE] transcript: {transcript_path}")
    print(f"[INFO] audio: rms={audio_check['rms']} peak={audio_check['peak']}")
    print(f"[INFO] segments: good={count_good} low_conf={count_low} rejected={count_rejected} raw={raw_count}")
    print(f"[INFO] quality_report={quality_report_path}")
    print(
        "[INFO] whisper config:",
        f"model={args.model}",
        f"device={device}",
        f"compute_type={args.compute_type}",
        f"beam_size={args.beam_size}",
        f"vad_filter={int(args.vad_filter)}",
    )
    if added_dll_paths:
        print(f"[INFO] added_dll_paths={len(added_dll_paths)}")


if __name__ == "__main__":
    main()
