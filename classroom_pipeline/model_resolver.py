from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"


MODEL_ALIASES: dict[str, tuple[str, ...]] = {
    "yolo11x.pt": ("models/yolo11x.pt", "yolo11x.pt"),
    "yolo11n-pose.pt": ("models/yolo11n-pose.pt", "yolo11n-pose.pt"),
    "runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt": (
        "models/behavior/official_yolo11s_detect_e150_v1_best.pt",
        "models/official_yolo11s_detect_e150_v1_best.pt",
        "runs/detect/official_yolo11s_detect_e150_v1/weights/best.pt",
    ),
    "runs/detect/case_yolo_train/weights/best.pt": (
        "models/behavior/case_yolo_train_best.pt",
        "models/case_yolo_train_best.pt",
        "runs/detect/case_yolo_train/weights/best.pt",
    ),
}


def _norm(raw: str) -> str:
    return str(raw or "").strip().replace("\\", "/")


def candidate_model_paths(model_arg: str, *, project_root: Path = PROJECT_ROOT) -> list[Path]:
    text = str(model_arg or "").strip()
    if not text:
        return []

    norm = _norm(text)
    candidates: list[Path] = []
    for rel in MODEL_ALIASES.get(norm, ()):
        candidates.append((project_root / rel).resolve())

    raw_path = Path(text)
    if raw_path.is_absolute():
        candidates.append(raw_path.resolve())
    else:
        candidates.extend(
            [
                (project_root / raw_path).resolve(),
                (MODELS_DIR / raw_path.name).resolve() if len(raw_path.parts) == 1 else (MODELS_DIR / raw_path).resolve(),
            ]
        )

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path).lower()
        if key not in seen:
            deduped.append(path)
            seen.add(key)
    return deduped


def resolve_model_reference(model_arg: str, *, project_root: Path = PROJECT_ROOT, required: bool = False) -> str:
    """Resolve local model files after centralizing weights under models/.

    If the argument is a model name rather than a path and no local file exists,
    keep the original value so libraries such as Ultralytics can still resolve
    built-in/downloadable model names.
    """
    text = str(model_arg or "").strip()
    if not text:
        if required:
            raise FileNotFoundError("empty model path")
        return text

    for path in candidate_model_paths(text, project_root=project_root):
        if path.exists():
            return str(path)

    norm = _norm(text)
    is_known_alias = norm in MODEL_ALIASES
    raw_path = Path(text)
    looks_like_path = raw_path.is_absolute() or len(raw_path.parts) > 1 or "/" in norm
    if required or is_known_alias or looks_like_path:
        tried = ", ".join(str(p) for p in candidate_model_paths(text, project_root=project_root))
        raise FileNotFoundError(f"model not found: {text}; tried: {tried}")

    return text


def try_resolve_model_reference(model_arg: str, *, project_root: Path = PROJECT_ROOT) -> str | None:
    try:
        return resolve_model_reference(model_arg, project_root=project_root, required=True)
    except FileNotFoundError:
        return None
