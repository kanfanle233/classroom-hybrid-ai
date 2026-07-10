from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _resolve_from_yaml(dataset_yaml: Path, raw_path: str) -> Path:
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate.resolve()
    return (dataset_yaml.parent / candidate).resolve()


def _list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _parse_label_dir(label_dir: Path, class_names: dict[int, str]) -> dict[str, object]:
    per_class = Counter()
    malformed_lines = 0
    empty_label_files = 0
    total_boxes = 0

    for label_path in sorted(label_dir.glob("*.txt")):
        raw_lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not raw_lines:
            empty_label_files += 1
            continue

        for line in raw_lines:
            parts = line.split()
            if len(parts) < 5:
                malformed_lines += 1
                continue
            try:
                class_id = int(float(parts[0]))
            except ValueError:
                malformed_lines += 1
                continue

            total_boxes += 1
            per_class[class_id] += 1

    class_box_counts = []
    for class_id, class_name in sorted(class_names.items()):
        class_box_counts.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "box_count": int(per_class.get(class_id, 0)),
            }
        )

    return {
        "total_boxes": total_boxes,
        "empty_label_files": empty_label_files,
        "malformed_lines": malformed_lines,
        "class_box_counts": class_box_counts,
    }


def _judge_readiness(train_images: int, val_images: int, total_boxes: int, class_box_counts: list[dict[str, object]]) -> dict[str, object]:
    min_boxes = min((int(item["box_count"]) for item in class_box_counts), default=0)
    max_boxes = max((int(item["box_count"]) for item in class_box_counts), default=0)
    imbalance_ratio = round(max_boxes / min_boxes, 3) if min_boxes else None

    detection_ready = train_images >= 3000 and val_images >= 500 and total_boxes >= 50000 and min_boxes >= 500
    strong_detection_ready = train_images >= 6000 and val_images >= 1000 and total_boxes >= 150000 and min_boxes >= 2000

    notes = []
    if detection_ready:
        notes.append("Official YOLO detection finetune is feasible with the current dataset.")
    else:
        notes.append("The dataset is below a conservative threshold for stable official YOLO detection finetune.")

    if strong_detection_ready:
        notes.append("The dataset is strong enough for repeated baseline and ablation runs.")
    else:
        notes.append("Ablations are still possible, but stability may depend more on the split and augmentation.")

    notes.append(
        "This dataset is a detection dataset. It is not, by itself, a fully labeled temporal sequence dataset for RNN/LSTM/Transformer action modeling."
    )

    return {
        "detection_ready": detection_ready,
        "strong_detection_ready": strong_detection_ready,
        "min_class_box_count": min_boxes,
        "max_class_box_count": max_boxes,
        "class_imbalance_ratio": imbalance_ratio,
        "notes": notes,
    }


def _to_markdown(summary: dict[str, object]) -> str:
    lines = [
        "# Dataset Readiness",
        "",
        f"- dataset_yaml: `{summary['dataset_yaml']}`",
        f"- dataset_root: `{summary['dataset_root']}`",
        f"- train_images: `{summary['train_images']}`",
        f"- val_images: `{summary['val_images']}`",
        f"- total_images: `{summary['total_images']}`",
        f"- total_boxes: `{summary['total_boxes']}`",
        f"- detection_ready: `{summary['readiness']['detection_ready']}`",
        f"- strong_detection_ready: `{summary['readiness']['strong_detection_ready']}`",
        f"- min_class_box_count: `{summary['readiness']['min_class_box_count']}`",
        f"- class_imbalance_ratio: `{summary['readiness']['class_imbalance_ratio']}`",
        "",
        "## Class Box Counts",
        "",
        "| class_id | class_name | box_count |",
        "| --- | --- | ---: |",
    ]

    for item in summary["class_box_counts"]:
        lines.append(f"| {item['class_id']} | {item['class_name']} | {item['box_count']} |")

    lines.extend(["", "## Notes", ""])
    for note in summary["readiness"]["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Check whether a YOLO detection dataset is strong enough for official finetuning.")
    parser.add_argument("--data", required=True, help="Path to dataset.yaml")
    parser.add_argument("--out_json", default="", help="Optional output JSON path")
    parser.add_argument("--out_md", default="", help="Optional output markdown path")
    args = parser.parse_args()

    dataset_yaml = Path(args.data).resolve()
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"dataset yaml not found: {dataset_yaml}")

    config = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8"))
    dataset_root = _resolve_from_yaml(dataset_yaml, str(config["path"]))
    train_dir = _resolve_from_yaml(dataset_yaml, str(config["train"]))
    val_dir = _resolve_from_yaml(dataset_yaml, str(config["val"]))
    train_label_dir = (dataset_root / "labels" / "train").resolve()
    val_label_dir = (dataset_root / "labels" / "val").resolve()

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError("train or val image directory does not exist")
    if not train_label_dir.exists() or not val_label_dir.exists():
        raise FileNotFoundError("train or val label directory does not exist")

    names = config.get("names", {})
    class_names = {int(k): str(v) for k, v in names.items()}

    train_images = _list_images(train_dir)
    val_images = _list_images(val_dir)
    train_label_stats = _parse_label_dir(train_label_dir, class_names)
    val_label_stats = _parse_label_dir(val_label_dir, class_names)

    merged_counts = Counter()
    for item in train_label_stats["class_box_counts"]:
        merged_counts[int(item["class_id"])] += int(item["box_count"])
    for item in val_label_stats["class_box_counts"]:
        merged_counts[int(item["class_id"])] += int(item["box_count"])

    class_box_counts = []
    for class_id, class_name in sorted(class_names.items()):
        class_box_counts.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "box_count": int(merged_counts.get(class_id, 0)),
            }
        )

    total_boxes = int(train_label_stats["total_boxes"]) + int(val_label_stats["total_boxes"])
    readiness = _judge_readiness(len(train_images), len(val_images), total_boxes, class_box_counts)

    summary = {
        "dataset_yaml": str(dataset_yaml),
        "dataset_root": str(dataset_root),
        "train_images": len(train_images),
        "val_images": len(val_images),
        "total_images": len(train_images) + len(val_images),
        "total_boxes": total_boxes,
        "train_empty_label_files": int(train_label_stats["empty_label_files"]),
        "val_empty_label_files": int(val_label_stats["empty_label_files"]),
        "train_malformed_lines": int(train_label_stats["malformed_lines"]),
        "val_malformed_lines": int(val_label_stats["malformed_lines"]),
        "class_box_counts": class_box_counts,
        "readiness": readiness,
    }

    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    print(payload)

    if args.out_json:
        out_json = Path(args.out_json).resolve()
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(payload + "\n", encoding="utf-8")

    if args.out_md:
        out_md = Path(args.out_md).resolve()
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(_to_markdown(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
