# -*- coding: utf-8 -*-
"""Crop Laboro Tomato fruit boxes into maturity-level patch folders.

Expected input layout:

data/laboro_tomato/
  raw/
    ... COCO json files and source images ...

Output layout:

data/laboro_tomato/
  crops/
    mature/
    turning/
    immature/
  crop_manifest.csv

Laboro Tomato category names are mapped as:
- fully_ripened -> mature
- half_ripened -> turning
- green -> immature

The category prefixes `b_` and `l_` are ignored when present.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("Missing Pillow. Install it with: pip install pillow") from exc


RIPENESS_TO_LEVEL = {
    "fully_ripened": "mature",
    "half_ripened": "turning",
    "green": "immature",
}
LEVELS = ("mature", "turning", "immature")


def category_name_to_level(name: str) -> str | None:
    normalized = name.lower()
    if normalized.startswith(("b_", "l_")):
        normalized = normalized[2:]
    return RIPENESS_TO_LEVEL.get(normalized)


def find_coco_jsons(raw_dir: Path) -> list[Path]:
    found = []
    for path in raw_dir.rglob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, dict) and {"images", "annotations", "categories"}.issubset(data):
            found.append(path)
    return sorted(found)


def build_image_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in raw_dir.rglob("*"):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            index[path.name].append(path)
    return index


def resolve_image_path(file_name: str, json_path: Path, image_index: dict[str, list[Path]]) -> Path | None:
    direct = (json_path.parent / file_name).resolve()
    if direct.is_file():
        return direct
    hits = image_index.get(Path(file_name).name, [])
    return hits[0] if hits else None


def crop_laboro_tomato(data_root: Path, padding: int = 4, min_side: int = 20) -> Path:
    raw_dir = data_root / "raw"
    crops_dir = data_root / "crops"
    if not raw_dir.is_dir():
        raise SystemExit(f"Missing raw directory: {raw_dir}")

    for level in LEVELS:
        (crops_dir / level).mkdir(parents=True, exist_ok=True)

    json_paths = find_coco_jsons(raw_dir)
    if not json_paths:
        raise SystemExit(f"No COCO json files found under {raw_dir}")

    image_index = build_image_index(raw_dir)
    manifest_rows = []
    level_counter = Counter()
    split_level_counter: dict[str, Counter] = defaultdict(Counter)
    skipped_small = 0
    skipped_no_image = 0
    unmapped_categories = Counter()

    for json_path in json_paths:
        split = json_path.stem
        coco = json.loads(json_path.read_text(encoding="utf-8"))

        category_id_to_level = {}
        for category in coco["categories"]:
            level = category_name_to_level(category["name"])
            if level is None:
                unmapped_categories[category["name"]] += 1
            else:
                category_id_to_level[category["id"]] = level

        image_id_to_meta = {image["id"]: image for image in coco["images"]}
        annotations_by_image: dict[int, list[dict]] = defaultdict(list)
        for annotation in coco["annotations"]:
            annotations_by_image[annotation["image_id"]].append(annotation)

        opened_images: dict[Path, Image.Image] = {}
        for image_id, meta in image_id_to_meta.items():
            annotations = annotations_by_image.get(image_id, [])
            if not annotations:
                continue

            image_path = resolve_image_path(meta["file_name"], json_path, image_index)
            if image_path is None:
                skipped_no_image += 1
                continue

            if image_path not in opened_images:
                try:
                    opened_images[image_path] = Image.open(image_path).convert("RGB")
                except Exception:
                    skipped_no_image += 1
                    continue

            image = opened_images[image_path]
            width, height = image.size

            for annotation in annotations:
                level = category_id_to_level.get(annotation["category_id"])
                if level is None:
                    continue

                x, y, box_w, box_h = annotation["bbox"]
                if min(box_w, box_h) < min_side:
                    skipped_small += 1
                    continue

                x0 = max(0, int(round(x - padding)))
                y0 = max(0, int(round(y - padding)))
                x1 = min(width, int(round(x + box_w + padding)))
                y1 = min(height, int(round(y + box_h + padding)))
                if x1 <= x0 or y1 <= y0:
                    skipped_small += 1
                    continue

                patch = image.crop((x0, y0, x1, y1))
                stem = image_path.stem
                out_name = f"{split}_{stem}_img{image_id}_ann{annotation['id']}.jpg"
                out_path = crops_dir / level / out_name
                patch.save(out_path, quality=95)

                level_counter[level] += 1
                split_level_counter[split][level] += 1
                manifest_rows.append(
                    {
                        "split": split,
                        "level": level,
                        "orig_category_id": annotation["category_id"],
                        "src_image": os.path.relpath(image_path, data_root),
                        "image_id": image_id,
                        "ann_id": annotation["id"],
                        "bbox_x": round(x, 1),
                        "bbox_y": round(y, 1),
                        "bbox_w": round(box_w, 1),
                        "bbox_h": round(box_h, 1),
                        "patch": os.path.relpath(out_path, data_root),
                    }
                )

        for image in opened_images.values():
            image.close()

    manifest_path = data_root / "crop_manifest.csv"
    fieldnames = list(manifest_rows[0].keys()) if manifest_rows else ["split", "level"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    print("COCO files:")
    for path in json_paths:
        print(f"  {path}")
    print("Patch counts:")
    for level in LEVELS:
        print(f"  {level}: {level_counter[level]}")
    print(f"  total: {sum(level_counter.values())}")
    print("Counts by split:")
    for split in sorted(split_level_counter):
        counts = "  ".join(f"{level}={split_level_counter[split][level]}" for level in LEVELS)
        print(f"  {split}: {counts}")
    if unmapped_categories:
        print(f"Unmapped categories: {dict(unmapped_categories)}")
    print(f"Skipped: small_bbox={skipped_small} no_image={skipped_no_image}")
    print(f"manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--min_side", type=int, default=20)
    args = parser.parse_args()
    crop_laboro_tomato(args.data_root, padding=args.padding, min_side=args.min_side)


if __name__ == "__main__":
    main()
