from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


CLASS_NAMES = ("ripe", "turning", "unripe")
REQUIRED_COLUMNS = ("image_path", "class", "split")


@dataclass(frozen=True)
class CropRecord:
    image_path: Path
    class_name: str
    split: str
    metadata: dict[str, str] = field(default_factory=dict)

    def to_episode_item(self) -> dict[str, str]:
        item = {
            "image_path": self.image_path.as_posix(),
            "class": self.class_name,
            "split": self.split,
        }
        item.update(self.metadata)
        return item


def load_crop_index(
    index_path: str | Path,
    *,
    dataset_root: str | Path | None = None,
    classes: Iterable[str] = CLASS_NAMES,
) -> list[CropRecord]:
    index_path = Path(index_path)
    root = Path(dataset_root) if dataset_root is not None else index_path.parent
    expected_classes = tuple(classes)
    expected_set = set(expected_classes)
    records: list[CropRecord] = []

    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"Empty crop index: {index_path}")
        missing = [column for column in REQUIRED_COLUMNS if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"Crop index missing columns: {', '.join(missing)}")

        for row_number, row in enumerate(reader, start=2):
            class_name = (row.get("class") or "").strip()
            split = (row.get("split") or "").strip()
            raw_image_path = (row.get("image_path") or "").strip()

            if class_name not in expected_set:
                raise ValueError(f"Unexpected class at row {row_number}: {class_name!r}")
            if not split:
                raise ValueError(f"Missing split at row {row_number}")
            if not raw_image_path:
                raise ValueError(f"Missing image_path at row {row_number}")

            image_path = Path(raw_image_path)
            if not image_path.is_absolute():
                image_path = root / image_path
            if not image_path.exists():
                raise FileNotFoundError(f"Crop image does not exist at row {row_number}: {image_path}")

            metadata = {
                key: value
                for key, value in row.items()
                if key not in REQUIRED_COLUMNS and value is not None
            }
            records.append(CropRecord(image_path=image_path, class_name=class_name, split=split, metadata=metadata))

    discovered = {record.class_name for record in records}
    if discovered != expected_set:
        missing = sorted(expected_set - discovered)
        extra = sorted(discovered - expected_set)
        details = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError("Crop index must contain exactly the expected classes: " + ", ".join(details))

    return sorted(records, key=lambda record: (record.split, record.class_name, record.image_path.as_posix()))


def sample_episode(
    records: Iterable[CropRecord],
    *,
    k_shot: int,
    query_per_class: int | None = None,
    support_split: str = "support_pool",
    query_split: str = "query_test",
    seed: int = 0,
    classes: Iterable[str] = CLASS_NAMES,
) -> dict[str, object]:
    if k_shot <= 0:
        raise ValueError("k_shot must be positive")
    if query_per_class is not None and query_per_class <= 0:
        raise ValueError("query_per_class must be positive when provided")

    class_names = tuple(classes)
    rng = random.Random(seed)
    by_split_class: dict[tuple[str, str], list[CropRecord]] = {}
    for record in records:
        by_split_class.setdefault((record.split, record.class_name), []).append(record)

    support_items: list[dict[str, str]] = []
    query_items: list[dict[str, str]] = []
    for class_name in class_names:
        support_pool = sorted(
            by_split_class.get((support_split, class_name), []),
            key=lambda record: record.image_path.as_posix(),
        )
        query_pool = sorted(
            by_split_class.get((query_split, class_name), []),
            key=lambda record: record.image_path.as_posix(),
        )
        if len(support_pool) < k_shot:
            raise ValueError(
                f"Not enough support crops for class {class_name!r}: "
                f"need {k_shot}, found {len(support_pool)}"
            )

        query_count = len(query_pool) if query_per_class is None else query_per_class
        if len(query_pool) < query_count:
            raise ValueError(
                f"Not enough query crops for class {class_name!r}: "
                f"need {query_count}, found {len(query_pool)}"
            )

        support_items.extend(record.to_episode_item() for record in rng.sample(support_pool, k_shot))
        query_items.extend(record.to_episode_item() for record in rng.sample(query_pool, query_count))

    return {
        "k_shot": k_shot,
        "seed": seed,
        "support_split": support_split,
        "query_split": query_split,
        "classes": list(class_names),
        "support": support_items,
        "query": query_items,
    }
