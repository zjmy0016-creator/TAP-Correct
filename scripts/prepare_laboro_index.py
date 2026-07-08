"""Convert laboro_tomato crop_manifest.csv to crop_index.csv format.

Maps:
  level -> class (mature/turning/immature)
  patch -> image_path
  split -> split
"""
import csv
from pathlib import Path

def main():
    data_root = Path("data/laboro_tomato")
    manifest_path = data_root / "crop_manifest.csv"
    output_path = data_root / "crop_index.csv"

    print(f"Reading: {manifest_path}")

    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "image_path": r["patch"],  # crops\turning\test_IMG_1122_img0_ann0.jpg
                "class": r["level"],        # mature / turning / immature
                "split": r["split"]         # train / test
            })

    print(f"Total crops: {len(rows)}")
    print(f"Classes: {set(r['class'] for r in rows)}")
    print(f"Splits: {set(r['split'] for r in rows)}")

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "class", "split"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    main()
