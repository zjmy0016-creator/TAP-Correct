"""Step 1: Strawberry-DS (Roboflow YOLO) crop + 6-class -> 3-class mapping.

Roboflow export structure: {train,valid,test}/images/*.jpg + {train,valid,test}/labels/*.txt
data.yaml names: 0=Early-Turning 1=Green 2=Late-Turning 3=Red 4=Turning 5=White

Mapping:
  Green(1), White(5) -> unripe
  Early-Turning(0), Late-Turning(2), Turning(4) -> turning
  Red(3) -> ripe

Usage: python scripts/strawberryds_prep.py --data D:/data/strawberryds
"""
import argparse
from pathlib import Path
from PIL import Image

MAP_6_TO_3 = {1: "unripe", 5: "unripe", 0: "turning", 2: "turning", 4: "turning", 3: "ripe"}

def run(data_root: Path, out_root: Path):
    for c in ("unripe", "turning", "ripe"):
        (out_root / c).mkdir(parents=True, exist_ok=True)
    crop_id = 0
    for split in ("train", "valid", "test"):
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        if not lbl_dir.exists():
            print(f"skip {split}: no labels dir")
            continue
        for lbl_file in lbl_dir.glob("*.txt"):
            img_file = img_dir / (lbl_file.stem + ".jpg")
            if not img_file.exists():
                img_file = img_dir / (lbl_file.stem + ".png")
            if not img_file.exists():
                continue
            img = Image.open(img_file).convert("RGB")
            W, H = img.size
            for line in lbl_file.read_text(encoding="utf-8").strip().split("\n"):
                if not line.strip():
                    continue
                parts = line.split()
                cls3 = MAP_6_TO_3.get(int(parts[0]))
                if cls3 is None:
                    continue
                xc, yc, w, h = map(float, parts[1:5])
                x1, y1 = max(0, int((xc - w / 2) * W)), max(0, int((yc - h / 2) * H))
                x2, y2 = min(W, int((xc + w / 2) * W)), min(H, int((yc + h / 2) * H))
                if x2 <= x1 or y2 <= y1:
                    continue
                img.crop((x1, y1, x2, y2)).save(out_root / cls3 / f"sbds_{crop_id:04d}.jpg")
                crop_id += 1
    print(f"Cropped {crop_id} fruits -> {out_root}")
    for c in ("unripe", "turning", "ripe"):
        print(f"  {c}: {len(list((out_root / c).glob('*.jpg')))}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    run(args.data, args.out or args.data / "crops")
