"""Stage: Extract CLIP features for Strawberry-DS crops (open_clip, matches main pipeline).

Output: outputs/features_strawberryds_vitb32.npz (all samples split=test).
NOTE: default model ViT-B-32 to match main features_vitb32.npz used by eval.

Usage: python scripts/strawberryds_extract_features.py --crops D:/data/strawberryds/crops
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, open_clip
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CLASSES = ("ripe", "turning", "unripe")
PROMPTS = {
    "ripe": ["a photo of a ripe strawberry", "a photo of a mature strawberry", "a photo of a red strawberry",
             "a photo of a fully ripe strawberry", "a photo of a harvestable strawberry"],
    "turning": ["a photo of a turning strawberry", "a photo of a partially ripe strawberry",
                "a photo of a pink strawberry", "a photo of a transitional strawberry",
                "a photo of a borderline strawberry"],
    "unripe": ["a photo of an unripe strawberry", "a photo of an immature strawberry", "a photo of a green strawberry",
               "a photo of a white strawberry", "a photo of an early-stage strawberry"],
}

def run(crop_dir: Path, model_name: str, pretrained: str, out_npz: Path, batch_size: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device} model={model_name}/{pretrained}")
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    paths, classes_arr = [], []
    for cls in CLASSES:
        for f in sorted((crop_dir / cls).glob("*.jpg")):
            paths.append(f); classes_arr.append(cls)
    print(f"total crops: {len(paths)}")

    feats = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)
    with torch.no_grad():
        batch, idxs = [], []
        for i, p in enumerate(paths):
            batch.append(preprocess(Image.open(p).convert("RGB"))); idxs.append(i)
            if len(batch) == batch_size or i == len(paths) - 1:
                x = torch.stack(batch).to(device)
                f = model.encode_image(x); f = f / f.norm(dim=-1, keepdim=True)
                feats[idxs] = f.cpu().numpy().astype(np.float32)
                batch, idxs = [], []

    text_feats = {}
    with torch.no_grad():
        for cls in CLASSES:
            toks = tokenizer(PROMPTS[cls]).to(device)
            tf = model.encode_text(toks); tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats[cls] = tf.cpu().numpy().astype(np.float32)

    save = {"image_feats": feats, "image_paths": np.array([str(p) for p in paths]),
            "classes": np.array(classes_arr), "splits": np.full(len(classes_arr), "test")}
    for cls in CLASSES:
        save[f"text_{cls}"] = text_feats[cls]; save[f"prompts_{cls}"] = np.array(PROMPTS[cls])
    np.savez(out_npz, **save)
    print(f"saved {len(paths)} -> {out_npz}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", type=Path, required=True)
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "features_strawberryds_vitb32.npz")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()
    run(args.crops, args.model, args.pretrained, args.out, args.batch_size)
