"""Extract and cache frozen CLIP features for all strawberry crops.

Zero training, zero backprop: pure forward passes with a frozen CLIP encoder.
Caches image features (per crop) and text features (per prompt) to .npz so that
every downstream experiment reuses the same features. Run once.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np
import torch
import open_clip
from PIL import Image


def load_index(index_path):
    with open(index_path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=Path, default=Path("data/strawberry_crops"))
    ap.add_argument("--out", type=Path, default=Path("outputs/features_vitb16.npz"))
    ap.add_argument("--model", default="ViT-B-16")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  model={args.model}/{args.pretrained}")

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    index = load_index(args.data_dir / "crop_index.csv")
    print(f"crops in index: {len(index)}")

    paths = [args.data_dir / r["image_path"] for r in index]
    classes = [r["class"] for r in index]
    splits = [r["split"] for r in index]

    feats = np.zeros((len(paths), model.visual.output_dim), dtype=np.float32)
    with torch.no_grad():
        batch, idxs = [], []
        for i, p in enumerate(paths):
            batch.append(preprocess(Image.open(p).convert("RGB")))
            idxs.append(i)
            if len(batch) == args.batch_size or i == len(paths) - 1:
                x = torch.stack(batch).to(device)
                f = model.encode_image(x)
                f = f / f.norm(dim=-1, keepdim=True)
                feats[idxs] = f.cpu().numpy().astype(np.float32)
                batch, idxs = [], []
                if (i + 1) % 1024 < args.batch_size:
                    print(f"  encoded {i + 1}/{len(paths)}")

    prompt_bank = json.loads((args.data_dir / "prompt_bank.json").read_text())
    text_feats = {}
    with torch.no_grad():
        for cls, prompts in prompt_bank.items():
            toks = tokenizer(prompts).to(device)
            tf = model.encode_text(toks)
            tf = tf / tf.norm(dim=-1, keepdim=True)
            text_feats[cls] = tf.cpu().numpy().astype(np.float32)

    np.savez(args.out,
        image_feats=feats,
        image_paths=np.array([str(p) for p in paths]),
        classes=np.array(classes),
        splits=np.array(splits),
        text_ripe=text_feats["ripe"], text_turning=text_feats["turning"], text_unripe=text_feats["unripe"],
        prompts_ripe=np.array(prompt_bank["ripe"]),
        prompts_turning=np.array(prompt_bank["turning"]),
        prompts_unripe=np.array(prompt_bank["unripe"]))
    print(f"saved -> {args.out}  image_feats={feats.shape}")


if __name__ == "__main__":
    main()
