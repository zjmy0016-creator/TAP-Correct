"""Zero-shot CLIP classification on laboro_tomato test set.

Verify whether turning collapse also occurs on tomato data.
"""
import argparse
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def class_metric_rows(true_labels, pred_labels, classes):
    """Return per-class precision/recall/F1/support rows."""
    precisions, recalls, f1s, supports = precision_recall_fscore_support(
        true_labels, pred_labels, labels=classes, zero_division=0
    )
    return [
        {
            "class": cls,
            "precision": float(precisions[i]),
            "recall": float(recalls[i]),
            "f1": float(f1s[i]),
            "support": int(supports[i]),
        }
        for i, cls in enumerate(classes)
    ]


def find_collapsed_class(metric_rows):
    """Identify the weakest class by F1, using recall as the tie breaker."""
    if not metric_rows:
        raise ValueError("metric_rows must not be empty")
    return min(metric_rows, key=lambda row: (row["f1"], row["recall"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=Path, default=Path("outputs/features_laboro_tomato_vitb16.npz"))
    args = ap.parse_args()

    print(f"Loading features: {args.npz}")
    data = np.load(args.npz, allow_pickle=True)

    # Load test split
    test_mask = data["splits"] == "test"
    test_feats = data["image_feats"][test_mask]
    test_labels = data["classes"][test_mask]

    # Get actual class order from data
    unique_classes = sorted(np.unique(test_labels))
    print(f"\nTest samples: {len(test_feats)}")
    print(f"Classes in data: {unique_classes}")

    # Count distribution
    for cls in unique_classes:
        count = (test_labels == cls).sum()
        print(f"  {cls:10s}: {count:4d}")

    # Build text prototypes (mean of 5 prompts per class, then L2 normalize)
    text_protos = {}
    for cls in unique_classes:
        text_vecs = data[f"text_{cls}"]  # (5, 512)
        mean_vec = text_vecs.mean(axis=0)
        text_protos[cls] = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)

    # Zero-shot classification: argmax similarity to text prototypes
    similarities = []
    for cls in unique_classes:
        sim = test_feats @ text_protos[cls]
        similarities.append(sim)
    similarities = np.stack(similarities, axis=1)  # (N, 3)

    pred_idx = similarities.argmax(axis=1)
    pred_labels = np.array([unique_classes[i] for i in pred_idx])

    # Evaluation
    print("\n" + "="*60)
    print("Zero-shot Classification Report (Text Prototypes)")
    print("="*60)
    print(classification_report(test_labels, pred_labels,
                                target_names=unique_classes,
                                digits=3,
                                zero_division=0))

    print("\nConfusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(test_labels, pred_labels, labels=unique_classes)
    print("             ", "  ".join(f"{c:>8s}" for c in unique_classes))
    for i, cls in enumerate(unique_classes):
        print(f"{cls:>12s}", "  ".join(f"{cm[i,j]:8d}" for j in range(len(unique_classes))))

    metric_rows = class_metric_rows(test_labels, pred_labels, unique_classes)
    collapsed = find_collapsed_class(metric_rows)

    print(f"\n{'='*60}")
    print("KEY COLLAPSE METRIC")
    print(f"   Collapsed class: {collapsed['class']}")
    print(f"   Precision:       {collapsed['precision']:.4f}")
    print(f"   Recall:          {collapsed['recall']:.4f}")
    print(f"   F1-score:        {collapsed['f1']:.4f}")
    print(f"{'='*60}")

    if collapsed["f1"] < 0.15:
        print("Collapsed-class failure CONFIRMED (F1 < 0.15)")
    elif collapsed["f1"] < 0.40:
        print("Collapsed-class failure is moderate (0.15 <= F1 < 0.40)")
    else:
        print("No severe collapsed-class failure observed (F1 >= 0.40)")


if __name__ == "__main__":
    main()
