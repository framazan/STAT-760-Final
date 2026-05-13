"""
evaluate.py — Standalone evaluation, prediction, and confusion matrix.

Usage:
    # Evaluate a checkpoint on the test set
    python evaluate.py --checkpoint checkpoints/firenet_best.pt

    # Predict on arbitrary images
    python evaluate.py --checkpoint checkpoints/firenet_best.pt --predict path/to/images/
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config as C
from data import get_dataloaders, get_eval_transform, list_image_paths, ImagePathDataset
from models import build_model
from utils import get_device, set_seed, Metrics, log


def evaluate(model, loader, criterion, device) -> Metrics:
    """Evaluate model, return loss + accuracy."""
    model.eval()
    total_loss = total_correct = total_count = 0
    all_preds = []
    all_labels = []
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, targets)
            preds = logits.argmax(1)
            total_loss += loss.item() * targets.size(0)
            total_correct += (preds == targets).sum().item()
            total_count += targets.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
    return Metrics(total_loss / total_count, total_correct / total_count), \
           np.array(all_preds), np.array(all_labels)


def confusion_matrix_2x2(preds, labels, classes):
    """Print a 2-class confusion matrix."""
    from collections import Counter
    cm = {}
    for true_c in range(len(classes)):
        for pred_c in range(len(classes)):
            cm[(true_c, pred_c)] = 0
    for p, l in zip(preds, labels):
        cm[(l, p)] += 1

    log(f"\nConfusion Matrix:")
    log(f"{'':>12} | {'Pred ' + classes[0]:>15} | {'Pred ' + classes[1]:>15}")
    log(f"{'-'*50}")
    for true_c in range(len(classes)):
        row = f"{'True ' + classes[true_c]:>12} |"
        for pred_c in range(len(classes)):
            row += f" {cm[(true_c, pred_c)]:>15} |"
        log(row)

    # Precision, Recall, F1 for positive class
    tp = cm[(1, 1)]
    fp = cm[(0, 1)]
    fn = cm[(1, 0)]
    tn = cm[(0, 0)]
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-8)
    log(f"\nPositive class metrics:")
    log(f"  Precision: {precision:.4f}")
    log(f"  Recall:    {recall:.4f}")
    log(f"  F1 Score:  {f1:.4f}")
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": precision, "recall": recall, "f1": f1}





def predict_images(model, image_path, device, classes):
    """Predict class for images at a given path."""
    eval_tf = get_eval_transform()
    paths = list_image_paths(image_path)
    dataset = ImagePathDataset(paths, transform=eval_tf)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)

    model.eval()
    log(f"\nPredicting {len(dataset)} images...")
    with torch.inference_mode():
        for images, fpaths in loader:
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            confs, indices = probs.max(dim=1)
            for fp, conf, idx, p in zip(fpaths, confs, indices, probs):
                cls = classes[int(idx)]
                prob_str = ", ".join(f"{classes[i]}={float(v):.4f}"
                                     for i, v in enumerate(p))
                log(f"  {fp}\t{cls}\tconf={float(conf):.4f}\t{prob_str}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate wildfire models")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--predict", type=str, default=None,
                        help="Image file or directory to classify")
    # Test-Time Augmentation (TTA) support removed
    parser.add_argument("--data_root", type=str, default=C.DATA_ROOT)
    parser.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    args = parser.parse_args()

    set_seed(C.SEED)
    device = get_device()
    log(f"Device: {device}")

    # load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model_name = ckpt.get("model_name", "simple")
    classes = ckpt.get("classes", list(C.CLASSES))
    log(f"Model: {model_name}  |  Classes: {classes}")

    model = build_model(model_name, num_classes=len(classes), pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    if args.predict:
        predict_images(model, args.predict, device, classes)
        return

    # evaluate on test set
    _, _, test_ds, _, _, test_loader = get_dataloaders(
        args.data_root, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    metrics, preds, labels = evaluate(model, test_loader, criterion, device)
    log(f"\nTest Loss: {metrics.loss:.4f}  |  Test Acc: {metrics.acc:.4f}")
    confusion_matrix_2x2(preds, labels, classes)

    # TTA removed — no additional test-time augmentation performed


if __name__ == "__main__":
    main()
