"""
train.py — Unified training script for all wildfire detection models.

Usage examples:
    # Train FireNet (custom CNN) — default
    python train.py --model firenet

    # Train SimpleCNN baseline
    python train.py --model simple --epochs 16 --lr 5e-3

    # Fine-tune ConvNeXt-Tiny
    python train.py --model convnext --transfer

    # Fine-tune ViT-Base (pretrained)
    python train.py --model vit --transfer
"""

import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import v2

import config as C
from data import get_dataloaders, compute_class_weights
from models import build_model
from utils import get_device, set_seed, Metrics, log


# ── Training / evaluation loop ──────────────────────────────────────────

def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    epoch: int | None = None,
    grad_clip: float = 0.0,
    scheduler=None,
    mixup_fn=None,
) -> Metrics:
    """Run one epoch of training or evaluation."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = len(loader)
    freq = max(1, n_batches // 5)
    phase = "Train" if is_train else "Eval "

    for i, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if is_train and mixup_fn is not None:
            images, mix_targets = mixup_fn(images, targets)
        else:
            mix_targets = targets

        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, mix_targets)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        # Compute accuracy using the original targets
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_count += bs

        if i % freq == 0 or i == n_batches - 1:
            rl = total_loss / total_count
            ra = total_correct / total_count
            log(f"  {phase} ep {epoch or '?'} | batch {i+1:>4}/{n_batches} | "
                f"loss {rl:.4f}  acc {ra:.4f}")

    if scheduler is not None and is_train:
        scheduler.step()

    return Metrics(loss=total_loss / total_count,
                   acc=total_correct / total_count)


# ── Transfer-learning helpers ────────────────────────────────────────────

def freeze_backbone(model: nn.Module):
    """Freeze all parameters, then unfreeze the classifier head."""
    for p in model.parameters():
        p.requires_grad = False
    # timm models expose .get_classifier() or .head / .classifier
    head = None
    if hasattr(model, "get_classifier"):
        head = model.get_classifier()
    elif hasattr(model, "head"):
        head = model.head
    elif hasattr(model, "classifier"):
        head = model.classifier
    if head is not None:
        for p in head.parameters():
            p.requires_grad = True


def unfreeze_all(model: nn.Module):
    for p in model.parameters():
        p.requires_grad = True


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Wildfire Detection Training")
    parser.add_argument("--model", type=str, default="firenet",
                        choices=["simple", "firenet", "convnext", "vit"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=C.DROPOUT)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    parser.add_argument("--transfer", action="store_true",
                        help="Use progressive unfreezing for timm models")
    parser.add_argument("--enable_mixup", dest="enable_mixup", action="store_true",
                        help="Enable MixUp/CutMix during training")
    parser.add_argument("--disable_mixup", dest="enable_mixup", action="store_false",
                        help="Disable MixUp/CutMix during training")
    parser.set_defaults(enable_mixup=C.ENABLE_MIXUP)
    parser.add_argument("--data_root", type=str, default=C.DATA_ROOT)
    parser.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--save_name", type=str, default=None,
                        help="Override checkpoint filename (saved in checkpoints/)")
    args = parser.parse_args()

    # ── Resolve defaults per model type ──────────────────────────────
    is_transfer    = args.transfer and args.model in ("convnext", "vit")

    epochs = args.epochs or (C.TL_EPOCHS if is_transfer else C.EPOCHS)
    lr = args.lr or (C.TL_LR_HEAD if is_transfer else C.LR)
    wd = args.weight_decay or C.WEIGHT_DECAY

    save_name = args.save_name or f"{args.model}_best.pt"
    save_path = os.path.join(C.SAVE_DIR, save_name)

    # ── Setup ────────────────────────────────────────────────────────
    set_seed(C.SEED)
    device = get_device()
    log(f"Device: {device}")

    train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader = \
        get_dataloaders(args.data_root, C.IMAGE_SIZE, args.batch_size, args.workers)

    log(f"Classes: {train_ds.classes}")
    log(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}  Test: {len(test_ds)}")

    # ── Model ────────────────────────────────────────────────────────
    model = build_model(args.model, num_classes=len(train_ds.classes),
                        pretrained=is_transfer, dropout=args.dropout)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("=" * 60)
    print(f"Total Parameters: {n_params:,}")
    print(f"Trainable Parameters: {n_train:,}")
    print("=" * 60)
    log(f"Model: {args.model}  |  params: {n_params:,}  |  trainable: {n_train:,}")

    # ── Loss ─────────────────────────────────────────────────────────
    if args.no_class_weights:
        weights = None
    else:
        weights = compute_class_weights(train_ds, device)
        log(f"Class weights: {weights.tolist()}")

    criterion = nn.CrossEntropyLoss(
        weight=weights,
        label_smoothing=C.LABEL_SMOOTH,
    )

    # ── MixUp / CutMix hook (applied at batch level)
    mixup_fn = None
    if args.enable_mixup:
        mixup_fn = v2.RandomChoice([
            v2.CutMix(num_classes=len(train_ds.classes), alpha=C.CUTMIX_ALPHA),
            v2.MixUp(num_classes=len(train_ds.classes), alpha=C.MIXUP_ALPHA),
        ])
        log(f"Using MixUp/CutMix (mixup_alpha={C.MIXUP_ALPHA}, cutmix_alpha={C.CUTMIX_ALPHA})")

    # ── Optimizer & scheduler ────────────────────────────────────────
    if is_transfer:
        freeze_backbone(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup_epochs = 3
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs - warmup_epochs), T_mult=1, eta_min=1e-6)

    # ── Training loop ────────────────────────────────────────────────
    best_val_acc = -1.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # progressive unfreezing for transfer learning
        if is_transfer and epoch == C.TL_FREEZE_EPOCHS + 1:
            log(">>> Unfreezing full backbone")
            unfreeze_all(model)
            optimizer = optim.AdamW(model.parameters(),
                                   lr=C.TL_LR_BACKBONE, weight_decay=wd)
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=max(1, epochs - epoch), T_mult=1, eta_min=1e-7)

        # linear warmup
        if epoch <= warmup_epochs:
            warmup_lr = lr * epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = warmup_lr

        train_m = run_epoch(model, train_loader, criterion, optimizer,
                    device, epoch, grad_clip=C.GRAD_CLIP, scheduler=scheduler,
                    mixup_fn=mixup_fn)
        val_m   = run_epoch(model, valid_loader, criterion, None,
                            device, epoch)

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        log(f"Epoch {epoch:03d}/{epochs:03d} | "
            f"train loss {train_m.loss:.4f} acc {train_m.acc:.4f} | "
            f"val loss {val_m.loss:.4f} acc {val_m.acc:.4f} | "
            f"lr {cur_lr:.2e} | {elapsed:.0f}s")

        if val_m.acc > best_val_acc:
            best_val_acc = val_m.acc
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": args.model,
                "classes": train_ds.classes,
                "image_size": C.IMAGE_SIZE,
                "mean": C.MEAN,
                "std": C.STD,
                "epoch": epoch,
                "val_acc": val_m.acc,
                "val_loss": val_m.loss,
            }, save_path)
            log(f"  ★ Saved best model (val acc {best_val_acc:.4f}) → {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                log(f"Early stopping at epoch {epoch} (patience={args.patience})")
                break

    # ── Final test evaluation ────────────────────────────────────────
    log("\n── Loading best checkpoint for test evaluation ──")
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model_test = build_model(args.model, num_classes=len(train_ds.classes),
                             pretrained=False, dropout=args.dropout)
    model_test.load_state_dict(ckpt["model_state_dict"])
    model_test.to(device)

    test_m = run_epoch(model_test, test_loader, criterion, None, device)
    log(f"\n{'='*60}")
    log(f"  FINAL TEST  |  loss {test_m.loss:.4f}  |  acc {test_m.acc:.4f}")
    log(f"  Best val acc: {best_val_acc:.4f}  (epoch {ckpt['epoch']})")
    log(f"  Checkpoint: {save_path}")
    log(f"{'='*60}")


if __name__ == "__main__":
    main()
