"""
Notes: 
# Train AdvancedCNN (report one)
python train.py --model advancedcnn --epochs 50 --lr 5e-3

# Fine-tune ConvNeXt-Tiny
python train.py --model convnext --transfer
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
from utils import get_device, set_seed, Metrics


def run_epoch(model, loader, criterion, optimizer, device, epoch, grad_clip=None, scheduler=None, mixup_fn=None):
    to_train = optimizer is not None
    model.train(to_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_batches = len(loader)
    freq = max(1, n_batches // 5)
    phase = "Train" if to_train else "Evaluation"

    for i, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)

        if to_train and mixup_fn is not None:
            images, mix_targets = mixup_fn(images, targets)
        else:
            mix_targets = targets

        with torch.set_grad_enabled(to_train):
            logits = model(images)
            loss = criterion(logits, mix_targets)

        if to_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = targets.size(0)
        total_loss += loss.item() * bs
        total_correct += (logits.argmax(1) == targets).sum().item()
        total_count += bs

        if i % freq == 0 or i == n_batches - 1:
            rl = total_loss / total_count
            ra = total_correct / total_count
            print(f"{phase} ep {epoch or 'last eval'}, batch {i+1:>4} / {n_batches}, loss {rl:.4f} acc {ra:.4f}")

    if scheduler is not None and to_train:
        scheduler.step()

    return Metrics(loss=total_loss / total_count, acc=total_correct / total_count)


def freeze_backbone(model: nn.Module):
    # Use this to freeze all parameters, then unfreeze the classifier head
    # when fine-tuning, as per https://cs231n.stanford.edu/slides/2026/lecture_6.pdf
    for p in model.parameters():
        p.requires_grad = False
    # timm models expose .get_classifier() or .head / .classifier,
    # had to use GitHub copilot here because I could figure out the error
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="advancedcnn",
                        choices=["simple", "advancedcnn", "convnext", "vit"])
    parser.add_argument("--epochs", type=int, default=C.EPOCHS)
    parser.add_argument("--batch_size", type=int, default=C.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=C.LR)
    parser.add_argument("--weight_decay", type=float, default=C.WEIGHT_DECAY)
    parser.add_argument("--dropout", type=float, default=C.DROPOUT)
    parser.add_argument("--patience", type=int, default=C.PATIENCE)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--enable_mixup", dest="enable_mixup", action="store_true")
    parser.add_argument("--disable_mixup", dest="enable_mixup", action="store_false")
    parser.set_defaults(enable_mixup=C.ENABLE_MIXUP)
    parser.add_argument("--data_root", type=str, default=C.DATA_ROOT)
    parser.add_argument("--workers", type=int, default=C.NUM_WORKERS)
    parser.add_argument("--save_name", type=str, default=None)
    args = parser.parse_args()

    is_transfer = args.transfer and args.model in ("convnext", "vit")

    epochs = args.epochs
    lr = args.lr
    wd = args.weight_decay

    save_name = args.save_name or f"{args.model}_best.pt"
    save_path = os.path.join(C.SAVE_DIR, save_name)

    set_seed(C.SEED)
    device = get_device()
    print(f"Device: {device}")

    train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader = \
        get_dataloaders(args.data_root, C.IMAGE_SIZE, args.batch_size, args.workers)

    print(f"Classes: {train_ds.classes}")
    print(f"Train: {len(train_ds)}  Valid: {len(valid_ds)}  Test: {len(test_ds)}")

    model = build_model(args.model, pretrained=is_transfer, dropout=args.dropout)
    model.to(device)

    n_params = sum(p.numel() for p in model.parameters())
    n_train  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {args.model}, params: {n_params:,},  trainable: {n_train:,}")
    weights = compute_class_weights(train_ds, device)
    print(f"Class weights: {weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=weights)
    mixup_fn = None
    if args.enable_mixup:
        mixup_fn = v2.RandomChoice([
            v2.CutMix(num_classes=len(train_ds.classes), alpha=C.CUTMIX_ALPHA),
            v2.MixUp(num_classes=len(train_ds.classes), alpha=C.MIXUP_ALPHA),
        ])
        print(f"Using MixUp/CutMix (mixup_alpha={C.MIXUP_ALPHA}, cutmix_alpha={C.CUTMIX_ALPHA})")

    if is_transfer:
        freeze_backbone(model)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=lr, weight_decay=wd)
    else:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    warmup_epochs = 3
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=max(1, epochs - warmup_epochs), T_mult=1, eta_min=1e-6)

    best_val_acc = -1.0
    patience_counter = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        # progressive unfreezing for transfer learning
        if is_transfer and epoch == C.TL_FREEZE_EPOCHS + 1:
            print("Unfreezing full backbone")
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

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, epoch, grad_clip=C.GRAD_CLIP, scheduler=scheduler, mixup_fn=mixup_fn)
        val_m = run_epoch(model, valid_loader, criterion, None, device, epoch)

        elapsed = time.time() - t0
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:03d}/{epochs:03d}, train loss {train_m.loss:.4f} acc {train_m.acc:.4f}, val loss {val_m.loss:.4f} acc {val_m.acc:.4f}, lr {cur_lr:.2e}, {elapsed:.0f}s")

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
            print(f"Saved best model (val acc {best_val_acc:.4f}) at {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Stopping at epoch {epoch}")
                break
    # Test eval
    ckpt = torch.load(save_path, map_location=device, weights_only=True)
    model_test = build_model(args.model, pretrained=False, dropout=args.dropout)
    model_test.load_state_dict(ckpt["model_state_dict"])
    model_test.to(device)

    test_m = run_epoch(model_test, test_loader, criterion, None, device)
    print(f"Final Test, loss {test_m.loss:.4f}, acc {test_m.acc:.4f}")

if __name__ == "__main__":
    main()
