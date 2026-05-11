import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.transforms import v2

from models import SimpleCNN


SEED = 16


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@dataclass(frozen=True)
class Metrics:
    loss: float
    acc: float


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths: list[Path], transform) -> None:
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        path = self.image_paths[index]
        image = pil_loader(str(path))
        return self.transform(image), str(path)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
    device: torch.device,
    epoch: int | None = None,
) -> Metrics:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    total_batches = len(loader) if hasattr(loader, "__len__") else None
    freq = max(1, (total_batches // 10) if total_batches else 10)
    phase = "Train" if is_train else "Eval"
    if epoch is not None:
        print(f"{phase} epoch {epoch} started ({'training' if is_train else 'evaluating'})")
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            logits = model(images)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total_count += batch_size

        if (batch_idx % freq == 0) or (batch_idx == total_batches - 1 if total_batches else False):
            running_loss = total_loss / max(total_count, 1)
            running_acc = total_correct / max(total_count, 1)
            if total_batches:
                print(
                    f"{phase} epoch {epoch or '?'} | batch {batch_idx+1}/{total_batches} "
                    f"running loss {running_loss:.4f} acc {running_acc:.4f}",
                    flush=True,
                )
            else:
                print(
                    f"{phase} epoch {epoch or '?'} | batch {batch_idx+1} "
                    f"running loss {running_loss:.4f} acc {running_acc:.4f}",
                    flush=True,
                )

    return Metrics(loss=total_loss / max(total_count, 1), acc=total_correct / max(total_count, 1))


def list_image_paths(path: str) -> list[Path]:
    input_path = Path(path).expanduser()
    if input_path.is_file():
        if input_path.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Unsupported image file extension: {input_path}")
        return [input_path]
    if input_path.is_dir():
        image_paths = [
            candidate
            for candidate in sorted(input_path.rglob("*"))
            if candidate.is_file() and candidate.suffix.lower() in IMG_EXTENSIONS
        ]
        if not image_paths:
            raise ValueError(f"No supported image files found in directory: {input_path}")
        return image_paths
    raise FileNotFoundError(f"Image path does not exist: {input_path}")


def build_transforms(image_size: int, mean: list[float], std: list[float]):
    train_tf = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.75, 1.0)),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    eval_tf = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(image_size + 32, image_size + 32)),
            v2.CenterCrop(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, eval_tf


def compute_train_mean_std(
    train_dir: str,
    image_size: int,
    batch_size: int,
    workers: int,
    seed: int,
) -> tuple[list[float], list[float]]:
    base_tf = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize(size=(image_size + 32, image_size + 32)),
            v2.CenterCrop(size=(image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    ds = torchvision.datasets.ImageFolder(train_dir, transform=base_tf)
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )

    channel_sum = torch.zeros(3, dtype=torch.float64)
    channel_sumsq = torch.zeros(3, dtype=torch.float64)
    total_pixels = 0

    total_batches = len(loader) if hasattr(loader, "__len__") else None
    freq = max(1, (total_batches // 10) if total_batches else 10)
    with torch.inference_mode():
        for batch_idx, (images, _) in enumerate(loader):
            if images.shape[1] != 3:
                raise ValueError(f"Expected 3-channel images, got shape {tuple(images.shape)}")

            images = images.contiguous()
            b, c, h, w = images.shape
            pixels = b * h * w
            channel_sum += images.sum(dim=(0, 2, 3), dtype=torch.float64)
            channel_sumsq += images.square().sum(dim=(0, 2, 3), dtype=torch.float64)
            total_pixels += pixels
            if (batch_idx % freq == 0) or (batch_idx == total_batches - 1 if total_batches else False):
                processed = batch_idx + 1
                if total_batches:
                    print(f"Computing mean/std: processed {processed}/{total_batches} batches", flush=True)
                else:
                    print(f"Computing mean/std: processed {processed} batches", flush=True)

    mean = channel_sum / max(total_pixels, 1)
    var = channel_sumsq / max(total_pixels, 1) - mean * mean
    std = torch.sqrt(torch.clamp(var, min=1e-12))

    mean_list = [float(x) for x in mean]
    std_list = [float(x) for x in std]
    return mean_list, std_list


def predict_images(
    image_path: str,
    checkpoint_path: str,
    batch_size: int,
    workers: int,
    device: torch.device,
    fallback_image_size: int,
    fallback_mean: list[float],
    fallback_std: list[float],
    dropout: float,
) -> None:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError(f"Checkpoint does not contain class names: {checkpoint_path}")

    image_size = int(checkpoint.get("image_size", fallback_image_size))
    mean = checkpoint.get("mean", fallback_mean)
    std = checkpoint.get("std", fallback_std)
    _, eval_tf = build_transforms(image_size, mean=mean, std=std)

    image_paths = list_image_paths(image_path)
    dataset = ImagePathDataset(image_paths, transform=eval_tf)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
    )

    model = SimpleCNN(num_classes=len(classes), dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Using checkpoint: {checkpoint_path}")
    print(f"Using device: {device}")
    print(f"Images: {len(dataset)}")
    with torch.inference_mode():
        for images, paths in loader:
            images = images.to(device)
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1).cpu()
            confidences, indices = probabilities.max(dim=1)
            for path, confidence, index, probs in zip(paths, confidences, indices, probabilities):
                class_name = classes[int(index)]
                prob_text = ", ".join(
                    f"{classes[class_idx]}={float(prob):.4f}" for class_idx, prob in enumerate(probs)
                )
                print(f"{path}\t{class_name}\tconfidence={float(confidence):.4f}\t{prob_text}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./dataset/combined_smoke_fire_classification_256_balanced",
    )
    parser.add_argument("--stats", choices=["imagenet", "dataset"], default="imagenet")
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--stats_workers",
        type=int,
        default=0,
        help="DataLoader workers for dataset mean/std calculation. Default 0 avoids macOS worker shutdown hangs.",
    )
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--save_path", type=str, default="./cnn_best.pt")
    parser.add_argument(
        "--predict",
        type=str,
        default=None,
        help="Image file or directory to classify with a saved checkpoint. If set, training is skipped.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for --predict. Defaults to --save_path.",
    )
    args = parser.parse_args()

    set_seed(SEED)
    device = get_device()
    print(f"Using device: {device}")
    train_dir = os.path.join(args.data_root, "train")
    valid_dir = os.path.join(args.data_root, "valid")
    test_dir = os.path.join(args.data_root, "test")

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]
    if args.predict is not None:
        predict_images(
            image_path=args.predict,
            checkpoint_path=args.checkpoint or args.save_path,
            batch_size=args.batch_size,
            workers=args.workers,
            device=device,
            fallback_image_size=args.image_size,
            fallback_mean=imagenet_mean,
            fallback_std=imagenet_std,
            dropout=args.dropout,
        )
        return

    if args.stats == "dataset":
        mean, std = compute_train_mean_std(
            train_dir=train_dir,
            image_size=args.image_size,
            batch_size=args.batch_size,
            workers=args.stats_workers,
            seed=SEED,
        )
        print(f"Dataset mean: {mean}")
        print(f"Dataset std:  {std}")
    else:
        mean, std = imagenet_mean, imagenet_std
        print(f"Using ImageNet mean/std")

    train_tf, eval_tf = build_transforms(args.image_size, mean=mean, std=std)

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    valid_ds = torchvision.datasets.ImageFolder(valid_dir, transform=eval_tf)
    test_ds = torchvision.datasets.ImageFolder(test_dir, transform=eval_tf)

    print(f"Classes: {train_ds.classes}")
    print(f"Train samples: {len(train_ds)}")
    print(f"Valid samples: {len(valid_ds)}")
    print(f"Test samples: {len(test_ds)}")

    generator = torch.Generator()
    generator.manual_seed(SEED)
    pin_memory = device.type == "cuda"
    persistent_workers = args.workers > 0 and device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=persistent_workers,
    )

    model = SimpleCNN(num_classes=len(train_ds.classes), dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)
        val_metrics = run_epoch(model, valid_loader, criterion, None, device, epoch=epoch)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train loss {train_metrics.loss:.4f} acc {train_metrics.acc:.4f} | "
            f"val loss {val_metrics.loss:.4f} acc {val_metrics.acc:.4f}"
        )
        if val_metrics.acc > best_val_acc:
            best_val_acc = val_metrics.acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": train_ds.classes,
                    "image_size": args.image_size,
                    "mean": mean,
                    "std": std,
                    "seed": SEED,
                },
                args.save_path,
            )

    test_metrics = run_epoch(model, test_loader, criterion, None, device, epoch=None)
    print(f"Test loss {test_metrics.loss:.4f} acc {test_metrics.acc:.4f}")
    print(f"Best checkpoint saved to: {args.save_path}")


if __name__ == "__main__":
    main()
