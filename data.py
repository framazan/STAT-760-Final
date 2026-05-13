import os
import torch
import torchvision
from pathlib import Path
from torch.utils.data import DataLoader
from torchvision.datasets.folder import IMG_EXTENSIONS, pil_loader
from torchvision.transforms import v2
import config as C
from utils import seed_worker

def get_train_transform(image_size: int = C.IMAGE_SIZE,
                        mean: list = C.MEAN,
                        std: list = C.STD):
    return v2.Compose([
        v2.ToImage(),
        v2.RandomResizedCrop(size=(image_size, image_size), scale=(0.5, 1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(degrees=15),
        v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        v2.RandomGrayscale(p=0.05),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
        v2.RandomErasing(p=0.2),
    ])


def get_eval_transform(image_size: int = C.IMAGE_SIZE,
                       mean: list = C.MEAN,
                       std: list = C.STD):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize(size=(image_size + 32, image_size + 32)),
        v2.CenterCrop(size=(image_size, image_size)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ])

def compute_class_weights(dataset: torchvision.datasets.ImageFolder,
                          device: torch.device) -> torch.Tensor:
    # inverse-frequency class weighting, because the data is imbalanced!
    counts = torch.zeros(len(dataset.classes))
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(dataset.classes) # normalize
    return weights.to(device)

def get_dataloaders(data_root: str = C.DATA_ROOT, image_size: int = C.IMAGE_SIZE, batch_size: int = C.BATCH_SIZE, num_workers: int = C.NUM_WORKERS, seed: int = C.SEED):
    # written with GitHub copilot
    train_dir = os.path.join(data_root, "train")
    valid_dir = os.path.join(data_root, "valid")
    test_dir  = os.path.join(data_root, "test")

    train_tf = get_train_transform(image_size)
    eval_tf  = get_eval_transform(image_size)

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=train_tf)
    valid_ds = torchvision.datasets.ImageFolder(valid_dir, transform=eval_tf)
    test_ds  = torchvision.datasets.ImageFolder(test_dir,  transform=eval_tf)

    generator = torch.Generator().manual_seed(seed)

    common = dict(
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=(num_workers > 0),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **common)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **common)

    return train_ds, valid_ds, test_ds, train_loader, valid_loader, test_loader


class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        path = self.image_paths[index]
        image = pil_loader(str(path))
        return self.transform(image), str(path)


def list_image_paths(path):
    # taken from StackOverflow, i use this for testing only though.
    p = Path(path).expanduser()
    if p.is_file():
        if p.suffix.lower() not in IMG_EXTENSIONS:
            raise ValueError(f"Unsupported extension: {p}")
        return [p]
    if p.is_dir():
        paths = sorted(
            c for c in p.rglob("*") if c.is_file() and c.suffix.lower() in IMG_EXTENSIONS
        )
        if not paths:
            raise ValueError(f"No images in {p}")
        return paths
    raise FileNotFoundError(f"Not found: {p}")
