import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import datasets
import cv2
import numpy as np


DATASET_STATS = {
    "cifar10":  ((0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)),
    "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    "mnist":    ((0.1307,), (0.3081,)),
    "pets":     ((0.485, 0.456, 0.406),     (0.229, 0.224, 0.225))
}


def get_transforms(dataset_name, train, image_size):
    mean, std = DATASET_STATS.get(dataset_name, DATASET_STATS["pets"])
    if train:
        return A.Compose([
            A.PadIfNeeded(min_height=image_size+image_size/8, min_width=image_size+image_size/8, border_mode=cv2.BORDER_REFLECT),
            A.RandomCrop(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.Rotate(limit=10, p=0.3),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.6),
            A.CoarseDropout(
                max_holes=1,
                max_height=8,
                max_width=8,
                min_height=4,
                min_width=4,
                fill_value=0,
                p=0.25
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])


class AlbumentationsDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)

        augmented = self.transform(image=img)
        img = augmented["image"]
        return img, label


def get_dataset(name: str, train=True):
    name = name.lower()
    root = "./data"


    if name in ["cifar100","cifar-100"]:
        base = datasets.CIFAR100(root, train=train, download=True)
    elif name in ["cifar10","cifar-10"]:
        base = datasets.CIFAR10(root, train=train, download=True)
    elif name == "mnist":
        base = datasets.MNIST(root, train=train, download=True)
    elif name in ["pets", "oxfordiiitpet"]:
        split = "train" if train else "test"
        base = datasets.OxfordIIITPet(root, split=split, download=True)
    else:
        raise ValueError(f"Unknown dataset: {name}")


    sample_img, _ = base[0]
    sample_np = np.array(sample_img)

    if sample_np.ndim == 2:
        H, W = sample_np.shape
    else:
        H, W, _ = sample_np.shape

    image_size = min(H, W)
    transform = get_transforms(name, train=train, image_size=image_size)

    return AlbumentationsDataset(base, transform)