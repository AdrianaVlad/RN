import torch.optim as optim
import torch_optimizer as optim_extra
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from torchvision.transforms.v2 import MixUp, CutMix, RandomChoice


def tta_predict(model, x, device):
    x = x.to(device)
    B, C, H, W = x.shape
    aug_list = []

    aug_list.append(x)
    aug_list.append(torch.flip(x, dims=[3]))

    padded = F.pad(x, (H//8, H//8, W//8, W//8))
    aug_list.append(padded[:, :, H//8: H//8+H, W//8: W//8+W])
    aug_list.append(padded[:, :, H//16: H//16+H, W//16: W//16+W])

    preds = []
    for aug in aug_list:
        preds.append(model(aug))

    return torch.mean(torch.stack(preds, dim=0), dim=0)

def get_optimizer(name: str, model, lr: float, momentum: float, weight_decay: float):
    name = name.lower()
    params = model.parameters()

    if lr is None:
        if name == "sgd":
            lr = 0.05
        elif name == "adamw":
            lr = 0.0005
        else:
            lr = 0.001

    if name in ["sgd"]:
        return optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )

    if name in ["adam"]:
        return optim.Adam(
            params,
            lr=lr,
            weight_decay=weight_decay
        )
    if name in ["adamw"]:
        return optim.AdamW(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

    if name in ["muon"]:
        return optim_extra.Muon(
            params,
            lr=lr,
            weight_decay=weight_decay
        )

    if name in ["sam"]:
        base_optimizer = optim.SGD
        return optim_extra.SAM(
            params,
            base_optimizer,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    raise ValueError(
        f"Unknown optimizer '{name}'. Supported: SGD, Adam, AdamW, Muon, SAM"
    )


class EarlyStopper:
    def __init__(self, patience=6, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = -float("inf")

    def check(self, metric):
        if metric > self.best + self.min_delta:
            self.best = metric
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def apply_batch_size_scheduler(loader, dataset, epoch, interval, factor, max_bs, num_workers):
    if epoch % interval != 0:
        return loader

    old_bs = loader.batch_size
    new_bs = min(int(old_bs * factor), max_bs)

    if new_bs == old_bs:
        return loader

    print(f"[BatchSizeScheduler] Increasing batch size: {old_bs} → {new_bs}")

    new_loader = DataLoader(
        dataset,
        batch_size=new_bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    return new_loader


