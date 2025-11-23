import warnings

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    category=UserWarning,
)

import argparse
import torch
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from model_v4 import get_model
from dataset_v4 import get_dataset
from utils_v4 import EarlyStopper, apply_batch_size_scheduler, get_optimizer, tta_predict
import wandb
import time
from torchvision.transforms import v2

def train_one_epoch(model, loader, criterion, optimizer, device, args, mix_transform,scaler):
    model.train()
    total = 0
    correct = 0

    for x, y in loader:  
        if args.mix_prob > 0 and torch.rand(1).item() < args.mix_prob:
            x, y = mix_transform(x, y)
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        with autocast(device_type=device, dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if y.ndim == 2:
            preds = out.argmax(dim=1)
            y_hard = y.argmax(dim=1)
            correct += (preds == y_hard).sum().item()
        else:
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
        total += y.size(0)
        

    return correct / total


def test(model, loader, device, use_tta):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        if use_tta:
            for x, y in loader:
                pred = tta_predict(model, x, device)
                preds = pred.argmax(dim=1).cpu()
                correct += (preds == y).sum().item()
                total += y.size(0)
        else:
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

    return correct / total

def safe_config(args):
    safe_keys = {
        "dataset", "model", "epochs", "bs", "lr",
        "opt", "scheduler", "scheduler_step_size",
        "scheduler_gamma", "scheduler_patience",
        "momentum", "weight_decay", "bs_growth_factor",
        "bs_growth_interval", "max_bs", "pretrained",
        "use_tta", "dropout", "label_smoothing",
        "mix_prob", "mix_alpha"
    }
    return {k: getattr(args, k) for k in safe_keys}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model", default="resnest26d")
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--bs", default=64, type=int)
    parser.add_argument("--lr", default=None, type=float)
    parser.add_argument("--opt", default="sgd")
    parser.add_argument("--scheduler", default="step")
    parser.add_argument("--scheduler_step_size", type=int, default=5)
    parser.add_argument("--scheduler_gamma", type=float, default=0.1)
    parser.add_argument("--scheduler_patience", type=int, default=3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--bs_growth_factor", type=float, default=1.5)
    parser.add_argument("--bs_growth_interval", type=int, default=5)
    parser.add_argument("--max_bs", type=int, default=256)
    parser.add_argument("--use_tta", type=int, default=0, choices=[0,1])
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--mix_prob", type=float, default=1.0)
    parser.add_argument("--mix_alpha", type=float, default=0.5)
    parser.add_argument("--pretrained", type=int, default=0, choices=[0,1])
    parser.add_argument("--target_size", type=int, default=64)
    parser.add_argument("--early_stop", type=int, default=5)
    parser.add_argument("--log", type=int, default=1, choices=[0,1])

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    train_ds = get_dataset(args.dataset, train=True)
    test_ds = get_dataset(args.dataset, train=False)
    num_classes = len(train_ds.classes)

    mix_transform = v2.RandomChoice([
        v2.CutMix(num_classes=num_classes, alpha=args.mix_alpha),
        v2.MixUp(num_classes=num_classes, alpha=args.mix_alpha)
    ])

    train_loader = DataLoader(train_ds, batch_size=args.bs, num_workers=2, pin_memory=True, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.bs, num_workers=2, pin_memory=True)

    model = get_model(
        args.model,
        num_classes=len(train_ds.classes),
        dropout_prob=args.dropout,
        pretrained=args.pretrained,
        target_size=args.target_size
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = get_optimizer(args.opt, model, args.lr, args.momentum, args.weight_decay)
    scaler = GradScaler(enabled=True)
    if args.scheduler == "step":
        scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.scheduler_gamma)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, patience=args.scheduler_patience, factor=args.scheduler_gamma)
    if(args.log):
        wandb.init(project="hw3-cifar100_final", config=safe_config(args))

    early_stop = EarlyStopper(args.early_stop)
    start_time = time.time()

    acc = 0.0
    test_acc = 0.0

    try:
        for epoch in range(1, args.epochs + 1):
            acc = train_one_epoch(model, train_loader, criterion, optimizer, device, args, mix_transform,scaler)

            test_acc = test(model, test_loader, device, args.use_tta)

            print(f"Epoch {epoch}: train={acc:.3f}, test={test_acc:.3f}")
            
            if(args.log):
                wandb.log({"train_acc": acc, "test_acc": test_acc, "epoch": epoch})

            if args.scheduler == "step":
                scheduler.step()
            else:
                scheduler.step(acc)

            train_loader = apply_batch_size_scheduler(
                train_loader, train_ds, epoch,
                args.bs_growth_interval, args.bs_growth_factor, args.max_bs,
                num_workers=2
            )

            if early_stop.check(acc):
                print("Early stopping triggered!")
                break

    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user.")

    torch.save(model.state_dict(), "best_model.pth")
    print("Best model saved.")

    total_time = time.time() - start_time
    if(args.log):
        wandb.log({"total_training_time": total_time})
        wandb.summary["final_test_acc"] = test_acc
        wandb.summary["final_train_acc"] = acc
        wandb.summary["total_time"] = total_time
        wandb.finish()


if __name__ == "__main__":
    main()

