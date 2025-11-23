import timm
import torch.nn as nn
import torch.nn.functional as F

class ImageAdapter(nn.Module):

    def __init__(self, target_size=64):
        super().__init__()
        self.target_size = target_size

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] not in [1, 3]:
            x = x.permute(0, 3, 1, 2)

        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 2)

        if x.shape[1] > 3:
            x = x[:, :3, :, :]

        if x.shape[-1] != self.target_size:
            x = F.interpolate(
                x,
                size=(self.target_size, self.target_size),
                mode="bilinear",
                align_corners=False
            )

        return x

class WrappedModel(nn.Module):
    def __init__(self, adapter, backbone):
        super().__init__()
        self.adapter = adapter
        self.backbone = backbone

    def forward(self, x):
        x = self.adapter(x)
        return self.backbone(x)



def add_dropout(model, p=0.3):
    if hasattr(model, "fc") and isinstance(model.fc, nn.Linear):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, model.num_classes),
        )
    elif hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p),
            nn.Linear(in_features, model.num_classes),
        )
    return model

def get_model(name: str, num_classes: int, dropout_prob=0.25, target_size=64, pretrained=0):
    name = name.lower()
    adapter = ImageAdapter(target_size=target_size)

    use_pretrained = bool(pretrained)

    if name in ["resnet18", "rn18"]:
        backbone = timm.create_model(
            "resnet18",
            pretrained=use_pretrained,
            num_classes=num_classes
        )
        backbone = add_dropout(backbone, p=dropout_prob)
        return WrappedModel(adapter, backbone)

    if name in ["resnet50", "rn50"]:
        backbone = timm.create_model(
            "resnet50",
            pretrained=use_pretrained,
            num_classes=num_classes
        )
        backbone = add_dropout(backbone, p=dropout_prob)
        return WrappedModel(adapter, backbone)

    if name in ["resnest14d", "resnest14", "rn14d"]:
        backbone = timm.create_model(
            "resnest14d",
            pretrained=use_pretrained,
            num_classes=num_classes
        )
        backbone = add_dropout(backbone, p=dropout_prob)
        return WrappedModel(adapter, backbone)

    if name in ["resnest26d", "resnest26", "rn26d"]:
        backbone = timm.create_model(
            "resnest26d",
            pretrained=use_pretrained,
            num_classes=num_classes
        )
        backbone = add_dropout(backbone, p=dropout_prob)
        return WrappedModel(adapter, backbone)

    if name in ["mlp", "basic_mlp"]:
        adapter = ImageAdapter(target_size=target_size)
        backbone = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * target_size * target_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        return WrappedModel(adapter, backbone)

    raise ValueError("Unknown model")
