"""
models.py — Feature extractor, iCaRL-compatible model, and Narcissus surrogate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18
from avalanche.models import make_icarl_net, initialize_icarl_net


# ── iCaRL model ────────────────────────────────────────────────────────────────

class CifarFeatureExtractor(nn.Module):
    """ResNet-18 backbone truncated to a 128-d embedding."""

    def __init__(self, feature_dim: int = 128):
        super().__init__()
        backbone = resnet18(pretrained=False)
        self.model = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(512, feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.model(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class IcarlNet(nn.Module):
    def __init__(self, feature_dim: int = 128, n_classes: int = 100):
        super().__init__()
        self.feature_extractor = CifarFeatureExtractor(feature_dim)
        self.classifier = nn.Linear(feature_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.feature_extractor(x))


def build_icarl_model(num_classes: int = 100, device: torch.device = None) -> nn.Module:
    """Return an initialised iCaRL-ready ResNet-32 model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = make_icarl_net(num_classes=num_classes).to(device)
    model.apply(initialize_icarl_net)
    return model


# ── Narcissus surrogate ────────────────────────────────────────────────────────

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, 1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class NarcissusSurrogate(nn.Module):
    """ResNet-18-style surrogate used for Narcissus trigger generation."""

    def __init__(self, n_pood_classes: int = 201, n_target_classes: int = 0):
        super().__init__()
        self.in_planes = 64
        self.conv1  = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64,  2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.head   = nn.Linear(512, n_pood_classes)
        self.target_head = (nn.Linear(512, n_target_classes)
                            if n_target_classes > 0 else None)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (num_blocks - 1)
        layers  = []
        for s in strides:
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def backbone(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        return out.view(out.size(0), -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))

    def forward_target(self, x: torch.Tensor) -> torch.Tensor:
        assert self.target_head is not None
        return self.target_head(self.backbone(x))
