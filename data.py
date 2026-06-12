"""
data.py — CIFAR-100 benchmark setup and transforms.
"""

import random
import numpy as np
import torch
import torchvision.transforms as transforms
from avalanche.benchmarks.classic import SplitCIFAR100


# ── Class-order constants ──────────────────────────────────────────────────────

FINE_TO_COARSE = {
    0: 4, 1: 1, 2: 14, 3: 8, 4: 0, 5: 6, 6: 7, 7: 7, 8: 18, 9: 3,
    10: 3, 11: 14, 12: 9, 13: 18, 14: 7, 15: 11, 16: 3, 17: 9, 18: 7, 19: 11,
    20: 6, 21: 11, 22: 5, 23: 10, 24: 7, 25: 6, 26: 13, 27: 15, 28: 3, 29: 15,
    30: 0, 31: 11, 32: 1, 33: 10, 34: 12, 35: 14, 36: 16, 37: 9, 38: 11, 39: 5,
    40: 5, 41: 19, 42: 8, 43: 8, 44: 15, 45: 13, 46: 14, 47: 17, 48: 18, 49: 10,
    50: 16, 51: 4, 52: 17, 53: 4, 54: 2, 55: 0, 56: 17, 57: 4, 58: 18, 59: 17,
    60: 10, 61: 3, 62: 2, 63: 12, 64: 12, 65: 16, 66: 12, 67: 1, 68: 9, 69: 19,
    70: 2, 71: 10, 72: 0, 73: 1, 74: 16, 75: 12, 76: 9, 77: 13, 78: 15, 79: 13,
    80: 16, 81: 19, 82: 2, 83: 4, 84: 6, 85: 19, 86: 5, 87: 5, 88: 8, 89: 19,
    90: 18, 91: 1, 92: 2, 93: 15, 94: 6, 95: 0, 96: 17, 97: 8, 98: 14, 99: 13,
}

FIXED_ORDER = [
    87, 0, 52, 58, 44, 91, 68, 97, 51, 15,
    94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
    84, 96, 18, 24, 32, 45, 88, 11, 4, 67,
    69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
    17, 81, 41, 12, 37, 59, 25, 20, 80, 73,
    1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
    38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
    60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
    40, 30, 23, 85, 2, 95, 56, 48, 71, 64,
    98, 13, 99, 7, 34, 55, 54, 26, 35, 39,
]

# ── Transforms ─────────────────────────────────────────────────────────────────

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)

_normalize = transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD)

icarl_cifar100_augment_data = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
])

train_transform = transforms.Compose([
    transforms.ToTensor(),
    icarl_cifar100_augment_data,
    _normalize,
])

eval_transform = transforms.Compose([
    transforms.ToTensor(),
    _normalize,
])


def denormalize(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).to(img_tensor.device)
    std  = torch.tensor(CIFAR100_STD).view(3, 1, 1).to(img_tensor.device)
    return img_tensor * std + mean


def normalize_func(img_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR100_MEAN).view(3, 1, 1).to(img_tensor.device)
    std  = torch.tensor(CIFAR100_STD).view(3, 1, 1).to(img_tensor.device)
    return (img_tensor - mean) / std


# ── Benchmark ──────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_scenario(n_experiences: int = 10, dataset_root: str = "./data"):
    """Return a SplitCIFAR100 scenario with raw [0,1] train tensors."""
    return SplitCIFAR100(
        n_experiences=n_experiences,
        train_transform=transforms.ToTensor(),
        eval_transform=eval_transform,
        fixed_class_order=FIXED_ORDER,
        dataset_root=dataset_root,
    )
