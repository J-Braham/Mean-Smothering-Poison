"""
backdoor.py — Poisoning helpers: dataset wrappers, experience poisoning,
              and the Narcissus trigger pipeline.
"""

import os
import copy
import time
import zipfile
import shutil
import requests

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset
from torchvision.datasets import ImageFolder
from avalanche.benchmarks.utils import make_avalanche_dataset

from src.data import (
    FINE_TO_COARSE, FIXED_ORDER,
    icarl_cifar100_augment_data, normalize_func,
    denormalize,
)
from src.models import NarcissusSurrogate


# ── Narcissus hyper-parameters ─────────────────────────────────────────────────
NARC_L_INF_R          = 12 / 255
NARC_SURROGATE_EPOCHS = 200
NARC_WARMUP_EPOCHS    = 5
NARC_WARMUP_LR        = 0.01
NARC_GEN_ROUNDS       = 1000
NARC_GEN_LR           = 0.01
NARC_TRAIN_BS         = 512
_NARC_DELTA_SCALE     = 0.5   # trigger lives in [-1,1]; pixel space is [0,1]
NARC_TEST_AMPLIFY     = 2.0

# ── Tiny-ImageNet paths ────────────────────────────────────────────────────────
TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
TINY_IMAGENET_DIR = "./tiny-imagenet-200"
TINY_IMAGENET_ZIP = "./tiny-imagenet-200.zip"

# ── Global state ───────────────────────────────────────────────────────────────
_NARCISSUS_TRIGGERS:    dict = {}
_NARCISSUS_PRETRAINED:  bool = False
_POOD_SURROGATE_PRETRAINED: bool = False

_NARC_NORM = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

_POOD_TRANSFORM = transforms.Compose([
    transforms.Resize(32),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _NARC_NORM,
])

_TARGET_TRANSFORM = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    _NARC_NORM,
])

# ── Dataset wrappers ───────────────────────────────────────────────────────────

class CustomPoisonedDataset(Dataset):
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        return self.xs[idx], self.ys[idx]


class TransformAfterPoison(Dataset):
    """Wraps a poisoned dataset and applies augment+normalise on the fly."""

    def __init__(self, poisoned_dataset, transform):
        self.dataset   = poisoned_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        return self.transform(x), y


class _NarcissusTargetDataset(Dataset):
    """Renormalises Avalanche [0,1] tensors into [-1,1] for the surrogate."""

    def __init__(self, avl_dataset, indices):
        self.ds      = avl_dataset
        self.indices = indices
        self._norm   = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y, *_ = self.ds[self.indices[i]]
        return self._norm(x), int(y)


class _NarcissusAugDataset(Dataset):
    """With spatial augmentation — used during poi-warm-up."""

    def __init__(self, avl_dataset, indices):
        self.ds      = avl_dataset
        self.indices = indices
        self._tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        x, y, *_ = self.ds[self.indices[i]]
        return self._tf(x), int(y)


# ── Trigger application ────────────────────────────────────────────────────────

def add_trigger(
    img: torch.Tensor,
    class_id: int,
    amplify: float = 1.0,
) -> torch.Tensor:
    """Apply the Narcissus trigger to a raw [0,1] tensor (3, 32, 32)."""
    delta_narc  = get_narcissus_trigger(class_id).to(img.device)
    delta_pixel = delta_narc * _NARC_DELTA_SCALE * amplify
    return (img + delta_pixel).clamp(0.0, 1.0)


def get_narcissus_trigger(coarse_id: int) -> torch.Tensor:
    """Return the cached trigger delta for *coarse_id* (on CPU)."""
    global _NARCISSUS_TRIGGERS, _NARCISSUS_PRETRAINED
    if not _NARCISSUS_PRETRAINED:
        raise RuntimeError(
            f"No Narcissus trigger for coarse class {coarse_id}. "
            "Call run_narcissus_pretrain() first."
        )
    if coarse_id not in _NARCISSUS_TRIGGERS:
        raise KeyError(f"Trigger for coarse class {coarse_id} not found.")
    return _NARCISSUS_TRIGGERS[coarse_id]


# ── Experience poisoning ───────────────────────────────────────────────────────

def should_poison_task(
    task_id: int,
    experience,
    poison_mode: str,
    poisoned_tasks: list,
    backdoor_target_course: list,
    poison_classes: list,
) -> bool:
    dataset = experience.dataset
    if poison_mode == "task":
        return task_id in poisoned_tasks
    elif poison_mode == "superclass":
        coarse_in_task = {FINE_TO_COARSE[int(dataset[i][1])] for i in range(len(dataset))}
        return bool(coarse_in_task & set(backdoor_target_course))
    elif poison_mode == "class":
        fine_in_task = {int(dataset[i][1]) for i in range(len(dataset))}
        return bool(fine_in_task & set(poison_classes))
    raise ValueError(f"Unknown poison_mode: {poison_mode}")


def make_poisoned_experience(
    experience,
    backdoor_target_course: list,
    poison_rate: float = 0.7,
    seed: int = 42,
    poison_mode: str = "superclass",
    poison_classes: list = None,
):
    orig_dataset = experience.dataset
    n = len(orig_dataset)

    if poison_mode in ("task", "superclass"):
        target_indices = [
            i for i in range(n)
            if FINE_TO_COARSE[int(orig_dataset[i][1])] in backdoor_target_course
        ]
    elif poison_mode == "class":
        target_indices = [
            i for i in range(n)
            if int(orig_dataset[i][1]) in (poison_classes or [])
        ]
    else:
        raise ValueError(f"Unknown poison_mode: {poison_mode}")

    rng      = np.random.default_rng(seed)
    n_poison = int(len(target_indices) * poison_rate)
    poisoned_idx = set(
        rng.choice(target_indices, size=n_poison, replace=False).tolist()
    ) if n_poison > 0 and target_indices else set()

    print(f"  [{poison_mode}] Poisoning {n_poison}/{len(target_indices)} "
          f"target-coarse samples (total task samples: {n})")

    xs, ys = [], []
    for i in range(n):
        sample = orig_dataset[i]
        x, y   = sample[0], int(sample[1])
        if i in poisoned_idx:
            coarse_y = FINE_TO_COARSE[y]
            x = add_trigger(x, class_id=coarse_y)
        xs.append(x)
        ys.append(y)

    inner   = CustomPoisonedDataset(xs, ys)
    wrapped = TransformAfterPoison(
        inner,
        transforms.Compose([icarl_cifar100_augment_data,
                             transforms.Normalize((0.5071, 0.4867, 0.4408),
                                                  (0.2675, 0.2565, 0.2761))]),
    )
    poisoned_avl = make_avalanche_dataset(
        wrapped,
        data_attributes=list(orig_dataset._data_attributes.values()),
    )
    poisoned_exp = copy.copy(experience)
    poisoned_exp.dataset = poisoned_avl
    return poisoned_exp


# ── Tiny-ImageNet helpers ──────────────────────────────────────────────────────

def _download_tiny_imagenet(verbose: bool = True) -> None:
    if os.path.isdir(TINY_IMAGENET_DIR):
        if verbose:
            print("Tiny-ImageNet already present, skipping download.")
        return
    if verbose:
        print("Downloading Tiny-ImageNet (~237 MB) ...")
    t0 = time.time()
    r  = requests.get(TINY_IMAGENET_URL, stream=True)
    r.raise_for_status()
    with open(TINY_IMAGENET_ZIP, "wb") as f:
        for chunk in r.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    with zipfile.ZipFile(TINY_IMAGENET_ZIP, "r") as z:
        z.extractall(".")
    if verbose:
        print(f"  Extracted in {time.time() - t0:.0f}s.")


def _fix_tiny_imagenet_val() -> None:
    val_dir  = os.path.join(TINY_IMAGENET_DIR, "val")
    sentinel = os.path.join(val_dir, ".reorganised")
    if os.path.exists(sentinel):
        return
    anno_path = os.path.join(val_dir, "val_annotations.txt")
    img_dir   = os.path.join(val_dir, "images")
    if not os.path.isfile(anno_path):
        return
    with open(anno_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            fname, cls = parts[0], parts[1]
            cls_dir = os.path.join(val_dir, cls)
            os.makedirs(cls_dir, exist_ok=True)
            src = os.path.join(img_dir, fname)
            dst = os.path.join(cls_dir, fname)
            if os.path.isfile(src):
                shutil.move(src, dst)
    open(sentinel, "w").close()


def load_tiny_imagenet_pood(max_samples: int = 100_000, verbose: bool = True) -> DataLoader:
    _download_tiny_imagenet(verbose=verbose)
    _fix_tiny_imagenet_val()
    train_dir = os.path.join(TINY_IMAGENET_DIR, "train")
    dataset   = ImageFolder(train_dir, transform=_POOD_TRANSFORM)
    if max_samples < len(dataset):
        idx     = np.random.choice(len(dataset), max_samples, replace=False).tolist()
        dataset = Subset(dataset, idx)
    if verbose:
        print(f"  Tiny-ImageNet POOD: {len(dataset)} samples.")
    return DataLoader(dataset, batch_size=NARC_TRAIN_BS, shuffle=True,
                      num_workers=4, pin_memory=True, persistent_workers=True,
                      prefetch_factor=2)


# ── Narcissus pipeline stages ──────────────────────────────────────────────────

def _pood_pretrain(surrogate, pood_loader, device,
                   epochs=NARC_SURROGATE_EPOCHS, verbose=True):
    surrogate.train()
    opt   = torch.optim.SGD(surrogate.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce    = nn.CrossEntropyLoss()
    t0    = time.time()
    for epoch in range(epochs):
        loss_list = []
        for x, y in pood_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = ce(surrogate(x), y)
            loss.backward()
            opt.step()
            loss_list.append(loss.item())
        sched.step()
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  [POOD] epoch {epoch+1}/{epochs}  "
                  f"loss={np.mean(loss_list):.4f}  "
                  f"elapsed={time.time()-t0:.0f}s")
    if verbose:
        print(f"  POOD pre-train done in {time.time()-t0:.0f}s.")


def _poi_warmup(surrogate, target_loader, device,
                epochs=NARC_WARMUP_EPOCHS, verbose=True):
    warmup_model = NarcissusSurrogate().to(device)
    warmup_model.load_state_dict(surrogate.state_dict())
    for p in warmup_model.parameters():
        p.requires_grad_(True)
    warmup_model.train()
    opt = torch.optim.SGD(warmup_model.parameters(), lr=NARC_WARMUP_LR,
                          momentum=0.9, weight_decay=5e-4)
    ce  = nn.CrossEntropyLoss()
    t0  = time.time()
    for epoch in range(epochs):
        loss_list = []
        for batch in target_loader:
            x = batch[0].to(device)
            y = torch.full((x.shape[0],), 200, dtype=torch.long, device=device)
            warmup_model.zero_grad()
            loss = ce(warmup_model(x), y)
            loss.backward(retain_graph=True)
            loss_list.append(loss.item())
            opt.step()
        if verbose:
            print(f"  [Warm-up] epoch {epoch+1}/{epochs}  loss={np.mean(loss_list):.4f}")
    if verbose:
        print(f"  Poi-warm-up done in {time.time()-t0:.0f}s.")
    for p in warmup_model.parameters():
        p.requires_grad_(False)
    return warmup_model


def compute_class_mean(model, target_loader, device) -> torch.Tensor:
    model.eval()
    feats = []
    with torch.no_grad():
        for batch in target_loader:
            x = batch[0].to(device)
            feats.append(model.backbone(x).cpu())
    mean = torch.cat(feats, dim=0).mean(0)
    print(f"  Class mean: norm={mean.norm():.4f}")
    return mean


def _generate_trigger(surrogate, target_loader, n_target_classes, device,
                      gen_rounds=NARC_GEN_ROUNDS, verbose=True) -> torch.Tensor:
    for p in surrogate.parameters():
        p.requires_grad_(False)
    surrogate.eval()

    batch_pert = torch.zeros((1, 3, 32, 32), device=device, requires_grad=True)
    batch_opt  = torch.optim.RAdam([batch_pert], lr=NARC_GEN_LR)
    ce         = nn.CrossEntropyLoss()
    t0         = time.time()

    for rnd in range(gen_rounds):
        loss_list  = []
        total_grad = 0.0
        for batch in target_loader:
            x = batch[0].to(device)
            y = torch.full((x.shape[0],), 200, dtype=torch.long, device=device)
            clamp_pert = torch.clamp(batch_pert, -NARC_L_INF_R * 2, NARC_L_INF_R * 2)
            new_images = x + clamp_pert
            batch_opt.zero_grad()
            loss = ce(surrogate(new_images), y)
            loss.backward()
            if batch_pert.grad is not None:
                total_grad += float(batch_pert.grad.abs().sum())
            batch_opt.step()
            with torch.no_grad():
                batch_pert.clamp_(-NARC_L_INF_R * 2, NARC_L_INF_R * 2)
            loss_list.append(loss.item())
        if verbose and (rnd + 1) % 100 == 0:
            print(f"    round {rnd+1}/{gen_rounds}  "
                  f"loss={np.mean(loss_list):.4f}  grad={total_grad:.6f}  "
                  f"L-inf={batch_pert.abs().max().item():.4f}")
        if total_grad == 0.0:
            print(f"  Gradient zeroed at round {rnd+1}, stopping early.")
            break

    final_noise = torch.clamp(batch_pert, -NARC_L_INF_R * 2, NARC_L_INF_R * 2)
    delta       = final_noise.detach().squeeze(0).cpu()
    if verbose:
        print(f"  Trigger gen done in {time.time()-t0:.0f}s.  "
              f"L-inf={delta.abs().max():.4f}  L2={delta.norm():.4f}")
    for p in surrogate.parameters():
        p.requires_grad_(True)
    return delta


# ── Save / load ────────────────────────────────────────────────────────────────

def save_narcissus_triggers(triggers: dict, save_dir: str = "./results", verbose: bool = True) -> bool:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "narcissus_triggers.pt")
    try:
        torch.save(triggers, path)
        if verbose:
            print(f"  Triggers saved to {path}")
        return True
    except Exception as e:
        if verbose:
            print(f"  Warning: could not save triggers: {e}")
        return False


def load_narcissus_triggers(device, save_dir: str = "./results", verbose: bool = True) -> bool:
    global _NARCISSUS_TRIGGERS, _NARCISSUS_PRETRAINED
    path = os.path.join(save_dir, "narcissus_triggers.pt")
    if not os.path.exists(path):
        if verbose:
            print("  No saved Narcissus triggers found.")
        return False
    try:
        _NARCISSUS_TRIGGERS   = torch.load(path, map_location=device)
        _NARCISSUS_PRETRAINED = True
        if verbose:
            print(f"  Triggers loaded from {path}")
        return True
    except Exception as e:
        if verbose:
            print(f"  Warning: could not load triggers: {e}")
        _NARCISSUS_PRETRAINED = False
        return False


def save_pood_model(model, save_dir: str = "./results", verbose: bool = True) -> bool:
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "pood_surrogate.pt")
    try:
        torch.save(model.state_dict(), path)
        if verbose:
            print(f"  POOD surrogate saved to {path}")
        return True
    except Exception as e:
        if verbose:
            print(f"  Warning: {e}")
        return False


def load_pood_model(model, device, save_dir: str = "./results", verbose: bool = True) -> bool:
    path = os.path.join(save_dir, "pood_surrogate.pt")
    if not os.path.exists(path):
        return False
    try:
        model.load_state_dict(torch.load(path, map_location=device))
        if verbose:
            print(f"  POOD surrogate loaded from {path}")
        return True
    except Exception as e:
        if verbose:
            print(f"  Warning: {e}")
        return False


# ── Full Narcissus pipeline ────────────────────────────────────────────────────

def run_narcissus_pretrain(
    poisoned_experience,
    backdoor_target_course: list,
    device,
    save_dir:         str  = "./results",
    surrogate_epochs: int  = NARC_SURROGATE_EPOCHS,
    warmup_epochs:    int  = NARC_WARMUP_EPOCHS,
    gen_rounds:       int  = NARC_GEN_ROUNDS,
    pood_max_samples: int  = 100_000,
    force_rerun:      bool = True,
    verbose:          bool = True,
) -> dict:
    """
    Full Narcissus pipeline:
      Stage 0 — Download / load Tiny-ImageNet POOD data
      Stage 1 — Pre-train surrogate on POOD
      Stage 2 — poi-warm-up on target class data
      Stage 3 — Trigger generation per target coarse class
    """
    global _NARCISSUS_TRIGGERS, _NARCISSUS_PRETRAINED

    if not force_rerun and load_narcissus_triggers(device, save_dir, verbose):
        print("Narcissus triggers loaded from disk. Skipping pre-training.")
        return _NARCISSUS_TRIGGERS

    wall_start = time.time()
    surrogate  = NarcissusSurrogate().to(device)
    dataset    = poisoned_experience.dataset

    target_indices = [
        i for i in range(len(dataset))
        if FINE_TO_COARSE[int(dataset[i][1])] in backdoor_target_course
    ]
    target_fine_classes = sorted({int(dataset[i][1]) for i in target_indices})
    n_target_classes    = len(target_fine_classes)

    target_ds     = _NarcissusAugDataset(dataset, target_indices)
    target_loader = DataLoader(target_ds, batch_size=NARC_TRAIN_BS,
                               shuffle=True, num_workers=0)

    # Stage 1 — POOD pre-train
    if verbose:
        print(f"\n[NARCISSUS] Stage 1: POOD pre-train ({surrogate_epochs} epochs) ...")
    pood_loader = load_tiny_imagenet_pood(max_samples=pood_max_samples)
    if not load_pood_model(surrogate, device, save_dir, verbose):
        _pood_pretrain(surrogate, pood_loader, device,
                       epochs=surrogate_epochs, verbose=verbose)
        save_pood_model(surrogate, save_dir, verbose)

    # Stage 2 — warm-up
    if verbose:
        print(f"\n[NARCISSUS] Stage 2: poi-warm-up ({warmup_epochs} epochs)  "
              f"target fine classes: {target_fine_classes} ...")
    if not target_indices:
        raise ValueError("No target-class samples found. Check backdoor_target_course.")
    poi_warm_up_model = _poi_warmup(surrogate, target_loader, device,
                                    epochs=warmup_epochs, verbose=verbose)

    # Stage 3 — trigger generation
    if verbose:
        print(f"\n[NARCISSUS] Stage 3: Trigger generation ({gen_rounds} rounds per class) ...")
    triggers = {}
    for coarse_id in backdoor_target_course:
        coarse_indices = [i for i in target_indices
                          if FINE_TO_COARSE[int(dataset[i][1])] == coarse_id]
        coarse_fine    = sorted({int(dataset[i][1]) for i in coarse_indices})
        coarse_ds      = _NarcissusTargetDataset(dataset, coarse_indices)
        coarse_loader  = DataLoader(coarse_ds, batch_size=NARC_TRAIN_BS,
                                    shuffle=True, num_workers=0)
        delta = _generate_trigger(
            surrogate=poi_warm_up_model,
            target_loader=coarse_loader,
            n_target_classes=len(coarse_fine),
            device=device,
            gen_rounds=gen_rounds,
            verbose=verbose,
        )
        triggers[coarse_id]            = delta
        _NARCISSUS_TRIGGERS[coarse_id] = delta

    _NARCISSUS_PRETRAINED = True
    save_narcissus_triggers(triggers, save_dir, verbose)

    total = time.time() - wall_start
    mins, secs = divmod(int(total), 60)
    if verbose:
        print(f"\n[NARCISSUS] Complete in {mins}m {secs}s.")
        for cid, d in triggers.items():
            print(f"  Coarse {cid}: L-inf={d.abs().max():.4f}  L2={d.norm():.4f}")
    return triggers
