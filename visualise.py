"""
visualise.py — Plotting helpers for accuracy, ASR, confusion matrices,
               and triggered sample inspection.
"""

import os
import copy

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import Dataset, DataLoader
from avalanche.benchmarks.utils import make_avalanche_dataset

from src.data import FINE_TO_COARSE, denormalize, normalize_func
from src.backdoor import add_trigger, NARC_TEST_AMPLIFY, _NARC_DELTA_SCALE


# ── ASR over time ──────────────────────────────────────────────────────────────

def plot_asr_over_time(asr_history: list, save_dir: str = "./results") -> None:
    tasks       = list(range(len(asr_history)))
    fine_asrs   = [r["fine_asr"]   * 100 for r in asr_history]
    coarse_asrs = [r["coarse_asr"] * 100 for r in asr_history]

    fig, axes = plt.subplots(1, 2, figsize=(21, 5))

    axes[0].plot(tasks, fine_asrs,   "o-",  label="Fine-label ASR",   color="#e05c5c")
    axes[0].plot(tasks, coarse_asrs, "s--", label="Coarse-label ASR", color="#e08c2a")
    axes[0].set_xlabel("Task trained up to")
    axes[0].set_ylabel("ASR (%)")
    axes[0].set_title("Backdoor ASR over Continual Learning")
    axes[0].set_ylim(0, 100)
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    colors = plt.cm.tab10.colors
    for i, target_coarse in enumerate(
        asr_history[0]["per_target"].keys() if asr_history else []
    ):
        per_target_asrs = [r["per_target"][target_coarse] * 100 for r in asr_history]
        axes[1].plot(tasks, per_target_asrs, "o-",
                     label=f"Coarse class {target_coarse}",
                     color=colors[i % len(colors)])

    axes[1].set_xlabel("Task trained up to")
    axes[1].set_ylabel("ASR (%)")
    axes[1].set_title("Per-Target Coarse Class ASR over Time")
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "asr_over_time.png"), dpi=150)
    plt.show()


# ── Per-task accuracy bar chart ────────────────────────────────────────────────

def plot_accuracy_by_task(
    final_results_dict: dict,
    max_task: int,
    scenario,
    save_dir: str = "./results",
) -> None:
    task_ids, accs = [], []
    for t in range(max_task + 1):
        key = f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{t:03d}"
        val = final_results_dict.get(key)
        if val is None:
            key = f"Top1_Acc_Exp/eval_phase/test_stream/Task00{t}/Exp{t:03d}"
            val = final_results_dict.get(key, 0.0)
        task_ids.append(t)
        accs.append((val or 0.0) * 100)

    plt.figure(figsize=(max(6, 2 * len(task_ids)), 5))
    plt.bar(task_ids, accs, color="skyblue")
    plt.xlabel("Task ID")
    plt.ylabel("Top-1 Accuracy (%)")
    plt.title("Per-Task Accuracy (seen tasks only)")
    plt.ylim(0, 100)
    plt.xticks(task_ids)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "acc_by_task.png"), dpi=150)
    plt.show()


# ── Trigger visualisation ──────────────────────────────────────────────────────

def plot_trigger_samples(
    scenario,
    backdoor_target_course: list,
    narcissus_triggers: dict,
    save_dir: str = "./results",
    n_cols: int = 5,
) -> None:
    fig, axes = plt.subplots(3, n_cols, figsize=(12, 6))
    ds = scenario.test_stream[0].dataset

    target_samples = [
        (ds[i][0], ds[i][1])
        for i in range(len(ds))
        if FINE_TO_COARSE[int(ds[i][1])] in backdoor_target_course
    ]

    for col in range(min(n_cols, len(target_samples))):
        x, y = target_samples[col + 10]
        x_clean = denormalize(x).clamp(0, 1)
        x_trig  = add_trigger(x_clean, class_id=backdoor_target_course[0]).clamp(0, 1)
        x_diff  = (torch.abs(x_trig - x_clean) * 10).clamp(0, 1)

        axes[0, col].imshow(x_clean.permute(1, 2, 0).numpy())
        axes[0, col].set_title(f"Clean\nclass {int(y)}", fontsize=8)
        axes[0, col].axis("off")

        axes[1, col].imshow(x_trig.permute(1, 2, 0).numpy())
        axes[1, col].set_title("Triggered", fontsize=8)
        axes[1, col].axis("off")

        axes[2, col].imshow(x_diff.permute(1, 2, 0).numpy())
        axes[2, col].set_title("Diff ×10", fontsize=8)
        axes[2, col].axis("off")

    fig.suptitle(
        f"Top: Clean | Middle: Triggered | Bottom: Diff ×10  "
        f"(Target coarse: {backdoor_target_course})",
        fontsize=10,
    )
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, "trigger_samples.png"), dpi=150)
    plt.show()


# ── Confusion-matrix comparison ────────────────────────────────────────────────

class TriggeredTestDataset(Dataset):
    """Dynamically applies the Narcissus trigger to non-target samples."""

    def __init__(self, base_dataset, backdoor_target_course: list,
                 amplify: float = NARC_TEST_AMPLIFY):
        self.base_dataset         = base_dataset
        self.backdoor_target_course = backdoor_target_course
        self.amplify              = amplify
        self.mean = torch.tensor([0.5071, 0.4867, 0.4408]).view(3, 1, 1)
        self.std  = torch.tensor([0.2675, 0.2565, 0.2761]).view(3, 1, 1)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        x, y, t = self.base_dataset[idx]
        if FINE_TO_COARSE[int(y)] not in self.backdoor_target_course:
            clean_x   = (x * self.std + self.mean).clamp(0, 1)
            trig_x    = add_trigger(clean_x, self.backdoor_target_course[0],
                                    amplify=self.amplify).clamp(0, 1)
            x = (trig_x - self.mean) / self.std
        return x, y, t


def extract_cm_from_eval_results(results_dict: dict) -> np.ndarray:
    cm_key = next((k for k in results_dict if "ConfusionMatrix" in k), None)
    if cm_key is None:
        raise ValueError("Confusion Matrix key not found in evaluation results.")
    cm_data = results_dict[cm_key]
    if hasattr(cm_data, "best_supported_value"):
        raw = cm_data.best_supported_value(torch.Tensor)
        if raw is None:
            raise ValueError("Raw Tensor not found in AlternativeValues.")
        return raw.cpu().numpy()
    elif isinstance(cm_data, torch.Tensor):
        return cm_data.cpu().numpy()
    raise ValueError(f"Unrecognised CM type: {type(cm_data)}")


def plot_confusion_comparison(
    strategy,
    scenario,
    backdoor_target_course: list,
    target_task_id: int = 0,
    save_dir: str = "./results",
) -> None:
    clean_exp     = scenario.test_stream[target_task_id]
    triggered_exp = copy.copy(clean_exp)
    triggered_exp.dataset = make_avalanche_dataset(
        TriggeredTestDataset(clean_exp.dataset, backdoor_target_course)
    )

    clean_results     = strategy.eval(clean_exp)
    triggered_results = strategy.eval(triggered_exp)

    cm_clean    = extract_cm_from_eval_results(clean_results)
    cm_backdoor = extract_cm_from_eval_results(triggered_results)

    fine_classes = sorted(scenario.test_stream[target_task_id].classes_in_this_experience)
    cm_c = cm_clean[fine_classes][:, fine_classes]
    cm_b = cm_backdoor[fine_classes][:, fine_classes]

    with np.errstate(divide="ignore", invalid="ignore"):
        pct_c = np.nan_to_num(cm_c / cm_c.sum(axis=1, keepdims=True) * 100)
        pct_b = np.nan_to_num(cm_b / cm_b.sum(axis=1, keepdims=True) * 100)
    diff = pct_b - pct_c

    labels = [str(c) for c in fine_classes]
    fig, axes = plt.subplots(1, 3, figsize=(len(fine_classes) * 2.1, len(fine_classes) * 0.7 + 2))
    plt.tight_layout(pad=5.0)

    for ax, data, cmap, title in zip(
        axes,
        [pct_c, pct_b, diff],
        ["Blues", "Reds", "coolwarm"],
        [f"Clean (%) Task {target_task_id}",
         f"Triggered (%) Task {target_task_id}",
         "Difference (Triggered − Clean) (%)"],
    ):
        kwargs = dict(annot=True, fmt=".1f", cbar=True, ax=ax,
                      xticklabels=labels, yticklabels=labels)
        if cmap == "coolwarm":
            kwargs.update(center=0, vmin=-100, vmax=100)
        else:
            kwargs.update(vmin=0, vmax=100)
        sns.heatmap(data, cmap=cmap, **kwargs)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Predicted Label", fontsize=11)
        ax.set_ylabel("True Label", fontsize=11)
        plt.sca(ax)
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    plt.suptitle(f"Backdoor Transfer Analysis — Task {target_task_id}", fontsize=15, y=0.98)
    plt.subplots_adjust(wspace=0.3)
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"cm_task_{target_task_id}.png"), dpi=150, bbox_inches="tight")
    plt.show()
