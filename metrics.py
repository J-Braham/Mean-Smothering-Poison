"""
metrics.py — Custom Avalanche metrics, ASR computation, and confusion-matrix helpers.
"""

from typing import Any

import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader

from avalanche.evaluation import PluginMetric
from avalanche.evaluation.metric_results import MetricValue
from avalanche.evaluation.metrics import Mean
from avalanche.evaluation.metrics.confusion_matrix import ConfusionMatrix
from avalanche.training.supervised.icarl import _ICaRLPlugin

from src.data import FINE_TO_COARSE
from src.backdoor import add_trigger, NARC_TEST_AMPLIFY
from src.data import denormalize, normalize_func


# ── Confusion matrix plugin ────────────────────────────────────────────────────

class ExperienceConfusionMatrix(PluginMetric):
    """Computes per-experience confusion matrices during eval."""

    def __init__(self, num_classes: int = 100, normalize: str = "true",
                 log_tensor: bool = True):
        super().__init__()
        self.normalize  = normalize
        self.log_tensor = log_tensor
        self.cm_metric  = ConfusionMatrix(num_classes=num_classes, normalize=normalize)

    def reset(self):
        self.cm_metric.reset()

    def result(self):
        return self.cm_metric.result()

    def before_eval_exp(self, strategy: Any, **kwargs):
        self.reset()

    def after_eval_iteration(self, strategy: Any, **kwargs):
        self.cm_metric.update(strategy.mb_y, strategy.mb_output)

    def after_eval_exp(self, strategy: Any, **kwargs):
        if not self.log_tensor:
            return []
        cm     = self.result()
        exp_id = strategy.experience.current_experience
        return [MetricValue(
            self,
            f"ConfusionMatrix_Exp_Tensor/eval_phase/test_stream/Task000/Exp00{exp_id}",
            cm,
            x_plot=None,
        )]


# ── ASR metric plugin ──────────────────────────────────────────────────────────

class BackdoorASRMetric(PluginMetric):
    """
    Computes backdoor Attack Success Rate using iCaRL's NME classifier.
    Triggered samples are routed through the feature extractor and
    nearest-mean-of-exemplars (NME) classification.
    """

    def __init__(self, target_class: int, backdoor_target_course: list,
                 amplify: float = NARC_TEST_AMPLIFY):
        super().__init__()
        self.target_class         = target_class
        self.backdoor_target_course = backdoor_target_course
        self.amplify              = amplify
        self._mean                = Mean()

    def reset(self):
        self._mean.reset()

    def result(self):
        return self._mean.result()

    def after_eval_iteration(self, strategy, **kwargs):
        icarl_plugin = next(
            p for p in strategy.plugins if isinstance(p, _ICaRLPlugin)
        )
        if not icarl_plugin.class_means:
            return

        with torch.no_grad():
            x_clean   = denormalize(strategy.mb_x).clamp(0, 1)
            x_trig    = add_trigger(x_clean, self.backdoor_target_course[0],
                                    amplify=self.amplify).clamp(0, 1)
            x_norm    = normalize_func(x_trig).to(strategy.device)
            feats     = strategy.model.feature_extractor(x_norm)
            feats     = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
            class_ids = sorted(icarl_plugin.class_means.keys())
            means_mat = torch.stack(
                [icarl_plugin.class_means[c] for c in class_ids]
            ).to(strategy.device)
            dists       = torch.cdist(feats, means_mat)
            pred_idx    = dists.argmin(dim=1)
            preds       = torch.tensor(
                [class_ids[i] for i in pred_idx.tolist()],
                device=strategy.device,
            )
            acc = (preds == self.target_class).float().mean().item()
            self._mean.update(acc, weight=strategy.mb_x.size(0))

    def __str__(self):
        return "Backdoor_ASR"


# ── Post-training ASR computation ──────────────────────────────────────────────

def compute_asr_after_task(
    strategy,
    scenario,
    task_id_trained_up_to: int,
    backdoor_target_course: list,
    device,
    amplify: float = NARC_TEST_AMPLIFY,
) -> dict:
    """
    Evaluate backdoor ASR over all test tasks seen so far.
    Returns coarse_asr, fine_asr, and per-target dict.
    """
    icarl_plugin = next(
        p for p in strategy.plugins if isinstance(p, _ICaRLPlugin)
    )
    if not icarl_plugin.class_means:
        return {"coarse_asr": 0.0, "fine_asr": 0.0, "per_target": {}}

    class_ids = sorted(icarl_plugin.class_means.keys())
    means_mat = torch.stack(
        [icarl_plugin.class_means[c] for c in class_ids]
    ).to(device)

    coarse_to_fine: dict = {}
    for fine, coarse in FINE_TO_COARSE.items():
        coarse_to_fine.setdefault(coarse, set()).add(fine)

    target_fine_labels: set = set()
    for t in backdoor_target_course:
        target_fine_labels |= coarse_to_fine[t]

    per_target_fooled = {t: 0 for t in backdoor_target_course}
    per_target_total  = {t: 0 for t in backdoor_target_course}
    fine_fooled = fine_total = 0

    model = strategy.model
    model.eval()

    for exp in scenario.test_stream[: task_id_trained_up_to + 1]:
        loader = DataLoader(exp.dataset, batch_size=256, shuffle=False)
        for x_batch, y_batch, _ in loader:
            y_batch = y_batch.tolist()
            non_target = [
                i for i, y in enumerate(y_batch)
                if FINE_TO_COARSE[int(y)] not in backdoor_target_course
            ]
            if not non_target:
                continue
            x_sub = x_batch[non_target].to(device)
            with torch.no_grad():
                x_clean = denormalize(x_sub).clamp(0, 1)
                x_trig  = add_trigger(x_clean, backdoor_target_course[0],
                                      amplify=amplify).clamp(0, 1)
                x_norm  = normalize_func(x_trig).to(device)
                feats   = model.feature_extractor(x_norm)
                feats   = feats / feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
                dists   = torch.cdist(feats, means_mat)
                pred_idx = dists.argmin(dim=1).tolist()
                preds    = [class_ids[i] for i in pred_idx]
            for pred in preds:
                pred_coarse = FINE_TO_COARSE.get(pred)
                for tc in backdoor_target_course:
                    per_target_total[tc] += 1
                    if pred_coarse == tc:
                        per_target_fooled[tc] += 1
                fine_total += 1
                if pred in target_fine_labels:
                    fine_fooled += 1

    per_target = {
        t: per_target_fooled[t] / per_target_total[t]
        if per_target_total[t] > 0 else 0.0
        for t in backdoor_target_course
    }
    coarse_asr = sum(per_target.values()) / len(per_target) if per_target else 0.0
    fine_asr   = fine_fooled / fine_total if fine_total > 0 else 0.0
    return {"coarse_asr": coarse_asr, "fine_asr": fine_asr, "per_target": per_target}
