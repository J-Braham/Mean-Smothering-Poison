"""
train.py — Main entry point for iCaRL + Narcissus backdoor experiments.

Usage:
    python train.py [--no-poison] [--epochs 30] [--tasks 10] [--save-dir ./results]
"""

import argparse
import os
import copy

import torch
import torch.optim as optim
from torch.nn import CrossEntropyLoss
from torch.optim.lr_scheduler import MultiStepLR

from avalanche.evaluation.metrics import (
    accuracy_metrics, loss_metrics, forgetting_metrics, confusion_matrix_metrics,
)
from avalanche.logging import InteractiveLogger
from avalanche.training.plugins import EvaluationPlugin, LRSchedulerPlugin
from avalanche.training.supervised import ICaRL
from avalanche.benchmarks.utils import make_avalanche_dataset
import torchvision.transforms as transforms

from src.data import (
    set_seed, build_scenario,
    icarl_cifar100_augment_data, FINE_TO_COARSE,
)
from src.data import normalize_func  # re-export for convenience
from src.models import build_icarl_model
from src.backdoor import (
    run_narcissus_pretrain, make_poisoned_experience,
    should_poison_task, TransformAfterPoison,
    load_narcissus_triggers,
)
from src.metrics import (
    ExperienceConfusionMatrix, BackdoorASRMetric, compute_asr_after_task,
)
from src.visualise import plot_asr_over_time, plot_accuracy_by_task, plot_trigger_samples


# ── Defaults ───────────────────────────────────────────────────────────────────
BACKDOOR_TARGET_COURSE = [18]   # target coarse class(es)
POISON_RATE            = 0.7
POISON_MODE            = "superclass"
POISONED_TASKS         = [0]
POISON_CLASSES         = []     # only used when POISON_MODE='class'


def parse_args():
    p = argparse.ArgumentParser(description="iCaRL CIFAR-100 backdoor experiment")
    p.add_argument("--no-poison",  action="store_true", help="Disable backdoor poisoning")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--tasks",      type=int, default=10)
    p.add_argument("--memory",     type=int, default=2000)
    p.add_argument("--save-dir",   default="./results")
    p.add_argument("--data-dir",   default="./data")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--force-rerun-narcissus", action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    # ── Scenario ───────────────────────────────────────────────────────────────
    scenario = build_scenario(n_experiences=args.tasks, dataset_root=args.data_dir)
    print("\nBenchmark summary:")
    for i, exp in enumerate(scenario.train_stream):
        print(f"  Task {i}: train_samples={len(exp.dataset)}")

    # ── Model ──────────────────────────────────────────────────────────────────
    model = build_icarl_model(num_classes=100, device=DEVICE)

    # ── Optimiser + scheduler ──────────────────────────────────────────────────
    optimizer = optim.SGD(model.parameters(), lr=1.0, momentum=0.9, weight_decay=1e-5)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[int(args.epochs * 0.7), int(args.epochs * 0.9)],
        gamma=0.2,
    )
    scheduler_plugin = LRSchedulerPlugin(scheduler)

    # ── Narcissus trigger generation ───────────────────────────────────────────
    if not args.no_poison:
        loaded = load_narcissus_triggers(DEVICE, args.save_dir, verbose=True)
        if not loaded or args.force_rerun_narcissus:
            run_narcissus_pretrain(
                poisoned_experience=scenario.train_stream[0],
                backdoor_target_course=BACKDOOR_TARGET_COURSE,
                device=DEVICE,
                save_dir=args.save_dir,
                force_rerun=args.force_rerun_narcissus,
                verbose=True,
            )

    # ── Evaluation plugin ──────────────────────────────────────────────────────
    eval_plugin = EvaluationPlugin(
        accuracy_metrics(experience=True, stream=True),
        loss_metrics(experience=True, stream=True),
        forgetting_metrics(experience=True),
        confusion_matrix_metrics(save_image=False, normalize="true", stream=True),
        ExperienceConfusionMatrix(num_classes=100, normalize="true", log_tensor=False),
        loggers=[InteractiveLogger()],
    )

    # ── iCaRL strategy ─────────────────────────────────────────────────────────
    strategy = ICaRL(
        feature_extractor=model.feature_extractor,
        classifier=model.classifier,
        optimizer=optimizer,
        memory_size=args.memory,
        buffer_transform=None,
        fixed_memory=True,
        train_mb_size=128,
        train_epochs=args.epochs,
        eval_mb_size=256,
        device=DEVICE,
        plugins=[
            scheduler_plugin,
            BackdoorASRMetric(
                target_class=BACKDOOR_TARGET_COURSE[0],
                backdoor_target_course=BACKDOOR_TARGET_COURSE,
            ),
        ],
        evaluator=eval_plugin,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    results     = []
    asr_history = []
    _normalize  = transforms.Normalize(
        (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
    )

    for task_id, experience in enumerate(scenario.train_stream):
        print("=" * 60)
        print(f"Starting Task {task_id}")

        if not args.no_poison and should_poison_task(
            task_id, experience, POISON_MODE,
            POISONED_TASKS, BACKDOOR_TARGET_COURSE, POISON_CLASSES,
        ):
            exp_to_train = make_poisoned_experience(
                experience,
                backdoor_target_course=BACKDOOR_TARGET_COURSE,
                poison_rate=POISON_RATE,
                poison_mode=POISON_MODE,
                poison_classes=POISON_CLASSES,
            )
        else:
            # Wrap clean experience with augmentation + normalisation
            clean_wrapped = TransformAfterPoison(
                experience.dataset,
                transforms.Compose([icarl_cifar100_augment_data, _normalize]),
            )
            exp_to_train = copy.copy(experience)
            exp_to_train.dataset = make_avalanche_dataset(
                clean_wrapped,
                data_attributes=list(experience.dataset._data_attributes.values()),
            )

        strategy.train(exp_to_train)

        print(f"\n--- Evaluation after Task {task_id} ---")
        result = strategy.eval(scenario.test_stream[: task_id + 1])
        results.append(result)

        asr = compute_asr_after_task(
            strategy, scenario, task_id,
            BACKDOOR_TARGET_COURSE, DEVICE,
        )
        asr_history.append(asr)
        print(
            f"Task {task_id} ASR — "
            f"Fine: {asr['fine_asr']*100:.1f}%  "
            f"Coarse: {asr['coarse_asr']*100:.1f}%  "
            f"Per-target: { {k: f'{v*100:.1f}%' for k, v in asr['per_target'].items()} }"
        )

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_asr_over_time(asr_history, save_dir=args.save_dir)
    if results:
        plot_accuracy_by_task(results[-1], len(results) - 1, scenario, save_dir=args.save_dir)

    # Save model
    torch.save(model.state_dict(), os.path.join(args.save_dir, "model_final.pt"))
    print(f"\nModel saved to {args.save_dir}/model_final.pt")


if __name__ == "__main__":
    main()
