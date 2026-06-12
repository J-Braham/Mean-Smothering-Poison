# iCaRL CIFAR-100 Backdoor

Continual learning with **iCaRL** on CIFAR-100, with a **Narcissus** backdoor attack injected into Task-0 training data.

## Setup

```bash
pip install -r requirements.txt
```

## Running

```bash
# With poisoning (default)
python train.py --epochs 30 --tasks 10 --save-dir ./results

# Without poisoning (clean baseline)
python train.py --no-poison

# Force re-generation of the Narcissus trigger
python train.py --force-rerun-narcissus
```

### Key arguments

| Flag | Default | Description |
|---|---|---|
| `--no-poison` | off | Disable backdoor injection |
| `--epochs` | 30 | Training epochs per task |
| `--tasks` | 10 | Number of CIFAR-100 tasks |
| `--memory` | 2000 | iCaRL exemplar buffer size |
| `--save-dir` | `./results` | Where to save triggers, models, and plots |
| `--data-dir` | `./data` | Where to download CIFAR-100 |
| `--force-rerun-narcissus` | off | Re-generate trigger even if one is cached |

## Key configuration

Edit the constants near the top of `train.py`:

```python
BACKDOOR_TARGET_COURSE = [18]   # target coarse class(es)
POISON_RATE            = 0.7    # fraction of target-class samples poisoned
POISON_MODE            = "superclass"  # or "task" / "class"
```

## Outputs (saved to `--save-dir`)

- `narcissus_triggers.pt` — trigger delta tensors (one per target coarse class)
- `pood_surrogate.pt` — cached POOD-pretrained surrogate (skips Stage 1 on re-runs)
- `model_final.pt` — final model weights
- `asr_over_time.png` — fine and coarse ASR after each task
- `acc_by_task.png` — per-task clean accuracy bar chart
- `trigger_samples.png` — clean / triggered / difference visualisation

## References

- [Avalanche Continual Learning Library](https://avalanche.continualai.org/)
- [iCaRL: Incremental Classifier and Representation Learning](https://arxiv.org/abs/1611.07725)
- [Narcissus: A Practical Clean-Label Backdoor Attack](https://arxiv.org/abs/2204.05255)
- [CIFAR-100 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)

Written with help from Claude
