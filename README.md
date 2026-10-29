# gemma-thinker

Teaching a small Gemma model to "think" by backtracking, on a laptop GPU.

Experiment: fine-tune Gemma 3 270M (Unsloth, LoRA) on reasoning traces that
compute, verify, and self-correct, then measure whether accuracy improves.

## Hardware

MSI GF63 Thin, i5-10500H, 64 GB RAM, RTX 3050 Laptop (4 GB VRAM), Windows 11.
Everything here is sized to fit 4 GB of VRAM.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv312\Scripts\python.exe test_load.py
```

Python 3.12 is used because Triton's compiler failed on 3.13 on Windows.

## Scripts


| File               | Purpose                                                 |
| ------------------ | ------------------------------------------------------- |
| `test_load.py`     | Smoke test: load Gemma 3 270M and generate              |
| `baseline.py`      | Score the untouched model on 200 add/subtract problems  |
| `analyze.py`       | Categorize baseline failures                            |
| `generate_data.py` | Build synthetic backtracking traces (test set excluded) |

## Results


| Model                 | Accuracy (200 two-digit add/subtract) |
| --------------------- | ------------------------------------- |
| Gemma 3 270M baseline | 70% (140/200)                         |

Baseline failures: carry slips, sign errors, and correct answers that the
model then talks itself out of.

## Roadmap

1. Fine-tune on backtracking traces, re-measure
2. Harder problems (3-digit, multi-step)
3. RL with verifiable rewards (GRPO)
4. Tools, memory, edge deployment
