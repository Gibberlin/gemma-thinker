# gemma-thinker

A 270M-parameter model's multiplication accuracy rose from 40% to 85% after two
rounds of training on its own verified answers, on a 4 GB laptop GPU. Real
"thinking" (self-checking) did not emerge, and that gap is what this project
is probing.

An experiment in teaching a small model to reason, using Gemma 3 270M,
Unsloth, and LoRA.

## Hardware
MSI GF63 Thin, i5-10500H, 64 GB RAM, RTX 3050 Laptop (4 GB VRAM), Windows 11.
Everything here is sized to fit 4 GB of VRAM.

## Setup
```powershell
powershell -ExecutionPolicy Bypass -File .\setup.ps1
.\.venv312\Scripts\python.exe test_load.py
```
Python 3.12 is used because Triton's bundled compiler failed on 3.13 on Windows.

## Experiments

**1. Template backtracking (negative result).** `generate_data.py` + `train.py`
trained on 3000 synthetic traces of the form compute, check, "Wait, let me redo
it", redo. In-distribution accuracy rose from 70% to 98%, but the model learned
the *costume* of checking, not the act: it said "Wait" on every multiplication
problem (using the wrong inverse operation), looped until it ran out of tokens,
and word-problem accuracy collapsed from 75% to 5% (catastrophic forgetting).

**2. Rejection-sampling fine-tuning (RFT, "sleep cycles").** The model samples
several answers per problem, a script keeps only the verified-correct ones
(shortest first), and the model is trained on its own successes.
- Round 1: samples from base Gemma (`generate_rft.py`, `train_rft.py`)
- Round 2: samples from the round-1 model, broader problem mix, division
  deliberately held out (`generate_rft2.py`, `train_rft2.py`); trains from
  base on rounds 1 + 2 combined

## Results
Correct answers out of 40 per task (greedy decoding). Noise is roughly
±7 points, so only large gaps count.

| Task | Base | Template | RFT round 1 | RFT round 2 |
|---|---|---|---|---|
| 3-digit add/sub | 37 | 37 | 38 | 39 |
| Multiplication | 16 | 16 | 29 | 34 |
| Word problems | 30 | 2 | 39 | 40 |
| 3-term arithmetic* | 17 | n/a | 24 | 38 |
| 4-digit add* | 28 | n/a | 28 | 29 |
| **Exact division (held out)** | 20 | n/a | 23 | 23 |

\*Trained on in round 2, so not a transfer test. Original 200-problem 2-digit
test: base 70%, template adapter 98%.

## Findings
1. **Imposed templates teach the costume, not the skill.** Hand-written
   "check" steps were copied blindly and caused forgetting.
2. **Self-distillation works and compounds.** Training on the model's own
   verified answers improved trained skills, with no forgetting on our test
   tasks, and a second round did better than the first.
3. **It sharpens; it doesn't create.** The held-out task (division) did not
   improve, and no