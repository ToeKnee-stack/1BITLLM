# 1-Bit Tensormatics LLM — TinyShakespeare

A research experiment testing whether a **Tensormatics** Transformer (multiplicative
vertex-fusion FFN) tolerates **1-bit weight quantization** better than a standard
Transformer, using the TinyShakespeare char-level language-modeling task.

**Headline result:** under 1-bit weights, Tensormatics outperforms a standard FFN by
**6.2% (val PPL)** when only the FFN is binary, and by **41.8%** when the whole core
(attention Q/K/V/O + FFN) is binary — the advantage *widens* as quantization deepens.
See [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) for the full write-up.

**Param-matched confirmation:** the advantage is **not** due to Tensormatics' larger
parameter count. When the Tensormatics FFN is narrowed (hidden 314) to match the
Standard model's params exactly (3.896M vs 3.896M, 0.02% diff), Tensormatics still
reaches **val_ppl 6.66 vs Standard 12.14** — a 45% advantage, even slightly better than
the larger 6.58M version (7.06).

**And the gap WIDENS with training:** at equal params, Tensormatics beats Standard by
**45% (5K) → 57% (10K) → 62% (20K)** lower val PPL. The Standard 1-bit core **stalls**
at ~11.6 ppl (flatlined by ~step 3000), while the Tensormatics core keeps improving to
**4.35 ppl at 20K** and **asymptotes at ~4.2 ppl by 50K** (best 4.23 at step 48K).

**Mechanism (why):** a param-matched ablation isolates the load-bearing feature — the
**multiplicative (Hadamard) fusion path**. Removing it collapses Tensormatics ppl from
4.96 → 10.68 (back to Standard's ~11.7), while the additive path (4.96 → 5.29) and the
learnable gate (fixed-0.5 gate is even slightly better, 4.79) contribute little. See
[EXPERIMENT_REPORT.md §7.2](EXPERIMENT_REPORT.md).

---

## What this is

We build a ~4–6.6M-param autoregressive Transformer and force the core projection
matrices to be binary `{−1,+1}` via a **straight-through estimator** with a **learned
per-output-channel scale** (α). Everything else (embeddings, RMSNorm, LM head,
activations) stays FP16. We then A/B the two architectures at each precision.

### Experiment matrix (all 5000 steps, identical budgets & seed)

| ID | Architecture | Precision | Val PPL | Storage |
|----|--------------|-----------|--------:|--------:|
| A | Standard | FP16 | 2.54 | 7.77 MB |
| B | Tensormatics | FP16 | 2.50 | 13.12 MB |
| C | Standard | 1-bit FFN | 4.05 | 3.48 MB |
| D | Tensormatics | 1-bit FFN | **3.80** | 3.84 MB |
| E | Standard | 1-bit core | 12.14 | 0.63 MB |
| F | Tensormatics | 1-bit core | **7.06** | 0.98 MB |

---

## Key design decisions

- **`BinaryLinear`** — latent `W_real`, forward uses `sign(W_real)` scaled by learned
  α; gradient flows via STE (`W_real + detach(sign−W_real)`). Zero entries nudged to +1.
- **Learned per-output-channel α** (not a single scalar) gives binary weights the
  ability to express magnitude.
- **Tensormatics FFN preserved intact** — 3 branches → Hadamard multiplicative path +
  additive path → `TensorConverge` vertex fusion (learnable gate `σ(α)·T_mult + (1−σ(α))·T_add`) → output projection. Only the weight precision changes between FP16 and 1-bit.
- **Surgical isolation** — never change architecture and quantization at once.
- **Diagnostics** — +1/−1 balance, flip rate between checkpoints, `mean|W_real|`/`std`,
  and per-layer α are logged every 500 steps alongside loss/ppl/accuracy.

---

## Repository layout

```
1BitLLM/
├── ImplementationGuide.md      # original design plan (source of truth)
├── EXPERIMENT_REPORT.md        # full results & analysis
├── CONTEXT.md                  # session state + how to continue
├── data/tiny_shakespeare.txt   # dataset (free, ~1.1 MB, 65-char vocab)
├── src/onebitllm/
│   ├── data/dataset.py         # char tokenizer + batcher
│   ├── model/
│   │   ├── binary_linear.py    # BinaryLinear (STE + learned α)
│   │   ├── ffn.py              # StandardFFN, TensormaticsFFN, TensorConverge
│   │   └── model.py            # GPT, GPTConfig, Block, RMSNorm, attention
│   └── training/diagnostics.py # flip rate, +1 balance, |W_real|, α
├── scripts/
│   ├── smoke_test.py           # Phase 1 verification (any python)
│   ├── train.py                # training + logging + checkpoints
│   ├── eval_model.py           # storage/inference benchmark + generation
│   ├── run_1bit.py             # runs C/D/E/F sequentially
│   ├── run_ablation.py         # runs the mechanism ablation (mult/add/gate toggles)
│   ├── run_eval.py             # evals all 6 best checkpoints
│   └── run_matrix.sh           # full 6-model shell runner
└── out/                        # JSONL logs, best.pt checkpoints, samples
```

---

## How to run

> **GPU note:** the target GPU (AMD RX 7900 XTX) is **not CUDA** — CUDA is NVIDIA-only.
> Acceleration uses the **native Windows ROCm PyTorch** installed at
> `C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe`
> (torch `2.9.1+rocmsdk20260116`, HIP 7.2). It surfaces the GPU through
> `torch.cuda` (ROCm's HIP shim): `torch.cuda.is_available() == True`. The code
> auto-detects this and needs no changes.

```bash
# Phase 1 verification (works with any python incl. the hermes venv CPU build)
python scripts/smoke_test.py

# Train a single model on the GPU (use the ROCm python)
"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe" \
    scripts/train.py --ffn tensormatics --binary-ffn --steps 5000 --out out

# Run the full 6-model matrix (C/D/E/F; A/B already ran)
"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe" scripts/run_1bit.py

# Evaluate + generate + storage/inference benchmark
"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe" \
    scripts/eval_model.py --ckpt out/TM-1bit-ffn_D-tm-1bit-ffn/best.pt --bench --gen 250
```

### train.py options

`--ffn standard|tensormatics`, `--binary-ffn`, `--binary-attn`, `--steps`,
`--batch`, `--block`, `--lr`, `--warmup`, `--eval-interval`, `--eval-iters`,
`--seed`, `--out`, `--tag`.

Tensormatics ablation flags: `--no-tm-mult` (disable Hadamard path), `--no-tm-add`
(disable additive path), `--no-tm-gate` (fix TensorConverge gate at 0.5, not learned).
These toggle structural features while keeping the param count identical, for isolating
WHY the Tensormatics FFN wins under 1-bit (see §7.2 of the report).

Note: `--binary-ffn` + `--binary-attn` together = "1-bit core". Embeddings, norms and
LM head are always FP16 by design.

---

## Status

- ✅ All 6 models trained (5000 steps each) and benchmarked
- ✅ Phase 1 (BinaryLinear + STE) verified
- ✅ Param-matched validation (hidden 314) — TM advantage is architectural, not capacity
- ✅ Extended training (10K, 20K) — gap WIDENS: 45%→57%→62%; Std stalls, TM keeps improving
- ✅ 50K run of F2 — asymptotic PPL pinned at ~4.2 (best 4.23 @ step 48K); also surfaced ~0.3-ppl same-seed ROCm nondeterminism
- ✅ Multiple seeds (3× E/F2 @ 10K) — advantage robust: F2 beats E by 51–59% (mean 56%, σ 4.3%) at every seed
- ✅ Mechanism ablation — the multiplicative (Hadamard) fusion path is the load-bearing feature (removing it: 4.96 → 10.68 ppl)
- ✅ Full results in EXPERIMENT_REPORT.md
- ⬜ (Optional) bit-serial inference kernel for real speedup
