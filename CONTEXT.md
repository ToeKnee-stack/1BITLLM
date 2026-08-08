# CONTEXT — Session State & How to Continue

> **Purpose:** This file lets us resume the 1-Bit Tensormatics LLM project tomorrow
> without re-deriving anything. It captures the goal, environment, exact commands,
> results, decisions made, and the precise next steps. Read this first, then
> EXPERIMENT_REPORT.md for the full analysis.

---

## 1. TL;DR of the project

Build a small Transformer on TinyShakespeare whose core weights are forced to 1-bit
{−1,+1} (straight-through estimator + learned scale), and test whether a **Tensormatics**
FFN survives extreme discretization better than a **standard** FFN.

**It does.** Under 1-bit weights, Tensormatics beats Standard by 6.2% (1-bit FFN) and
41.8% (1-bit core) on validation PPL. All 6 models are trained and benchmarked. The
code is complete and reproducible.

---

## 2. Environment — READ THIS BEFORE RUNNING ANYTHING

- **Working directory:** `E:\AI-Workspace\1BitLLM`
- **GPU:** AMD **RX 7900 XTX** (24 GB). This is an **AMD** GPU.
  - ⚠️ **CUDA does NOT work on it.** CUDA is NVIDIA-only.
  - ✅ Use the **native Windows ROCm PyTorch** that is already installed.
- **ROCm Python (USE THIS for training/eval):**
  `C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe`
  - torch version: `2.9.1+rocmsdk20260116`, HIP 7.2
  - It reports the GPU via `torch.cuda` (ROCm HIP shim):
    `torch.cuda.is_available() == True`, name = "AMD Radeon RX 7900 XTX"
- **Hermes venv python (default `python` on PATH):** torch `2.12.1+cpu`, **CPU-only**.
  - Fine for `smoke_test.py`. NOT for training.
  - ⚠️ Earlier I briefly installed `torch-directml` here, then **uninstalled it and
    restored torch to `2.12.1+cpu`**. It's clean now. Don't reinstall directml; the
    ROCm python is the correct path.
- **Other Python installs found:** `Python310` has torch `2.7.0+cu118` (CUDA build —
  useless here since GPU is AMD). `Python312` (ROCm) is the one that matters.
- **WSL2** exists (Ubuntu) but is **not needed** — the native ROCm Windows build works.
- **ROCm drivers:** `C:\Program Files\AMD\ROCm\5.7` present.

### How the GPU python is invoked
```bash
PY=/c/Users/antho/AppData/Local/Programs/Python/Python312/python.exe
$PY scripts/train.py --ffn tensormatics --binary-ffn --steps 5000 --out out
```

---

## 3. Project structure (current)

```
E:\AI-Workspace\1BitLLM\
├── ImplementationGuide.md      # original plan (source of truth for intent)
├── EXPERIMENT_REPORT.md        # full results & analysis (done yesterday)
├── README.md                   # quick overview + how to run
├── CONTEXT.md                  # THIS FILE — session state
├── data\tiny_shakespeare.txt   # dataset, 1.1 MB, 65-char vocab
├── src\onebitllm\
│   ├── __init__.py
│   ├── data\dataset.py         # CharDataset: encode/decode/get_batch
│   ├── model\
│   │   ├── binary_linear.py    # BinaryLinear (STE + learned per-channel α)
│   │   ├── ffn.py              # StandardFFN, TensormaticsFFN, TensorConverge
│   │   └── model.py            # GPT, GPTConfig, Block, RMSNorm, CausalSelfAttention
│   └── training\diagnostics.py # snapshot_binary_state, compute_flip_rate, collect_binary_stats
├── scripts\
│   ├── smoke_test.py           # Phase 1 verification
│   ├── train.py                # training + JSONL logging + checkpointing
│   ├── eval_model.py           # storage/inference benchmark + generation
│   ├── run_1bit.py             # runs C/D/E/F sequentially
│   ├── run_eval.py             # evals all 6 best checkpoints → out/eval_all.txt
│   └── run_matrix.sh           # shell runner for the full matrix
├── checkpoints\                # EMPTY — unused (checkpoints actually go in out/)
└── out\
    ├── <Model>_log.jsonl        # full per-model diagnostics
    ├── run_*.log                # console captures
    ├── eval_all.txt             # all 6 eval + sample outputs
    └── <Model>\best.pt          # best-checkpoint weights + config
```

**Note:** `src` package is `onebitllm` (not `1bitllm`) because `1bitllm` is not a valid
Python identifier (starts with a digit). Scripts insert `src/` onto `sys.path`.

---

## 4. Results (measured, from `out/*_log.jsonl`)

All models: TinyShakespeare, ctx 128, emb 252, 6 layers, 6 heads, FFN hidden 756,
batch 64, lr 3e-4 (AdamW, 200-step warmup, clip 1.0), **5000 steps**, seed 1337, GELU.

| ID | Model | Params | Binary params | Val Loss | Val PPL | Val Acc | Storage (packed) | Ratio | Tok/s |
|----|-------|-------:|--------------:|---------:|--------:|--------:|-----------------:|------:|------:|
| A | Std-FP16 | 3.88M | 0 | 0.932 | **2.54** | 70.2% | 7.77 MB | 1.0× | 180.5 |
| B | TM-FP16 | 6.56M | 0 | 0.916 | **2.50** | 70.9% | 13.12 MB | 1.0× | 107.8 |
| C | Std-1bit-FFN | 3.89M | 2.29M | 1.398 | 4.05 | 56.5% | 3.48 MB | 2.2× | 142.8 |
| D | TM-1bit-FFN | 6.58M | 4.97M | 1.334 | **3.80** | 57.9% | 3.84 MB | 3.4× | 76.3 |
| E | Std-1bit-core | 3.90M | 3.82M | 2.496 | 12.14 | 27.3% | 0.63 MB | 12.4× | 120.8 |
| F | TM-1bit-core | 6.59M | 6.50M | 1.955 | **7.06** | 41.3% | 0.98 MB | 13.4× | 69.2 |

### Headline finding — the advantage widens with quantization depth
```
FP16:        TM 2.50 vs Std 2.54  →  TM better by  1.6%
1-bit FFN:   TM 3.80 vs Std 4.05  →  TM better by  6.2%
1-bit core:  TM 7.06 vs Std 12.14 →  TM better by 41.8%
```

### Diagnostics summary (all healthy)
- **+1 fraction:** ~50% in every binary layer of every model. **No sign collapse.**
- **Flip rate:** D averages ~3.6% (settling slowly); F averages ~2.7% with several
  layers dropping to **0.18–0.5%** — clearly settling toward discrete configs.
- **Latent weights:** mean|W_real| ~0.043–0.047, std ~0.055–0.06 (still moving).
- **Learned α:** converges near 1.0 with small spread (per-channel adjustments).

---

## 5. Decisions made & rationale (recorded for continuity)

1. **Activation = GELU** in the FFN pathway, not TenSu. (Per user-confirmed T-RAE
   convention: GELU for non-linear encoders, TenSu only at fusion/vertex.)
2. **Tensormatics FFN structure:** 3 branches → Hadamard mult-path + additive-path →
   `TensorConverge` (learned gate `σ(α)·T_mult + (1−σ(α))·T_add`) → output proj.
3. **Learned per-output-channel α** (not single scalar) for weight scaling.
4. **Never binary-ize** embeddings, RMSNorm, or LM head — isolates the experiment.
5. **Param-count caveat:** TM models have ~1.7× the params of Std (wider 3-branch FFN
   at same hidden width). This is the main confound. See §8 of the report.
6. **GPU path:** native Windows ROCm at Python312, not WSL, not CUDA, not directml.
7. **`checkpoints/` dir unused** — checkpoints are saved to `out/<tag>/best.pt`.

---

## 6. How to reproduce / rerun

```bash
cd /e/AI-Workspace/1BitLLM
PY=/c/Users/antho/AppData/Local/Programs/Python/Python312/python.exe

# smoke test (any python)
python scripts/smoke_test.py

# single model (GPU)
$PY scripts/train.py --ffn tensormatics --binary-ffn --steps 5000 --out out --tag D-tm-1bit-ffn

# all 1-bit models C/D/E/F sequentially
$PY scripts/run_1bit.py

# eval + benchmark + generate all 6
$PY scripts/run_eval.py
# → writes out/eval_all.txt
```

**train.py flag semantics:** `--binary-ffn` alone = "1-bit FFN". `--binary-ffn
--binary-attn` together = "1-bit core". The output tag is `cfg.name` (e.g.
`TM-1bit-core`) + optional `--tag`.

---

## 7. Next steps (priority order) — pick up here tomorrow

### 7.1 [HIGH] Param-matched comparison (removes the confound)
The biggest gap in the current result: TM models have 1.7× more params. Options:
- **Narrow the TM FFN** so its total params ≈ Std (e.g. reduce branches to 2, or
  reduce hidden width so TM ≈ 3.9M).
- **Or widen the Std FFN** so both ≈ 6.5M.
Then rerun C/D (1-bit FFN) and E/F (1-bit core) to confirm the 6%/42% advantage holds
at equal parameter counts. This is the scientific crux.

### 7.2 [HIGH] Longer runs (50k–200k steps)
FP16 curves were still descending at 5000. Longer runs establish asymptotic PPL and
whether the TM advantage persists/narrows at convergence. ~64 ms/step on GPU → 50k ≈
54 min, 200k ≈ 3.5 hr per model.

### 7.3 [MED] Multiple seeds
All runs used seed 1337. Re-run key models (D, F) with 2–3 seeds to quantify variance
in the ppl gap and flip-rate settling.

### 7.4 [MED] Isolate the mechanism: multiplicative fusion vs learned α
E (Std 1-bit core) stalls. Test whether adding a learned per-channel α to the
*standard* FFN's binary layers closes part of the gap — this isolates whether the
advantage is the multiplicative interaction or just the presence of learned scales.

### 7.5 [LOW] Bit-serial inference kernel
Storage is compressed 13× but speed is not (matmul isn't bit-serial). A real speedup
needs a custom HIP/bit-serial kernel — separate from the storage experiment.

---

## 8. Known issues / gotchas

- **Don't run training with the hermes venv python** (`python` → torch 2.12.1+cpu) —
  it's CPU-only and slow. Use the ROCm Python312.
- **`out/` contains run artifacts**; `checkpoints/` is empty and unused.
- **Don't reinstall torch-directml** into the hermes venv (already reverted once).
- **Single seed** — all conclusions are from seed 1337.
- **Param confound** — see §7.1; don't over-claim the 41.8% until param-matched.
- **Inference speed** — 1-bit does not speed up generation; only storage shrinks.

---

## 9. Files that matter

| File | Why |
|------|-----|
| `EXPERIMENT_REPORT.md` | Full write-up: design, results, analysis, limitations, next steps |
| `README.md` | Quick overview + run instructions |
| `ImplementationGuide.md` | The original plan — source of truth for experimental intent |
| `src/onebitllm/model/binary_linear.py` | The core `BinaryLinear` (STE + α) |
| `src/onebitllm/model/ffn.py` | StandardFFN + TensormaticsFFN + TensorConverge |
| `src/onebitllm/model/model.py` | GPT, config, blocks |
| `out/<Model>_log.jsonl` | Raw per-model metrics + binary diagnostics |
| `out/eval_all.txt` | Generation samples for all 6 models |
| `out/<Model>/best.pt` | Best checkpoint weights + config |

---

*Context captured: 2026-08-08. Project directory: `E:\AI-Workspace\1BitLLM`.*
