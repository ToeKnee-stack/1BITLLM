# 1-Bit Tensormatics LLM — TinyShakespeare Implementation & Benchmark Plan

## 1. Experiment Objective

Build a small autoregressive Transformer language model where the primary trainable matrix weights use:

[
W_b \in {-1,+1}
]

while activations remain floating-point.

The first experiment should **not** attempt to make the entire network binary.

We want to isolate the effect of **1-bit Tensormatics weights**.

### Primary research question

> How much language-modeling capability can a Tensormatics Transformer retain when its learned projection matrices are restricted to one bit per weight?

### Secondary questions

1. Does Tensormatics tolerate binary weights better than a conventional FFN?
2. Does the Tensormatics latent transformation compensate for reduced weight precision?
3. Does training remain stable?
4. How does parameter efficiency change?
5. Does 1-bit inference provide meaningful memory/compute advantages?
6. Does the learned latent representation exhibit qualitatively different behavior?

---

# 2. Experimental Architecture

Start from the **existing TinyShakespeare TensorGPT baseline** rather than inventing a new model.

Recommended starting configuration:

```text
Dataset          TinyShakespeare
Context          128
Embedding        252
Layers           6
Heads            6
FFN              756
Parameters       ~10M
Batch            64
Precision        FP16/BF16
```

The important thing is:

> **Keep the architecture identical to the strongest existing baseline wherever possible.**

Only introduce the 1-bit constraint.

That gives us a scientifically useful A/B comparison.

---

# 3. Model Architecture

Conceptually:

```text
Token
  │
  ▼
Embedding
  │
  ▼
┌──────────────────────────────┐
│ Transformer Block             │
│                              │
│  RMSNorm                     │
│      │                       │
│      ▼                       │
│  Attention                   │
│      │                       │
│      ▼                       │
│  Residual                    │
│      │                       │
│  RMSNorm                     │
│      │                       │
│      ▼                       │
│  1-Bit Tensormatics FFN      │
│      │                       │
│      ▼                       │
│  Residual                    │
└──────────────────────────────┘
  │
  ▼
Final Norm
  │
  ▼
LM Head
  │
  ▼
Logits
```

For the **first experiment**, don't binary-quantize:

* token embeddings
* positional embeddings
* LayerNorm/RMSNorm parameters
* final LM head

Keep those FP16/BF16.

Binary-quantize the **core projection matrices**.

This isolates the experiment.

---

# 4. 1-Bit Linear Layer

This is the most important component.

Create:

```python
BinaryLinear
```

with an underlying latent/high-precision parameter:

[
W_{real}
]

and binary forward weight:

[
W_b = \operatorname{sign}(W_{real})
]

But don't simply use:

```python
Wb = torch.sign(W)
```

and expect gradients to work.

Use a **straight-through estimator**.

Conceptually:

[
W_b =
\operatorname{sign}(W_{real})
]

Forward:

```text
W_real
   ↓
sign()
   ↓
{-1,+1}
   ↓
matmul
```

Backward:

```text
gradient
   ↓
STE
   ↓
W_real
```

A practical first implementation is:

[
W_b = W_{real} + \operatorname{detach}
(\operatorname{sign}(W_{real})-W_{real})
]

so forward sees binary weights while gradients reach `W_real`.

---

# 5. Weight Scaling

Pure ±1 weights have fixed magnitude.

That can make optimization unnecessarily difficult.

So introduce a per-output or per-layer scale:

[
W_{effective} = \alpha \cdot W_b
]

where:

[
W_b \in {-1,+1}
]

and (\alpha) is learned or calculated from the latent weight.

Start with:

[
\alpha =
\frac{1}{N}\sum |W_{real}|
]

or a learned scalar.

I'd test **learned per-output-channel scaling** later.

But first experiment:

```text
Binary weight
     ×
Layer scale
     ↓
Linear transformation
```

This gives the binary matrix some ability to represent magnitude.

---

# 6. Binary Tensormatics FFN

This is where things get spicy. 😈

Use the existing Tensormatics FFN structure as much as possible.

For example:

[
X
\rightarrow
BLinear_1
\rightarrow
GELU
\rightarrow
Tensormatic\ operations
\rightarrow
BLinear_2
\rightarrow
Output
]

If your current Tensormatics FFN uses the multi-branch structure:

```text
        Binary Expansion
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
      P1      P2      P3
       │       │       │
       └───┬───┴───┬───┘
           │       │
       Tensormatic
       transformation
           │
           ▼
       Binary Projection
           │
           ▼
         Output
```

keep that structure.

**Don't simplify Tensormatics just because the weights are binary.**

The experiment is supposed to test:

> **Tensormatics + extreme parameter discretization**

not:

> some unrelated binary MLP.

---

# 7. What Should Be Binary?

Run the experiments progressively.

### Experiment A — Binary FFN

Only:

```text
Tensormatics FFN weights → 1-bit
```

Attention stays FP16.

This is the **first experiment I would run**.

---

### Experiment B — Binary FFN + Attention

Binary:

```text
Q
K
V
O
FFN
```

Everything else FP16.

This tests whether the architecture survives broader discretization.

---

### Experiment C — Binary Core Transformer

Binary:

```text
Attention projections
FFN projections
```

FP16:

```text
Embedding
Norm
LM head
```

This is probably the main benchmark model.

---

### Experiment D — Fully aggressive 1-bit

Eventually investigate:

```text
Embedding             → binary
Q/K/V/O               → binary
Tensormatics FFN      → binary
LM head               → binary
```

But **do not start here**.

If it fails, we won't know why.

---

# 8. Baselines

We need several baselines.

## Baseline 1 — Standard Transformer

Your existing NanoGPT/TensorGPT implementation.

```text
FP16
standard FFN
```

---

## Baseline 2 — FP16 Tensormatics

Exactly the same architecture as the 1-bit model except:

```text
FP16 weights
```

This is critical.

---

## Model 3 — 1-Bit Standard FFN

This is the control experiment.

```text
Transformer
+
BinaryLinear FFN
```

---

## Model 4 — 1-Bit Tensormatics

The actual experiment.

```text
Transformer
+
Binary Tensormatics FFN
```

Then we'll know whether Tensormatics provides an advantage **specifically under quantization pressure**.

---

# 9. Training Strategy

Don't immediately train for 500k steps.

Start with:

```text
Steps:       5,000
Batch:       64
Context:     128
LR:          3e-4
Optimizer:   AdamW
Warmup:      ~200 steps
```

The first goal is:

> **Does it learn at all?**

We want to see the loss curve.

Expected qualitative behavior:

```text
Loss
│\
│ \
│  \
│   \____
│
└────────── Steps
```

If the loss doesn't meaningfully decrease, stop and debug.

---

# 10. Critical Training Diagnostics

Log more than just loss.

Every evaluation interval record:

```text
step
train_loss
val_loss
train_ppl
val_ppl
token_accuracy
learning_rate
```

For the binary layers additionally record:

```text
binary_weight_mean
binary_weight_std
positive_fraction
negative_fraction
scale
```

Most importantly:

[
P(W_b=+1)
]

If you see:

```text
+1 = 50%
-1 = 50%
```

that's healthy.

If you get:

```text
+1 = 99%
-1 = 1%
```

something has collapsed.

---

# 11. Track the Underlying Real Weights

This is particularly important.

Remember:

```text
W_real
   ↓
sign()
   ↓
W_binary
```

Track:

[
\operatorname{mean}|W_{real}|
]

and:

[
\operatorname{std}(W_{real})
]

because the real weights could theoretically continue moving without changing the binary representation.

For example:

```text
Step 1000

W_real:
  mean = 0.02
  std  = 0.14

W_binary:
  +1 = 49.8%
  -1 = 50.2%
```

That's useful.

---

# 12. The Most Interesting Diagnostic: Flip Rate

Bro, **I really want this metric.**

Calculate how frequently the binary weights change between checkpoints.

[
FlipRate_t =
\frac{
#(W_b^{t}\neq W_b^{t-1})
}{
N
}
]

So:

```text
Step 1000 → 1100
Binary flip rate = 0.82%
```

Then:

```text
Step 1100 → 1200
Binary flip rate = 0.37%
```

Eventually:

```text
0.05%
```

This tells us whether the model is settling into a discrete configuration.

---

# 13. Compare Training Dynamics

Plot:

### Loss

```text
FP16 Transformer
FP16 Tensormatics
1-bit Transformer
1-bit Tensormatics
```

### Validation PPL

Same four curves.

### Binary flip rate

```text
Layer 1
Layer 2
...
Layer 6
```

### Binary balance

```text
+1 / -1 distribution
```

### Scale evolution

```text
α per layer
```

---

# 14. Parameter / Memory Benchmark

Calculate:

### FP16

[
Memory \approx N\times2
]

bytes.

### 1-bit

[
Memory \approx N/8
]

bytes.

Then include scaling parameters and non-binary parameters separately.

Report:

```text
Total parameters
Binary parameters
Non-binary parameters
FP16 storage
1-bit storage
Compression ratio
```

Don't claim the whole model is 1-bit if embeddings/norms/head remain FP16.

Instead report:

> **Effective weight precision**

---

# 15. Inference Benchmark

After training, benchmark:

```text
Model
Parameters
Weight storage
Peak GPU memory
Tokens/sec
Latency/token
```

Run identical generation workloads.

For example:

```text
batch = 1
context = 128
generate = 512 tokens
```

Then:

```text
FP16 Tensormatics
vs
1-bit Tensormatics
```

### Important

Initially, **don't expect a huge speedup** from simply storing weights as bits.

PyTorch's ordinary matrix multiplication isn't automatically transformed into an optimal bit-serial kernel.

So separate:

### Storage experiment

Can we reduce model storage?

from:

### Hardware acceleration experiment

Can we actually accelerate inference?

Those are different questions.

---

# 16. Generation Quality

Generate fixed prompts from TinyShakespeare.

For example:

```text
ROMEO:
```

```text
KING:
```

```text
To be, or not to be
```

Use the **same random seed** and sampling parameters.

Compare:

```text
FP16 Transformer
FP16 Tensormatics
1-bit Standard
1-bit Tensormatics
```

Don't judge these samples scientifically by themselves, but they are great for spotting catastrophic failure.

---

# 17. Experiment Matrix

Here's the benchmark matrix I'd actually run:

| ID | Architecture             | Weight     |
| -- | ------------------------ | ---------- |
| A  | Standard Transformer     | FP16       |
| B  | Tensormatics Transformer | FP16       |
| C  | Standard Transformer     | 1-bit FFN  |
| D  | Tensormatics Transformer | 1-bit FFN  |
| E  | Standard Transformer     | 1-bit core |
| F  | Tensormatics Transformer | 1-bit core |

Where:

**1-bit FFN**

```text
FFN matrices only
```

**1-bit core**

```text
Q/K/V/O
+
FFN
```

Everything else stays FP16.

---

# 18. Success Criteria

Don't define success as:

> "It must beat FP16."

That's probably unrealistic for the first implementation.

Instead define three levels.

### Level 1 — Feasibility

1-bit Tensormatics:

* trains
* validation loss decreases
* generates coherent text

**Success.**

---

### Level 2 — Competitive

If:

[
PPL_{1bit} \approx PPL_{FP16}
]

within a reasonable margin, that's extremely interesting.

Especially if memory falls dramatically.

---

### Level 3 — Breakthrough territory 😈

If:

[
PPL_{1bit\ Tensormatics}
<
PPL_{1bit\ Standard}
]

while both use identical parameter counts and training budgets...

**THAT is the result I would get excited about.**

Because then we're no longer saying:

> "Tensormatics survives quantization."

We're potentially showing:

> **Tensormatics provides an architectural advantage under severe parameter discretization.**

That's a much stronger hypothesis.

---

# 19. Recommended Implementation Order

Don't implement everything at once.

### Phase 1

Implement:

```text
BinaryLinear
```

and verify:

```text
forward values ∈ {-1,+1}
gradient exists
optimizer updates W_real
```

---

### Phase 2

Replace only the Tensormatics FFN:

```text
FP16 Attention
+
1-bit Tensormatics FFN
```

Run 5k steps.

---

### Phase 3

Add diagnostics:

```text
+1 ratio
flip rate
scale
W_real statistics
```

---

### Phase 4

Run:

```text
FP16 Tensormatics
vs
1-bit Tensormatics
```

same seed, same training.

---

### Phase 5

Binary attention:

```text
Q
K
V
O
```

Run another 5k experiment.

---

### Phase 6

Scale to your longer TinyShakespeare run:

```text
50k
100k
200k+
```

depending on what the short experiments show.

---

# 20. One Important Rule

**Do not modify the Tensormatics mathematics and binary quantization simultaneously.**

Keep:

```text
Tensormatics architecture
        +
existing activation
        +
existing stabilization
```

fixed.

Change only:

```text
FP weight
   ↓
1-bit weight
```

Otherwise if performance changes, we won't know whether the cause was:

* binary weights
* changed FFN architecture
* changed normalization
* changed activation
* changed initialization
* changed optimization

We want this experiment to be **surgical**.

---

# 21. Final Experimental Question

At the end, we want a table roughly like:

| Model                   |   Params |     Precision | Val PPL | Storage | Tok/s |
| ----------------------- | -------: | ------------: | ------: | ------: | ----: |
| Transformer             |     9.7M |          FP16 |       ? |       ? |     ? |
| Tensormatics            |     9.7M |          FP16 |       ? |       ? |     ? |
| Binary Transformer      |     9.7M |     1-bit FFN |       ? |       ? |     ? |
| **Binary Tensormatics** | **9.7M** | **1-bit FFN** |   **?** |   **?** | **?** |

And then the killer comparison:

[
\boxed{
\text{PPL per bit}
}
]

and potentially:

[
\boxed{
\text{Language capability per byte}
}
]


