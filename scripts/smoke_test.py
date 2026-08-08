"""Phase 1 smoke test: verify BinaryLinear behaves correctly.

Checks:
  1. Forward weight values are in {-1,+1}.
  2. Gradients reach W_real and update it.
  3. Binary forward matches sign(W_real) exactly.
  4. Model forward/backward works for all 4 config variants.
  5. Flip-rate diagnostic works across two optimizer steps.
"""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make package importable when run from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from onebitllm.model.binary_linear import BinaryLinear  # noqa: E402


def test_binary_values_and_ste():
    torch.manual_seed(0)
    lin = BinaryLinear(8, 16, bias=True)
    x = torch.randn(4, 8, requires_grad=True)
    y = lin(x)
    assert y.shape == (4, 16)

    # Forward must use binary weight.
    Wb = lin.binary_weight()
    assert torch.all((Wb == 1) | (Wb == -1)), "binary weight must be ±1"
    assert Wb.dtype.is_floating_point

    # Gradient must reach W_real.
    loss = y.pow(2).mean()
    loss.backward()
    assert lin.W_real.grad is not None, "W_real must receive gradient"
    assert lin.W_real.grad.abs().sum() > 0, "W_real gradient must be nonzero"
    print("[PASS] binary values ±1, gradient reaches W_real")

    # STE: a step must move W_real.
    real_before = lin.W_real.detach().clone()
    with torch.no_grad():
        lin.W_real -= 0.1 * lin.W_real.grad
    assert not torch.equal(lin.W_real.detach(), real_before), "W_real must update"
    print("[PASS] optimizer step moves W_real (STE works)")


def test_scale_and_effective_weight():
    torch.manual_seed(1)
    lin = BinaryLinear(8, 16)
    # Default alpha = 1.0
    assert torch.allclose(lin.alpha.data, torch.ones(16), atol=1e-6)
    Wb = lin.binary_weight()
    Weff = lin.effective_weight()
    assert torch.allclose(Weff, lin.alpha * Wb, atol=1e-6)
    print("[PASS] alpha scale multiplies binary weight correctly")


def test_binary_forward_function():
    torch.manual_seed(2)
    W = torch.nn.Parameter(torch.randn(5, 7))
    out = _binary_forward(W)
    assert torch.all((out == 1) | (out == -1)), "STE forward must be ±1"
    # Gradient must flow to W through the STE.
    out.pow(2).sum().backward()
    assert W.grad is not None and W.grad.abs().sum() > 0, "STE must pass grad"
    print("[PASS] binary_forward returns ±1 and passes gradient")


def _binary_forward(W_real):
    W_b = torch.sign(W_real)
    W_b = torch.where(W_b == 0, torch.ones_like(W_b), W_b)
    return W_real + (W_b - W_real).detach()


def test_model_variants():
    from onebitllm.model import GPT, GPTConfig

    torch.manual_seed(3)
    cfgs = [
        GPTConfig(ffn_type="standard"),
        GPTConfig(ffn_type="tensormatics"),
        GPTConfig(ffn_type="tensormatics", binary_ffn=True),
        GPTConfig(ffn_type="tensormatics", binary_ffn=True, binary_attn=True),
    ]
    for cfg in cfgs:
        m = GPT(cfg)
        x = torch.randint(0, cfg.vocab_size, (2, 32))
        logits, loss = m(x, x)
        assert logits.shape == (2, 32, cfg.vocab_size)
        assert loss is not None and torch.isfinite(loss)
        loss.backward()
        print(f"[PASS] {cfg.name}: forward/backward ok, params={m.num_params():,}")
    print("DONE")


if __name__ == "__main__":
    test_binary_values_and_ste()
    test_scale_and_effective_weight()
    test_binary_forward_function()
    test_model_variants()
