"""1-bit linear layer with straight-through estimator (STE).

Core experiment component. Holds a latent fp weight `W_real`, derives a
binary `W_b = sign(W_real)` used in the forward pass, and scales by a
learned per-output-channel scalar alpha. Gradients pass through to `W_real`
via the identity straight-through estimator.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def binary_forward(W_real: torch.Tensor) -> torch.Tensor:
    """Return W_b in {-1,+1} with STE gradient flowing to W_real.

    Forward:  W_b = sign(W_real)
    Backward: grad flows to W_real as identity (STE) so W_real can move.
    """
    W_b = torch.sign(W_real)
    # For exact zero, sign gives 0; nudge to +1 to keep values in {-1,+1}.
    W_b = torch.where(W_b == 0, torch.ones_like(W_b), W_b)
    # Straight-through estimator: forward sees binary, backward sees identity.
    return W_real + (W_b - W_real).detach()


class BinaryLinear(nn.Module):
    """Linear layer whose weight is binary {+1,-1} with learned per-channel scale.

    Effective forward: y = x @ (alpha * W_b).T + bias
    - W_real: latent fp weights (learned, trained via STE).
    - alpha:  learned per-output-channel scaling, length out_features.
    - bias:   optional fp bias (kept fp — not part of the experiment).
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        scale_init: float = 1.0,
        init_std: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        # Latent real-valued weight.
        self.W_real = nn.Parameter(
            torch.randn(in_features, out_features) * init_std
        )
        # Learned per-output-channel scale.
        self.alpha = nn.Parameter(torch.full((out_features,), float(scale_init)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.W_real, std=0.05)
        # Scale such that initial |W_eff| ~ sqrt(fan_in) like a normal linear
        # would have, but keep it learnable.
        with torch.no_grad():
            self.alpha.fill_(1.0)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def binary_weight(self) -> torch.Tensor:
        """Return the binary weight used in the forward pass (detached values)."""
        W_b = torch.sign(self.W_real)
        return torch.where(W_b == 0, torch.ones_like(W_b), W_b)

    def effective_weight(self) -> torch.Tensor:
        """alpha * W_b — the weight the forward pass actually applies."""
        return self.alpha * self.binary_weight()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W_b = binary_forward(self.W_real)  # (in, out) in {-1,+1}, STE grad
        W_eff = self.alpha * W_b            # (out,) * (in, out)
        y = F.linear(x, W_eff.t(), self.bias)
        return y

    def binary_stats(self) -> dict:
        """Diagnostics for the binary weight and latent weight."""
        with torch.no_grad():
            W_b = self.binary_weight().float()
            W_real = self.W_real.detach().float()
            n = W_b.numel()
            pos = (W_b > 0).float().mean().item()
            return {
                "binary_mean": W_b.mean().item(),
                "binary_std": W_b.std().item(),
                "positive_fraction": pos,
                "negative_fraction": 1.0 - pos,
                "real_mean_abs": W_real.abs().mean().item(),
                "real_std": W_real.std().item(),
                "real_mean": W_real.mean().item(),
                "alpha_mean": self.alpha.mean().item(),
                "alpha_std": self.alpha.std().item(),
                "n_binary": n,
            }

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
