"""Feed-forward networks for the experiment.

Two FFN variants:
  - StandardFFN:  linear -> GELU -> linear  (the control architecture)
  - TensormaticsFFN: multi-branch projection -> multiplicative fusion ->
    vertex stabilization (TensorConverge) -> output projection.

Both support optional binary projections (Q/K/V/O and FFN matrices) to test
the effect of 1-bit weights. Keeps the Tensormatics mathematics identical
when toggling precision (per guide §20, don't change math and precision
simultaneously).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_linear import BinaryLinear, binary_forward


def make_linear(
    in_f: int,
    out_f: int,
    binary: bool,
    bias: bool = True,
) -> nn.Module:
    """Return a BinaryLinear if binary else a normal nn.Linear."""
    if binary:
        return BinaryLinear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


class StandardFFN(nn.Module):
    """Conventional MLP: proj_up -> GELU -> proj_down. Optionally binary."""

    def __init__(self, dim: int, hidden: int, binary: bool = False) -> None:
        super().__init__()
        self.w1 = make_linear(dim, hidden, binary=binary)
        self.w2 = make_linear(hidden, dim, binary=binary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.gelu(self.w1(x)))

    def binary_layers(self) -> list:
        return [l for l in (self.w1, self.w2) if isinstance(l, BinaryLinear)]


class TensorConverge(nn.Module):
    """Vertex fusion: multiplicative path vs additive path, learnable gate.

    Per Tensormatics: V = sigmoid(alpha) * T_mult + (1 - sigmoid(alpha)) * T_add
    alpha is a single learnable scalar (diagnostic of structure type).
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(1))  # init -> 0.5 gate

    def forward(self, t_mult: torch.Tensor, t_add: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.alpha)
        return g * t_mult + (1 - g) * t_add


class TensormaticsFFN(nn.Module):
    """Tensormatics FFN with multi-branch multiplicative fusion.

    X
      -> Binary Expansion: split into P branches (via projections)
      -> per-branch latent (GELU)
      -> Tensormatic fusion: multiplicative path (Hadamard) + additive path
      -> TensorConverge vertex stabilization (learnable gate)
      -> output projection

    If binary=True, the branch and output projections are BinaryLinear.
    """

    def __init__(
        self,
        dim: int,
        hidden: int,
        n_branches: int = 3,
        binary: bool = False,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.hidden = hidden
        self.n_branches = n_branches
        self.binary = binary

        # Branch projections: dim -> hidden for each branch.
        self.branch_projs = nn.ModuleList(
            [
                make_linear(dim, hidden, binary=binary)
                for _ in range(n_branches)
            ]
        )
        # Fusion projection: hidden -> dim (used to build the additive path
        # and to give multiplicative path a compatible shape).
        self.fusion = make_linear(hidden, dim, binary=binary)
        self.gelu = nn.GELU()

        # TensorConverge vertex stabilization.
        self.converge = TensorConverge(dim)

        # Output projection: dim -> dim.
        self.out_proj = make_linear(dim, dim, binary=binary)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branches = [self.gelu(p(x)) for p in self.branch_projs]

        # Additive path: sum of branch activations projected to dim.
        add_path = self.fusion(torch.stack(branches, dim=0).sum(dim=0))

        # Multiplicative path: Hadamard product across branches.
        mult = branches[0]
        for b in branches[1:]:
            mult = mult * b
        mult_path = self.fusion(mult)

        # Vertex fusion via TensorConverge.
        fused = self.converge(mult_path, add_path)

        # Output projection.
        return self.out_proj(fused)

    def binary_layers(self) -> list:
        layers = list(self.branch_projs) + [self.fusion, self.out_proj]
        return [l for l in layers if isinstance(l, BinaryLinear)]
