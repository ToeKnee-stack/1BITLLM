"""Diagnostics for binary layers.

Tracks the metrics the experiment guide emphasizes:
  - +1 / -1 balance (positive_fraction)
  - flip rate between checkpoints (how fast the binary config settles)
  - mean|W_real| and std(W_real) (do latent weights keep moving?)
  - per-layer learned scale alpha
"""
from __future__ import annotations

import torch

from ..model.binary_linear import BinaryLinear


def snapshot_binary_state(model) -> dict[str, list]:
    """Capture the current binary sign state of every BinaryLinear, keyed by
    a stable per-layer label, for flip-rate computation on the next call."""
    state: dict[str, torch.Tensor] = {}
    labels: dict[str, str] = {}
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, BinaryLinear):
            # Build a readable label: block index + role (attn qkv/proj / ffn)
            state[str(layer_idx)] = m.binary_weight().clone()
            labels[str(layer_idx)] = name
            layer_idx += 1
    return state


def compute_flip_rate(prev: dict, curr: dict) -> dict[str, float]:
    """Flip rate between two binary snapshots: #changed / N per layer."""
    rates: dict[str, float] = {}
    for k in curr:
        if k in prev:
            a = prev[k]
            b = curr[k]
            n = a.numel()
            changed = (a != b).float().mean().item()
            rates[k] = changed
    return rates


def collect_binary_stats(model) -> dict:
    """Aggregate binary balance, latent-weight stats, and scale per layer."""
    out: dict[str, dict] = {}
    layer_idx = 0
    for name, m in model.named_modules():
        if isinstance(m, BinaryLinear):
            stats = m.binary_stats()
            out[f"bin_{layer_idx}"] = stats
            out[f"bin_{layer_idx}"]["name"] = name
            layer_idx += 1
    return out


def model_summary(model, binary_layer_count: int) -> str:
    total = sum(p.numel() for p in model.parameters())
    return f"params={total:,} binary_layers={binary_layer_count}"
