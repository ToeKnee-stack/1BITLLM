"""Config-driven GPT-style transformer for the 1-bit experiment.

Architecture (config-driven):
  - token embedding (always fp)
  - N transformer blocks:
      RMSNorm -> attention -> residual
      RMSNorm -> FFN (Standard or Tensormatics) -> residual
  - final norm -> LM head (always fp)

`binary_core` quantizes attention Q/K/V/O + FFN projections (Experiment C/D).
`binary_ffn` quantizes only FFN projections (Experiment A).
Only the attention and FFN projections are ever binary; embedding, norms and
the LM head always stay fp (per guide §7).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from .binary_linear import BinaryLinear
from .ffn import StandardFFN, TensormaticsFFN


@dataclass
class GPTConfig:
    vocab_size: int = 65          # TinyShakespeare char vocab
    block_size: int = 128         # context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 252             # embedding dim
    ffn_hidden: int = 756         # FFN hidden dim
    ffn_type: str = "standard"    # "standard" | "tensormatics"
    n_branches: int = 3           # Tensormatics branches
    tm_use_mult: bool = True      # Tensormatics: use multiplicative (Hadamard) path
    tm_use_add: bool = True       # Tensormatics: use additive (sum) path
    tm_learn_gate: bool = True    # Tensormatics: learn the TensorConverge gate (else fixed 0.5)
    binary_ffn: bool = False      # quantize FFN projections to 1-bit
    binary_attn: bool = False     # quantize Q/K/V/O to 1-bit
    bias: bool = False            # standard GPT-2 convention
    dropout: float = 0.0

    @property
    def binary_core(self) -> bool:
        return self.binary_ffn and self.binary_attn

    @property
    def name(self) -> str:
        parts = []
        parts.append("TM" if self.ffn_type == "tensormatics" else "Std")
        if self.binary_ffn and self.binary_attn:
            parts.append("1bit-core")
        elif self.binary_ffn:
            parts.append("1bit-ffn")
        else:
            parts.append("fp16")
        return "-".join(parts)


class LayerNorm(nn.Module):
    """LayerNorm, but with fp weights (never binary)."""

    def __init__(self, dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (self.weight.shape[0],), self.weight, self.bias, 1e-5)


class RMSNorm(nn.Module):
    """RMSNorm (fp parameters, never binary)."""

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return x * rms * self.weight


def build_linear(in_f, out_f, binary: bool, bias: bool):
    if binary:
        return BinaryLinear(in_f, out_f, bias=bias)
    return nn.Linear(in_f, out_f, bias=bias)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.binary = cfg.binary_attn

        self.c_attn = build_linear(
            cfg.n_embd, 3 * cfg.n_embd, binary=cfg.binary_attn, bias=cfg.bias
        )
        self.c_proj = build_linear(
            cfg.n_embd, cfg.n_embd, binary=cfg.binary_attn, bias=cfg.bias
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).view(
            1, 1, cfg.block_size, cfg.block_size
        )
        self.register_buffer("mask", mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = qkv.split(self.n_head * self.head_dim, dim=2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B,H,T,hd)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim**0.5))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        y = att @ v  # (B,H,T,hd)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

    def binary_layers(self) -> list:
        return [l for l in (self.c_attn, self.c_proj) if isinstance(l, BinaryLinear)]


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln_1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = RMSNorm(cfg.n_embd)
        if cfg.ffn_type == "tensormatics":
            self.mlp = TensormaticsFFN(
                cfg.n_embd, cfg.ffn_hidden, cfg.n_branches, binary=cfg.binary_ffn,
                use_mult=cfg.tm_use_mult, use_add=cfg.tm_use_add,
                learn_gate=cfg.tm_learn_gate,
            )
        else:
            self.mlp = StandardFFN(cfg.n_embd, cfg.ffn_hidden, binary=cfg.binary_ffn)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.resid_dropout(self.mlp(self.ln_2(x)))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)  # always fp
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.block_size, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = RMSNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)  # always fp

        self.apply(self._init_weights)

    def _init_weights(self, module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.cfg.block_size
        tok = self.tok_emb(idx)  # (B,T,C)
        pos = self.pos_emb[:, :T, :]
        x = self.drop(tok + pos)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (B,T,vocab)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.block_size else idx[:, -self.cfg.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def binary_layers(self) -> list:
        layers = []
        for blk in self.blocks:
            layers.extend(blk.attn.binary_layers())
            layers.extend(blk.mlp.binary_layers())
        return layers

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def binary_parameter_count(self) -> int:
        n = 0
        for p in self.parameters():
            # A BinaryLinear's W_real is the binary storage (1 bit/weight).
            if isinstance(getattr(p, "requires_grad", False), bool):
                pass
        # Compute from module registry instead.
        total = 0
        for m in self.modules():
            if isinstance(m, BinaryLinear):
                total += m.W_real.numel()
        return total
