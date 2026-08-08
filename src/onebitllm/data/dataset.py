"""Character-level data loading for TinyShakespeare."""
from __future__ import annotations

import os

import torch


class CharDataset:
    """Load a text file, build a char-level vocab, produce token batches."""

    def __init__(self, text_path: str, block_size: int, device: str = "cpu"):
        with open(text_path, "r", encoding="utf-8") as f:
            text = f.read()

        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.block_size = block_size
        self.device = device

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)
        self.n_tokens = len(self.data)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[ch] for ch in s], dtype=torch.long)

    def decode(self, ids) -> str:
        if torch.is_tensor(ids):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def get_batch(self, batch_size: int) -> tuple:
        """Random (x, y) sequences of length block_size."""
        ix = torch.randint(0, self.n_tokens - self.block_size - 1, (batch_size,))
        x = torch.stack(
            [self.data[i : i + self.block_size] for i in ix]
        ).to(self.device)
        y = torch.stack(
            [self.data[i + 1 : i + 1 + self.block_size] for i in ix]
        ).to(self.device)
        return x, y
