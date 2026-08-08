"""Generate text from a trained checkpoint and report storage/memory benchmark.

Usage:
  python scripts/eval_model.py --ckpt out/Std-fp16/best.pt [--gen 200] [--bench]

For binary checkpoints, prints a storage comparison (fp16 vs 1-bit packed)
and an effective-precision breakdown per guide §14/§15.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from onebitllm.data import CharDataset
from onebitllm.model import GPT, GPTConfig


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="data/tiny_shakespeare.txt")
    p.add_argument("--prompt", default="ROMEO:\n")
    p.add_argument("--gen", type=int, default=200)
    p.add_argument("--bench", action="store_true")
    p.add_argument("--device", default="auto")
    return p.parse_args()


def storage_report(model, cfg):
    """Count binary vs non-binary params; report fp16 vs 1-bit storage."""
    total_params = sum(p.numel() for p in model.parameters())
    bin_params = 0
    nonbin_params = 0
    for name, p in model.named_parameters():
        # Binary storage lives in W_real of BinaryLinear (1 bit/weight).
        if "W_real" in name or ".alpha" in name:
            bin_params += p.numel()
        else:
            nonbin_params += p.numel()

    fp16_bytes = total_params * 2
    # Binary weights stored packed at 1 bit each; alpha + rest at fp16.
    bin_storage_bits = bin_params  # 1 bit per binary weight
    nonbin_bytes = nonbin_params * 2
    packed_bytes = nonbin_bytes + bin_storage_bits / 8
    comp = fp16_bytes / packed_bytes if packed_bytes > 0 else 0

    print("\n=== Storage benchmark ===")
    print(f"total_params      : {total_params:,}")
    print(f"  binary params   : {bin_params:,}")
    print(f"  non-binary params: {nonbin_params:,}")
    print(f"fp16 storage      : {fp16_bytes/1e6:.2f} MB ({fp16_bytes:,.0f} B)")
    print(f"1-bit packed      : {packed_bytes/1e6:.3f} MB ({packed_bytes:,.0f} B)")
    print(f"compression ratio : {comp:.1f}x")
    print(f"effective precision: {bin_params/total_params*100:.1f}% of weights are 1-bit")


def main():
    args = parse_args()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = GPTConfig(**{k: v for k, v in ckpt["cfg"].items() if k in GPTConfig.__dataclass_fields__})
    ds = CharDataset(args.data, cfg.block_size, device=device)
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"loaded {args.ckpt} (step {ckpt['step']}, val_loss {ckpt['val_loss']:.4f}, val_ppl {ckpt['val_ppl']:.2f})")
    print(f"arch: {cfg.name} ffn={cfg.ffn_type}")

    storage_report(model, cfg)

    if args.bench:
        print("\n=== Inference benchmark (batch=1, ctx=128, gen=256) ===")
        model.eval()
        # warmup
        idx = ds.encode(args.prompt).unsqueeze(0).to(device)
        with torch.no_grad():
            model.generate(idx, 8)
        torch.cuda.synchronize() if device == "cuda" else None

        idx = ds.encode(args.prompt).unsqueeze(0).to(device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(idx, 256)
        torch.cuda.synchronize() if device == "cuda" else None
        dt = time.time() - t0
        n_tok = 256
        print(f"generated {n_tok} tokens in {dt:.2f}s")
        print(f"tokens/sec      : {n_tok/dt:.1f}")
        print(f"ms/token        : {dt/n_tok*1000:.1f}")
        if device == "cuda":
            print(f"peak GPU mem    : {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # Always show a sample.
    idx = ds.encode(args.prompt).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(idx, args.gen, temperature=0.8)
    text = ds.decode(out[0].tolist())
    print(f"\n=== Sample ({args.prompt!r}) ===")
    print(text)


if __name__ == "__main__":
    main()
