"""Train a config-driven GPT variant on TinyShakespeare with full diagnostics.

Usage:
  python scripts/train.py --ffn tensormatics --binary-ffn
  python scripts/train.py --ffn standard --binary-ffn --binary-attn

Logs train/val loss + ppl + token accuracy every eval_interval, and for binary
models also logs +1/-1 balance, flip rate, |W_real| stats, and alpha per layer.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from onebitllm.data import CharDataset
from onebitllm.model import GPT, GPTConfig
from onebitllm.training import (
    snapshot_binary_state,
    compute_flip_rate,
    collect_binary_stats,
    model_summary,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ffn", choices=["standard", "tensormatics"], default="standard")
    p.add_argument("--ffn-hidden", type=int, default=None,
                   help="Override FFN hidden width (default: 756). For param-matched TM runs.")
    p.add_argument("--binary-ffn", action="store_true")
    p.add_argument("--binary-attn", action="store_true")
    p.add_argument("--data", default="data/tiny_shakespeare.txt")
    p.add_argument("--out", default="out")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--block", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=200)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", default="auto")
    p.add_argument("--tag", default="")
    return p.parse_args()


def set_seed(seed):
    torch.manual_seed(seed)
    import random

    random.seed(seed)


@torch.no_grad()
def evaluate(model, ds, batch, iters, block_size):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    total_correct = 0
    for _ in range(iters):
        x, y = ds.get_batch(batch)
        logits, loss = model(x, y)
        total_loss += loss.item()
        preds = logits.argmax(-1)
        # token accuracy (ignore positions where y == -1, but there are none here)
        correct = (preds == y).sum().item()
        total_tokens += y.numel()
        total_correct += correct
    model.train()
    val_loss = total_loss / iters
    return val_loss, math.exp(val_loss), total_correct / total_tokens


def main():
    args = parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    ds = CharDataset(args.data, args.block, device=device)
    cfg = GPTConfig(
        vocab_size=ds.vocab_size,
        block_size=args.block,
        ffn_type=args.ffn,
        ffn_hidden=args.ffn_hidden if args.ffn_hidden else 756,
        binary_ffn=args.binary_ffn,
        binary_attn=args.binary_attn,
    )
    model = GPT(cfg).to(device)
    n_params = model.num_params()
    n_binary = model.binary_parameter_count()
    print(f"model={cfg.name} ffn={cfg.ffn_type} binary_ffn={cfg.binary_ffn} binary_attn={cfg.binary_attn}")
    print(model_summary(model, len(model.binary_layers())))
    print(f"  total_params={n_params:,} binary_params={n_binary:,}")

    os.makedirs(args.out, exist_ok=True)
    out_tag = cfg.name + (f"_{args.tag}" if args.tag else "")
    log_path = os.path.join(args.out, f"{out_tag}_log.jsonl")
    ckpt_dir = os.path.join(args.out, out_tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95))
    best_val = float("inf")

    log_entries = []
    prev_binary = snapshot_binary_state(model) if (args.binary_ffn or args.binary_attn) else None

    def lr_at(step):
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        return args.lr

    print(f"starting {args.steps} steps (eval every {args.eval_interval})")
    t0 = time.time()
    for step in range(1, args.steps + 1):
        for g in optimizer.param_groups:
            g["lr"] = lr_at(step)
        x, y = ds.get_batch(args.batch)
        logits, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == args.steps:
            val_loss, val_ppl, val_acc = evaluate(
                model, ds, args.batch, args.eval_iters, args.block
            )
            train_ppl = math.exp(loss.item()) if loss.item() > -700 else float("inf")
            entry = {
                "step": step,
                "train_loss": round(loss.item(), 4),
                "val_loss": round(val_loss, 4),
                "train_ppl": round(train_ppl, 2),
                "val_ppl": round(val_ppl, 2),
                "val_acc": round(val_acc, 4),
                "lr": round(lr_at(step), 6),
                "elapsed_s": round(time.time() - t0, 1),
            }

            # Binary diagnostics.
            if prev_binary is not None:
                curr_binary = snapshot_binary_state(model)
                flip = compute_flip_rate(prev_binary, curr_binary)
                stats = collect_binary_stats(model)
                entry["flip_rate"] = {k: round(v, 5) for k, v in flip.items()}
                entry["binary_stats"] = stats
                prev_binary = curr_binary

            log_entries.append(entry)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

            # Compact console line.
            extra = ""
            if "flip_rate" in entry:
                fr = list(entry["flip_rate"].values())
                avg = sum(fr) / len(fr) if fr else 0
                stats = entry["binary_stats"]
                pos = [s["positive_fraction"] for s in stats.values()]
                avg_pos = sum(pos) / len(pos) if pos else 0
                extra = f" avg_flip={avg*100:.2f}% avg_pos={avg_pos*100:.1f}%"
            print(
                f"step {step:>5} | tr_loss {loss.item():.4f} | val_loss {val_loss:.4f} "
                f"| val_ppl {val_ppl:.2f} | val_acc {val_acc:.3f}{extra}"
            )

            if val_loss < best_val:
                best_val = val_loss
                torch.save(
                    {
                        "cfg": cfg.__dict__,
                        "model": model.state_dict(),
                        "step": step,
                        "val_loss": val_loss,
                        "val_ppl": val_ppl,
                    },
                    os.path.join(ckpt_dir, "best.pt"),
                )
            torch.save(
                {
                    "cfg": cfg.__dict__,
                    "model": model.state_dict(),
                    "step": step,
                    "val_loss": val_loss,
                },
                os.path.join(ckpt_dir, f"step{step}.pt"),
            )

    # Final summary line.
    print(f"done. best_val_loss={best_val:.4f} log={log_path}")
    print(f"JSONL log: {log_path}")


if __name__ == "__main__":
    main()
