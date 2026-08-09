"""Tensormatics mechanism ablation — WHY does Tensormatics work under 1-bit?

Runs the param-matched TM 1-bit core (hidden 314) at 10K steps, seed 1337, with
each structural feature toggled off. All variants have identical param counts
(3,895,890), so any ppl difference is purely structural.

Variants (vs F2 baseline = 4.96 ppl @ 10K s1337):
  F2-full        : mult + add + learned gate   (the baseline, already run)
  A-no-mult      : additive path only          (--no-tm-mult)
  B-no-add       : multiplicative path only    (--no-tm-add)
  C-no-gate      : mult + add, fixed 0.5 gate  (--no-tm-gate)
  D-no-mult-gate : additive only, fixed gate   (--no-tm-mult --no-tm-gate)

~5 runs, ~30 min sequential GPU.
"""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

# (tag, extra_flags)
JOBS = [
    ("A-no-mult",      ["--no-tm-mult"]),
    ("B-no-add",       ["--no-tm-add"]),
    ("C-no-gate",      ["--no-tm-gate"]),
    ("D-no-mult-gate", ["--no-tm-mult", "--no-tm-gate"]),
]

def main():
    for tag, flags in JOBS:
        log = os.path.join("out", f"run_ablation-{tag}.log")
        print(f"\n########## ablation-{tag} ##########", flush=True)
        cmd = [PY, "scripts/train.py",
               "--ffn", "tensormatics", "--ffn-hidden", "314",
               "--binary-ffn", "--binary-attn",
               "--steps", "10000",
               "--eval-interval", "1000", "--eval-iters", "50",
               "--seed", "1337",
               "--out", "out", "--tag", f"ablation-{tag}"] + flags
        f = open(log, "w")
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        f.close()
        print(f"ablation-{tag} exit_code={r.returncode}", flush=True)

    print("\nABLATION DONE")

if __name__ == "__main__":
    main()
