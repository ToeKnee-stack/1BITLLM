"""Run 10k-step training for the param-matched 1-bit-core pair:
  E  = Std 1-bit core (hidden 756)
  F2 = TM 1-bit core (hidden 314, param-matched)

Sequential to avoid GPU contention. ~11 min each.
"""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

jobs = [
    # E: standard 1-bit core, 10k
    ("E-std-core-10k", ["--ffn", "standard", "--binary-ffn", "--binary-attn", "--tag", "E-10k"]),
    # F2: tensormatics 1-bit core param-matched, 10k
    ("F2-tm-core-10k", ["--ffn", "tensormatics", "--ffn-hidden", "314", "--binary-ffn", "--binary-attn", "--tag", "F2-10k"]),
]

for tag, extra in jobs:
    print(f"\n########## {tag} ##########", flush=True)
    cmd = [PY, "scripts/train.py", "--steps", "10000", "--eval-interval", "1000",
           "--eval-iters", "50", "--out", "out"] + extra
    log = open(f"out/run_{tag}.log", "w")
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f"{tag} exit_code={r.returncode}", flush=True)

print("\nALL 10K DONE")
