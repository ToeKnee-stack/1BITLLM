"""Run 50k-step long training for the param-matched 1-bit-core pair:
  E  = Std 1-bit core (hidden 756)
  F2 = TM 1-bit core (hidden 314, param-matched)

Sequential to avoid GPU contention. ~53 min each.
"""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

jobs = [
    # E: standard 1-bit core, 50k
    ("E-std-core-50k", ["--ffn", "standard", "--binary-ffn", "--binary-attn", "--tag", "E-50k"]),
    # F2: tensormatics 1-bit core param-matched, 50k
    ("F2-tm-core-50k", ["--ffn", "tensormatics", "--ffn-hidden", "314", "--binary-ffn", "--binary-attn", "--tag", "F2-50k"]),
]

for tag, extra in jobs:
    print(f"\n########## {tag} ##########", flush=True)
    cmd = [PY, "scripts/train.py", "--steps", "50000", "--eval-interval", "2000",
           "--eval-iters", "50", "--out", "out"] + extra
    log = open(f"out/run_{tag}.log", "w")
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f"{tag} exit_code={r.returncode}", flush=True)

print("\nALL 50K DONE")
