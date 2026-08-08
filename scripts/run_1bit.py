"""Run 1-bit experiment models C/D/E/F sequentially on GPU."""
import subprocess
import sys
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

jobs = [
    ("C-std-1bit-ffn", ["--ffn", "standard", "--binary-ffn"]),
    ("D-tm-1bit-ffn", ["--ffn", "tensormatics", "--binary-ffn"]),
    ("E-std-1bit-core", ["--ffn", "standard", "--binary-ffn", "--binary-attn"]),
    ("F-tm-1bit-core", ["--ffn", "tensormatics", "--binary-ffn", "--binary-attn"]),
]

for tag, extra in jobs:
    print(f"\n########## {tag} ##########", flush=True)
    cmd = [PY, "scripts/train.py", "--steps", "5000", "--eval-interval", "500",
           "--eval-iters", "50", "--out", "out", "--tag", tag] + extra
    log = open(f"out/run_{tag}.log", "w")
    r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    log.close()
    print(f"{tag} exit_code={r.returncode}", flush=True)

print("\nALL 1-BIT DONE")
