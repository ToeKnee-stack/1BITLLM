"""Run 50k-step training for F2 (TM 1-bit core, hidden 314, param-matched).

Extend the still-learning Tensormatics model to pin down its asymptotic PPL.
~53 min GPU (extrapolated from 20K @ ~22 min).
"""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

print("\n########## F2-tm-core-50k ##########", flush=True)
cmd = [PY, "scripts/train.py", "--ffn", "tensormatics", "--ffn-hidden", "314",
       "--binary-ffn", "--binary-attn", "--steps", "50000",
       "--eval-interval", "2000", "--eval-iters", "50", "--out", "out",
       "--tag", "F2-50k"]
log = open("out/run_F2-tm-core-50k.log", "w")
r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
log.close()
print(f"F2-tm-core-50k exit_code={r.returncode}", flush=True)
print("\nF2 50K DONE")
