"""Run 20k-step training for F2 (TM 1-bit core, hidden 314, param-matched).

E (Std core) has stalled/saturated, so we extend only the still-learning
Tensormatics model to see its asymptotic PPL.
~22 min.
"""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

print("\n########## F2-tm-core-20k ##########", flush=True)
cmd = [PY, "scripts/train.py", "--ffn", "tensormatics", "--ffn-hidden", "314",
       "--binary-ffn", "--binary-attn", "--steps", "20000",
       "--eval-interval", "2000", "--eval-iters", "50", "--out", "out",
       "--tag", "F2-20k"]
log = open("out/run_F2-tm-core-20k.log", "w")
r = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
log.close()
print(f"F2-tm-core-20k exit_code={r.returncode}", flush=True)
print("\nF2 20K DONE")
