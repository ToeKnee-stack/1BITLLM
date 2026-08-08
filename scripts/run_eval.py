"""Run eval_model.py on all 6 trained checkpoints."""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

ckpts = [
    ("A-std-fp16", "out/Std-fp16/best.pt"),
    ("B-tm-fp16", "out/TM-fp16/best.pt"),
    ("C-std-1bit-ffn", "out/Std-1bit-ffn_C-std-1bit-ffn/best.pt"),
    ("D-tm-1bit-ffn", "out/TM-1bit-ffn_D-tm-1bit-ffn/best.pt"),
    ("E-std-1bit-core", "out/Std-1bit-core_E-std-1bit-core/best.pt"),
    ("F-tm-1bit-core", "out/TM-1bit-core_F-tm-1bit-core/best.pt"),
]

for tag, ckpt in ckpts:
    print(f"\n{'='*70}\n### {tag} ###\n{'='*70}")
    cmd = [PY, "scripts/eval_model.py", "--ckpt", ckpt, "--bench", "--gen", "250"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    print(r.stdout)
    if r.stderr:
        print("STDERR:", r.stderr[-500:])
