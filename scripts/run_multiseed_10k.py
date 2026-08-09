"""Multi-seed 10K runs to quantify ROCm run-to-run variance.

Runs the param-matched 1-bit-core pair E (Std) and F2 (TM) at 10K steps across
3 seeds each. Seed 1337 re-runs the original 10K config as a reproducibility
control; seeds 42 and 2024 are new.

Each model x seed writes:
  out/<Model>-10k-s<seed>_log.jsonl + out/run_<tag>.log

~6 runs, ~45 min sequential GPU.
"""
import subprocess
import os

PY = r"C:\Users\antho\AppData\Local\Programs\Python\Python312\python.exe"
os.chdir(r"E:\AI-Workspace\1BitLLM")

SEEDS = [1337, 42, 2024]

# (model_key, ffn, ffn_hidden, binary_ffn, binary_attn)
JOBS = [
    ("E",  "standard",     None, True, True),  # Std 1-bit core (hidden 756)
    ("F2", "tensormatics", 314,  True, True),  # TM 1-bit core param-matched
]

def main():
    for key, ffn, ffn_hidden, bffn, battn in JOBS:
        for seed in SEEDS:
            tag = f"{key}-10k-s{seed}"
            log = os.path.join("out", f"run_{tag}.log")
            print(f"\n########## {tag} ##########", flush=True)
            cmd = [PY, "scripts/train.py",
                   "--ffn", ffn]
            if ffn_hidden:
                cmd += ["--ffn-hidden", str(ffn_hidden)]
            cmd += ["--binary-ffn", "--binary-attn",
                    "--steps", "10000",
                    "--eval-interval", "1000", "--eval-iters", "50",
                    "--seed", str(seed),
                    "--out", "out", "--tag", tag]
            f = open(log, "w")
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
            f.close()
            print(f"{tag} exit_code={r.returncode}", flush=True)

    print("\nMULTI-SEED 10K DONE")

if __name__ == "__main__":
    main()
