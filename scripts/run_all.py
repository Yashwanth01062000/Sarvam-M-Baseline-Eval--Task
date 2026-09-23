"""Validate and run the complete pipeline."""
import subprocess
import sys

commands = [
    [sys.executable, "-m", "src.run_eval"],
    [sys.executable, "-m", "src.scoring"],
    [sys.executable, "-m", "src.analyze"],
]
for cmd in commands:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)
print("Evaluation pipeline completed.")
