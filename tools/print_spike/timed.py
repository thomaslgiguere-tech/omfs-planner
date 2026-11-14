"""Spike: run a command with a time limit and print its wall time. python3 timed.py SECONDS cmd args..."""
import subprocess
import sys
import time

limit, cmd = float(sys.argv[1]), sys.argv[2:]
start = time.time()
try:
    code = subprocess.run(cmd, capture_output=True, timeout=limit).returncode
    print(f"{time.time() - start:.2f} s (code {code})")
except subprocess.TimeoutExpired:
    print(f"> {limit:.0f} s (stopped)")
