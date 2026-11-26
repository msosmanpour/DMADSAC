#!/usr/bin/env python3
"""Small reproducible demonstration of CPU subprocess parallelism. NOT study data."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from subprocess import run
from sys import executable
from time import perf_counter
from pathlib import Path
import sys

# Worker is a separate interpreter process, not a thread doing CPU-bound Python work.
WORKER = "import hashlib; h=b'abc'; [(h:=hashlib.sha256(h).digest()) for _ in range(90000)]; print(h.hex()[:16])"

def worker(i):
    start=perf_counter()
    result=run([executable,'-c',WORKER],capture_output=True,text=True,check=True)
    return (i, round(perf_counter()-start,3), result.stdout.strip())

if __name__=='__main__':
    t0=perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs=[pool.submit(worker,i) for i in range(6)]
        for future in as_completed(futs):
            i,sec,digest=future.result();print(f'demo_worker={i} seconds={sec} digest={digest}')
    print(f'demo_elapsed_seconds={perf_counter()-t0:.3f}')
    print('DEMO_ONLY_NOT_AN_ARCHIVED_SCIENTIFIC_RUN')
