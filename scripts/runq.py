#!/usr/bin/env python
"""Run a job list N at a time on one GPU, in file order; finished runs are skipped.

    python scripts/runq.py JOBS.jsonl --jobs 2
Each line: {"out": DIR, "args": ["--arm", "paged", ...]} (passed to python -m pagegram.train)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time


def busy() -> int:
    """Number of pagegram.train processes running."""
    out = subprocess.run(["pgrep", "-f", "-c", "pagegram.train"], capture_output=True, text=True).stdout.strip()
    return int(out or 0)


def started_elsewhere(out: str) -> bool:
    log = os.path.join(out, "stdout.log")
    return os.path.exists(log) and time.time() - os.path.getmtime(log) < 300


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jobfile")
    p.add_argument("--jobs", type=int, default=2)
    p.add_argument("--data", default="/workspace/data/pg")
    a = p.parse_args()
    todo = []
    for line in open(a.jobfile):
        if line.strip():
            j = json.loads(line)
            if not os.path.exists(os.path.join(j["out"], "summary.json")) and not started_elsewhere(j["out"]):
                todo.append(j)
    print(f"{len(todo)} runs to do", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < a.jobs and busy() < a.jobs:
            j = todo.pop(0)
            os.makedirs(j["out"], exist_ok=True)
            fh = open(os.path.join(j["out"], "stdout.log"), "w")
            cmd = [sys.executable, "-m", "pagegram.train", "--data", a.data, "--out", j["out"]] + j["args"]
            running.append((j["out"], subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] start {j['out']}", flush=True)
        for r in list(running):
            if r[1].poll() is not None:
                r[2].close()
                running.remove(r)
                ok = os.path.exists(os.path.join(r[0], "summary.json"))
                print(f"[{time.strftime('%H:%M:%S', time.gmtime())}] {'done' if ok else 'FAILED'} {r[0]}", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
