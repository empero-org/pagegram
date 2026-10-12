"""Run a grid a few jobs at a time on one GPU, seed by seed; finished runs are skipped.

    python -m pagegram.suite --grid probe --data DATA --out runs/probe
    python -m pagegram.suite --grid main  --data DATA --out runs/main --mul-flat 70 --mul-multi 70
"""
from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import time

GRIDS = {
    "probe": dict(arms=("flat", "paged"), seeds=(0,), muls=(20, 70, 200), tokens=3e7, eval_every=50),
    "main": dict(arms=("none", "flat", "global", "paged", "paged_ctx", "paged_q"), seeds=(0, 1, 2),
                 muls=(None,), tokens=1e8, eval_every=100),
    "slots": dict(arms=("paged",), seeds=(0,), muls=(None,), slots=(1, 4, 60), tokens=6e7, eval_every=100),
}


def jobs(grid: str, a):
    g = GRIDS[grid]
    tokens = a.tokens or g["tokens"]
    for seed, arm, mul, slots in itertools.product(g["seeds"], g["arms"], g["muls"], g.get("slots", (15,))):
        if mul is None:
            mul = a.mul_flat if arm == "flat" else a.mul_multi
        sub = {"probe": f"{arm}/mul{mul:g}/s{seed}", "slots": f"{arm}/S{slots}/s{seed}"}.get(grid, f"{arm}/s{seed}")
        out = os.path.join(a.out, sub)
        cmd = [sys.executable, "-m", "pagegram.train", "--data", a.data, "--arm", arm, "--seed", str(seed),
               "--tokens", str(tokens), "--table-lr-mul", str(mul), "--eval-every", str(g["eval_every"]),
               "--slots", str(slots), "--out", out] + a.extra
        yield out, cmd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=sorted(GRIDS), required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--tokens", type=float, default=0)
    p.add_argument("--mul-flat", type=float, default=70.0)
    p.add_argument("--mul-multi", type=float, default=70.0)
    p.add_argument("extra", nargs=argparse.REMAINDER)
    a = p.parse_args()
    a.extra = [x for x in a.extra if x != "--"]
    todo = [(o, c) for o, c in jobs(a.grid, a) if not os.path.exists(os.path.join(o, "summary.json"))]
    print(f"{len(todo)} runs to do", flush=True)
    running = []
    while todo or running:
        while todo and len(running) < a.jobs:
            out, cmd = todo.pop(0)
            os.makedirs(out, exist_ok=True)
            fh = open(os.path.join(out, "stdout.log"), "w")
            running.append((out, subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT), fh))
            print(f"start {out}", flush=True)
        for r in list(running):
            if r[1].poll() is not None:
                r[2].close()
                running.remove(r)
                ok = os.path.exists(os.path.join(r[0], "summary.json"))
                print(f"{'done' if ok else 'FAILED'} {r[0]} (exit {r[1].returncode})", flush=True)
        time.sleep(5)


if __name__ == "__main__":
    main()
