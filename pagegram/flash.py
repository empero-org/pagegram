"""Flash cost of deployment-size tables on held-out tokens: blocks read per token, LRU page-cache
misses, and direct-read throughput.

    python -m pagegram.flash --holdout DATA/holdout --out runs/io [--bench-dir /workspace/iobench]

A 4-bit head-row (64 values + fp16 scale) is 34 bytes, so a 4 KiB block holds 120 head-rows: one
PageGram page at K = 8, S = 15, or 120 rows of a global table."""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import subprocess

import numpy as np

from .memory import primes_le

HERE = os.path.dirname(os.path.abspath(__file__))
TABLES = {"3.2B": 3.2e9, "10B": 1.0e10, "20B": 2.0e10}
CACHE_MIB = (0, 64, 256, 1024, 4096)


def build(work: str) -> str:
    exe = os.path.join(work, "iosim")
    subprocess.run(["cc", "-O2", "-pthread", os.path.join(HERE, "iosim.c"), "-o", exe], check=True)
    return exe


def token_file(holdout: str, n: int, path: str) -> int:
    files = sorted(glob.glob(os.path.join(holdout, "**", "*.bin"), recursive=True))
    got = 0
    with open(path, "wb") as fh:
        for f in files:
            t = np.fromfile(f, dtype=np.uint16)[: n - got]
            t.tofile(fh)
            got += len(t)
            if got >= n:
                break
    return got


ROW_BYTES = 34


def geo_args(layout: str, params: float, K: int = 8, r: int = 64, S: int = 15):
    """iosim args and the number of 4 KiB blocks. Paged tables pack whole pages into blocks."""
    per_order = int(params) // r // 2
    if layout == "paged":
        P2, P3 = primes_le(per_order // (K * S), 2)
        ppb = max(1, 4096 // (K * S * ROW_BYTES))
        return ["paged", str(K), str(S), str(P2), str(P3), "0", str(ppb)], -(-P2 // ppb) - (-P3 // ppb)
    sizes = primes_le(per_order // K, 2 * K)
    return ["global", str(K), str(S), "0", "0", ",".join(map(str, sizes)), str(K * S)], sum(sizes) // (K * S) + 1


def run_json(cmd) -> dict:
    out = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--bench-dir", default="")
    p.add_argument("--bench-table", default="20B")
    p.add_argument("--slots", default="15,4")
    p.add_argument("--no-bench", action="store_true")
    a = p.parse_args()
    slots = [int(s) for s in a.slots.split(",")]
    os.makedirs(a.out, exist_ok=True)
    exe = build(a.out)
    tf = os.path.join(a.out, "tokens.u16")
    n = token_file(a.holdout, a.tokens, tf)
    res = {"tokens": n, "sim": [], "bench": []}
    caps = ",".join(str(m * 256) for m in CACHE_MIB)
    for name, params in TABLES.items():
        for layout, S in [("paged", s) for s in slots] + [("global", 15)]:
            g, pages = geo_args(layout, params, S=S)
            r = run_json([exe, "sim", tf, str(n), "65536", *g, caps])
            r.update(table=name, table_params=params, file_gib=pages * 4096 / 2**30, slots=S if layout == "paged" else None)
            res["sim"].append(r)
            print(json.dumps(r), flush=True)
            json.dump(res, open(os.path.join(a.out, "io.json"), "w"), indent=1)
    if a.bench_dir and not a.no_bench:
        os.makedirs(a.bench_dir, exist_ok=True)
        params = TABLES[a.bench_table]
        size = max(geo_args(l, params)[1] for l in ("paged", "global")) * 4096
        free = shutil.disk_usage(a.bench_dir).free
        res["bench_table"] = a.bench_table
        if free < size * 1.2:
            res["bench_skipped"] = f"needs {size / 2**30:.1f} GiB, {free / 2**30:.1f} GiB free"
        else:
            path = os.path.join(a.bench_dir, "table.bin")
            if not os.path.exists(path) or os.path.getsize(path) < size:
                rng = np.random.default_rng(0)
                chunk = rng.integers(0, 256, size=64 << 20, dtype=np.uint8).tobytes()
                with open(path, "wb") as fh:
                    left = size
                    while left > 0:
                        fh.write(chunk[: min(left, len(chunk))])
                        left -= len(chunk)
                    fh.flush()
                    os.fsync(fh.fileno())
            for threads, ntok in ((1, 20_000), (2, 40_000), (16, 200_000), (64, 400_000)):
                for layout in ("paged", "global"):
                    g, _ = geo_args(layout, params)
                    r = run_json([exe, "bench", tf, str(ntok), "65536", *g, path, str(threads)])
                    r.update(table=a.bench_table)
                    res["bench"].append(r)
                    print(json.dumps(r), flush=True)
                    json.dump(res, open(os.path.join(a.out, "io.json"), "w"), indent=1)
    json.dump(res, open(os.path.join(a.out, "io.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
