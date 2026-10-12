#!/usr/bin/env python
"""Collision stats for the global and paged layouts on real tokens (numpy only).

Per order and head: per-row load vs Poisson, the share of each row's occurrences that come from other
n-grams, and how many other heads two colliding n-grams also share.

    python scripts/collisions.py --holdout DATA/holdout --tokens 20000000 --table-params 2e8"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pagegram.memory import A, EOS, M, primes_le  # noqa: E402

V = 65536


def stream(holdout: str, n: int) -> np.ndarray:
    files = sorted(glob.glob(os.path.join(holdout, "**", "*.bin"), recursive=True))
    out, got = [], 0
    for f in files:
        t = np.fromfile(f, dtype=np.uint16)[: n - got]
        out.append(t)
        got += len(t)
        if got >= n:
            break
    return np.concatenate(out).astype(np.int64)


def keys(x: np.ndarray):
    pad = np.int64(V)
    p1 = np.concatenate([[pad], x[:-1]])
    p2 = np.concatenate([[pad], p1[:-1]])
    start = x == EOS
    p2 = np.where(start | (p1 == EOS), pad, p2)
    p1 = np.where(start, pad, p1)
    return x, p1, p2


def stats(rows: np.ndarray, gram: np.ndarray, size: int) -> dict:
    """rows, gram: per occurrence (row within the head, compact n-gram id)."""
    key = rows * (1 << 26) + gram
    uk, inv, pair_cnt = np.unique(key, return_inverse=True, return_counts=True)
    row_total = np.bincount(rows, minlength=size)
    per_row = np.bincount(uk >> 26, minlength=size)
    foreign = 1.0 - pair_cnt[inv] / row_total[rows]
    lam = len(uk) / size
    return {"distinct_ngrams": int(len(uk)), "lambda": lam,
            "foreign_share": float(foreign.mean()),
            "collided_occurrences": float((foreign > 0).mean()),
            "empty_rows": float((per_row == 0).mean()), "empty_poisson": math.exp(-lam),
            "var_over_mean": float(per_row.var() / lam)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--holdout", required=True)
    p.add_argument("--tokens", type=int, default=20_000_000)
    p.add_argument("--table-params", type=float, default=2e8)
    p.add_argument("--out", default="")
    a = p.parse_args()
    x, p1, p2 = keys(stream(a.holdout, a.tokens))
    K, S, r = 8, 15, 64
    per_order = int(a.table_params) // r // 2
    P2, P3 = primes_le(per_order // (K * S), 2)
    sizes = primes_le(per_order // K, 2 * K)
    b = (p1 * M[0]) ^ (x * M[1])
    c = (p2 * M[2]) ^ (p1 * M[3]) ^ (x * M[4])
    res = {"tokens": int(len(x)), "table_params": a.table_params, "P": [P2, P3], "sizes": sizes, "orders": {}}
    for n, (name, base, P) in enumerate((("bigram", b, P2), ("trigram", c, P3))):
        gid = p1 * (V + 1) + x if n == 0 else (p2 * (V + 1) + p1) * (V + 1) + x
        _, gram = np.unique(gid, return_inverse=True)
        page = base % P
        out = {"global": [], "paged": [], "pair_sharing": {}}
        rows_g, rows_p = [], []
        for k in range(K):
            ak = A[n][k]
            u = (p1 * ak[0]) ^ (x * ak[1]) if n == 0 else (p2 * ak[0]) ^ (p1 * ak[1]) ^ (x * ak[2])
            rg = u % sizes[n * K + k]
            rp = page * S + u % S
            rows_g.append(rg)
            rows_p.append(rp)
            out["global"].append(stats(rg, gram, sizes[n * K + k]))
            out["paged"].append(stats(rp, gram, P * S))
            print(name, k, "global", json.dumps(out["global"][-1]), flush=True)
            print(name, k, "paged ", json.dumps(out["paged"][-1]), flush=True)
        # n-gram pairs that collide in head 0: how many other heads do they share?
        for layout, rows in (("global", rows_g), ("paged", rows_p)):
            first = np.unique(gram, return_index=True)[1]
            g = np.arange(len(first))
            r0 = rows[0][first]
            order = np.argsort(r0, kind="stable")
            r0s, gs = r0[order], g[order]
            same = r0s[1:] == r0s[:-1]
            ia, ib = gs[:-1][same], gs[1:][same]
            shared = np.zeros(len(ia), dtype=np.int64)
            for k in range(1, K):
                rk = rows[k][first]
                shared += rk[ia] == rk[ib]
            out["pair_sharing"][layout] = {"pairs": int(len(ia)),
                                           "mean_other_heads_shared": float(shared.mean()) if len(ia) else 0.0,
                                           "share_ge1_other": float((shared >= 1).mean()) if len(ia) else 0.0}
            print(name, layout, "pairs sharing head 0:", json.dumps(out["pair_sharing"][layout]), flush=True)
        res["orders"][name] = out
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
