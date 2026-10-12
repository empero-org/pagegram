"""Per-arm mean and std over seeds, paired deltas vs no table and flat, NF4 readout, throughput.
Writes <dir>/analysis.md and <dir>/analysis.json.

    python -m pagegram.analyze runs/main
    python -m pagegram.analyze runs/probe --best      # prints the best table-lr multiplier per kind
"""
from __future__ import annotations

import glob
import json
import os
import statistics
import sys
from collections import defaultdict


def _ms(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0), len(xs)


def load(root: str):
    out = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(root, "**", "summary.json"), recursive=True)):
        s = json.load(open(f))
        if not isinstance(s, dict) or "final_val_loss" not in s:
            continue
        key = os.path.relpath(os.path.dirname(os.path.dirname(f)), root)
        out[key].append(s)
    return out


def best_muls(root: str) -> dict:
    res = {}
    for key, runs in load(root).items():
        arm, mul = key.split("/")
        loss = _ms([r["final_val_loss"] for r in runs if not r.get("diverged")])[0]
        if loss is not None and (arm not in res or loss < res[arm][1]):
            res[arm] = (float(mul[3:]), loss)
    return res


def main():
    root = sys.argv[1]
    if "--best" in sys.argv:
        b = best_muls(root)
        print(json.dumps({k: v[0] for k, v in b.items()}))
        return
    data = load(root)
    table = {}
    for arm, runs in data.items():
        ok = [r for r in runs if not r.get("diverged")]
        table[arm] = {
            "n": len(ok), "diverged": len(runs) - len(ok),
            "loss": _ms([r["final_val_loss"] for r in ok]),
            "nf4": _ms([r.get("val_loss_nf4") for r in ok]),
            "tok_s": _ms([r["tok_s"] for r in ok]),
            "table_params": ok[0]["geometry"]["params"] if ok else None,
            "per_seed": {r["args"]["seed"]: r["final_val_loss"] for r in ok},
        }
    base = table.get("none", {}).get("per_seed", {})
    flat = table.get("flat", {}).get("per_seed", {})
    for arm, t in table.items():
        for name, ref in (("d_none", base), ("d_flat", flat)):
            ds = [t["per_seed"][s] - ref[s] for s in t["per_seed"] if s in ref]
            t[name] = _ms(ds)
    order = ["none", "flat", "global", "paged", "paged_ctx", "paged_q"]
    lines = ["| arm | table params | n | final val loss | vs none (paired) | vs flat (paired) | NF4 table | tok/s |",
             "|---|---|---|---|---|---|---|---|"]
    for arm in sorted(table, key=lambda a: order.index(a) if a in order else 99):
        t = table[arm]
        f = lambda m, d=4: "--" if m[0] is None else f"{m[0]:.{d}f} ± {m[1]:.{d}f}"
        lines.append(f"| {arm} | {t['table_params'] or 0:,} | {t['n']} | {f(t['loss'])} | {f(t['d_none'])} | "
                     f"{f(t['d_flat'])} | {f(t['nf4'])} | {f(t['tok_s'], 0)} |")
    md = "\n".join(lines)
    print(md)
    open(os.path.join(root, "analysis.md"), "w").write(md + "\n")
    json.dump(table, open(os.path.join(root, "analysis.json"), "w"), indent=1, default=list)


if __name__ == "__main__":
    main()
