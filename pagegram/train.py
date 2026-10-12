"""Train one arm and log held-out loss.

    python -m pagegram.train --data DATA --arm paged --seed 0 --tokens 1e8 --out runs/main/paged/s0

DATA has S1/ (training shards) and holdout/. Arms with the same seed share the core init and the data
order. Writes <out>/log.jsonl and <out>/summary.json."""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from .data import Shards
from .memory import RowRMS, geometry
from .model import LM, Config

ARMS = {
    "none": ("none", "scalar", 0.0),
    "flat": ("flat", "scalar", 1.0),
    "global": ("global", "scalar", 1.0),
    "paged": ("paged", "scalar", 1.0),
    "paged_ctx": ("paged", "context", 1.0),
    "paged_q": ("paged", "scalar", 0.25),
}


def schedule(step: int, steps: int, warmup: int, decay_frac: float) -> float:
    if step < warmup:
        return (step + 1) / warmup
    start = int(steps * (1 - decay_frac))
    return 1.0 if step < start else max(0.0, (steps - step) / max(1, steps - start))


@torch.no_grad()
def evaluate(model, val: torch.Tensor, micro: int, table=None) -> float:
    model.eval()
    total, n = 0.0, 0
    for i in range(0, val.shape[0], micro):
        b = val[i:i + micro].cuda(non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(b[:, :-1], b[:, 1:], table=table)
        total += loss.item() * b.shape[0]
        n += b.shape[0]
    model.train()
    return total / n


def run(a) -> dict:
    os.makedirs(a.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    kind, gate, scale = ARMS[a.arm]
    cfg = Config(d=a.d, layers=a.layers, heads=a.heads, seq=a.seq, every=a.every, gate=gate)
    geo = geometry(kind, int(a.table_params * scale), a.d, slots=a.slots)
    init_seed = a.seed if a.init_seed < 0 else a.init_seed
    memory_seed = a.seed if a.memory_seed < 0 else a.memory_seed
    data_seed = a.seed if a.data_seed < 0 else a.data_seed
    model = LM(cfg, geo, init_seed, memory_seed).cuda()
    if a.compile:
        model.compile_blocks()
    mem = model.memory
    decay, rest = [], []
    for n, p in model.named_parameters():
        if n == "memory.table":
            continue
        (decay if p.ndim == 2 and n != "embed.weight" else rest).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": rest, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
    rowopt = RowRMS(mem.table) if mem is not None else None
    dense = decay + rest

    train, hold = Shards(os.path.join(a.data, "S1")), Shards(os.path.join(a.data, "holdout"))
    val = hold.fixed(a.eval_tokens // a.seq, a.seq)
    rng = np.random.default_rng(1000 + data_seed)
    accum = a.batch_tokens // (a.micro * a.seq)
    steps = int(a.tokens // a.batch_tokens)
    warmup = max(1, int(steps * a.warmup_frac))

    log, t0, tok_time, tok_count, ema = [], time.time(), 0.0, 0, None
    f = open(os.path.join(a.out, "log.jsonl"), "w")
    for step in range(steps + 1):
        if step % a.eval_every == 0 or step == steps:
            r = {"step": step, "tokens": step * a.batch_tokens, "val_loss": evaluate(model, val, a.micro),
                 "train_loss": ema, "tok_s": tok_count / tok_time if tok_time else None,
                 "elapsed": time.time() - t0}
            log.append(r)
            f.write(json.dumps(r) + "\n")
            f.flush()
            print(json.dumps(r), flush=True)
        if step == steps:
            break
        torch.cuda.synchronize()
        ts = time.time()
        s = schedule(step, steps, warmup, a.decay_frac)
        for _ in range(accum):
            b = train.windows(rng, a.micro, a.seq).pin_memory().cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(b[:, :-1], b[:, 1:]) / accum
            loss.backward()
            ema = loss.item() * accum if ema is None else 0.98 * ema + 0.02 * loss.item() * accum
        if not math.isfinite(ema):
            log.append({"step": step, "diverged": True})
            break
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        for g in opt.param_groups:
            g["lr"] = a.lr * s
        opt.step()
        opt.zero_grad(set_to_none=True)
        if rowopt is not None:
            rowopt.step(mem.table, a.lr * a.table_lr_mul * s, step + 1)
        torch.cuda.synchronize()
        if step >= 10:
            tok_time += time.time() - ts
            tok_count += a.batch_tokens
    f.close()

    summary = {"args": vars(a), "arm": a.arm, "kind": kind, "gate": gate,
               "geometry": {"kind": geo.kind, "rows": geo.rows, "width": geo.width, "heads": geo.heads,
                            "slots": geo.slots, "pages": list(geo.pages), "params": geo.params},
               "core_params": sum(p.numel() for p in model.core_parameters()),
               "memory_dense_params": sum(p.numel() for n, p in model.named_parameters()
                                          if n.startswith("memory.") and n != "memory.table"),
               "steps": steps, "final_val_loss": log[-1].get("val_loss"),
               "diverged": any(r.get("diverged") for r in log),
               "tok_s": tok_count / tok_time if tok_time else None,
               "seconds": time.time() - t0, "peak_gib": torch.cuda.max_memory_allocated() / 2**30}
    if mem is not None and not summary["diverged"]:
        summary["val_loss_nf4"] = evaluate(model, val, a.micro, table=mem.quantized_table())
        summary["rows_touched_frac"] = float((rowopt.t_last > 0).float().mean())
        summary["table_rms"] = float(mem.table.detach().pow(2).mean().sqrt())
    with open(os.path.join(a.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    return summary


def parser():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", required=True)
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-seed", type=int, default=-1)
    p.add_argument("--memory-seed", type=int, default=-1)
    p.add_argument("--data-seed", type=int, default=-1)
    p.add_argument("--tokens", type=float, default=1e8)
    p.add_argument("--table-params", type=float, default=2e8)
    p.add_argument("--batch-tokens", type=int, default=131072)
    p.add_argument("--micro", type=int, default=32)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--d", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--every", type=int, default=4)
    p.add_argument("--slots", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--table-lr-mul", type=float, default=70.0)
    p.add_argument("--wd", type=float, default=0.1)
    p.add_argument("--warmup-frac", type=float, default=0.05)
    p.add_argument("--decay-frac", type=float, default=0.2)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-tokens", type=int, default=2_097_152)
    p.add_argument("--compile", type=int, default=1)
    p.add_argument("--out", required=True)
    return p


if __name__ == "__main__":
    s = run(parser().parse_args())
    print(json.dumps({k: v for k, v in s.items() if k != "args"}))
