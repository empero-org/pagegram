"""Harness self-checks: CPU checks first, then a short GPU run per arm.

    python -m pagegram.test_harness
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import torch

from .memory import EOS, Geometry, RowRMS, addresses, geometry
from .model import LM, Config

RESULTS = []
D, V = 64, 512


def check(name, ok, detail=""):
    RESULTS.append((name, bool(ok)))
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   [{detail}]" if detail else ""), flush=True)


def test_geometry():
    print("geometry")
    budget = 2e8
    g = {k: geometry(k, int(budget), 384) for k in ("flat", "global", "paged")}
    rel = {k: abs(v.params - budget) / budget for k, v in g.items()}
    check("flat, global and paged tables hold the same parameters (within 0.5%)",
          max(rel.values()) < 5e-3, ", ".join(f"{k} {v.params / 1e6:.2f}M" for k, v in g.items()))
    q = geometry("paged", int(budget / 4), 384)
    check("the quarter table holds a quarter", abs(q.params / g["paged"].params - 0.25) < 5e-3)
    check("global: 16 distinct prime table sizes", len(set(g["global"].sizes)) == 16)


def test_addresses():
    print("addresses")
    torch.manual_seed(0)
    idx = torch.randint(3, V, (4, 300))
    for kind in ("flat", "global", "paged"):
        geo = geometry(kind, 400_000, D, heads=8, head_dim=16, slots=15)
        rows, sidx = addresses(idx, geo, V)
        ok = bool((rows >= 0).all() and (rows < geo.rows).all() and (sidx >= 0).all() and (sidx < 4096).all())
        if kind == "flat":
            half = geo.rows // 2
            ok &= bool((rows[..., 0] < half).all() and (rows[..., 1] >= half).all())
        elif kind == "paged":
            K, S, (P2, P3) = geo.heads, geo.slots, geo.pages
            R = K * S
            for n, base in ((0, 0), (1, P2 * R)):
                r = rows[..., n * K:(n + 1) * K] - base
                ok &= bool((r // R == r[..., :1] // R).all())
                ok &= bool(((r % R) // S == torch.arange(K)).all())
        else:
            lo = np.cumsum((0,) + geo.sizes[:-1])
            hi = lo + np.array(geo.sizes)
            for c in range(rows.shape[-1]):
                ok &= bool((rows[..., c] >= int(lo[c])).all() and (rows[..., c] < int(hi[c])).all())
        check(f"{kind}: every address in range and in its region", ok)
    geo = geometry("paged", 400_000, D, heads=8, head_dim=16, slots=15)
    a = torch.randint(3, V, (1, 40))
    a[0, 20] = EOS
    b = a.clone()
    b[0, :20] = torch.randint(3, V, (20,))
    ra, _ = addresses(a, geo, V)
    rb, _ = addresses(b, geo, V)
    check("n-grams never reach back across an EOS", torch.equal(ra[0, 20:], rb[0, 20:]))


def test_init():
    print("initialisation")
    cfg = Config(vocab=V, d=D, layers=2, heads=4, seq=32, every=1)
    torch.manual_seed(1)
    idx = torch.randint(3, V, (2, 33))
    x, y = idx[:, :-1], idx[:, 1:]
    m0 = LM(cfg, geometry("none", 0, D), seed=3)
    l0 = m0(x, y)
    same_core, same_loss = True, True
    for kind, gate in (("flat", "scalar"), ("global", "scalar"), ("paged", "scalar"), ("paged", "context")):
        c = Config(**{**cfg.__dict__, "gate": gate})
        m = LM(c, geometry(kind, 200_000, D, heads=8, head_dim=16), seed=3)
        same_core &= all(torch.equal(p, q) for p, q in zip(m.core_parameters(), m0.core_parameters()))
        same_loss &= torch.equal(m(x, y), l0)
    check("every arm's core equals the table-free model's for a seed", same_core)
    check("with the table at zero every arm computes the table-free loss, bitwise", same_loss)
    m1 = LM(cfg, geometry("none", 0, D), seed=4)
    check("another seed draws another core",
          not all(torch.equal(p, q) for p, q in zip(m1.core_parameters(), m0.core_parameters())))


def test_row_optimizer():
    print("row optimizer")
    torch.manual_seed(2)
    cfg = Config(vocab=V, d=D, layers=2, heads=4, seq=32, every=1)
    m = LM(cfg, geometry("paged", 200_000, D, heads=8, head_dim=16), seed=0)
    with torch.no_grad():
        m.memory.table.normal_(0, 0.1)
    idx = torch.randint(3, V, (2, 33))
    m(idx[:, :-1], idx[:, 1:]).backward()
    g = m.memory.table.grad.coalesce()
    dense = torch.zeros_like(m.memory.table)
    dense.index_put_((g.indices()[0],), g.values(), accumulate=True)
    m2 = LM(cfg, geometry("paged", 200_000, D, heads=8, head_dim=16), seed=0)
    with torch.no_grad():
        m2.memory.table.copy_(m.memory.table)
    rows, sidx = addresses(idx[:, :-1], m2.memory.geo, V)
    t = m2.memory.table.detach().clone().requires_grad_(True)
    h = torch.nn.functional.embedding(rows, t) * m2.memory.signs[sidx]
    e = m2.memory.value(h.flatten(-2))
    x = m2.embed(idx[:, :-1])
    from .model import rope_tables
    cos, sin = rope_tables(32, D // 4, 10000.0, "cpu")
    for i, blk in enumerate(m2.blocks):
        x = m2.memory.inject(i, x, e, None)
        x = blk(x, cos, sin)
    logits = m2.head(m2.norm(x))
    torch.nn.functional.cross_entropy(logits.flatten(0, 1), idx[:, 1:].flatten()).backward()
    check("sparse table gradient == the dense reference", torch.allclose(dense, t.grad, atol=1e-6, rtol=1e-5),
          f"max err {(dense - t.grad).abs().max().item():.1e}")
    opt = RowRMS(m.memory.table)
    w0 = m.memory.table.detach().clone()
    ids = g.indices()[0]
    opt.step(m.memory.table, 0.1, 3)
    ms = g.values().square().mean(1)
    ref = w0.clone()
    ref[ids] -= 0.1 * g.values() / (ms.sqrt() + 1e-8)[:, None]
    ok = torch.allclose(m.memory.table.detach(), ref, atol=1e-6)
    gv = torch.randn(len(ids), g.values().shape[1])
    m.memory.table.grad = torch.sparse_coo_tensor(ids[None], gv, m.memory.table.shape)
    w1 = m.memory.table.detach().clone()
    opt.step(m.memory.table, 0.1, 7)
    d = 0.95 ** 4
    v = 0.05 * ms * d + 0.05 * gv.square().mean(1)
    ws = 0.05 * d + 0.05
    ref = w1.clone()
    ref[ids] -= 0.1 * gv / ((v / ws).sqrt() + 1e-8)[:, None]
    ok &= torch.allclose(m.memory.table.detach(), ref, atol=1e-6)
    check("row-RMS update and its replayed decay == the hand-computed rule", ok)


def test_nf4():
    print("4-bit readout")
    cfg = Config(vocab=V, d=D, layers=2, heads=4, seq=32, every=1)
    m = LM(cfg, geometry("paged", 200_000, D, heads=8, head_dim=16), seed=0)
    with torch.no_grad():
        m.memory.table.normal_(0, 0.3)
        m.memory.table[:7] = 0
    q = m.memory.quantized_table()
    err = (q - m.memory.table).abs().reshape(-1, 64).amax(1) / m.memory.table.abs().reshape(-1, 64).amax(1).clamp(min=1e-12)
    check("NF4: zero rows stay zero, error within half the widest level gap",
          bool((q[:7] == 0).all()) and err.max().item() <= 0.161, f"max rel err {err.max().item():.3f}")


def test_flash():
    print("flash simulator")
    from .flash import build, geo_args
    with tempfile.TemporaryDirectory() as tmp:
        exe = build(tmp)
        rng = np.random.default_rng(1)
        tok = rng.integers(3, 65536, size=5000).astype(np.uint16)
        tok[::97] = EOS
        tf = os.path.join(tmp, "t.u16")
        tok.tofile(tf)
        idx = torch.from_numpy(tok.astype(np.int64))[None]
        for layout in ("paged", "global"):
            g, _ = geo_args(layout, 3.2e9)
            out = subprocess.run([exe, "pages", tf, "5000", "65536", *g], check=True, capture_output=True,
                                 text=True).stdout.splitlines()
            c_pages = [sorted(map(int, line.split())) for line in out]
            if layout == "paged":
                K, S, P2, P3 = int(g[1]), int(g[2]), int(g[3]), int(g[4])
                geo = Geometry("paged", (P2 + P3) * K * S, 64, K, S, (P2, P3))
                rows, _ = addresses(idx, geo, 65536)
                py = [sorted({int(r) // (K * S) for r in row}) for row in rows[0]]
            else:
                sizes = tuple(int(s) for s in g[5].split(","))
                geo = Geometry("global", sum(sizes), 64, 8, sizes=sizes)
                rows, _ = addresses(idx, geo, 65536)
                py = [sorted({int(r) // 120 for r in row}) for row in rows[0]]
            check(f"{layout}: the C page ids == the model's addresses, every token", c_pages == py)


def test_gpu_runs():
    if not torch.cuda.is_available():
        print("gpu runs: skipped (no CUDA)")
        return
    print("gpu runs")
    with tempfile.TemporaryDirectory() as tmp:
        rng = np.random.default_rng(0)
        for part in ("S1", "holdout"):
            os.makedirs(os.path.join(tmp, "data", part, "src"))
            t = rng.integers(3, 65536, size=3_000_000).astype(np.uint16)
            t[::500] = EOS
            t.tofile(os.path.join(tmp, "data", part, "src", "000.bin"))
        for arm in ("none", "flat", "global", "paged", "paged_ctx"):
            out = os.path.join(tmp, arm)
            r = subprocess.run([sys.executable, "-m", "pagegram.train", "--data", os.path.join(tmp, "data"),
                                "--arm", arm, "--tokens", str(20 * 131072), "--eval-every", "10",
                                "--eval-tokens", str(65536), "--table-params", "2e7", "--out", out,
                                "--compile", "1" if arm == "paged_ctx" else "0"],
                               capture_output=True, text=True)
            ok = r.returncode == 0 and os.path.exists(os.path.join(out, "summary.json"))
            detail = ""
            if ok:
                s = json.load(open(os.path.join(out, "summary.json")))
                detail = f"val {s['final_val_loss']:.3f}, {s['tok_s'] or 0:,.0f} tok/s, peak {s['peak_gib']:.1f} GiB"
            else:
                detail = (r.stdout + r.stderr)[-600:]
            check(f"{arm}: 20 steps train, evaluate and summarise", ok, detail)


if __name__ == "__main__":
    torch.set_num_threads(4)
    for t in (test_geometry, test_addresses, test_init, test_row_optimizer, test_nf4, test_flash, test_gpu_runs):
        t()
    n = sum(ok for _, ok in RESULTS)
    print(f"\n{n}/{len(RESULTS)} passed")
    sys.exit(0 if n == len(RESULTS) else 1)
