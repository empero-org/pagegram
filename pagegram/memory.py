"""Hashed n-gram memories: flat rows, multi-head with global addressing, and PageGram."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EOS = 2
SIGN_ROWS = 4096
M = (0x5BD1E995, 0x27D4EB2F, 0x165667B1, 0x61C88647, 0x2545F491, 0x3C6EF372, 0x4F1BBCDD)
A = (
    ((0x7f5885ed, 0x48ecdc53, 0x47183b11), (0x468f9463, 0x559e0b89, 0x4dd332af),
     (0x4d6a04e7, 0x719fe317, 0x51d40e7d), (0x7ac5a7ef, 0x438ce42b, 0x6b2f068f),
     (0x6bc310ef, 0x6349370d, 0x5a896ed7), (0x65c10571, 0x7c01717d, 0x7f57b517),
     (0x413d6137, 0x50669347, 0x4a29bb65), (0x64575d93, 0x79887bcf, 0x6c96af2f)),
    ((0x49ad3517, 0x78574a7f, 0x7b41221d), (0x6d6f659b, 0x786913ef, 0x61d66543),
     (0x571f7899, 0x6c284c17, 0x6b6f07c3), (0x6392b4ad, 0x7b05c7bb, 0x4f91118b),
     (0x5fde71d7, 0x5170f3f9, 0x5ca28ab7), (0x7b51ee09, 0x42ba98f7, 0x4c06b121),
     (0x61538671, 0x77c73d95, 0x5fe57185), (0x48f95c27, 0x5737c629, 0x4b3e4c63)),
)
KINDS = ("none", "flat", "global", "paged")
NF4 = (-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
       -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
       0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
       0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0)


def is_prime(p: int) -> bool:
    if p < 2 or p % 2 == 0:
        return p == 2
    return all(p % f for f in range(3, math.isqrt(p) + 1, 2))


def primes_le(n: int, k: int) -> list[int]:
    out, p = [], n
    while len(out) < k:
        if p < 2:
            raise ValueError(f"fewer than {k} primes <= {n}")
        if is_prime(p):
            out.append(p)
        p -= 1
    return out


@dataclass
class Geometry:
    kind: str
    rows: int                      # table rows (head-rows for multi-head kinds)
    width: int                     # row width: d for flat, r for multi-head
    heads: int = 0                 # K per order
    slots: int = 0                 # S per head per page (paged)
    pages: tuple = ()              # (P2, P3) pages per order (paged)
    sizes: tuple = ()              # 2K per-head table sizes, order-major (global)

    @property
    def params(self) -> int:
        return self.rows * self.width


def geometry(kind: str, table_params: int, d: int, heads: int = 8, head_dim: int = 64,
             slots: int = 15) -> Geometry:
    """Table geometry for a parameter budget: flat rows of width d, or head-rows of width r."""
    if kind == "none" or table_params <= 0:
        return Geometry("none", 0, 0)
    if kind == "flat":
        rows = table_params // d
        return Geometry("flat", rows - rows % 2, d)
    per_order = table_params // head_dim // 2
    if kind == "paged":
        P2, P3 = primes_le(per_order // (heads * slots), 2)
        return Geometry("paged", (P2 + P3) * heads * slots, head_dim, heads, slots, (P2, P3))
    if kind == "global":
        sizes = tuple(primes_le(per_order // heads, 2 * heads))
        return Geometry("global", sum(sizes), head_dim, heads, sizes=sizes)
    raise ValueError(kind)


def ngram_keys(idx: torch.Tensor, vocab: int):
    """Token and its two predecessors; before the window or across an EOS they read the pad id (vocab)."""
    x = idx.long()
    pad = torch.full_like(x[:, :1], vocab)
    p1 = torch.cat([pad, x[:, :-1]], 1)
    p2 = torch.cat([pad, p1[:, :-1]], 1)
    start = x == EOS
    p2 = p2.masked_fill(start | (p1 == EOS), vocab)
    p1 = p1.masked_fill(start, vocab)
    return x, p1, p2


def _sign(h: torch.Tensor, mult: int) -> torch.Tensor:
    return (((h & 0xFFFFFFFF) * mult) >> 13) & (SIGN_ROWS - 1)


def addresses(idx: torch.Tensor, geo: Geometry, vocab: int, cons: torch.Tensor | None = None):
    """[B, T] tokens -> (rows [B, T, C], sign rows [B, T, C]); C = 2 (flat) or 2K."""
    x, p1, p2 = ngram_keys(idx, vocab)
    b = (p1 * M[0]) ^ (x * M[1])
    c = (p2 * M[2]) ^ (p1 * M[3]) ^ (x * M[4])
    if geo.kind == "flat":
        half = geo.rows // 2
        rows = torch.stack([b % half, half + c % (geo.rows - half)], -1)
        return rows, torch.stack([_sign(b, M[5]), _sign(c, M[6])], -1)
    K = geo.heads
    if cons is None:
        cons = torch.tensor(A, dtype=torch.long, device=idx.device)
    a2, a3 = cons[0, :K], cons[1, :K]
    x, p1, p2 = x[..., None], p1[..., None], p2[..., None]
    u2 = (p1 * a2[:, 0]) ^ (x * a2[:, 1])
    u3 = (p2 * a3[:, 0]) ^ (p1 * a3[:, 1]) ^ (x * a3[:, 2])
    if geo.kind == "paged":
        S, (P2, P3) = geo.slots, geo.pages
        R = K * S
        off = torch.arange(0, R, S, device=idx.device)
        rows = torch.cat([(b % P2)[..., None] * R + off + u2 % S,
                          P2 * R + (c % P3)[..., None] * R + off + u3 % S], -1)
    else:
        sizes = torch.tensor(geo.sizes, dtype=torch.long, device=idx.device)
        bases = torch.cumsum(sizes, 0) - sizes
        rows = torch.cat([bases[:K] + u2 % sizes[:K], bases[K:] + u3 % sizes[K:]], -1)
    return rows, torch.cat([_sign(u2, M[5]), _sign(u3, M[6])], -1)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * self.weight


class NgramMemory(nn.Module):
    """Signed rows -> e: summed for flat, concatenated and decoded by W_V for multi-head.

    scalar gate:  x + g_j * e
    context gate: x + g_j * sigmoid(q_j(x) . k / sqrt(d)) * e, k = RMSNorm(W_K e)   (Engram)
    The table starts at zero, so at step 0 the model matches the no-table model."""

    def __init__(self, geo: Geometry, d: int, vocab: int, n_inject: int, gate: str = "scalar",
                 seed: int = 0, init_std: float = 0.02):
        super().__init__()
        self.geo, self.d, self.vocab, self.gate_kind = geo, d, vocab, gate
        self.table = nn.Parameter(torch.zeros(geo.rows, geo.width))
        gen = torch.Generator().manual_seed(12345)
        self.register_buffer("signs", (torch.randint(0, 2, (SIGN_ROWS, geo.width), generator=gen)
                                       * 2 - 1).float(), persistent=False)
        self.register_buffer("cons", torch.tensor(A, dtype=torch.long), persistent=False)
        self.value = None
        if geo.kind != "flat":
            c = 2 * geo.heads * geo.width
            self.value = nn.Linear(c, d, bias=False)
            g = torch.Generator().manual_seed(seed * 1000 + 4242)
            q, r = torch.linalg.qr(torch.randn(max(d, c), min(d, c), generator=g, dtype=torch.float64))
            sgn = torch.sign(torch.diagonal(r))
            q = q * torch.where(sgn == 0, torch.ones_like(sgn), sgn)
            w = (q if d >= c else q.T) * math.sqrt(2 * d / c)
            with torch.no_grad():
                self.value.weight.copy_(w.float())
        self.gates = nn.Parameter(torch.full((n_inject,), (2.0 if gate == "context" else 1.0) / n_inject))
        if gate == "context":
            self.key = nn.Linear(d, d, bias=False)
            g = torch.Generator().manual_seed(seed * 1000 + 4343)
            with torch.no_grad():
                self.key.weight.copy_(torch.randn(d, d, generator=g) * init_std)
            self.k_norm = RMSNorm(d)
            self.q_norms = nn.ModuleList([RMSNorm(d) for _ in range(n_inject)])

    def lookup(self, idx: torch.Tensor, table: torch.Tensor | None = None) -> torch.Tensor:
        rows, sidx = addresses(idx, self.geo, self.vocab, self.cons)
        h = F.embedding(rows, self.table if table is None else table, sparse=table is None)
        h = h * self.signs[sidx]
        if self.value is None:
            return h.sum(-2)
        return self.value(h.flatten(-2))

    def key_of(self, e: torch.Tensor):
        return self.k_norm(self.key(e)) if self.gate_kind == "context" else None

    def inject(self, j: int, x: torch.Tensor, e: torch.Tensor, k: torch.Tensor | None):
        if k is None:
            return x + self.gates[j] * e
        a = torch.sigmoid((self.q_norms[j](x).float() * k.float()).sum(-1, keepdim=True) / math.sqrt(self.d))
        return x + self.gates[j] * a.to(x.dtype) * e

    @torch.no_grad()
    def quantized_table(self, block: int = 64) -> torch.Tensor:
        """NF4 copy of the table with one fp16 absmax scale per 64 values (the deploy format)."""
        t = self.table.detach()
        flat = t.reshape(-1, block)
        out = torch.empty_like(flat)
        levels = torch.tensor(NF4, device=t.device)
        mids = (levels[1:] + levels[:-1]) / 2
        for i in range(0, flat.shape[0], 1 << 18):
            x = flat[i:i + (1 << 18)]
            scale = x.abs().amax(1, keepdim=True).half().float()
            z = x / scale.clamp(min=1e-12)
            out[i:i + (1 << 18)] = levels[torch.bucketize(z, mids)] * scale
        return out.reshape_as(t)


class RowRMS:
    """Row-wise RMS update on touched rows: x -= lr * g / sqrt(v / w), one second moment per row,
    missed steps replayed as decay."""

    def __init__(self, table: torch.Tensor, beta2: float = 0.95, eps: float = 1e-8):
        n = table.shape[0]
        self.beta2, self.eps = beta2, eps
        self.v = torch.zeros(n, device=table.device)
        self.wsum = torch.zeros(n, device=table.device)
        self.t_last = torch.zeros(n, dtype=torch.int32, device=table.device)

    @torch.no_grad()
    def step(self, table: nn.Parameter, lr: float, t: int) -> int:
        g = table.grad
        if g is None:
            return 0
        g = g.coalesce()
        ids, vals = g.indices()[0], g.values()
        decay = (self.beta2 ** (t - self.t_last[ids]).double()).float()
        v = self.v[ids] * decay + (1 - self.beta2) * vals.square().mean(1)
        ws = self.wsum[ids] * decay + (1 - self.beta2)
        table.index_add_(0, ids, -lr * vals / ((v / ws).sqrt() + self.eps)[:, None])
        self.v[ids], self.wsum[ids] = v, ws
        self.t_last.index_fill_(0, ids, t)
        table.grad = None
        return ids.numel()
