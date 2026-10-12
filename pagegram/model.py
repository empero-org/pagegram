"""Small decoder for the memory study: pre-norm RMSNorm, RoPE attention, GELU MLP, untied embeddings.
The memory vector is added before every `every`-th block."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .memory import Geometry, NgramMemory, RMSNorm


@dataclass
class Config:
    vocab: int = 65536
    d: int = 384
    layers: int = 6
    heads: int = 6
    seq: int = 1024
    mlp: int = 4
    every: int = 4
    gate: str = "scalar"
    rope_base: float = 10000.0
    init_std: float = 0.02


def rope_tables(T: int, dh: int, base: float, device):
    inv = 1.0 / (base ** (torch.arange(0, dh, 2, device=device, dtype=torch.float32) / dh))
    f = torch.outer(torch.arange(T, device=device, dtype=torch.float32), inv)
    return f.cos(), f.sin()


def apply_rope(x, cos, sin):
    h = x.shape[-1] // 2
    x1, x2 = x[..., :h].float(), x[..., h:].float()
    c, s = cos[None, None], sin[None, None]
    return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], -1).to(x.dtype)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d
        self.h, self.dh = cfg.heads, d // cfg.heads
        self.n1, self.n2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.up = nn.Linear(d, cfg.mlp * d, bias=False)
        self.down = nn.Linear(cfg.mlp * d, d, bias=False)

    def forward(self, x, cos, sin):
        B, T, d = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, d))
        return x + self.down(F.gelu(self.up(self.n2(x))))


def _ce_chunk(x, w, t):
    return F.cross_entropy(F.linear(x, w).float(), t, reduction="sum")


class LM(nn.Module):
    def __init__(self, cfg: Config, geo: Geometry, seed: int = 0, memory_seed: int | None = None):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab, cfg.d)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.layers))
        self.norm = RMSNorm(cfg.d)
        self.head = nn.Linear(cfg.d, cfg.vocab, bias=False)
        self.inject = [i for i in range(cfg.layers) if i % cfg.every == 0]
        self.reset_core(seed)
        self.memory = None
        if geo.kind != "none":
            self.memory = NgramMemory(geo, cfg.d, cfg.vocab, len(self.inject), cfg.gate,
                                      seed if memory_seed is None else memory_seed, cfg.init_std)
        self.compiled = None

    @torch.no_grad()
    def reset_core(self, seed: int):
        """Init the core from a seeded generator in module order, so all arms of a seed share it."""
        g = torch.Generator().manual_seed(seed)
        std, out_std = self.cfg.init_std, self.cfg.init_std / math.sqrt(2 * self.cfg.layers)
        for name, p in self.named_parameters():
            if name.endswith("n1.weight") or name.endswith("n2.weight") or name == "norm.weight":
                p.fill_(1.0)
            else:
                s = out_std if name.endswith(("proj.weight", "down.weight")) else std
                p.copy_(torch.randn(p.shape, generator=g) * s)

    def core_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("memory.")]

    def compile_blocks(self):
        self.compiled = [torch.compile(b, dynamic=False) for b in self.blocks]

    def forward(self, idx, targets, table=None, chunk: int = 4096):
        B, T = idx.shape
        cos, sin = rope_tables(T, self.cfg.d // self.cfg.heads, self.cfg.rope_base, idx.device)
        e = k = None
        if self.memory is not None:
            e = self.memory.lookup(idx, table)
            k = self.memory.key_of(e)
        x = self.embed(idx)
        blocks = self.compiled or self.blocks
        for i, blk in enumerate(blocks):
            if e is not None and i in self.inject:
                x = self.memory.inject(self.inject.index(i), x, e.to(x.dtype), k)
            x = blk(x, cos, sin)
        x = self.norm(x).flatten(0, 1)
        t = targets.flatten()
        total = x.new_zeros((), dtype=torch.float32)
        for i in range(0, x.shape[0], chunk):
            xs, ts = x[i:i + chunk], t[i:i + chunk]
            if torch.is_grad_enabled():
                total = total + checkpoint(_ce_chunk, xs, self.head.weight, ts, use_reentrant=False)
            else:
                total = total + _ce_chunk(xs, self.head.weight, ts)
        return total / t.numel()
