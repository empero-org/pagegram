"""uint16 token shards: seeded random training windows and fixed held-out windows."""
from __future__ import annotations

import glob
import os

import numpy as np
import torch


class Shards:
    def __init__(self, root: str):
        files = sorted(glob.glob(os.path.join(root, "**", "*.bin"), recursive=True))
        if not files:
            raise FileNotFoundError(f"no .bin shards under {root}")
        self.maps = [np.memmap(f, dtype=np.uint16, mode="r") for f in files]
        self.sizes = np.array([len(m) for m in self.maps], dtype=np.int64)
        self.total = int(self.sizes.sum())

    def windows(self, rng: np.random.Generator, n: int, T: int) -> torch.Tensor:
        """n windows of T + 1 tokens; files picked in proportion to their size."""
        p = np.maximum(self.sizes - T - 1, 0).astype(np.float64)
        f = rng.choice(len(self.maps), size=n, p=p / p.sum())
        out = np.empty((n, T + 1), dtype=np.int64)
        for i, fi in enumerate(f):
            s = int(rng.integers(0, self.sizes[fi] - T - 1))
            out[i] = self.maps[fi][s:s + T + 1]
        return torch.from_numpy(out)

    def fixed(self, n: int, T: int) -> torch.Tensor:
        """n fixed held-out windows, the same for every run."""
        rng = np.random.default_rng(20261012)
        return self.windows(rng, n, T)
