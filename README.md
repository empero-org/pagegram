# PageGram

Code and results for the Empero AI technical report

> **PageGram: Multi-Head N-gram Memory at the Read Cost of a Flat Table**
> Leon Lehmann, Empero AI Research, October 2026. ([PDF](paper.pdf))

Multi-head n-gram tables hash every n-gram with K independent heads, one table per head. On a device the
table sits in flash, so a token costs one random read per head: 16 reads for 8 heads and two n-gram
orders. PageGram keeps the heads but puts all K head-rows of an order in one page: the n-gram picks a
page, each head picks one of S slots inside it. A token reads 2 pages, the same as a flat table.

At equal size (200M parameters, small decoder), held-out loss:

| table | flash reads / token | 60M tokens (5 seeds) | 120M tokens (3 seeds) |
|---|---|---|---|
| none | 0 | 5.484 | 4.220 |
| flat | 2 | 5.151 | 4.159 |
| Engram-style (8 heads, global) | 16 | 5.225 | 4.125 |
| **PageGram, S = 4** | **2** | **5.084** (better than flat and Engram-style on 5/5 seeds) | 4.136 |

PageGram leads early in training and matches Engram-style addressing later, while reading 8x less from
flash (0.99 vs 9.18 reads per token with a 1 GiB page cache on a 20B table). Per head its collisions are
as frequent as Engram-style addressing's; they are correlated across heads, and S sets how much.

## Layout

```
pagegram/memory.py      hashing, the three table layouts, the row-wise RMS optimiser, NF4 readout
pagegram/model.py       the small decoder
pagegram/train.py       one run
pagegram/suite.py       grids of runs on one GPU
pagegram/analyze.py     per-arm tables over seeds
pagegram/test_harness.py  self-checks (run these first)
pagegram/flash.py, iosim.c  flash simulator: blocks per token, LRU cache misses, direct-read throughput
scripts/collisions.py   per-head collision statistics on real tokens
scripts/runq.py         run a list of jobs, two at a time
results/                every run behind the report (summary.json and log.jsonl per run)
```

## Running it

```bash
pip install -e .
python -m pagegram.test_harness
python -m pagegram.train --data DATA --arm paged --slots 4 --seed 0 --tokens 6e7 --out runs/paged_s4
python -m pagegram.flash --holdout DATA/holdout --out runs/flash --bench-dir /path/on/nvme
```

`DATA` needs two folders of uint16 token shards, `S1/` for training and `holdout/` for evaluation. Token
id 2 is the document separator, and the vocabulary is 65,536. Our runs used a sample of a private
multilingual pretraining mixture; any tokenised corpus in this format works.

Arms: `none`, `flat`, `global` (Engram-style), `paged`, `paged_ctx` (with Engram's context gate),
`paged_q` (a quarter of the table). `--slots` sets S for the paged arms.

## Citation

```bibtex
@techreport{lehmann2026pagegram,
  title       = {PageGram: Multi-Head N-gram Memory at the Read Cost of a Flat Table},
  author      = {Lehmann, Leon},
  institution = {Empero AI Research},
  type        = {Empero AI Technical Report},
  year        = {2026},
  month       = {October},
  url         = {https://github.com/empero-org/pagegram}
}
```

## License

Apache License 2.0, see [LICENSE](LICENSE).
