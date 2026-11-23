"""Session-level inference for the two full campaigns.

The 150 runs inside one cell share one governor state, so treating them as
150 independent observations inflates significance. This script collapses
each cell to 5 session medians per device, then reports:

  - per-cell effect sizes with a session-bootstrap CI (n=5 sessions/side)
  - precision-level Mann-Whitney over 30 session medians per side
  - Holm correction over the three precision tests

Output: research/analysis_v4_session.json
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
import pandas as pd
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data baru", "result")
OUT = os.path.join(ROOT, "research", "analysis_v4_session.json")
RNG = np.random.default_rng(20261124)
PREC = ["fp16", "8bit", "4bit"]

d50 = pd.read_csv(os.path.join(RES, "rtx5050_results.csv"))
d4 = pd.read_csv(os.path.join(RES, "t4_colab_results.csv"))
for d in (d50, d4):
    d["model_short"] = d.model.str.split("/").str[-1]


def jnum(x):
    if x is None:
        return None
    v = float(x)
    if math.isnan(v) or math.isinf(v):
        return None
    if v != 0.0 and abs(v) < 1e-3:
        return float(f"{v:.3g}")
    return round(v, 4)


def sess_med(df, col):
    return (df.groupby(["model_short", "precision", "prompt_tokens", "session_id"])[col]
              .median().reset_index())


a = sess_med(d50, "decode_tok_per_s").rename(columns={"decode_tok_per_s": "d50"})
b = sess_med(d4, "decode_tok_per_s").rename(columns={"decode_tok_per_s": "d4"})
at = sess_med(d50, "ttft_ms").rename(columns={"ttft_ms": "t50"})
bt = sess_med(d4, "ttft_ms").rename(columns={"ttft_ms": "t4"})
key = ["model_short", "precision", "prompt_tokens"]
dec = a.merge(b, on=key + ["session_id"])
ttf = at.merge(bt, on=key + ["session_id"])


def boot_ratio_session(x, y, n=20000):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ix = RNG.integers(0, x.size, size=(n, x.size))
    iy = RNG.integers(0, y.size, size=(n, y.size))
    r = np.median(y[iy], axis=1) / np.median(x[ix], axis=1)
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def cliffs(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return (int(np.sum(a[:, None] < b[None, :])) -
            int(np.sum(a[:, None] > b[None, :]))) / (a.size * b.size)


# ---------------------------------------------------------------- per cell
cells = []
for k, g in dec.groupby(key):
    h = ttf[(ttf.model_short == k[0]) & (ttf.precision == k[1])
            & (ttf.prompt_tokens == k[2])]
    dlo, dhi = boot_ratio_session(g.d50, g.d4)
    tlo, thi = boot_ratio_session(h.t50, h.t4)
    # MWU on 5-vs-5 sessions: smallest attainable two-sided p is 0.0079
    pd_ = stats.mannwhitneyu(g.d50, g.d4, alternative="two-sided").pvalue
    pt = stats.mannwhitneyu(h.t50, h.t4, alternative="two-sided").pvalue
    cells.append(dict(
        model=k[0], precision=k[1], ctx=int(k[2]), n_sessions=len(g),
        decode_med_5050=jnum(g.d50.median()), decode_med_t4=jnum(g.d4.median()),
        decode_ratio=jnum(g.d4.median() / g.d50.median()),
        decode_ci_lo=jnum(dlo), decode_ci_hi=jnum(dhi),
        ttft_med_5050=jnum(h.t50.median()), ttft_med_t4=jnum(h.t4.median()),
        ttft_ratio=jnum(h.t4.median() / h.t50.median()),
        ttft_ci_lo=jnum(tlo), ttft_ci_hi=jnum(thi),
        decode_p_session=jnum(pd_), ttft_p_session=jnum(pt),
        decode_delta=jnum(cliffs(g.d50, g.d4)),
        ttft_delta=jnum(cliffs(h.t50, h.t4))))
cells.sort(key=lambda r: (PREC.index(r["precision"]), r["model"], r["ctx"]))

for f in ("decode_p_session", "ttft_p_session"):
    p = np.array([c[f] for c in cells])
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    for c, v in zip(cells, adj):
        c[f + "_holm"] = jnum(v)

# ---------------------------------------------------------- precision level
prec = []
for pr in PREC:
    g = dec[dec.precision == pr]
    h = ttf[ttf.precision == pr]
    pd_ = stats.mannwhitneyu(g.d50, g.d4, alternative="two-sided").pvalue
    pt = stats.mannwhitneyu(h.t50, h.t4, alternative="two-sided").pvalue
    dlo, dhi = boot_ratio_session(g.d50, g.d4)
    tlo, thi = boot_ratio_session(h.t50, h.t4)
    prec.append(dict(
        precision=pr, n_sessions_per_side=len(g),
        decode_ratio=jnum(g.d4.median() / g.d50.median()),
        decode_ci_lo=jnum(dlo), decode_ci_hi=jnum(dhi),
        decode_p=jnum(pd_), decode_delta=jnum(cliffs(g.d50, g.d4)),
        ttft_ratio=jnum(h.t4.median() / h.t50.median()),
        ttft_ci_lo=jnum(tlo), ttft_ci_hi=jnum(thi),
        ttft_p=jnum(pt), ttft_delta=jnum(cliffs(h.t50, h.t4))))
for f in ("decode_p", "ttft_p"):
    p = np.array([r[f] for r in prec])
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    for r, v in zip(prec, adj):
        r[f + "_holm"] = jnum(v)

out = {
    "unit": "session median (5 sessions per cell per device)",
    "note": "cell-level p floored at 0.0079 for 5-vs-5; 0.1429 after Holm over 18",
    "cells": cells,
    "by_precision": prec,
    "cell_level_significant_after_holm_decode": int(
        sum(1 for c in cells if c["decode_p_session_holm"] < 0.05)),
    "cell_level_significant_after_holm_ttft": int(
        sum(1 for c in cells if c["ttft_p_session_holm"] < 0.05)),
    "precision_level_significant_after_holm_decode": int(
        sum(1 for r in prec if r["decode_p_holm"] < 0.05)),
    "precision_level_significant_after_holm_ttft": int(
        sum(1 for r in prec if r["ttft_p_holm"] < 0.05)),
}
with open(OUT, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1, ensure_ascii=False)
print("wrote", OUT)
for r in prec:
    print(r)
print("cell significant after Holm: decode", out["cell_level_significant_after_holm_decode"],
      "ttft", out["cell_level_significant_after_holm_ttft"])
