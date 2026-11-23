"""Regenerate every figure in the manuscript from data baru.

Two full campaigns (RTX 5050, Tesla T4), 2.700 runs each, plus the 9-column
pilot on both devices. All numbers come from the CSVs; the only hard-coded
values are vendor declared bandwidth and the model geometry used for the
roofline ceiling.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, NullFormatter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "data baru", "result")
FIG = os.path.join(ROOT, "manuscript", "FIGURES")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman"],
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

COLORS = {"fp16": "#1f4e79", "8bit": "#b5651d", "4bit": "#2e7d5b"}
MARKERS = {"fp16": "o", "8bit": "s", "4bit": "^"}
LABELS = {"fp16": "fp16", "8bit": "INT8 (LLM.int8)", "4bit": "NF4 4-bit"}
MODEL_TITLES = {"meta-llama/Llama-3.2-1B": "Llama-3.2-1B",
                "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B"}
DEV_TITLES = {"rtx5050": "RTX 5050 Laptop", "t4": "Tesla T4"}
DEV_COLORS = {"rtx5050": "#1f4e79", "t4": "#8c2f39"}
DEV_LS = {"rtx5050": "-", "t4": "--"}
DEV_MARK = {"rtx5050": "o", "t4": "s"}
BW = {"rtx5050": 384.0, "t4": 320.0}
BASE = {"rtx5050": 1500.0, "t4": 700.0}
BITS = {"fp16": 16.0, "8bit": 8.0, "4bit": 4.5}
MODELS = {"meta-llama/Llama-3.2-1B": dict(params=1.235, layers=16, kv=8, hd=64),
          "Qwen/Qwen2.5-1.5B": dict(params=1.543, layers=28, kv=2, hd=128)}
PREC_ORDER = ["fp16", "8bit", "4bit"]

frames = {}
for key, fname in (("rtx5050", "rtx5050_results.csv"),
                   ("t4", "t4_colab_results.csv")):
    df = pd.read_csv(os.path.join(RES, fname))
    df["model_short"] = df.model.str.split("/").str[-1]
    df["on_base"] = df.sm_clock_post_mhz < BASE[key]
    df["prefill_share_pct"] = df.ttft_ms / df.e2e_ms * 100
    frames[key] = df

pilot5050 = pd.read_csv(os.path.join(RES, "local_rtx5050_benchmark_results.csv"))
pilot_t4 = pd.read_csv(os.path.join(RES, "colab_t4_benchmark_results.csv"))
RNG = np.random.default_rng(7)


def boot_ci(v, stat=np.median, n=4000):
    v = np.asarray(v, float)
    idx = RNG.integers(0, v.size, size=(n, v.size))
    d = stat(v[idx], axis=1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def saved(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p)
    plt.close(fig)
    print("wrote", name, os.path.getsize(p), "bytes")


def cell_medians(df):
    return (df.groupby(["model", "precision", "prompt_tokens"])
              .agg(dec=("decode_tok_per_s", "median"),
                   ttft=("ttft_ms", "median"))
              .reset_index())


# ---------------------------------------------------------------- fig 01/02
for metric, ylab, name in [
    ("ttft_ms", "TTFT (ms)", "fig01_ttft_context.png"),
    ("decode_tok_per_s", "Throughput decode (token/detik)",
     "fig02_decode_context.png"),
]:
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4), sharex=True)
    for ax, model in zip(axes, MODELS):
        for prec in PREC_ORDER:
            for dev in ("rtx5050", "t4"):
                h = frames[dev][(frames[dev].model == model)
                                & (frames[dev].precision == prec)]
                med = h.groupby("prompt_tokens")[metric].median()
                lo, hi = zip(*[boot_ci(h[h.prompt_tokens == c][metric])
                               for c in med.index])
                ax.plot(med.index, med.values, color=COLORS[prec],
                        linestyle=DEV_LS[dev], marker=DEV_MARK[dev],
                        markersize=4.5, linewidth=1.3,
                        label=f"{LABELS[prec]} — {DEV_TITLES[dev]}")
                ax.fill_between(med.index, lo, hi, color=COLORS[prec],
                                alpha=0.10, linewidth=0)
        ax.set_xscale("log")
        ax.set_xticks([128, 512, 1024])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Panjang prompt (token)")
        ax.set_title(MODEL_TITLES[model], loc="left", fontweight="bold")
        ax.grid(True, color="#e6e6e6", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel(ylab)
    axes[1].set_ylabel("")
    if metric == "decode_tok_per_s":
        axes[0].set_ylim(0, 72)
        axes[1].set_ylim(0, 72)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=7.6,
               bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout()
    saved(fig, name)

# ---------------------------------------------------------------- fig 03
fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.4))
groups = [(m, d) for m in MODELS for d in ("rtx5050", "t4")]
x = np.arange(len(groups))
w = 0.26
for j, prec in enumerate(PREC_ORDER):
    dec, dec_e, vram = [], [], []
    for model, dev in groups:
        h = frames[dev][(frames[dev].model == model)
                        & (frames[dev].precision == prec)]
        dec.append(h.decode_tok_per_s.mean())
        dec_e.append(h.decode_tok_per_s.std() / np.sqrt(len(h)))
        vram.append(h.peak_vram_mb.mean())
    axes[0].bar(x + (j - 1) * w, dec, w, yerr=dec_e, capsize=2,
                color=COLORS[prec], edgecolor="white", linewidth=0.6,
                label=LABELS[prec], error_kw=dict(elinewidth=0.8))
    axes[1].bar(x + (j - 1) * w, vram, w, color=COLORS[prec],
                edgecolor="white", linewidth=0.6, label=LABELS[prec])
axes[0].set_ylabel("Decode rerata (token/detik)")
axes[1].set_ylabel("VRAM puncak rerata (MB)")
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels([f"{MODEL_TITLES[m]}\n{DEV_TITLES[d]}" for m, d in groups],
                       fontsize=8.2)
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_title("Kecepatan", loc="left", fontweight="bold")
axes[1].set_title("Jejak memori", loc="left", fontweight="bold")
axes[0].legend(loc="upper right", fontsize=7.6)
fig.tight_layout()
saved(fig, "fig03_precision_effect.png")

# ---------------------------------------------------------------- fig 04
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6),
                         gridspec_kw={"width_ratios": [1.15, 1]})
combos = [(d, p) for d in ("rtx5050", "t4") for p in PREC_ORDER]
for i, (dev, prec) in enumerate(combos):
    v = frames[dev][frames[dev].precision == prec].sm_clock_post_mhz.values
    jit = RNG.normal(0, 0.07, size=v.size)
    axes[0].scatter(np.full(v.size, i) + jit, v, s=1.4, alpha=0.16,
                    color=COLORS[prec], linewidths=0)
    axes[0].plot([i - 0.28, i + 0.28], [np.median(v)] * 2,
                 color="#111111", linewidth=1.6)
axes[0].set_xticks(range(len(combos)))
axes[0].set_xticklabels([f"{LABELS[p].split()[0]}\n"
                         f"{'RTX 5050' if d == 'rtx5050' else 'T4'}"
                         for d, p in combos], fontsize=8)
axes[0].set_ylabel("Clock SM sesudah run (MHz)")
axes[0].set_title("Sebaran clock per presisi dan perangkat",
                  loc="left", fontweight="bold")
for dev in ("rtx5050", "t4"):
    for prec in PREC_ORDER:
        h = frames[dev][frames[dev].precision == prec]
        axes[1].scatter(h.sm_clock_post_mhz, h.decode_tok_per_s, s=2.2,
                        alpha=0.22, color=COLORS[prec],
                        marker=DEV_MARK[dev], linewidths=0,
                        label=f"{LABELS[prec].split()[0]} — {DEV_TITLES[dev]}")
axes[1].set_xlabel("Clock SM (MHz)")
axes[1].set_ylabel("Decode (token/detik)")
axes[1].set_title("Clock terhadap throughput", loc="left", fontweight="bold")
for ax in axes:
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
saved(fig, "fig04_clock_regimes.png")

# ---------------------------------------------------------------- fig 05
fig, ax = plt.subplots(figsize=(6.6, 3.8))
for dev in ("rtx5050", "t4"):
    for prec in PREC_ORDER:
        h = frames[dev][frames[dev].precision == prec]
        for (model, ctx), g in h.groupby(["model", "prompt_tokens"]):
            spec = MODELS[model]
            w_b = spec["params"] * 1e9 * BITS[prec] / 8.0
            kv_b = 2 * spec["layers"] * spec["kv"] * spec["hd"] * ctx * 2
            gb = (w_b + kv_b) / 1e9
            dec = g.decode_tok_per_s.median()
            ax.scatter(gb, dec * gb, s=34, color=COLORS[prec],
                       marker=DEV_MARK[dev], alpha=0.85,
                       edgecolor="white", linewidth=0.5)
for dev, ls in (("rtx5050", "-"), ("t4", "--")):
    ax.axhline(BW[dev], color=DEV_COLORS[dev], linestyle=ls, linewidth=1.1)
    ax.text(0.63, BW[dev] * 1.04, f"batas {DEV_TITLES[dev]} {BW[dev]:.0f} GB/s",
            fontsize=7.6, color=DEV_COLORS[dev])
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(0.55, 4.2)
ax.set_ylim(12, 470)
ax.set_xticks([0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0])
ax.set_yticks([15, 20, 30, 50, 100, 200, 400])
ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("Byte yang dibaca per token (GB, skala log)")
ax.set_ylabel("Bandwidth terpakai (GB/s, skala log)")
ax.set_title("Semua konfigurasi jauh di bawah batas terpasang",
             loc="left", fontweight="bold")
ax.grid(True, which="major", color="#e6e6e6", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], color=COLORS[p], marker="o", linestyle="",
                      markersize=6, label=LABELS[p]) for p in PREC_ORDER]
handles += [plt.Line2D([], [], color="#555555", marker=DEV_MARK[d],
                       linestyle="", markersize=6, label=DEV_TITLES[d])
            for d in ("rtx5050", "t4")]
ax.legend(handles=handles, loc="lower right", fontsize=7.6, ncol=1)
fig.tight_layout()
saved(fig, "fig05_roofline.png")

# ---------------------------------------------------------------- fig 06
fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.4),
                         gridspec_kw={"width_ratios": [1, 1.1]})
flag = frames["rtx5050"][(frames["rtx5050"].model == "meta-llama/Llama-3.2-1B")
                         & (frames["rtx5050"].precision == "fp16")
                         & (frames["rtx5050"].prompt_tokens == 1024)]
sess = sorted(flag.session_id.unique())
share, cv, temps = [], [], []
for s in sess:
    h = flag[flag.session_id == s]
    share.append(h.on_base.mean() * 100)
    cv.append(h.decode_tok_per_s.std() / h.decode_tok_per_s.mean() * 100)
    temps.append(h.temp_post_c.max())
xs = np.arange(len(sess))
bars = axes[0].bar(xs, share, width=0.52,
                   color=["#c0504d"] * 3 + ["#4f81bd"] * 2,
                   edgecolor="white", linewidth=0.8)
for b, v, c, t in zip(bars, share, cv, temps):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.9, f"{v:.0f}%",
                 ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    axes[0].text(b.get_x() + b.get_width() / 2, 1.2, f"CV {c:.1f}%\n{t:.0f} °C",
                 ha="center", va="bottom", fontsize=7.4, color="white")
axes[0].axhline(10, color="#555555", linestyle=":", linewidth=0.9)
axes[0].set_xticks(xs)
axes[0].set_xticklabels(sess)
axes[0].set_ylabel("Run pada base clock, %")
axes[0].set_ylim(0, 48)
axes[0].set_title("RTX 5050, Llama fp16 konteks 1.024", loc="left",
                  fontweight="bold")

for dev in ("rtx5050", "t4"):
    h = frames[dev]
    cvs = (h.groupby(["model_short", "precision", "prompt_tokens", "session_id"])
             .decode_tok_per_s.apply(lambda x: x.std() / x.mean() * 100)
             .sort_values().values)
    axes[1].plot(np.arange(1, len(cvs) + 1), cvs, color=DEV_COLORS[dev],
                 linewidth=1.6, label=f"{DEV_TITLES[dev]} (maks {cvs.max():.1f}%)")
axes[1].axhline(10, color="#555555", linestyle=":", linewidth=0.9)
axes[1].text(1, 10.4, "ambang gerbang 10%", fontsize=7.4, color="#555555")
axes[1].set_xlabel("Sesi-konfigurasi (terurut)")
axes[1].set_ylabel("CV decode dalam sesi (%)")
axes[1].set_ylim(0, 17)
axes[1].set_title("Sebaran CV dalam sesi", loc="left", fontweight="bold")
axes[1].legend(loc="upper left", fontsize=7.6)
for ax in axes:
    ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
saved(fig, "fig06_session_stability.png")

# ---------------------------------------------------------------- fig 07
key = ["model", "precision", "prompt_tokens"]
m50 = cell_medians(frames["rtx5050"]).rename(columns={"dec": "d50", "ttft": "t50"})
m4 = cell_medians(frames["t4"]).rename(columns={"dec": "d4", "ttft": "t4"})
cc = m50.merge(m4, on=key)
cc["model_short"] = cc.model.str.split("/").str[-1]
cc["rt"] = cc.t4 / cc.t50
cc["rd"] = cc.d4 / cc.d50
cc["row"] = cc.model_short + " " + cc.precision + " " + cc.prompt_tokens.astype(str)
order = (cc.sort_values(["model_short", "precision", "prompt_tokens"])
           .assign(pp=lambda d: d.precision.map(PREC_ORDER.index))
           .sort_values(["pp", "model_short", "prompt_tokens"])
           .row.tolist())
cc["y"] = cc.row.map({r: i for i, r in enumerate(order)})

fig, axes = plt.subplots(1, 2, figsize=(7.8, 4.4), sharey=True)
for ax, col, title in ((axes[0], "rt", "TTFT: T4 / RTX 5050"),
                       (axes[1], "rd", "Decode: T4 / RTX 5050")):
    for prec in PREC_ORDER:
        h = cc[cc.precision == prec]
        ax.scatter(h[col], h.y, s=32, color=COLORS[prec], marker=MARKERS[prec],
                   edgecolor="white", linewidth=0.5, label=LABELS[prec], zorder=3)
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlim(0.55, 2.0)
    ax.set_xticks([0.6, 0.8, 1.0, 1.3, 1.6])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Rasio median")
    ax.grid(True, axis="x", color="#e8e8e8", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].set_yticks(cc.y)
axes[0].set_yticklabels(cc.row, fontsize=7.2)
axes[0].text(0.57, len(order) - 0.6, "T4 lebih cepat", fontsize=7.2,
             color="#333333")
axes[0].text(1.05, len(order) - 0.6, "5050 lebih cepat", fontsize=7.2,
             color="#333333")
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, 1.02),
               ncol=3, fontsize=7.4)
fig.tight_layout()
saved(fig, "fig07_cross_device.png")

# ---------------------------------------------------------------- fig 08
fig, ax = plt.subplots(figsize=(6.6, 3.6))
xs = {"meta-llama/Llama-3.2-1B": 0, "Qwen/Qwen2.5-1.5B": 1}
for dev, hatch in (("rtx5050", ""), ("t4", "//")):
    for j, prec in enumerate(PREC_ORDER):
        vals, errs, pos = [], [], []
        for model, xi in xs.items():
            h = frames[dev][(frames[dev].model == model)
                            & (frames[dev].precision == prec)
                            & (frames[dev].prompt_tokens == 128)]
            lat = 1000.0 / h.decode_tok_per_s
            vals.append(lat.median())
            errs.append(lat.std() / np.sqrt(len(lat)))
            pos.append(xi + (j - 1) * 0.22 + (-0.055 if dev == "t4" else 0.055))
        ax.bar(pos, vals, 0.10, yerr=errs, capsize=2, color=COLORS[prec],
               alpha=0.95 if dev == "rtx5050" else 0.55, edgecolor="white",
               linewidth=0.5, hatch=hatch,
               error_kw=dict(elinewidth=0.7))
ax.set_xticks([0, 1])
ax.set_xticklabels(["Llama-3.2-1B\n16 lapisan, 1,24 miliar parameter",
                    "Qwen2.5-1.5B\n28 lapisan, 1,54 miliar parameter"],
                   fontsize=8.6)
ax.set_ylabel("Latensi per token (ms)")
ax.set_title("Latensi menurut presisi dan perangkat, konteks 128",
             loc="left", fontweight="bold")
ax.grid(True, axis="y", color="#e6e6e6", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[p], label=LABELS[p])
           for p in PREC_ORDER]
handles += [plt.Rectangle((0, 0), 1, 1, facecolor="#999999", alpha=0.55,
                          hatch="//", edgecolor="white",
                          label="T4 (gelap: RTX 5050)")]
ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.01),
          ncol=4, fontsize=7.4)
ax.set_ylim(0, 142)
fig.tight_layout()
saved(fig, "fig08_layer_scaling.png")

# ---------------------------------------------------------------- fig 09
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for dev, pilot, mark in (("rtx5050", pilot5050, "o"), ("t4", pilot_t4, "s")):
    full = frames[dev]
    for prec in PREC_ORDER:
        f = (full[full.precision == prec]
             .groupby(["model", "prompt_tokens"]).decode_tok_per_s.median())
        p = (pilot[pilot.precision == prec]
             .groupby(["model", "prompt_tokens"]).throughput_tokens_per_sec.median())
        ax.scatter(p.values, f.loc[p.index].values, s=34, color=COLORS[prec],
                   marker=mark, edgecolor="white", linewidth=0.5,
                   alpha=0.95 if dev == "rtx5050" else 0.65)
lim = [3, 75]
ax.plot(lim, lim, color="#333333", linestyle="--", linewidth=1.0)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("Decode protokol pilot 9 kolom (token/detik)")
ax.set_ylabel("Decode protokol lengkap 35 kolom (token/detik)")
ax.set_title("Protokol menggeser angka pada kedua perangkat",
             loc="left", fontweight="bold")
ax.grid(True, color="#e6e6e6", linewidth=0.6)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)
handles = [plt.Line2D([], [], color=COLORS[p], marker="o", linestyle="",
                      markersize=6, label=LABELS[p]) for p in PREC_ORDER]
handles += [plt.Line2D([], [], color="#555555", marker=DEV_MARK[d],
                       linestyle="", markersize=6, label=DEV_TITLES[d])
            for d in ("rtx5050", "t4")]
handles.append(plt.Line2D([], [], color="#333333", linestyle="--",
                          label="identitas (tidak berubah)"))
ax.legend(handles=handles, loc="upper left", fontsize=7.4)
fig.tight_layout()
saved(fig, "fig09_instrument_drift.png")

print("done")
