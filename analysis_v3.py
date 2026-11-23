"""Rebuild every number the manuscript needs from the new datasets.

Inputs (all under ./data baru/result):
  rtx5050_results.csv            2700 runs, 35 columns, RTX 5050 full protocol
  t4_colab_results.csv           2700 runs, 35 columns, Tesla T4 full protocol
  local_rtx5050_benchmark_results.csv   90 runs, 9 columns, pilot instrument
  colab_t4_benchmark_results.csv        90 runs, 9 columns, pilot instrument

Outputs:
  research/analysis_v3.json      machine-readable, one blob per table in the paper
  research/analysis_v3.md        human-readable digest

Every reported value in the manuscript must be traceable to this file.
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
OUT_JSON = os.path.join(ROOT, "research", "analysis_v3.json")
OUT_MD = os.path.join(ROOT, "research", "analysis_v3.md")

RNG = np.random.default_rng(20261123)
N_BOOT = 10_000

# Declared peak bandwidth, GB/s (vendor datasheets used in the literature review)
BW = {
    "NVIDIA GeForce RTX 5050 Laptop GPU": 384.0,
    "Tesla T4": 320.0,
}
# Effective bits per weight including quantisation metadata
BITS = {"fp16": 16.0, "8bit": 8.0, "4bit": 4.5}
# Parameter counts (billions) and KV geometry: (layers, kv_heads, head_dim)
MODELS = {
    "meta-llama/Llama-3.2-1B": dict(short="Llama-3.2-1B", params=1.235,
                                    layers=16, kv_heads=8, head_dim=64),
    "Qwen/Qwen2.5-1.5B": dict(short="Qwen2.5-1.5B", params=1.543,
                              layers=28, kv_heads=2, head_dim=128),
}
DEVICES = {
    "rtx5050": "NVIDIA GeForce RTX 5050 Laptop GPU",
    "t4": "Tesla T4",
}
# Base-clock thresholds used to classify a run as sitting on base clock
BASE_MHZ = {"rtx5050": 1500.0, "t4": 700.0}
PREC_ORDER = ["fp16", "8bit", "4bit"]


def jnum(x):
    if x is None:
        return None
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        # keep scientific magnitude for tiny p-values instead of rounding to zero
        if v != 0.0 and abs(v) < 1e-3:
            return float(f"{v:.3g}")
        return round(v, 4)
    return x


def boot_ci(values, stat=np.median, n=N_BOOT):
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return None, None
    idx = RNG.integers(0, values.size, size=(n, values.size))
    draws = stat(values[idx], axis=1)
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def boot_ratio_ci(a, b, n=N_BOOT):
    """CI for median(a) / median(b), resampling each group independently."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ia = RNG.integers(0, a.size, size=(n, a.size))
    ib = RNG.integers(0, b.size, size=(n, b.size))
    ra = np.median(a[ia], axis=1)
    rb = np.median(b[ib], axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = ra / rb
    r = r[np.isfinite(r)]
    return float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def holm(pvals):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return adj


def cliffs_delta(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    gt = int(np.sum(a[:, None] > b[None, :]))
    lt = int(np.sum(a[:, None] < b[None, :]))
    return (gt - lt) / (a.size * b.size)


def roofline(model, precision, ctx):
    spec = MODELS[model]
    w_bytes = spec["params"] * 1e9 * BITS[precision] / 8.0
    kv_bytes = 2 * spec["layers"] * spec["kv_heads"] * spec["head_dim"] * ctx * 2
    return w_bytes, kv_bytes


def ols(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    res = stats.linregress(x, y)
    return dict(slope=float(res.slope), intercept=float(res.intercept),
                r=float(res.rvalue), r2=float(res.rvalue ** 2),
                p=float(res.pvalue), se=float(res.stderr), n=int(x.size))


def load_devices():
    frames = {}
    for key, gpu in DEVICES.items():
        path = os.path.join(RES, "rtx5050_results.csv" if key == "rtx5050"
                            else "t4_colab_results.csv")
        df = pd.read_csv(path)
        df["gpu"] = gpu
        df["device"] = key
        df["model_short"] = df["model"].map(lambda m: MODELS[m]["short"])
        df["base_threshold_mhz"] = BASE_MHZ[key]
        df["on_base_clock"] = df["sm_clock_post_mhz"] < BASE_MHZ[key]
        df["prefill_share_pct"] = df["ttft_ms"] / df["e2e_ms"] * 100
        frames[key] = df
    return frames


# ---------------------------------------------------------------------------
# device-level analysis, run once per campaign
# ---------------------------------------------------------------------------
def analyse_device(df, key, gpu):
    out = {}

    # ---- coverage -----------------------------------------------------
    cfg_keys = ["model", "precision", "prompt_tokens"]
    per_cfg = df.groupby(cfg_keys).size()
    out["coverage"] = {
        "runs": int(len(df)),
        "configs": int(per_cfg.size),
        "runs_per_config_min": int(per_cfg.min()),
        "runs_per_config_max": int(per_cfg.max()),
        "sessions": sorted(df.session_id.unique().tolist()),
        "runs_per_session": df.groupby("session_id").size().to_dict(),
        "generated_tokens": sorted(df.generated_tokens.unique().tolist()),
        "prompt_tokens": sorted(int(v) for v in df.prompt_tokens.unique()),
        "models": sorted(df.model.unique().tolist()),
        "precisions": sorted(df.precision.unique().tolist()),
        "attn_impl": sorted(df.attn_impl.unique().tolist()),
        "schema_version": sorted(int(v) for v in df.schema_version.unique()),
        "columns": int(df.shape[1]),
    }

    # ---- hardware / software ------------------------------------------
    out["hardware"] = {
        "gpu": gpu,
        "compute_capability": float(df.compute_capability.iloc[0]),
        "gpu_total_mem_mb": int(df.gpu_total_mem_mb.iloc[0]),
        "host_cpu": str(df.host_cpu.iloc[0]),
        "torch": sorted(df.torch.unique().tolist()),
        "transformers": sorted(df.transformers.unique().tolist()),
        "bitsandbytes": sorted(df.bitsandbytes.unique().tolist()),
        "cuda": sorted(str(v) for v in df.cuda.unique()),
        "snapshots": {MODELS[m]["short"]: sorted(
            df.loc[df.model == m, "model_snapshot"].unique().tolist())
            for m in MODELS},
    }

    out["session_windows"] = {
        s: dict(runs=int(g.shape[0]),
                start=str(g.timestamp_utc.min()),
                end=str(g.timestamp_utc.max()),
                duration_min=round((pd.to_datetime(g.timestamp_utc.max())
                                    - pd.to_datetime(g.timestamp_utc.min())
                                    ).total_seconds() / 60, 1))
        for s, g in df.groupby("session_id")
    }

    # ---- global distributions -----------------------------------------
    out["global"] = {}
    for col in ["ttft_ms", "decode_tok_per_s", "itl_p95_ms", "e2e_ms",
                "weights_vram_mb", "peak_vram_mb", "temp_pre_c", "temp_post_c",
                "sm_clock_post_mhz", "power_post_w"]:
        v = df[col]
        out["global"][col] = dict(
            mean=jnum(v.mean()), sd=jnum(v.std()), median=jnum(v.median()),
            min=jnum(v.min()), max=jnum(v.max()),
            p05=jnum(v.quantile(0.05)), p95=jnum(v.quantile(0.95)))

    # ---- by precision --------------------------------------------------
    out["by_precision"] = []
    for prec in PREC_ORDER:
        g = df[df.precision == prec]
        for model in [m for m in MODELS if m in g.model.unique()]:
            h = g[g.model == model]
            out["by_precision"].append(dict(
                device=key, model=MODELS[model]["short"], precision=prec,
                runs=int(len(h)),
                decode_mean=jnum(h.decode_tok_per_s.mean()),
                decode_median=jnum(h.decode_tok_per_s.median()),
                decode_cv_pct=jnum(h.decode_tok_per_s.std()
                                   / h.decode_tok_per_s.mean() * 100),
                ttft_mean=jnum(h.ttft_ms.mean()),
                ttft_median=jnum(h.ttft_ms.median()),
                ttft_cv_pct=jnum(h.ttft_ms.std() / h.ttft_ms.mean() * 100),
                peak_vram_mean=jnum(h.peak_vram_mb.mean()),
                weights_vram_mean=jnum(h.weights_vram_mb.mean()),
                clock_mean=jnum(h.sm_clock_post_mhz.mean()),
                clock_median=jnum(h.sm_clock_post_mhz.median()),
                pct_base_clock=jnum(h.on_base_clock.mean() * 100),
                power_mean=jnum(h.power_post_w.mean()),
                temp_post_max=jnum(h.temp_post_c.max()),
            ))

    # ---- by context ----------------------------------------------------
    out["by_context"] = []
    for model in MODELS:
        for prec in PREC_ORDER:
            h = df[(df.model == model) & (df.precision == prec)]
            if h.empty:
                continue
            med = h.groupby("prompt_tokens").ttft_ms.median()
            slope = ols(med.index.values, med.values)
            dec = h.groupby("prompt_tokens").decode_tok_per_s.median()
            out["by_context"].append(dict(
                device=key, model=MODELS[model]["short"], precision=prec,
                ttft_128=jnum(med.get(128)), ttft_512=jnum(med.get(512)),
                ttft_1024=jnum(med.get(1024)),
                slope_ms_per_token=jnum(slope["slope"]),
                r2=jnum(slope["r2"]),
                decode_128=jnum(dec.get(128)), decode_512=jnum(dec.get(512)),
                decode_1024=jnum(dec.get(1024)),
                decode_range_pct=jnum((dec.max() / dec.min() - 1) * 100),
            ))

    # ---- per-config detail --------------------------------------------
    out["configs"] = []
    for (model, prec, ctx), h in df.groupby(["model", "precision", "prompt_tokens"]):
        w_b, kv_b = roofline(model, prec, ctx)
        tot_gb = (w_b + kv_b) / 1e9
        med_dec = float(h.decode_tok_per_s.median())
        ceiling = BW[gpu] / tot_gb
        lo, hi = boot_ci(h.decode_tok_per_s)
        tlo, thi = boot_ci(h.ttft_ms)
        out["configs"].append(dict(
            device=key, model=MODELS[model]["short"], precision=prec,
            prompt_tokens=int(ctx), runs=int(len(h)),
            ttft_mean=jnum(h.ttft_ms.mean()), ttft_median=jnum(h.ttft_ms.median()),
            ttft_ci_lo=jnum(tlo), ttft_ci_hi=jnum(thi),
            decode_mean=jnum(h.decode_tok_per_s.mean()),
            decode_median=jnum(med_dec),
            decode_ci_lo=jnum(lo), decode_ci_hi=jnum(hi),
            decode_cv_pct=jnum(h.decode_tok_per_s.std()
                               / h.decode_tok_per_s.mean() * 100),
            peak_vram_mean=jnum(h.peak_vram_mb.mean()),
            itl_p95_mean=jnum(h.itl_p95_ms.mean()),
            e2e_mean=jnum(h.e2e_ms.mean()),
            clock_mean=jnum(h.sm_clock_post_mhz.mean()),
            pct_base_clock=jnum(h.on_base_clock.mean() * 100),
            roofline_eff_pct=jnum(med_dec / ceiling * 100),
            bytes_per_token_gb=jnum(tot_gb),
        ))

    # ---- telemetry by precision ---------------------------------------
    out["telemetry_by_precision"] = []
    for prec in PREC_ORDER:
        h = df[df.precision == prec]
        out["telemetry_by_precision"].append(dict(
            device=key, precision=prec, runs=int(len(h)),
            clock_mean=jnum(h.sm_clock_post_mhz.mean()),
            clock_p05=jnum(h.sm_clock_post_mhz.quantile(0.05)),
            clock_p95=jnum(h.sm_clock_post_mhz.quantile(0.95)),
            clock_min=jnum(h.sm_clock_post_mhz.min()),
            clock_max=jnum(h.sm_clock_post_mhz.max()),
            pct_base_clock=jnum(h.on_base_clock.mean() * 100),
            power_mean=jnum(h.power_post_w.mean()),
            power_max=jnum(h.power_post_w.max()),
            temp_post_max=jnum(h.temp_post_c.max()),
            temp_post_mean=jnum(h.temp_post_c.mean()),
            temp_pre_mean=jnum(h.temp_pre_c.mean()),
            thermal_hw_total=int(h.throttle_thermal_hw.sum()),
            powercap_total=int(h.throttle_powercap.sum()),
            powercap_pct=jnum(h.throttle_powercap.mean() * 100),
        ))

    # ---- thermal / throttle -------------------------------------------
    out["thermal"] = dict(
        temp_pre_mean=jnum(df.temp_pre_c.mean()),
        temp_post_mean=jnum(df.temp_post_c.mean()),
        temp_post_max=jnum(df.temp_post_c.max()),
        temp_pre_max=jnum(df.temp_pre_c.max()),
        cooled_ok_all=bool(df.cooled_ok.all()),
        thermal_throttle_total=int(df.throttle_thermal_hw.sum()),
        powercap_total=int(df.throttle_powercap.sum()),
        spearman_temp_clock=jnum(stats.spearmanr(df.temp_post_c,
                                                 df.sm_clock_post_mhz).statistic),
        spearman_temp_power=jnum(stats.spearmanr(df.temp_post_c,
                                                 df.power_post_w).statistic),
        spearman_power_clock=jnum(stats.spearmanr(df.power_post_w,
                                                  df.sm_clock_post_mhz).statistic),
    )

    # ---- clock elasticity ---------------------------------------------
    med = (df.groupby(["model", "precision", "prompt_tokens"])
             .agg(decode=("decode_tok_per_s", "median"),
                  clock=("sm_clock_post_mhz", "median"))
             .reset_index())
    med["ld"] = np.log(med.decode)
    med["lc"] = np.log(med.clock)
    reg = ols(med.lc, med.ld)
    by_prec = {}
    for prec, h in med.groupby("precision"):
        by_prec[prec] = dict(decode=jnum(h.decode.mean()),
                             clock=jnum(h.clock.mean()),
                             runs=int(len(h)))
    out["clock_elasticity"] = dict(
        slope=jnum(reg["slope"]), se=jnum(reg["se"]), r2=jnum(reg["r2"]),
        p=jnum(reg["p"]), n_configs=int(reg["n"]), by_precision=by_prec)

    # ---- roofline -----------------------------------------------------
    roof = {}
    for prec, h in df.groupby("precision"):
        eff = []
        for (model, ctx), g in h.groupby(["model", "prompt_tokens"]):
            w_b, kv_b = roofline(model, prec, int(ctx))
            ceiling = BW[gpu] / ((w_b + kv_b) / 1e9)
            eff.append(float(g.decode_tok_per_s.median()) / ceiling * 100)
        roof[prec] = dict(mean_eff_pct=jnum(np.mean(eff)),
                          min_eff_pct=jnum(np.min(eff)),
                          max_eff_pct=jnum(np.max(eff)),
                          n=jnum(len(eff)))
    out["roofline"] = roof

    # ---- layer scaling -------------------------------------------------
    layers = []
    for (prec, ctx), h in df.groupby(["precision", "prompt_tokens"]):
        lat = h.groupby("model").apply(
            lambda g: 1000.0 / g.decode_tok_per_s.median(), include_groups=False)
        if len(lat) == 2:
            layers.append(dict(
                device=key, precision=prec, prompt_tokens=int(ctx),
                lat_llama_ms=jnum(lat["meta-llama/Llama-3.2-1B"]),
                lat_qwen_ms=jnum(lat["Qwen/Qwen2.5-1.5B"]),
                lat_ratio=jnum(lat["Qwen/Qwen2.5-1.5B"]
                               / lat["meta-llama/Llama-3.2-1B"]),
                layer_ratio=jnum(28 / 16),
                param_ratio=jnum(1.543 / 1.235)))
    out["layer_scaling"] = layers

    # ---- speedup vs fp16 ----------------------------------------------
    base = (df[df.precision == "fp16"]
            .groupby(["model", "prompt_tokens"]).decode_tok_per_s.median())
    sp = []
    for (model, prec, ctx), h in df.groupby(["model", "precision", "prompt_tokens"]):
        if prec == "fp16":
            continue
        b = base.loc[(model, ctx)]
        lo, hi = boot_ratio_ci(h.decode_tok_per_s.values,
                              df[(df.model == model) & (df.precision == "fp16")
                                 & (df.prompt_tokens == ctx)].decode_tok_per_s.values)
        sp.append(dict(device=key, model=MODELS[model]["short"], precision=prec,
                       ctx=int(ctx), ratio=jnum(h.decode_tok_per_s.median() / b),
                       ci_lo=jnum(lo), ci_hi=jnum(hi)))
    out["speedups"] = sp

    # ---- between-session dispersion ------------------------------------
    sess = []
    for (model, prec, ctx), h in df.groupby(["model", "precision", "prompt_tokens"]):
        med = h.groupby("session_id").decode_tok_per_s.median()
        sess.append(dict(
            device=key, model=MODELS[model]["short"], precision=prec,
            prompt_tokens=int(ctx),
            session_medians={s: jnum(v) for s, v in med.items()},
            session_median_range_pct=jnum((med.max() / med.min() - 1) * 100),
            within_session_cv_max=jnum(
                h.groupby("session_id").decode_tok_per_s.apply(
                    lambda x: x.std() / x.mean() * 100).max()),
        ))
    out["between_session"] = sess

    # ---- bimodality of the noisiest config ------------------------------
    worst = max(sess, key=lambda r: r["within_session_cv_max"])
    flag = df[(df.model_short == worst["model"])
              & (df.precision == worst["precision"])
              & (df.prompt_tokens == worst["prompt_tokens"])]
    rows = []
    for s, h in flag.groupby("session_id"):
        base_m = h[h.on_base_clock]
        boost_m = h[~h.on_base_clock]
        rows.append(dict(
            session_id=s, runs=int(len(h)),
            base_runs=int(len(base_m)), base_pct=jnum(len(base_m) / len(h) * 100),
            decode_base=jnum(base_m.decode_tok_per_s.mean()) if len(base_m) else None,
            decode_boost=jnum(boost_m.decode_tok_per_s.mean()) if len(boost_m) else None,
            clock_base=jnum(base_m.sm_clock_post_mhz.mean()) if len(base_m) else None,
            clock_boost=jnum(boost_m.sm_clock_post_mhz.mean()) if len(boost_m) else None,
            power_base=jnum(base_m.power_post_w.mean()) if len(base_m) else None,
            power_boost=jnum(boost_m.power_post_w.mean()) if len(boost_m) else None,
            cv_pct=jnum(h.decode_tok_per_s.std() / h.decode_tok_per_s.mean() * 100),
            temp_max=jnum(h.temp_post_c.max()),
            decode_mean=jnum(h.decode_tok_per_s.mean())))
    out["bimodal"] = dict(model=worst["model"], precision=worst["precision"],
                          prompt_tokens=worst["prompt_tokens"],
                          within_session_cv_max=worst["within_session_cv_max"],
                          sessions=rows)
    vals = flag.decode_tok_per_s.values
    out["flagged_config"] = dict(
        n=int(vals.size), mean=jnum(vals.mean()), median=jnum(np.median(vals)),
        p10=jnum(np.percentile(vals, 10)), p90=jnum(np.percentile(vals, 90)),
        min=jnum(vals.min()), max=jnum(vals.max()),
        cv_pct=jnum(vals.std() / vals.mean() * 100),
        gap_between_modes=jnum(np.percentile(vals, 90) - np.percentile(vals, 10)))

    # ---- QA gate -------------------------------------------------------
    qa_rows = []
    for (model, prec, ctx, s), h in df.groupby(
            ["model", "precision", "prompt_tokens", "session_id"]):
        cv = float(h.decode_tok_per_s.std() / h.decode_tok_per_s.mean() * 100)
        qa_rows.append(dict(
            device=key, model=MODELS[model]["short"], precision=prec,
            prompt_tokens=int(ctx), session_id=s, run_count=int(len(h)),
            decode_cv_pct=jnum(cv), cv_ok=bool(cv <= 10.0),
            throttle_pct=jnum(h.throttle_thermal_hw.mean() * 100),
            thermal_ok=bool(h.throttle_thermal_hw.mean() == 0),
            complete=bool(len(h) == 30),
            missing_telemetry=int(h[["temp_pre_c", "temp_post_c",
                                     "sm_clock_post_mhz", "power_post_w"]]
                                  .isna().any(axis=1).sum())))
    out["qa"] = dict(
        cells_total=len(qa_rows),
        cells_pass=int(sum(r["cv_ok"] and r["thermal_ok"] and r["complete"]
                           and r["missing_telemetry"] == 0 for r in qa_rows)),
        cv_ok=int(sum(r["cv_ok"] for r in qa_rows)),
        thermal_ok=int(sum(r["thermal_ok"] for r in qa_rows)),
        complete=int(sum(r["complete"] for r in qa_rows)),
        missing_telemetry_total=int(sum(r["missing_telemetry"] for r in qa_rows)),
        max_cv=jnum(max(r["decode_cv_pct"] for r in qa_rows)),
        fail_rows=[r for r in qa_rows
                   if not (r["cv_ok"] and r["thermal_ok"] and r["complete"]
                           and r["missing_telemetry"] == 0)])

    # ---- prefill share --------------------------------------------------
    out["prefill_share"] = dict(
        overall_mean_pct=jnum(df.prefill_share_pct.mean()),
        overall_median_pct=jnum(df.prefill_share_pct.median()),
        overall_min_pct=jnum(df.prefill_share_pct.min()),
        overall_max_pct=jnum(df.prefill_share_pct.max()),
        by_precision={p: jnum(g.prefill_share_pct.mean())
                      for p, g in df.groupby("precision")})

    return out


# ---------------------------------------------------------------------------
# cross-device analysis, now n=150 per cell on both sides
# ---------------------------------------------------------------------------
def analyse_cross(d5050, t4):
    key = ["model", "precision", "prompt_tokens"]
    m50 = d5050.groupby(key).agg(
        ttft_med=("ttft_ms", "median"), dec_med=("decode_tok_per_s", "median"),
        vram=("peak_vram_mb", "mean"), ttft_mean=("ttft_ms", "mean"),
        dec_mean=("decode_tok_per_s", "mean"),
        ttft_cv=("ttft_ms", lambda s: s.std() / s.mean() * 100),
        dec_cv=("decode_tok_per_s", lambda s: s.std() / s.mean() * 100),
        clock=("sm_clock_post_mhz", "mean"),
        power=("power_post_w", "mean"), temp=("temp_post_c", "max"),
        base=("on_base_clock", "mean")).reset_index()
    mt4 = t4.groupby(key).agg(
        ttft_med=("ttft_ms", "median"), dec_med=("decode_tok_per_s", "median"),
        vram=("peak_vram_mb", "mean"), ttft_mean=("ttft_ms", "mean"),
        dec_mean=("decode_tok_per_s", "mean"),
        ttft_cv=("ttft_ms", lambda s: s.std() / s.mean() * 100),
        dec_cv=("decode_tok_per_s", lambda s: s.std() / s.mean() * 100),
        clock=("sm_clock_post_mhz", "mean"),
        power=("power_post_w", "mean"), temp=("temp_post_c", "max"),
        base=("on_base_clock", "mean")).reset_index()

    cells = []
    for (model, prec, ctx), h50 in d5050.groupby(key):
        h4 = t4[(t4.model == model) & (t4.precision == prec)
                & (t4.prompt_tokens == ctx)]
        t_ttft = stats.mannwhitneyu(h50.ttft_ms, h4.ttft_ms, alternative="two-sided")
        t_dec = stats.mannwhitneyu(h50.decode_tok_per_s, h4.decode_tok_per_s,
                                   alternative="two-sided")
        ttft_lo, ttft_hi = boot_ratio_ci(h4.ttft_ms.values, h50.ttft_ms.values)
        dec_lo, dec_hi = boot_ratio_ci(h4.decode_tok_per_s.values,
                                       h50.decode_tok_per_s.values)
        cells.append(dict(
            device="t4_over_rtx5050", model=MODELS[model]["short"],
            precision=prec, ctx=int(ctx),
            ttft_r50=h50.ttft_ms.median(), ttft_t4=h4.ttft_ms.median(),
            ttft_ratio=jnum(h4.ttft_ms.median() / h50.ttft_ms.median()),
            ttft_ratio_lo=jnum(ttft_lo), ttft_ratio_hi=jnum(ttft_hi),
            dtps_r50=h50.decode_tok_per_s.median(),
            dtps_t4=h4.decode_tok_per_s.median(),
            dtps_ratio=jnum(h4.decode_tok_per_s.median()
                            / h50.decode_tok_per_s.median()),
            dtps_ratio_lo=jnum(dec_lo), dtps_ratio_hi=jnum(dec_hi),
            p_ttft=jnum(t_ttft.pvalue), p_dtps=jnum(t_dec.pvalue),
            delta_ttft=jnum(cliffs_delta(h4.ttft_ms.values, h50.ttft_ms.values)),
            delta_dtps=jnum(cliffs_delta(h4.decode_tok_per_s.values,
                                         h50.decode_tok_per_s.values)),
            clock_r50=jnum(h50.sm_clock_post_mhz.mean()),
            clock_t4=jnum(h4.sm_clock_post_mhz.mean()),
            power_r50=jnum(h50.power_post_w.mean()),
            power_t4=jnum(h4.power_post_w.mean()),
            temp_r50=jnum(h50.temp_post_c.max()),
            temp_t4=jnum(h4.temp_post_c.max()),
            vram_r50=jnum(h50.peak_vram_mb.mean()),
            vram_t4=jnum(h4.peak_vram_mb.mean())))
    cells.sort(key=lambda c: (c["model"], PREC_ORDER.index(c["precision"]), c["ctx"]))
    cells.sort(key=lambda c: (PREC_ORDER.index(c["precision"]), c["model"], c["ctx"]))

    for field in ("p_ttft", "p_dtps"):
        adj = holm([c[field] for c in cells])
        for c, a in zip(cells, adj):
            c[field + "_holm"] = jnum(a)

    # pooled by precision
    pooled = []
    for prec in PREC_ORDER:
        a = d5050[d5050.precision == prec]
        b = t4[t4.precision == prec]
        if a.empty or b.empty:
            continue
        t1 = stats.mannwhitneyu(a.ttft_ms, b.ttft_ms, alternative="two-sided")
        t2 = stats.mannwhitneyu(a.decode_tok_per_s, b.decode_tok_per_s,
                                alternative="two-sided")
        lo1, hi1 = boot_ratio_ci(b.ttft_ms.values, a.ttft_ms.values)
        lo2, hi2 = boot_ratio_ci(b.decode_tok_per_s.values,
                                 a.decode_tok_per_s.values)
        pooled.append(dict(
            precision=prec, n_per_side=int(len(a)),
            ttft_ratio=jnum(b.ttft_ms.median() / a.ttft_ms.median()),
            ttft_ci_lo=jnum(lo1), ttft_ci_hi=jnum(hi1),
            ttft_p=jnum(t1.pvalue),
            dtps_ratio=jnum(b.decode_tok_per_s.median()
                            / a.decode_tok_per_s.median()),
            dtps_ci_lo=jnum(lo2), dtps_ci_hi=jnum(hi2),
            dtps_p=jnum(t2.pvalue),
            delta_ttft=jnum(cliffs_delta(b.ttft_ms.values, a.ttft_ms.values)),
            delta_dtps=jnum(cliffs_delta(b.decode_tok_per_s.values,
                                         a.decode_tok_per_s.values))))
    adj1 = holm([r["ttft_p"] for r in pooled])
    adj2 = holm([r["dtps_p"] for r in pooled])
    for r, a, b in zip(pooled, adj1, adj2):
        r["ttft_p_holm"] = jnum(a)
        r["dtps_p_holm"] = jnum(b)

    # aggregate per device
    agg = []
    for label, df in (("rtx5050", d5050), ("t4", t4)):
        lo1, hi1 = boot_ci(df.ttft_ms)
        lo2, hi2 = boot_ci(df.decode_tok_per_s)
        agg.append(dict(device=label, runs=int(len(df)),
                        ttft_mean=jnum(df.ttft_ms.mean()),
                        ttft_median=jnum(df.ttft_ms.median()),
                        ttft_ci_lo=jnum(lo1), ttft_ci_hi=jnum(hi1),
                        decode_mean=jnum(df.decode_tok_per_s.mean()),
                        decode_median=jnum(df.decode_tok_per_s.median()),
                        decode_ci_lo=jnum(lo2), decode_ci_hi=jnum(hi2),
                        vram_mean=jnum(df.peak_vram_mb.mean()),
                        vram_max=jnum(df.peak_vram_mb.max()),
                        clock_mean=jnum(df.sm_clock_post_mhz.mean()),
                        power_mean=jnum(df.power_post_w.mean()),
                        temp_max=jnum(df.temp_post_c.max()),
                        temp_mean=jnum(df.temp_post_c.mean())))

    wins_ttft = int(sum(1 for c in cells if c["ttft_ratio"] < 1))
    wins_dec = int(sum(1 for c in cells if c["dtps_ratio"] > 1))

    # interaction: do the T4/5050 log-ratios differ across precision?
    groups = {}
    for prec in PREC_ORDER:
        groups[prec] = [math.log(c["dtps_ratio"]) for c in cells
                        if c["precision"] == prec]
    inter = stats.kruskal(*groups.values())
    groups_t = {}
    for prec in PREC_ORDER:
        groups_t[prec] = [math.log(c["ttft_ratio"]) for c in cells
                          if c["precision"] == prec]
    inter_t = stats.kruskal(*groups_t.values())

    by_prec = []
    for prec in PREC_ORDER:
        sub = [c for c in cells if c["precision"] == prec]
        if not sub:
            continue
        by_prec.append(dict(
            precision=prec, n_cells=len(sub),
            ttft_ratio=jnum(float(np.median([c["ttft_ratio"] for c in sub]))),
            ttft_min=jnum(min(c["ttft_ratio"] for c in sub)),
            ttft_max=jnum(max(c["ttft_ratio"] for c in sub)),
            dtps_ratio=jnum(float(np.median([c["dtps_ratio"] for c in sub]))),
            dtps_min=jnum(min(c["dtps_ratio"] for c in sub)),
            dtps_max=jnum(max(c["dtps_ratio"] for c in sub)),
            t4_ttft_wins=int(sum(1 for c in sub if c["ttft_ratio"] < 1)),
            t4_dtps_wins=int(sum(1 for c in sub if c["dtps_ratio"] > 1))))

    return dict(
        aggregate=agg, cells=cells, pooled_by_precision=pooled,
        by_precision=by_prec,
        wins=dict(ttft_rtx=int(wins_ttft), dtps_rtx=int(wins_dec), cells=len(cells)),
        interaction=dict(dtps_h=jnum(inter.statistic), dtps_p=jnum(inter.pvalue),
                         ttft_h=jnum(inter_t.statistic),
                         ttft_p=jnum(inter_t.pvalue)),
        min_p_ttft=jnum(min(c["p_ttft"] for c in cells)),
        min_p_dtps=jnum(min(c["p_dtps"] for c in cells)),
        n_significant_holm_ttft=int(sum(1 for c in cells if c["p_ttft_holm"] < 0.05)),
        n_significant_holm_dtps=int(sum(1 for c in cells if c["p_dtps_holm"] < 0.05)),
    )


# ---------------------------------------------------------------------------
# instrument drift: pilot 9-column protocol vs full 35-column protocol
# ---------------------------------------------------------------------------
def analyse_drift(pilot, full, key):
    f = full.groupby(["model", "precision", "prompt_tokens"]).agg(
        dec_full=("decode_tok_per_s", "median"),
        ttft_full=("ttft_ms", "median"),
        vram_full=("peak_vram_mb", "median")).reset_index()
    p = pilot.groupby(["model", "precision", "prompt_tokens"]).agg(
        dec_pilot=("throughput_tokens_per_sec", "median"),
        ttft_pilot=("ttft_ms", "median"),
        vram_pilot=("peak_vram_mb", "median")).reset_index()
    d = f.merge(p, on=["model", "precision", "prompt_tokens"])
    d["dec_pct"] = (d.dec_full / d.dec_pilot - 1) * 100
    d["ttft_pct"] = (d.ttft_full / d.ttft_pilot - 1) * 100
    d["vram_pct"] = (d.vram_full / d.vram_pilot - 1) * 100
    by_prec = {}
    for prec, g in d.groupby("precision"):
        by_prec[prec] = dict(decode_median_pct=jnum(g.dec_pct.median()),
                             ttft_median_pct=jnum(g.ttft_pct.median()))
    return dict(
        device=key, n_configs=int(len(d)),
        decode_median_pct_by_precision={k: v["decode_median_pct"]
                                        for k, v in by_prec.items()},
        ttft_median_pct_by_precision={k: v["ttft_median_pct"]
                                      for k, v in by_prec.items()},
        decode_median_abs_pct=jnum(d.dec_pct.abs().median()),
        decode_min_pct=jnum(d.dec_pct.min()), decode_max_pct=jnum(d.dec_pct.max()),
        decode_max_abs_pct=jnum(d.dec_pct.abs().max()),
        ttft_median_abs_pct=jnum(d.ttft_pct.abs().median()),
        ttft_max_abs_pct=jnum(d.ttft_pct.abs().max()),
        vram_median_abs_pct=jnum(d.vram_pct.abs().median()),
        vram_max_abs_pct=jnum(d.vram_pct.abs().max()),
        rows=[dict(model=MODELS[r.model]["short"], precision=r.precision,
                   ctx=int(r.prompt_tokens),
                   decode_full=jnum(r.dec_full), decode_pilot=jnum(r.dec_pilot),
                   decode_pct=jnum(r.dec_pct),
                   ttft_full=jnum(r.ttft_full), ttft_pilot=jnum(r.ttft_pilot),
                   ttft_pct=jnum(r.ttft_pct))
              for r in d.itertuples()])


def main():
    frames = load_devices()
    out = {"generated_by": "research/analysis_v3.py",
           "bootstrap_resamples": N_BOOT,
           "seed": 20261123,
           "base_clock_threshold_mhz": BASE_MHZ,
           "declared_bandwidth_gbs": BW,
           "effective_bits": BITS}

    for key in DEVICES:
        out[key] = analyse_device(frames[key], key, DEVICES[key])
        print("analysed", key, len(frames[key]), "runs")

    d50, d4 = frames["rtx5050"], frames["t4"]
    out["cross"] = analyse_cross(d50, d4)
    print("cross cells:", len(out["cross"]["cells"]))

    pilot50 = pd.read_csv(os.path.join(RES, "local_rtx5050_benchmark_results.csv"))
    pilot4 = pd.read_csv(os.path.join(RES, "colab_t4_benchmark_results.csv"))
    out["drift"] = {
        "rtx5050": analyse_drift(pilot50, d50, "rtx5050"),
        "t4": analyse_drift(pilot4, d4, "t4"),
    }
    print("drift computed")

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, ensure_ascii=False)

    # ---------------- markdown digest ---------------------------------
    L = []
    L.append("# analysis_v3 — digest\n")
    for key in DEVICES:
        d = out[key]
        L.append(f"\n## {DEVICES[key]}\n")
        L.append(f"- runs {d['coverage']['runs']}, configs "
                 f"{d['coverage']['configs']}, sessions "
                 f"{len(d['coverage']['sessions'])}")
        L.append(f"- QA pass {d['qa']['cells_pass']}/{d['qa']['cells_total']}"
                 f" (max CV {d['qa']['max_cv']})")
        L.append(f"- decode median {d['global']['decode_tok_per_s']['median']}"
                 f" TTFT median {d['global']['ttft_ms']['median']}")
        for r in d["by_precision"]:
            L.append(f"  - {r['model']} {r['precision']}: "
                     f"{r['decode_mean']} tok/s mean, "
                     f"{r['decode_median']} median, "
                     f"clock {r['clock_mean']} MHz, "
                     f"base {r['pct_base_clock']}%, "
                     f"power {r['power_mean']} W")
    L.append("\n## cross\n")
    for r in out["cross"]["aggregate"]:
        L.append(f"- {r['device']}: ttft med {r['ttft_median']} "
                 f"decode med {r['decode_median']}")
    L.append(f"- wins rtx ttft {out['cross']['wins']['ttft_rtx']}/"
             f"{out['cross']['wins']['cells']}, decode "
             f"{out['cross']['wins']['dtps_rtx']}/{out['cross']['wins']['cells']}")
    L.append(f"- interaction decode p {out['cross']['interaction']['dtps_p']}, "
             f"ttft p {out['cross']['interaction']['ttft_p']}")
    for r in out["cross"]["by_precision"]:
        L.append(f"  - {r['precision']}: ttft {r['ttft_ratio']} "
                 f"decode {r['dtps_ratio']}")
    L.append("\n## drift\n")
    for key in DEVICES:
        d = out["drift"][key]
        L.append(f"- {key}: decode {d['decode_median_pct_by_precision']} "
                 f"median abs {d['decode_median_abs_pct']}%")

    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    print("wrote", OUT_JSON)
    print("wrote", OUT_MD)


if __name__ == "__main__":
    main()
