"""
Figure + table generation for the attention benchmark report.

Reads the JSON produced by benchmark.py / train_gpt.py and writes PNGs to ../figures
and markdown/CSV tables to ../results.

Design notes: single validated categorical palette assigned per *kernel* (colour
follows the entity, never its rank), one y-axis per chart, recessive grid, direct
end-labels on every line so identity never rests on colour alone.
"""

from __future__ import annotations

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
FIGS = os.path.join(HERE, "..", "figures")
os.makedirs(FIGS, exist_ok=True)

# --------------------------------------------------------------------------
# Design tokens (validated categorical palette, light surface)
# --------------------------------------------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
INK3 = "#8a8984"
GRID = "#e6e5e1"
AXIS = "#c9c8c3"

# fixed slot order; colour follows the entity across every figure
SERIES = {
    "flash":         ("FlashAttention (SDPA)",      "#2a78d6", "o", "-"),
    "naive":         ("Standard MHA (explicit NxN)", "#eb6834", "s", "-"),
    "mem_efficient": ("Mem-efficient (xFormers-style)", "#1baf7a", "^", "-"),
    "triton_flash":  ("FlashAttention (our Triton)", "#eda100", "D", "--"),
    "math":          ("PyTorch SDPA math backend",  "#e87ba4", "v", "-"),
    "cudnn":         ("cuDNN fused attention",      "#008300", "P", "--"),
    "auto":          ("SDPA automatic dispatch",    "#4a3aa7", "X", ":"),
    "tiled_flash":   ("Flash algorithm (pure PyTorch)", "#e34948", "*", ":"),
}
LAYOUT_COLORS = {"MHA": "#2a78d6", "GQA-8": "#eb6834", "GQA-4": "#1baf7a",
                 "GQA-2": "#eda100", "MQA": "#e87ba4"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.labelsize": 10.5,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5, "legend.frameon": False, "legend.fontsize": 9.5,
    "lines.linewidth": 2.0, "lines.markersize": 6.5, "figure.dpi": 150,
    "axes.axisbelow": True,
})


def style(ax, title=None, xlabel=None, ylabel=None, subtitle=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, loc="left", pad=16 if subtitle else 8)
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, fontsize=9.5,
                color=INK2, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    return ax


def end_label(ax, x, y, text, color, dx=1.03, fontsize=9, va="center"):
    """Direct label at the end of a line - relief for the low-contrast hues."""
    ax.annotate(text, xy=(x, y), xytext=(4, 0), textcoords="offset points",
                fontsize=fontsize, color=INK, va=va, ha="left",
                bbox=dict(boxstyle="round,pad=0.18", fc=SURFACE, ec="none", alpha=0.85))


def stagger_labels(ax, items, min_gap_px=15, fontsize=9):
    """Place end-of-line labels, nudging them apart so they never overlap.

    `items` is a list of (x, y, text). Positions are resolved in display space, so
    this works on log axes too.
    """
    if not items:
        return
    ax.figure.canvas.draw()
    trans = ax.transData
    inv = trans.inverted()
    pts = sorted(((trans.transform((x, y))[1], x, y, t) for x, y, t in items),
                 key=lambda p: p[0])
    placed = []
    for py, x, y, t in pts:
        if placed and py - placed[-1] < min_gap_px:
            py = placed[-1] + min_gap_px
        placed.append(py)
        px = trans.transform((x, y))[0]
        _, ydata = inv.transform((px, py))
        ax.annotate(t, xy=(x, ydata), xytext=(6, 0), textcoords="offset points",
                    fontsize=fontsize, color=INK, va="center", ha="left")


def save(fig, name):
    path = os.path.join(FIGS, name)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"  figure -> {os.path.relpath(path, os.path.join(HERE, '..'))}")


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        print(f"  (missing {name}, skipping dependent figures)")
        return None
    with open(p) as f:
        return json.load(f)


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def ok(rows):
    return [r for r in rows if r.get("status") == "ok"]


def series_for(rows, sweep, kernel, xkey="seq_len", ykey="ms_median"):
    rs = sorted(ok(sel(rows, sweep=sweep, kernel=kernel)), key=lambda r: r[xkey])
    return [r[xkey] for r in rs], [r[ykey] for r in rs]


def oom_points(rows, sweep, kernel, xkey="seq_len"):
    return sorted(r[xkey] for r in sel(rows, sweep=sweep, kernel=kernel)
                  if r.get("status") == "OOM")


def pow2_axis(ax, xs):
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.minorticks_off()


# ==========================================================================
# Kernel micro-benchmark figures
# ==========================================================================

MAIN_KERNELS = ["naive", "math", "mem_efficient", "triton_flash", "flash"]


def fig_latency(rows, sweep="A_main", fname="01_latency_vs_seqlen.png",
                title="Attention forward latency vs sequence length",
                sub="batch 16, 12 heads, head_dim 64, fp16, non-causal — RTX 5070"):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_x = set()
    oom_notes = []
    for kern in MAIN_KERNELS:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, sweep, kern)
        if not x:
            continue
        all_x |= set(x)
        ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        oo = oom_points(rows, sweep, kern)
        if oo:
            # extrapolate the N^2 trend one octave so the X sits where the point
            # *would* have been, rather than floating at an arbitrary height
            xo = min(oo)
            y_ext = y[-1] * (xo / x[-1]) ** 2
            ax.plot([xo], [y_ext], marker="X", color=color, ms=11, mew=1.6,
                    markeredgecolor=SURFACE, zorder=4)
            oom_notes.append((xo, y_ext, color))
    if oom_notes:
        xo, y_ext, _ = max(oom_notes, key=lambda t: t[1])
        ax.annotate("out of memory\n(12 GB card)", (xo, y_ext),
                    textcoords="offset points", xytext=(-6, 10), ha="right",
                    fontsize=9, color=INK2)
    ax.set_yscale("log")
    pow2_axis(ax, sorted(all_x))
    style(ax, title, "sequence length N (tokens)", "latency (ms, median of 50)", sub)
    ax.legend(loc="upper left", ncol=1)
    ax.margins(x=0.09)
    save(fig, fname)


def fig_speedup(rows, sweep="A_main", baseline="naive",
                fname="02_speedup_vs_seqlen.png"):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bx, by = series_for(rows, sweep, baseline)
    base = dict(zip(bx, by))
    all_x = set()
    labels = []
    for kern in ["flash", "cudnn", "triton_flash", "mem_efficient"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, sweep, kern)
        xs = [xi for xi in x if xi in base]
        sp = [base[xi] / yi for xi, yi in zip(x, y) if xi in base]
        if not xs:
            continue
        all_x |= set(xs)
        ax.plot(xs, sp, color=color, marker=marker, ls=ls, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        labels.append((xs[-1], sp[-1], f"{sp[-1]:.1f}×"))
    ax.axhline(1.0, color=INK3, lw=1.2, ls="--")
    ax.annotate("standard MHA baseline (1×)", (min(all_x), 1.0),
                textcoords="offset points", xytext=(2, 5), fontsize=8.5, color=INK2)
    pow2_axis(ax, sorted(all_x))
    style(ax, "Speed-up over standard multi-head attention",
          "sequence length N (tokens)", "speed-up (x, higher is better)",
          "same inputs, same output to within fp16 rounding — batch 16, 12 heads, fp16")
    ax.legend(loc="upper left")
    ax.margins(x=0.10)
    stagger_labels(ax, labels)
    save(fig, fname)


def fig_memory(rows, sweep="A_main", fname="03_peak_memory.png"):
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    all_x = set()
    tails = {}
    for kern in MAIN_KERNELS:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, sweep, kern, ykey="peak_mem_mb")
        if not x:
            continue
        all_x |= set(x)
        ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        tails[kern] = (x[-1], y[-1])
    # label only the two story lines; the legend carries the rest
    for kern in ("naive", "flash"):
        if kern in tails:
            x1, y1 = tails[kern]
            end_label(ax, x1, y1, f"{y1:,.0f} MB", SERIES[kern][1])
    xs = sorted(all_x)
    if xs:
        x0 = xs[0]
        base_q = 16 * 12 * x0 * 64 * 2 / 1e6
        ax.plot(xs, [base_q * 4 * (x / x0) ** 2 for x in xs], color=INK3, lw=1.1,
                ls=":", zorder=1)
        ax.annotate("O(N²)", (xs[len(xs) // 2], base_q * 4 * (xs[len(xs) // 2] / x0) ** 2),
                    textcoords="offset points", xytext=(6, -16), fontsize=9,
                    color=INK2)
        ax.plot(xs, [base_q * (x / x0) for x in xs], color=INK3, lw=1.1, ls=":",
                zorder=1)
        ax.annotate("O(N)", (xs[len(xs) // 2], base_q * (xs[len(xs) // 2] / x0)),
                    textcoords="offset points", xytext=(6, -16), fontsize=9,
                    color=INK2)
    ax.set_yscale("log")
    pow2_axis(ax, xs)
    style(ax, "Peak attention workspace memory",
          "sequence length N (tokens)", "peak extra GPU memory (MB, log scale)",
          "memory allocated by the attention op itself, excluding Q/K/V")
    ax.annotate("the mem-efficient, cuDNN and Triton kernels\n"
                "lie exactly on the FlashAttention line",
                (0.985, 0.055), xycoords="axes fraction", ha="right", fontsize=9,
                color=INK2)
    ax.legend(loc="upper left")
    ax.margins(x=0.13)
    save(fig, fname)


def fig_throughput(rows, sweep="A_main", fname="04_throughput_tflops.png",
                   peak_tflops=None):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    all_x = set()
    for kern in MAIN_KERNELS:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, sweep, kern, ykey="tflops")
        if not x:
            continue
        all_x |= set(x)
        ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        end_label(ax, x[-1], y[-1], f"{y[-1]:.0f}", color)
    if peak_tflops:
        ax.axhline(peak_tflops, color=INK3, lw=1.2, ls="--")
        ax.annotate(f"measured dense fp16 matmul peak ≈ {peak_tflops:.0f} TFLOP/s",
                    (max(all_x), peak_tflops), textcoords="offset points",
                    xytext=(-4, -14), ha="right", fontsize=8.5, color=INK2)
        ax.set_ylim(top=peak_tflops * 1.10)
    pow2_axis(ax, sorted(all_x))
    style(ax, "Achieved arithmetic throughput",
          "sequence length N (tokens)", "TFLOP/s (fp16)",
          "identical FLOP count for every kernel — the gap is memory traffic, not math")
    ax.legend(loc="center right")
    ax.margins(x=0.14)
    save(fig, fname)


def fig_accuracy(rows, sweep="A_main", fname="05_numerical_fidelity.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for ax, key, ttl, ylab in [
        (axes[0], "max_abs_err_vs_fp64", "Error against a float64 ground truth",
         "max absolute error"),
        (axes[1], "max_abs_err_vs_math", "Difference from the standard fp16 path",
         "max absolute difference"),
    ]:
        all_x = set()
        for kern in MAIN_KERNELS:
            label, color, marker, ls = SERIES[kern]
            if key == "max_abs_err_vs_math" and kern == "math":
                continue
            x, y = series_for(rows, sweep, kern, ykey=key)
            xy = [(a, b) for a, b in zip(x, y) if b == b and b > 0]
            if not xy:
                continue
            x, y = zip(*xy)
            all_x |= set(x)
            ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_yscale("log")
        pow2_axis(ax, sorted(all_x))
        style(ax, ttl, "sequence length N", ylab)
    axes[0].axhline(9.77e-4, color=INK3, ls="--", lw=1.2)
    axes[0].annotate("one fp16 ulp near |x|=1  (2⁻¹⁰ ≈ 9.8e-4)", (0.30, 0.955),
                     xycoords="axes fraction", fontsize=8.5, color=INK2, ha="left")
    axes[0].annotate("the fused kernels are CLOSER to the true answer\n"
                     "than the fp16 baseline: they accumulate the running\n"
                     "softmax in fp32, the baseline rounds S and P to fp16",
                     (0.03, 0.20), xycoords="axes fraction", fontsize=9, color=INK2)
    axes[0].legend(loc="upper right", ncol=1, fontsize=8.5)
    axes[0].margins(y=0.3)
    axes[1].annotate("every fused kernel agrees with the standard path\n"
                     "to fp16 rounding — none of them approximate",
                     (0.03, 0.86), xycoords="axes fraction", fontsize=9, color=INK2)
    axes[1].margins(y=0.22)
    fig.suptitle("FlashAttention is exact: the difference is rounding, not approximation",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_longctx(rows, fname="06_long_context.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for ax, ykey, ttl, ylab in [
        (axes[0], "ms_median", "Latency at long context", "latency (ms, log)"),
        (axes[1], "peak_mem_mb", "Peak workspace at long context",
         "peak extra memory (MB, log)")]:
        all_x = set()
        oom_x = []
        for kern in MAIN_KERNELS:
            label, color, marker, ls = SERIES[kern]
            x, y = series_for(rows, "B_longctx", kern, ykey=ykey)
            if not x:
                continue
            all_x |= set(x)
            ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            o = oom_points(rows, "B_longctx", kern)
            if o:
                oom_x.append(min(o))
        if oom_x:
            xmin = min(oom_x)
            ax.axvspan(xmin / 1.42, max(all_x) * 1.45, color="#e34948", alpha=0.06,
                       zorder=0)
            ax.annotate("standard MHA is out of\nmemory beyond this point",
                        (xmin / 1.35, ax.get_ylim()[0]), textcoords="offset points",
                        xytext=(6, 12), fontsize=9, color="#a33", ha="left",
                        va="bottom")
        ax.set_yscale("log")
        pow2_axis(ax, sorted(all_x))
        style(ax, ttl, "sequence length N (tokens)", ylab)
    axes[0].legend(loc="upper left")
    fig.suptitle("Long context, batch 4 — the memory wall the baseline hits",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_causal(rows, fname="07_causal.png"):
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bx, by = series_for(rows, "C_causal", "naive")
    base = dict(zip(bx, by))
    all_x = set()
    labels = []
    for kern in ["flash", "cudnn", "triton_flash", "mem_efficient"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, "C_causal", kern)
        pts = [(xi, base[xi] / yi) for xi, yi in zip(x, y) if xi in base]
        if not pts:
            continue
        xs, sp = zip(*pts)
        all_x |= set(xs)
        ax.plot(xs, sp, color=color, marker=marker, ls=ls, label=label,
                markeredgecolor=SURFACE, markeredgewidth=1.2)
        labels.append((xs[-1], sp[-1], f"{sp[-1]:.1f}×"))
    ax.axhline(1.0, color=INK3, lw=1.2, ls="--")
    pow2_axis(ax, sorted(all_x))
    style(ax, "Causal (decoder-style) attention: speed-up over standard MHA",
          "sequence length N (tokens)", "speed-up (x)",
          "batch 8, 12 heads — flash skips fully-masked tiles entirely")
    ax.legend(loc="upper left")
    ax.margins(x=0.10)
    stagger_labels(ax, labels)
    save(fig, fname)


def fig_batch_sensitivity(rows, fname="08_batch_sensitivity.png"):
    sweeps = sorted({r["sweep"] for r in rows if r["sweep"].startswith("E_batch")},
                    key=lambda s: int(s.replace("E_batch", "")))
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    bs_vals = [int(s.replace("E_batch", "")) for s in sweeps]
    for kern in ["flash", "cudnn", "triton_flash", "mem_efficient"]:
        label, color, marker, ls = SERIES[kern]
        ys = []
        xs = []
        for s, bs in zip(sweeps, bs_vals):
            b = ok(sel(rows, sweep=s, kernel="naive"))
            k = ok(sel(rows, sweep=s, kernel=kern))
            if b and k:
                xs.append(bs)
                ys.append(b[0]["ms_median"] / k[0]["ms_median"])
        if xs:
            ax.plot(xs, ys, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            end_label(ax, xs[-1], ys[-1], f"{ys[-1]:.1f}x", color)
    ax.axhline(1.0, color=INK3, lw=1.2, ls="--")
    pow2_axis(ax, bs_vals)
    style(ax, "Speed-up vs batch size at N = 1024",
          "batch size", "speed-up over standard MHA (x)",
          "small batches leave the GPU under-occupied, which caps the achievable gain")
    ax.legend(loc="lower right")
    ax.margins(x=0.16)
    save(fig, fname)


def fig_headdim(rows, fname="09_head_dim.png"):
    sweeps = sorted({r["sweep"] for r in rows if r["sweep"].startswith("D_headdim")},
                    key=lambda s: int(s.replace("D_headdim", "")))
    if not sweeps:
        return
    dims = [int(s.replace("D_headdim", "")) for s in sweeps]
    kerns = ["naive", "mem_efficient", "triton_flash", "flash"]
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    w = 0.8 / len(kerns)
    for i, kern in enumerate(kerns):
        label, color, marker, ls = SERIES[kern]
        vals = []
        for s in sweeps:
            r = ok(sel(rows, sweep=s, kernel=kern))
            vals.append(r[0]["ms_median"] if r else np.nan)
        pos = np.arange(len(dims)) + (i - (len(kerns) - 1) / 2) * w
        bars = ax.bar(pos, vals, width=w * 0.88, color=color, label=label,
                      edgecolor=SURFACE, linewidth=1.6)
        for p, v in zip(pos, vals):
            if v == v:
                ax.annotate(f"{v:.2f}", (p, v), textcoords="offset points",
                            xytext=(0, 3), ha="center", fontsize=8, color=INK2)
    ax.set_xticks(np.arange(len(dims)))
    ax.set_xticklabels([f"head_dim {d}\n({768//d} heads)" for d in dims])
    ax.set_yscale("log")
    style(ax, "Head-dimension sensitivity at N = 2048",
          None, "latency (ms, log scale)",
          "embed_dim held at 768, batch 8 — the fused kernels stay flat, the baseline does not")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.30), ncol=4, fontsize=9)
    save(fig, fname)


def fig_dtype(rows, fname="10_bf16_vs_fp16.png"):
    if not sel(rows, sweep="F_bf16"):
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    kerns = ["naive", "mem_efficient", "triton_flash", "flash"]
    xs = sorted({r["seq_len"] for r in ok(sel(rows, sweep="F_bf16"))})
    w = 0.8 / (2 * len(kerns))
    for i, kern in enumerate(kerns):
        label, color, marker, ls = SERIES[kern]
        for j, (sw, hatch, tag) in enumerate([("A_main", None, "fp16"),
                                              ("F_bf16", "///", "bf16")]):
            vals = []
            for N in xs:
                r = ok([q for q in sel(rows, sweep=sw, kernel=kern)
                        if q["seq_len"] == N])
                vals.append(r[0]["ms_median"] if r else np.nan)
            pos = np.arange(len(xs)) + (2 * i + j - (2 * len(kerns) - 1) / 2) * w
            ax.bar(pos, vals, width=w * 0.9, color=color, hatch=hatch,
                   edgecolor=SURFACE, linewidth=1.2,
                   label=f"{label.split(' (')[0]} · {tag}")
    ax.set_xticks(np.arange(len(xs)))
    ax.set_xticklabels([f"N={n:,}" for n in xs])
    ax.set_yscale("log")
    style(ax, "fp16 vs bfloat16", None, "latency (ms, log scale)",
          "batch 16 — hatched bars are bf16; the choice of 16-bit format is not the lever")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    save(fig, fname)


def fig_layer(rows, fname="11_full_layer.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for ax, ykey, ttl, ylab in [
        (axes[0], "ms_median", "Full MHA layer latency", "latency (ms, log)"),
        (axes[1], "peak_mem_mb", "Full MHA layer peak memory", "peak memory (MB, log)")]:
        all_x = set()
        for kern in ["math", "mem_efficient", "triton_flash", "cudnn", "flash"]:
            label, color, marker, ls = SERIES[kern]
            x, y = series_for(rows, "layer", kern, ykey=ykey)
            if not x:
                continue
            all_x |= set(x)
            ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
        ax.set_yscale("log")
        pow2_axis(ax, sorted(all_x))
        style(ax, ttl, "sequence length N (tokens)", ylab)
    axes[0].legend(loc="upper left")
    fig.suptitle("Complete attention layer, projections included "
                 "(batch 8, embed 768, 12 heads)",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_train_step(rows, fname="12_train_step.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    for ax, ykey, ttl, ylab in [
        (axes[0], "ms_median", "Training step latency (forward + backward)",
         "latency (ms, log)"),
        (axes[1], "peak_mem_mb", "Training step peak memory", "peak memory (MB, log)")]:
        all_x = set()
        for kern in ["math", "mem_efficient", "flash"]:
            label, color, marker, ls = SERIES[kern]
            x, y = series_for(rows, "train_step", kern, ykey=ykey)
            if not x:
                continue
            all_x |= set(x)
            ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
            o = oom_points(rows, "train_step", kern)
            if o:
                ax.annotate(f"OOM ≥ N={min(o):,}", (x[-1], y[-1]),
                            textcoords="offset points", xytext=(6, -4), fontsize=8.5,
                            color=color)
        ax.set_yscale("log")
        pow2_axis(ax, sorted(all_x))
        style(ax, ttl, "sequence length N (tokens)", ylab)
    axes[0].legend(loc="upper left")
    fig.suptitle("Backward pass: recomputation buys a large memory saving and is "
                 "still faster", x=0.007, ha="left", fontsize=13.5, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


# ==========================================================================
# Decode / architecture-level figures
# ==========================================================================

def fig_decode(dec, fname="13_decode_kv_cache.png"):
    rows = [r for r in dec if r.get("status") == "ok"]
    if not rows:
        return
    ctxs = sorted({r["context_len"] for r in rows})
    layouts = ["MHA", "GQA-8", "GQA-4", "GQA-2", "MQA"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    ax = axes[0]
    for lay in layouts:
        rs = sorted([r for r in rows if r["layout"] == lay],
                    key=lambda r: r["context_len"])
        if not rs:
            continue
        ax.plot([r["context_len"] for r in rs], [r["kv_cache_mb"] for r in rs],
                color=LAYOUT_COLORS[lay], marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=lay)
        end_label(ax, rs[-1]["context_len"], rs[-1]["kv_cache_mb"], lay,
                  LAYOUT_COLORS[lay])
    ax.set_yscale("log")
    pow2_axis(ax, ctxs)
    style(ax, "KV-cache footprint per layer", "context length (tokens)",
          "KV cache (MB, log)")
    ax.margins(x=0.18)

    ax = axes[1]
    for lay in layouts:
        rs = sorted([r for r in rows if r["layout"] == lay],
                    key=lambda r: r["context_len"])
        if not rs:
            continue
        ax.plot([r["context_len"] for r in rs], [r["tokens_per_s"] for r in rs],
                color=LAYOUT_COLORS[lay], marker="o", markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=lay)
        ax.plot([r["context_len"] for r in rs],
                [r["tokens_per_s_naive_repeat"] for r in rs],
                color=LAYOUT_COLORS[lay], marker="x", ls=":", lw=1.4, alpha=0.75)
        end_label(ax, rs[-1]["context_len"], rs[-1]["tokens_per_s"], lay,
                  LAYOUT_COLORS[lay])
    ax.set_yscale("log")
    pow2_axis(ax, ctxs)
    style(ax, "Decode throughput (batch 32, 1 token/step)", "context length (tokens)",
          "tokens/s (log)")
    ax.margins(x=0.18)
    ax.legend(handles=[Line2D([], [], color=INK2, ls="-", marker="o",
                              label="kernel reads the narrow KV cache (enable_gqa)"),
                       Line2D([], [], color=INK2, ls=":", marker="x",
                              label="caller expands KV to full width first")],
              loc="lower left")
    fig.suptitle("Architecture-level optimisation: GQA / MQA shrink the KV cache",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_gqa_trap(dec, fname="14_gqa_implementation_trap.png"):
    rows = [r for r in dec if r.get("status") == "ok"]
    if not rows:
        return
    T = max(r["context_len"] for r in rows)
    rs = [r for r in rows if r["context_len"] == T]
    order = ["MHA", "GQA-8", "GQA-4", "GQA-2", "MQA"]
    rs = sorted(rs, key=lambda r: order.index(r["layout"]) if r["layout"] in order else 9)
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(rs))
    mha = next((r for r in rs if r["layout"] == "MHA"), rs[0])
    ax.bar(x - 0.2, [mha["ms_median"] / r["ms_median"] for r in rs], width=0.38,
           color="#2a78d6", edgecolor=SURFACE, linewidth=1.6,
           label="kernel consumes the narrow KV cache (enable_gqa)")
    ax.bar(x + 0.2, [mha["ms_median"] / r["ms_median_naive_repeat"] for r in rs],
           width=0.38, color="#eb6834", edgecolor=SURFACE, linewidth=1.6,
           label="caller expands KV to full width first (repeat_kv)")
    for xi, r in zip(x, rs):
        ax.annotate(f"{mha['ms_median']/r['ms_median']:.1f}x",
                    (xi - 0.2, mha["ms_median"] / r["ms_median"]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=INK2)
        ax.annotate(f"{mha['ms_median']/r['ms_median_naive_repeat']:.1f}x",
                    (xi + 0.2, mha["ms_median"] / r["ms_median_naive_repeat"]),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=8.5, color=INK2)
    ax.axhline(1.0, color=INK3, ls="--", lw=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([r["layout"] for r in rs])
    style(ax, "The same architecture, two implementations", None,
          f"decode speed-up over MHA (x), context {T:,}",
          "expanding K/V before the kernel throws away the entire benefit of GQA")
    ax.legend(loc="upper left")
    save(fig, fname)


# ==========================================================================
# Training figures
# ==========================================================================

def fig_training_curves(tr, ss=None, fname="15_training_curves.png"):
    arch = tr["architecture_variants"]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    ax = axes[0]
    for r in arch:
        lay = r["name"]
        h = r["history"]
        ax.plot(h["step"], h["val_loss"], color=LAYOUT_COLORS.get(lay, INK2),
                marker="o", ms=3.5, markeredgecolor=SURFACE, markeredgewidth=0.7,
                label=lay)
    ax.set_xlim(0, max(h["step"]) * 1.03)
    style(ax, "Validation loss during training", "training step",
          "validation loss (nats/char)")
    ax.legend(loc="upper right")
    # the curves are indistinguishable at full scale, so zoom the tail explicitly
    inset = ax.inset_axes([0.42, 0.30, 0.54, 0.44])
    for r in arch:
        h = r["history"]
        pts = [(s, v) for s, v in zip(h["step"], h["val_loss"]) if s >= 1500]
        if pts:
            xs_, ys_ = zip(*pts)
            inset.plot(xs_, ys_, color=LAYOUT_COLORS.get(r["name"], INK2), marker="o",
                       ms=3.5, markeredgecolor=SURFACE, markeredgewidth=0.7)
    inset.set_facecolor(SURFACE)
    inset.tick_params(labelsize=7.5, colors=INK2)
    for s in ("top", "right"):
        inset.spines[s].set_visible(False)
    inset.grid(color=GRID, lw=0.7)
    inset.set_title("tail, zoomed", fontsize=8.5, color=INK2, loc="left", pad=3)

    ax = axes[1]
    for r in arch:
        lay = r["name"]
        ax.scatter([r["kv_cache_bytes_per_token"]], [r["final_val_loss"]], s=110,
                   color=LAYOUT_COLORS.get(lay, INK2), edgecolor=SURFACE, linewidth=1.8,
                   zorder=3)
        ax.annotate(f"  {lay}\n  ppl {r['final_val_ppl']:.2f}",
                    (r["kv_cache_bytes_per_token"], r["final_val_loss"]),
                    textcoords="offset points", xytext=(8, -4), fontsize=9, color=INK)
    # a seed-noise band makes the scatter honest: the layout gaps sit inside it
    note = "single run per layout"
    if ss:
        summ = ss["summary"]
        sd = float(np.mean([v["std"] for v in summ.values()]))
        mid = float(np.mean([r["final_val_loss"] for r in arch]))
        ax.axhspan(mid - sd, mid + sd, color=INK3, alpha=0.15, lw=0, zorder=0)
        ax.annotate(f"±1 s.d. of seed noise (±{sd:.4f})", (0.03, 0.06),
                    xycoords="axes fraction", fontsize=9, color=INK2)
        note = ("the gaps between layouts are the size of the seed-noise band "
                "— see figure 18")
    ax.set_xscale("log", base=2)
    xs = sorted(r["kv_cache_bytes_per_token"] for r in arch)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}" for x in xs])
    ax.minorticks_off()
    style(ax, "Quality vs KV-cache cost", "KV cache bytes per token (whole model, log)",
          "final validation loss (lower is better)", note)
    ax.margins(x=0.30, y=0.40)
    fig.suptitle("Architecture-level methods change the model, so each needs its own "
                 "training run", x=0.007, ha="left", fontsize=13.5, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, fname)


def fig_kernel_equivalence(tr, fname="16_kernel_equivalence.png"):
    rows = [r for r in tr["kernel_equivalence"] if r.get("status") == "ok"]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["fwd_ms"])
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))
    names = [r["kernel"] for r in rows]
    colors = [SERIES.get(n, ("", INK2, "", ""))[1] for n in names]
    y = np.arange(len(rows))

    ax = axes[0]
    losses = [r["val_loss"] for r in rows]
    lo, hi = min(losses), max(losses)
    # plotting the raw losses from zero would render seven identical bars; plot the
    # deviation from the best kernel instead, in millionths of a nat
    dev = [(v - lo) * 1e6 for v in losses]
    ax.barh(y, dev, color=colors, edgecolor=SURFACE, linewidth=1.8, height=0.62)
    for yi, r, d in zip(y, rows, dev):
        ax.annotate(f"  {d:+,.0f}    (loss {r['val_loss']:.6f})", (d, yi),
                    textcoords="offset points", xytext=(4, 0), ha="left",
                    va="center", fontsize=8.8, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(0, max(dev) * 2.9 if max(dev) else 1)
    style(ax, "Validation loss — identical weights, different kernels",
          "difference from the best kernel (millionths of a nat)", None,
          f"total spread {hi-lo:.2e} nats = {100*(hi-lo)/lo:.4f}% of the loss")
    ax.grid(axis="y", visible=False)

    ax = axes[1]
    fast = min(r["fwd_ms"] for r in rows)
    slow = max(r["fwd_ms"] for r in rows)
    ax.barh(y, [r["fwd_ms"] for r in rows], color=colors, edgecolor=SURFACE,
            linewidth=1.8, height=0.62)
    for yi, r in zip(y, rows):
        ax.annotate(f"{r['fwd_ms']:.2f} ms   ({r['fwd_ms']/fast:.1f}x)",
                    (r["fwd_ms"], yi), textcoords="offset points", xytext=(6, 0),
                    va="center", fontsize=9, color=INK)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    style(ax, "Forward-pass time for the same model", "ms per forward pass", None,
          f"identical loss to 4 decimal places, {slow/fast:.1f}x apart in speed")
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.30)
    fig.suptitle("Kernel-level optimisation: free speed, no change to the model",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_seed_study(ss, fname="18_seed_noise.png"):
    """Is the layout ranking real, or is it seed noise? Show every run."""
    summ = ss["summary"]
    order = [l for l in ["MHA", "GQA-8", "GQA-4", "GQA-2", "MQA"] if l in summ]
    if not order:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.9))

    ax = axes[0]
    for i, lay in enumerate(order):
        v = summ[lay]
        color = LAYOUT_COLORS.get(lay, INK2)
        ax.errorbar([i], [v["mean"]], yerr=[v["std"]], fmt="none", ecolor=color,
                    elinewidth=2.2, capsize=7, capthick=2.2, zorder=2)
        ax.scatter([i] * len(v["losses"]), v["losses"], s=42, color=color,
                   edgecolor=SURFACE, linewidth=1.4, zorder=3, alpha=0.95)
        ax.scatter([i], [v["mean"]], s=130, marker="_", color=INK, zorder=4,
                   linewidth=2.4)
        ax.annotate(f"{v['mean']:.4f}\n±{v['std']:.4f}", (i, v["mean"]),
                    textcoords="offset points", xytext=(14, -6), fontsize=9,
                    color=INK2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_xlim(-0.5, len(order) - 0.15)
    style(ax, "Every run, not just the mean", None, "final validation loss",
          f"{summ[order[0]]['n']} seeds per layout — dots are individual runs, "
          "bars are ±1 s.d.")

    ax = axes[1]
    means = [summ[l]["mean"] for l in order]
    between = max(means) - min(means)
    within = float(np.mean([summ[l]["std"] for l in order]))
    ax.bar([0, 1], [between, within],
           color=["#2a78d6", "#eb6834"], edgecolor=SURFACE, linewidth=1.8, width=0.5)
    for i, v, lab in [(0, between, "spread between layouts\n(max mean − min mean)"),
                      (1, within, "seed noise within a layout\n(mean s.d.)")]:
        ax.annotate(f"{v:.4f} nats", (i, v), textcoords="offset points",
                    xytext=(0, 5), ha="center", fontsize=10, color=INK)
        ax.annotate(lab, (i, 0), textcoords="offset points", xytext=(0, -34),
                    ha="center", fontsize=9, color=INK2)
    ax.set_xticks([])
    ax.set_ylim(0, max(between, within) * 1.35)
    verdict = ("the layouts are not distinguishable at this scale"
               if between < 2 * within else
               "the layout difference is larger than seed noise")
    style(ax, "Signal vs noise", None, "nats",
          f"ratio {between/within:.2f}× — {verdict}")
    ax.grid(axis="x", visible=False)
    fig.suptitle("Honest reading of the architecture sweep: how much of the "
                 "difference is real?", x=0.007, ha="left", fontsize=13.5,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, fname)


# ==========================================================================
# Why it works: the HBM traffic model
# ==========================================================================

def fig_hbm_traffic(rows, env, peak_tflops=None, hbm_gb_s=None,
                    fname="17_roofline_mechanism.png"):
    """Roofline: is each kernel limited by arithmetic or by memory traffic?

    Both kernels execute the same FLOPs, so the compute-bound floor is identical.
    They differ only in bytes moved, so only the memory-bound ceiling moves. If the
    IO-awareness story is right, the baseline should sit on its memory line and the
    fused kernel on the shared compute line — which is exactly what is measured.
    """
    B, H, D = 16, 12, 64
    Ns = sorted({r["seq_len"] for r in ok(sel(rows, sweep="A_main"))})
    if not Ns or not peak_tflops or not hbm_gb_s:
        return
    el = 2  # fp16 bytes
    peak = peak_tflops * 1e12
    bw = hbm_gb_s * 1e9

    def flops(N):
        return 4.0 * B * H * N * N * D

    def qkvo_bytes(N):
        return 4 * B * H * N * D * el

    def naive_bytes(N, passes):
        # S: matmul writes it, softmax reads it.  P: softmax writes it, matmul reads
        # it. `passes` counts the N^2 transfers; 4 is the minimum, PyTorch's softmax
        # adds a separate reduction pass over S, which pushes it towards 6.
        return qkvo_bytes(N) + passes * B * H * N * N * el

    t_compute = [flops(n) / peak * 1e3 for n in Ns]
    t_naive_lo = [max(flops(n) / peak, naive_bytes(n, 4) / bw) * 1e3 for n in Ns]
    t_naive_hi = [max(flops(n) / peak, naive_bytes(n, 6) / bw) * 1e3 for n in Ns]
    t_flash = [max(flops(n) / peak, qkvo_bytes(n) / bw) * 1e3 for n in Ns]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))
    ax = axes[0]
    ax.fill_between(Ns, t_naive_lo, t_naive_hi, color=SERIES["naive"][1], alpha=0.16,
                    lw=0, zorder=1)
    ax.plot(Ns, t_naive_hi, color=SERIES["naive"][1], ls="--", lw=1.4, zorder=2)
    ax.plot(Ns, t_naive_lo, color=SERIES["naive"][1], ls="--", lw=1.4, zorder=2)
    ax.plot(Ns, t_flash, color=SERIES["flash"][1], ls="--", lw=1.4, zorder=2)
    bx, by = series_for(rows, "A_main", "naive")
    fx, fy = series_for(rows, "A_main", "flash")
    ax.plot(bx, by, color=SERIES["naive"][1], marker="s", markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3, label="standard MHA — measured")
    ax.plot(fx, fy, color=SERIES["flash"][1], marker="o", markeredgecolor=SURFACE,
            markeredgewidth=1.2, zorder=3, label="FlashAttention — measured")
    ax.plot([], [], color=INK3, ls="--", label="predicted bound (dashed)")
    ax.set_yscale("log")
    pow2_axis(ax, Ns)
    ax.annotate("standard MHA is pinned to its\nmemory-traffic bound (shaded band\n"
                "= 4 to 6 passes over the N×N matrices)",
                (Ns[1], t_naive_hi[1]), textcoords="offset points", xytext=(10, -46),
                fontsize=8.8, color=INK2)
    ax.annotate("FlashAttention sits on the\nshared compute-bound floor",
                (Ns[-2], t_flash[-2]), textcoords="offset points", xytext=(-4, 24),
                ha="right", fontsize=8.8, color=INK2)
    style(ax, "Roofline: measured latency vs predicted bounds",
          "sequence length N", "latency (ms, log)",
          f"device ceilings measured on this card: {peak_tflops:.0f} TFLOP/s, "
          f"{hbm_gb_s:.0f} GB/s")
    ax.legend(loc="upper left", fontsize=8.8)

    ax = axes[1]
    for kern in ["naive", "mem_efficient", "triton_flash", "flash"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, "A_main", kern, ykey="tflops")
        if not x:
            continue
        ax.plot(x, [100 * v / peak_tflops for v in y], color=color, marker=marker,
                ls=ls, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=label.split(" (")[0] if kern != "triton_flash"
                else "FlashAttention (our Triton)")
        end_label(ax, x[-1], 100 * y[-1] / peak_tflops,
                  f"{100*y[-1]/peak_tflops:.0f}%", color)
    ax.axhline(100, color=INK3, ls="--", lw=1.2)
    ax.annotate("device dense-matmul peak", (Ns[0], 100), textcoords="offset points",
                xytext=(2, 5), fontsize=8.8, color=INK2)
    pow2_axis(ax, Ns)
    ax.set_ylim(0, 118)
    style(ax, "Fraction of the machine's arithmetic peak actually used",
          "sequence length N", "% of peak fp16 matmul throughput",
          "identical FLOP counts — the baseline simply cannot keep the ALUs fed")
    ax.legend(loc="center left", fontsize=8.8)
    ax.margins(x=0.10)
    fig.suptitle("Why it is faster: same arithmetic, far less memory traffic",
                 x=0.007, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


# ==========================================================================
# PagedAttention (vLLM-style KV cache) figures
# ==========================================================================

PAGED_COLORS = {"contiguous": "#eb6834", "paged": "#2a78d6", "ideal": "#8a8984",
                "paged_share": "#1baf7a", "paged_noshare": "#eda100"}


def fig_paged_fragmentation(pg, fname="19_paged_fragmentation.png"):
    rows = pg.get("fragmentation", [])
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["mean_gen"])
    xs = [r["mean_gen"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    ax = axes[0]
    ax.plot(xs, [r["contiguous_gb"] for r in rows], color=PAGED_COLORS["contiguous"],
            marker="s", markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="contiguous cache (pre-allocate max_len)")
    ax.plot(xs, [r["paged_gb"] for r in rows], color=PAGED_COLORS["paged"],
            marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="paged cache (blocks on demand)")
    ax.plot(xs, [r["ideal_gb"] for r in rows], color=PAGED_COLORS["ideal"],
            ls=":", lw=1.5, label="ideal (one slot per real token)")
    for r in rows:
        end_label(ax, r["mean_gen"], r["paged_gb"], f"{r['reduction_x']:.0f}× less",
                  PAGED_COLORS["paged"])
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs]); ax.minorticks_off()
    style(ax, "Allocated KV memory vs generation length",
          "mean generation length (tokens)", "allocated KV cache (GB, log)",
          "256 sequences, max_len 2048 — contiguous is flat because it always books max_len")
    ax.legend(loc="lower right")
    ax.margins(x=0.16)

    ax = axes[1]
    ax.plot(xs, [100 * r["paged_util"] for r in rows], color=PAGED_COLORS["paged"],
            marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2, label="paged")
    ax.plot(xs, [100 * r["contiguous_util"] for r in rows],
            color=PAGED_COLORS["contiguous"], marker="s", markeredgecolor=SURFACE,
            markeredgewidth=1.2, label="contiguous")
    for r in rows:
        end_label(ax, r["mean_gen"], 100 * r["paged_util"],
                  f"{100*r['paged_util']:.0f}%", PAGED_COLORS["paged"])
        end_label(ax, r["mean_gen"], 100 * r["contiguous_util"],
                  f"{100*r['contiguous_util']:.0f}%", PAGED_COLORS["contiguous"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs]); ax.minorticks_off()
    ax.set_ylim(0, 108)
    style(ax, "KV-cache utilisation (useful ÷ allocated)",
          "mean generation length (tokens)", "% of allocated memory holding real tokens",
          "paged wastes at most one partial block per sequence")
    ax.legend(loc="center right")
    ax.margins(x=0.16)
    fig.suptitle("PagedAttention removes KV-cache fragmentation "
                 "(vLLM-style, from scratch)", x=0.007, ha="left", fontsize=13.5,
                 fontweight="bold", color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_paged_concurrency(pg, fname="20_paged_concurrency.png"):
    conc = sorted(pg.get("concurrency", []), key=lambda r: r["budget_gb"])
    tput = [r for r in pg.get("decode_throughput", []) if r.get("status") == "ok"]
    if not conc:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    ax = axes[0]
    xs = [r["budget_gb"] for r in conc]
    ax.plot(xs, [r["paged_seqs"] for r in conc], color=PAGED_COLORS["paged"],
            marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2, label="paged")
    ax.plot(xs, [r["contiguous_seqs"] for r in conc],
            color=PAGED_COLORS["contiguous"], marker="s", markeredgecolor=SURFACE,
            markeredgewidth=1.2, label="contiguous")
    for r in conc:
        end_label(ax, r["budget_gb"], r["paged_seqs"],
                  f"{r['paged_seqs']}  ({r['admit_ratio']:.1f}×)",
                  PAGED_COLORS["paged"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:g}" for x in xs]); ax.minorticks_off()
    style(ax, "Concurrent sequences admitted in a fixed KV budget",
          "KV memory budget (GB)", "sequences served (log)",
          "mean gen 256, max_len 2048 — paged admits ~7× more by not over-booking")
    ax.legend(loc="upper left")
    ax.margins(x=0.18)

    ax = axes[1]
    if tput:
        b = [r["batch"] for r in tput]
        ax.plot(b, [r["tokens_per_s"] for r in tput], color=PAGED_COLORS["paged"],
                marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2)
        # mark the batch each scheme could fit at the 16 GB budget
        ref = next((r for r in conc if r["budget_gb"] == 16), conc[-1])
        for seqs, col, lab in [(ref["contiguous_seqs"], PAGED_COLORS["contiguous"],
                                "contiguous fits"),
                               (ref["paged_seqs"], PAGED_COLORS["paged"],
                                "paged fits")]:
            near = min(tput, key=lambda r: abs(r["batch"] - seqs))
            ax.scatter([near["batch"]], [near["tokens_per_s"]], s=150, color=col,
                       edgecolor=SURFACE, linewidth=2, zorder=4)
            ax.annotate(f"{lab}\n≈{seqs} seqs", (near["batch"], near["tokens_per_s"]),
                        textcoords="offset points", xytext=(-4, -34), ha="center",
                        fontsize=8.5, color=col)
        ax.set_xscale("log", base=2)
        ax.set_xticks(b); ax.set_xticklabels([str(x) for x in b]); ax.minorticks_off()
    style(ax, "Measured decode throughput vs batch size",
          "concurrent sequences (batch)", "tokens/s (real, flash decode)",
          "per-GPU throughput saturates by ~batch 8 — the win is concurrency, not peak tok/s")
    ax.margins(x=0.12)
    fig.suptitle("More concurrent sequences per GPU in the same memory budget "
                 "(~7×)", x=0.007, ha="left", fontsize=13.5, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_paged_prefix_sharing(pg, fname="21_paged_prefix_sharing.png"):
    rows = sorted(pg.get("prefix_sharing", []), key=lambda r: r["n_seqs"])
    if not rows:
        return
    xs = [r["n_seqs"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(xs, [r["contiguous_gb"] for r in rows], color=PAGED_COLORS["contiguous"],
            marker="s", markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="contiguous / paged without sharing")
    ax.plot(xs, [r["paged_share_gb"] for r in rows], color=PAGED_COLORS["paged_share"],
            marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="paged with copy-on-write prompt sharing")
    last = rows[-1]
    end_label(ax, last["n_seqs"], last["contiguous_gb"],
              f"{last['contiguous_gb']:.1f} GB", PAGED_COLORS["contiguous"])
    end_label(ax, last["n_seqs"], last["paged_share_gb"],
              f"{last['paged_share_gb']:.1f} GB  ({last['vs_contiguous_x']:.1f}× less)",
              PAGED_COLORS["paged_share"])
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([str(x) for x in xs]); ax.minorticks_off()
    style(ax, "Shared-prompt serving: memory vs number of sequences",
          "sequences sharing one 512-token prompt",
          "total KV cache (GB, log)",
          "each sequence adds 128 generated tokens — the prompt is stored once, not N times")
    ax.legend(loc="upper left")
    ax.margins(x=0.22)
    save(fig, fname)


def fig_paged_latency(pg, fname="22_paged_latency_tradeoff.png"):
    rows = [r for r in pg.get("paged_latency", []) if r.get("status") == "ok"]
    if not rows:
        return
    rows = sorted(rows, key=lambda r: r["context"])
    xs = [r["context"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.0))

    ax = axes[0]
    ax.plot(xs, [r["contiguous_ms"] for r in rows], color=PAGED_COLORS["contiguous"],
            marker="s", markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="one contiguous tensor + SDPA")
    ax.plot(xs, [r["paged_ms"] for r in rows], color=PAGED_COLORS["paged"],
            marker="o", markeredgecolor=SURFACE, markeredgewidth=1.2,
            label="paged: gather blocks, then SDPA (unfused)")
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(xs); ax.set_xticklabels([f"{x:,}" for x in xs]); ax.minorticks_off()
    style(ax, "Decode-step latency (batch 64, equal length)",
          "context length (tokens)", "ms per step (log)",
          "the cost side: the gather writes a scratch KV copy SDPA then re-reads")
    ax.legend(loc="upper left")
    ax.margins(x=0.12)

    ax = axes[1]
    ax.bar(range(len(xs)), [r["overhead_pct"] for r in rows], width=0.6,
           color=PAGED_COLORS["paged"], edgecolor=SURFACE, linewidth=1.6)
    for i, r in enumerate(rows):
        ax.annotate(f"+{r['overhead_pct']:.0f}%", (i, r["overhead_pct"]),
                    textcoords="offset points", xytext=(0, 4), ha="center",
                    fontsize=9, color=INK2)
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([f"{x:,}" for x in xs])
    style(ax, "Overhead of the unfused gather path", "context length (tokens)",
          "extra time vs contiguous (%)",
          "this is exactly what vLLM's fused paged kernel removes — see note")
    ax.grid(axis="x", visible=False)
    fig.suptitle("The trade-off: paging saves memory but the unfused gather costs "
                 "compute", x=0.007, ha="left", fontsize=13.5, fontweight="bold",
                 color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, fname)


def fig_summary(rows, dec, tr, fname="00_summary_dashboard.png"):
    fig = plt.figure(figsize=(13.5, 8.6))
    gs = fig.add_gridspec(2, 2, hspace=0.42, wspace=0.24)

    # latency
    ax = fig.add_subplot(gs[0, 0])
    all_x = set()
    for kern in ["naive", "mem_efficient", "flash"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, "A_main", kern)
        if not x:
            continue
        all_x |= set(x)
        ax.plot(x, y, color=color, marker=marker, ls=ls, label=label.split(" (")[0],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_yscale("log")
    pow2_axis(ax, sorted(all_x))
    style(ax, "Latency", "N", "ms (log)")
    ax.legend(loc="upper left", fontsize=8.5)

    # memory
    ax = fig.add_subplot(gs[0, 1])
    for kern in ["naive", "flash"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, "A_main", kern, ykey="peak_mem_mb")
        ax.plot(x, y, color=color, marker=marker, ls=ls, label=label.split(" (")[0],
                markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_yscale("log")
    pow2_axis(ax, sorted(all_x))
    style(ax, "Peak workspace memory", "N", "MB (log)")
    ax.legend(loc="upper left", fontsize=8.5)

    # speedup
    ax = fig.add_subplot(gs[1, 0])
    bx, by = series_for(rows, "A_main", "naive")
    base = dict(zip(bx, by))
    for kern in ["flash", "mem_efficient", "triton_flash"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, "A_main", kern)
        pts = [(a, base[a] / b) for a, b in zip(x, y) if a in base]
        if pts:
            xs, sp = zip(*pts)
            ax.plot(xs, sp, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.axhline(1, color=INK3, ls="--", lw=1.2)
    pow2_axis(ax, sorted(all_x))
    style(ax, "Speed-up over standard MHA", "N", "speed-up (x)")
    ax.legend(loc="upper left", fontsize=8.5)

    # accuracy
    ax = fig.add_subplot(gs[1, 1])
    for kern in ["naive", "flash", "mem_efficient", "triton_flash"]:
        label, color, marker, ls = SERIES[kern]
        x, y = series_for(rows, "A_main", kern, ykey="max_abs_err_vs_fp64")
        xy = [(a, b) for a, b in zip(x, y) if b == b and b > 0]
        if xy:
            x, y = zip(*xy)
            ax.plot(x, y, color=color, marker=marker, ls=ls, label=label,
                    markeredgecolor=SURFACE, markeredgewidth=1.2)
    ax.set_yscale("log")
    pow2_axis(ax, sorted(all_x))
    style(ax, "Max abs error vs float64 truth", "N", "max abs error")
    ax.legend(loc="lower left", fontsize=8.5)

    fig.suptitle("FlashAttention vs standard multi-head self-attention — RTX 5070, "
                 "batch 16, 12 heads, d=64, fp16",
                 x=0.007, ha="left", fontsize=14.5, fontweight="bold", color=INK)
    save(fig, fname)


# ==========================================================================
def main():
    print("Generating figures...")
    rows = load("kernel_benchmarks.json") or []
    layer = load("layer_benchmarks.json") or []
    train_step = load("train_step_benchmarks.json") or []
    dec = load("decode_benchmarks.json") or []
    tr = load("training_results.json")
    env = load("environment.json") or {}
    mm = load("matmul_peak.json") or {}
    peak = mm.get("peak_tflops")
    hbm = mm.get("hbm_gb_s")

    if rows:
        fig_latency(rows)
        fig_speedup(rows)
        fig_memory(rows)
        fig_throughput(rows, peak_tflops=peak)
        fig_accuracy(rows)
        fig_longctx(rows)
        fig_causal(rows)
        fig_batch_sensitivity(rows)
        fig_headdim(rows)
        fig_dtype(rows)
        fig_hbm_traffic(rows, env, peak_tflops=peak, hbm_gb_s=hbm)
    if layer:
        fig_layer(layer)
    if train_step:
        fig_train_step(train_step)
    if dec:
        fig_decode(dec)
        fig_gqa_trap(dec)
    ss = load("seed_study.json")
    if tr:
        fig_training_curves(tr, ss)
        fig_kernel_equivalence(tr)
    if ss:
        fig_seed_study(ss)
    pg = load("paged_benchmarks.json")
    if pg:
        fig_paged_fragmentation(pg)
        fig_paged_concurrency(pg)
        fig_paged_prefix_sharing(pg)
        fig_paged_latency(pg)
    if rows:
        fig_summary(rows, dec, tr)
    print("done.")


if __name__ == "__main__":
    main()
