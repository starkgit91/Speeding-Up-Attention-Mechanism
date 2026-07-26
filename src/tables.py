"""
Turn the benchmark JSON into markdown + CSV tables for the report.

Writes  ../results/tables.md  and  ../results/*.csv
"""

from __future__ import annotations

import csv
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

NAMES = {
    "naive": "Standard MHA (explicit N x N)",
    "math": "PyTorch SDPA - math backend",
    "mem_efficient": "Memory-efficient (xFormers-style)",
    "cudnn": "cuDNN fused attention",
    "flash": "FlashAttention (SDPA flash)",
    "auto": "PyTorch SDPA - auto dispatch",
    "triton_flash": "FlashAttention (our Triton kernel)",
    "tiled_flash": "Flash algorithm (pure PyTorch)",
}
ORDER = ["naive", "math", "tiled_flash", "mem_efficient", "cudnn", "triton_flash",
         "auto", "flash"]


def load(n):
    p = os.path.join(RESULTS, n)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def sel(rows, **kw):
    out = rows
    for k, v in kw.items():
        out = [r for r in out if r.get(k) == v]
    return out


def fmt(v, spec="{:.3f}", na="—"):
    if v is None:
        return na
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return na
    return spec.format(v)


def md_table(header, rows, align=None):
    align = align or ["---"] * len(header)
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def write_csv(name, header, rows):
    path = os.path.join(RESULTS, name)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def cell(r, key, spec="{:.3f}"):
    if r is None:
        return "—"
    if r.get("status") == "OOM":
        return "**OOM**"
    if r.get("status") != "ok":
        return r.get("status", "—")
    return fmt(r.get(key), spec)


# ---------------------------------------------------------------------------
def headline_table(rows, sweep="A_main", baseline="naive", opt="flash"):
    """The table the assignment asks for: speed, memory, output difference."""
    Ns = sorted({r["seq_len"] for r in sel(rows, sweep=sweep)})
    header = ["Seq. length N", "Standard MHA (ms)", "FlashAttention (ms)",
              "Speed-up", "Peak workspace: standard", "Peak workspace: flash",
              "Memory saved", "Max abs. difference", "Cosine similarity"]
    out, csv_rows = [], []
    for N in Ns:
        b = next((r for r in sel(rows, sweep=sweep, kernel=baseline)
                  if r["seq_len"] == N), None)
        f = next((r for r in sel(rows, sweep=sweep, kernel=opt)
                  if r["seq_len"] == N), None)
        if b is None or f is None:
            continue
        bad = b.get("status") != "ok"
        sp = ("— (baseline OOM)" if bad
              else f"**{b['ms_median']/f['ms_median']:.2f}x**")
        msave = ("—" if bad
                 else f"{100*(1-f['peak_mem_mb']/b['peak_mem_mb']):.1f}%")
        out.append([f"{N:,}",
                    cell(b, "ms_median"), cell(f, "ms_median"), sp,
                    ("**OOM**" if bad else f"{b['peak_mem_mb']:,.1f} MB"),
                    f"{f['peak_mem_mb']:,.1f} MB", msave,
                    cell(f, "max_abs_err_vs_math", "{:.2e}"),
                    cell(f, "cosine_vs_math", "{:.8f}")])
        csv_rows.append([N,
                         b.get("ms_median") if not bad else "OOM",
                         f.get("ms_median"),
                         (b["ms_median"] / f["ms_median"]) if not bad else "",
                         b.get("peak_mem_mb") if not bad else "OOM",
                         f.get("peak_mem_mb"),
                         (100 * (1 - f["peak_mem_mb"] / b["peak_mem_mb"]))
                         if not bad else "",
                         f.get("max_abs_err_vs_math"), f.get("cosine_vs_math")])
    write_csv("headline_results.csv", header, csv_rows)
    return md_table(header, out,
                    ["---:"] * 1 + ["---:"] * 8)


def full_kernel_table(rows, sweep, caption_ns=None):
    Ns = caption_ns or sorted({r["seq_len"] for r in sel(rows, sweep=sweep)})
    header = ["Kernel"] + [f"N={n:,}" for n in Ns] + ["peak mem @ largest N",
                                                      "TFLOP/s @ largest N",
                                                      "rel. err vs fp64"]
    out, csv_rows = [], []
    for k in ORDER:
        rs = {r["seq_len"]: r for r in sel(rows, sweep=sweep, kernel=k)}
        if not rs:
            continue
        cells = [cell(rs.get(n), "ms_median") for n in Ns]
        last_ok = [rs[n] for n in Ns if n in rs and rs[n].get("status") == "ok"]
        tail = last_ok[-1] if last_ok else None
        out.append([NAMES.get(k, k)] + cells
                   + [cell(tail, "peak_mem_mb", "{:,.1f} MB"),
                      cell(tail, "tflops", "{:.1f}"),
                      cell(tail, "rel_err_vs_fp64", "{:.2e}")])
        csv_rows.append([k] + [(rs[n].get("ms_median") if n in rs
                                and rs[n].get("status") == "ok" else
                                rs[n].get("status") if n in rs else "")
                               for n in Ns]
                        + [tail.get("peak_mem_mb") if tail else "",
                           tail.get("tflops") if tail else "",
                           tail.get("rel_err_vs_fp64") if tail else ""])
    write_csv(f"kernels_{sweep}.csv", header, csv_rows)
    return md_table(header, out, ["---"] + ["---:"] * (len(header) - 1))


def decode_table(dec):
    rows = [r for r in dec if r.get("status") == "ok"]
    if not rows:
        return ""
    ctxs = sorted({r["context_len"] for r in rows})
    order = ["MHA", "GQA-8", "GQA-4", "GQA-2", "MQA"]
    header = ["Layout", "KV heads", "KV cache @ 16k ctx", "params"] + \
             [f"tok/s @ {c:,}" for c in ctxs] + ["speed-up vs MHA @ longest",
                                                 "same, with repeat_kv"]
    out, csv_rows = [], []
    ref = {c: next((r for r in rows if r["layout"] == "MHA"
                    and r["context_len"] == c), None) for c in ctxs}
    for lay in order:
        rs = {r["context_len"]: r for r in rows if r["layout"] == lay}
        if not rs:
            continue
        big = ctxs[-1]
        a = rs.get(big)
        r0 = ref.get(big)
        sp = (f"**{r0['ms_median']/a['ms_median']:.2f}x**"
              if a and r0 else "—")
        spr = (f"{r0['ms_median']/a['ms_median_naive_repeat']:.2f}x"
               if a and r0 else "—")
        any_r = next(iter(rs.values()))
        out.append([lay, any_r["n_kv_heads"],
                    f"{a['kv_cache_mb']:,.1f} MB" if a else "—",
                    f"{any_r['params_m']:.2f} M"]
                   + [fmt(rs[c]["tokens_per_s"], "{:,.0f}") if c in rs else "—"
                      for c in ctxs] + [sp, spr])
        csv_rows.append([lay, any_r["n_kv_heads"],
                         a["kv_cache_mb"] if a else "", any_r["params_m"]]
                        + [rs[c]["tokens_per_s"] if c in rs else "" for c in ctxs]
                        + [(r0["ms_median"] / a["ms_median"]) if a and r0 else "",
                           (r0["ms_median"] / a["ms_median_naive_repeat"])
                           if a and r0 else ""])
    write_csv("decode_results.csv", header, csv_rows)
    return md_table(header, out, ["---"] + ["---:"] * (len(header) - 1))


def training_tables(tr):
    arch = tr["architecture_variants"]
    header = ["Layout", "KV heads", "Params", "KV cache / token",
              "Final val loss", "Val perplexity", "Δ loss vs MHA",
              "Train tokens/s", "Peak train mem"]
    base = next(r for r in arch if r["name"] == "MHA")
    out, csv_rows = [], []
    for r in arch:
        out.append([r["name"], r["config"]["n_kv_head"], f"{r['params_m']:.2f} M",
                    f"{r['kv_cache_bytes_per_token']:,} B",
                    f"{r['final_val_loss']:.4f}", f"{r['final_val_ppl']:.3f}",
                    f"{r['final_val_loss']-base['final_val_loss']:+.4f}",
                    f"{r['tokens_per_s']:,.0f}", f"{r['peak_mem_mb']:,.0f} MB"])
        csv_rows.append([r["name"], r["config"]["n_kv_head"], r["params_m"],
                         r["kv_cache_bytes_per_token"], r["final_val_loss"],
                         r["final_val_ppl"],
                         r["final_val_loss"] - base["final_val_loss"],
                         r["tokens_per_s"], r["peak_mem_mb"]])
    write_csv("training_architecture.csv", header, csv_rows)
    t1 = md_table(header, out, ["---"] + ["---:"] * (len(header) - 1))

    kr = [r for r in tr["kernel_equivalence"] if r.get("status") == "ok"]
    header2 = ["Kernel", "Val loss (same weights)", "Val perplexity",
               "Δ vs first kernel", "Forward (ms)", "Relative speed",
               "Peak memory"]
    fast = min(r["fwd_ms"] for r in kr) if kr else 1
    out2, csv2 = [], []
    for r in sorted(kr, key=lambda r: r["fwd_ms"]):
        out2.append([NAMES.get(r["kernel"], r["kernel"]), f"{r['val_loss']:.6f}",
                     f"{r['val_ppl']:.4f}", f"{r['delta_vs_first']:+.2e}",
                     f"{r['fwd_ms']:.3f}", f"{r['fwd_ms']/fast:.2f}x",
                     f"{r['peak_mem_mb']:,.1f} MB"])
        csv2.append([r["kernel"], r["val_loss"], r["val_ppl"], r["delta_vs_first"],
                     r["fwd_ms"], r["fwd_ms"] / fast, r["peak_mem_mb"]])
    write_csv("kernel_equivalence.csv", header2, csv2)
    t2 = md_table(header2, out2, ["---"] + ["---:"] * (len(header2) - 1))
    return t1, t2


def main():
    rows = load("kernel_benchmarks.json") or []
    layer = load("layer_benchmarks.json") or []
    tstep = load("train_step_benchmarks.json") or []
    dec = load("decode_benchmarks.json") or []
    tr = load("training_results.json")
    env = load("environment.json") or {}
    peak = load("matmul_peak.json") or {}

    parts = ["# Measured results\n",
             "All numbers measured on the machine described below. Every table here is "
             "generated directly from the benchmark JSON in `results/` — nothing is "
             "transcribed by hand.\n"]

    parts.append("## Environment\n")
    parts.append(md_table(["Item", "Value"],
                          [[k, str(v)] for k, v in env.items()], ["---", "---"]))
    if peak:
        parts.append(f"\nMeasured device ceilings: **{peak['peak_tflops']:.1f} TFLOP/s** "
                     f"dense fp16 matmul (N={peak['peak_at_n']}), "
                     f"**{peak['hbm_gb_s']:.0f} GB/s** HBM copy bandwidth.\n")

    if rows:
        parts.append("\n## 1. Headline: standard MHA vs FlashAttention\n")
        parts.append("Configuration: batch 16, embed 768, 12 heads, head_dim 64, fp16, "
                     "non-causal. Median of 50 timed runs after 10 warm-up iterations.\n")
        parts.append(headline_table(rows))
        parts.append("\n## 2. All kernels, forward latency (ms)\n")
        parts.append(full_kernel_table(rows, "A_main"))
        parts.append("\n## 3. Long context (batch 4) — where the baseline stops fitting\n")
        parts.append(full_kernel_table(rows, "B_longctx"))
        parts.append("\n## 4. Causal / decoder-style attention (batch 8)\n")
        parts.append(full_kernel_table(rows, "C_causal"))
        bsweeps = sorted({r["sweep"] for r in rows if r["sweep"].startswith("E_batch")},
                         key=lambda s: int(s.replace("E_batch", "")))
        if bsweeps:
            parts.append("\n## 5. Batch-size sensitivity at N=1024\n")
            hdr = ["Kernel"] + [f"batch {s.replace('E_batch','')}" for s in bsweeps]
            body = []
            for k in ORDER:
                cells = []
                any_ = False
                for s in bsweeps:
                    r = next(iter(sel(rows, sweep=s, kernel=k)), None)
                    if r:
                        any_ = True
                    cells.append(cell(r, "ms_median"))
                if any_:
                    body.append([NAMES.get(k, k)] + cells)
            parts.append(md_table(hdr, body, ["---"] + ["---:"] * len(bsweeps)))
        dsweeps = sorted({r["sweep"] for r in rows
                          if r["sweep"].startswith("D_headdim")},
                         key=lambda s: int(s.replace("D_headdim", "")))
        if dsweeps:
            parts.append("\n## 6. Head-dimension sensitivity at N=2048 (embed 768)\n")
            hdr = ["Kernel"] + [f"head_dim {s.replace('D_headdim','')}"
                                for s in dsweeps]
            body = []
            for k in ORDER:
                cells, any_ = [], False
                for s in dsweeps:
                    r = next(iter(sel(rows, sweep=s, kernel=k)), None)
                    if r:
                        any_ = True
                    cells.append(cell(r, "ms_median"))
                if any_:
                    body.append([NAMES.get(k, k)] + cells)
            parts.append(md_table(hdr, body, ["---"] + ["---:"] * len(dsweeps)))
        if sel(rows, sweep="F_bf16"):
            parts.append("\n## 7. bfloat16\n")
            parts.append(full_kernel_table(rows, "F_bf16"))

    if layer:
        parts.append("\n## 8. Complete MHA layer (Q/K/V/O projections included)\n")
        parts.append(full_kernel_table(layer, "layer"))
    if tstep:
        parts.append("\n## 9. Training step (forward + backward), causal, batch 4\n")
        parts.append(full_kernel_table(tstep, "train_step"))
    if dec:
        parts.append("\n## 10. Autoregressive decode with a KV cache "
                     "(batch 32, embed 1024, 16 heads)\n")
        parts.append(decode_table(dec))
    if tr:
        t1, t2 = training_tables(tr)
        parts.append("\n## 11. Architecture-level: models trained from scratch\n")
        parts.append(t1)
        parts.append("\n## 12. Kernel-level: one trained checkpoint, every kernel\n")
        parts.append(t2)
    ss = load("seed_study.json")
    if ss:
        summ = ss["summary"]
        parts.append("\n## 13. Seed-variance study — is the layout ranking real?\n")
        header = ["Layout", "KV cache / token", "Mean val loss", "Std dev",
                  "Min", "Max"] + [f"seed {s}" for s in ss["seeds"]]
        body, csv_rows = [], []
        for lay, v in summ.items():
            body.append([lay, f"{v['kv_bytes_per_token']:,} B", f"{v['mean']:.4f}",
                         f"{v['std']:.4f}", f"{v['min']:.4f}", f"{v['max']:.4f}"]
                        + [f"{x:.4f}" for x in v["losses"]])
            csv_rows.append([lay, v["kv_bytes_per_token"], v["mean"], v["std"],
                             v["min"], v["max"]] + v["losses"])
        write_csv("seed_study.csv", header, csv_rows)
        parts.append(md_table(header, body, ["---"] + ["---:"] * (len(header) - 1)))
        means = [v["mean"] for v in summ.values()]
        between = max(means) - min(means)
        within = sum(v["std"] for v in summ.values()) / len(summ)
        parts.append(
            f"\nSpread **between** layouts (max mean − min mean): **{between:.4f}** nats. "
            f"Average **within**-layout seed standard deviation: **{within:.4f}** nats. "
            f"Ratio **{between/within:.2f}×**.\n")

    pg = load("paged_benchmarks.json")
    if pg:
        m = pg["model"]
        bpt = 2 * m["n_layers"] * m["n_kv_heads"] * m["head_dim"] * m["dtype_bytes"]
        parts.append("\n## 14. PagedAttention — KV-cache fragmentation "
                     f"(model: {m['n_layers']} layers, {m['n_kv_heads']} KV heads, "
                     f"head_dim {m['head_dim']}, {bpt/1024:.0f} KB/token)\n")
        fr = sorted(pg["fragmentation"], key=lambda r: r["mean_gen"])
        header = ["Mean gen length", "Contiguous cache", "Paged cache", "Ideal",
                  "Contiguous util", "Paged util", "Memory saved", "Reduction"]
        body, csv_rows = [], []
        for r in fr:
            body.append([f"{r['mean_gen']:,}", f"{r['contiguous_gb']:.2f} GB",
                         f"{r['paged_gb']:.2f} GB", f"{r['ideal_gb']:.2f} GB",
                         f"{100*r['contiguous_util']:.1f}%",
                         f"{100*r['paged_util']:.1f}%",
                         f"{r['memory_saved_pct']:.1f}%", f"{r['reduction_x']:.1f}×"])
            csv_rows.append([r["mean_gen"], r["contiguous_gb"], r["paged_gb"],
                             r["ideal_gb"], r["contiguous_util"], r["paged_util"],
                             r["memory_saved_pct"], r["reduction_x"]])
        write_csv("paged_fragmentation.csv", header, csv_rows)
        parts.append(md_table(header, body, ["---"] + ["---:"] * (len(header) - 1)))

        parts.append("\n## 15. PagedAttention — concurrent sequences in a fixed "
                     "KV budget (mean gen 256, max_len 2048)\n")
        cc = sorted(pg["concurrency"], key=lambda r: r["budget_gb"])
        header = ["KV budget", "Contiguous seqs", "Paged seqs", "More"]
        body, csv_rows = [], []
        for r in cc:
            body.append([f"{r['budget_gb']:g} GB", f"{r['contiguous_seqs']:,}",
                         f"{r['paged_seqs']:,}", f"{r['admit_ratio']:.1f}×"])
            csv_rows.append([r["budget_gb"], r["contiguous_seqs"], r["paged_seqs"],
                             r["admit_ratio"]])
        write_csv("paged_concurrency.csv", header, csv_rows)
        parts.append(md_table(header, body, ["---"] + ["---:"] * (len(header) - 1)))

        parts.append("\n## 16. PagedAttention — prefix sharing (shared 512-token "
                     "prompt + 128 generated tokens each)\n")
        ps = sorted(pg["prefix_sharing"], key=lambda r: r["n_seqs"])
        header = ["Sequences", "Contiguous", "Paged (no sharing)",
                  "Paged (shared prompt)", "Sharing saves", "vs contiguous"]
        body, csv_rows = [], []
        for r in ps:
            body.append([f"{r['n_seqs']:,}", f"{r['contiguous_gb']:.3f} GB",
                         f"{r['paged_noshare_gb']:.3f} GB",
                         f"{r['paged_share_gb']:.3f} GB",
                         f"{r['share_saves_pct']:.0f}%",
                         f"{r['vs_contiguous_x']:.1f}×"])
            csv_rows.append([r["n_seqs"], r["contiguous_gb"], r["paged_noshare_gb"],
                             r["paged_share_gb"], r["share_saves_pct"],
                             r["vs_contiguous_x"]])
        write_csv("paged_prefix_sharing.csv", header, csv_rows)
        parts.append(md_table(header, body, ["---"] + ["---:"] * (len(header) - 1)))

        parts.append("\n## 17. PagedAttention — unfused gather latency overhead "
                     "(batch 64, equal length)\n")
        pl = sorted([r for r in pg["paged_latency"] if r.get("status") == "ok"],
                    key=lambda r: r["context"])
        header = ["Context length", "Contiguous + SDPA", "Paged gather + SDPA",
                  "Overhead"]
        body, csv_rows = [], []
        for r in pl:
            body.append([f"{r['context']:,}", f"{r['contiguous_ms']:.3f} ms",
                         f"{r['paged_ms']:.3f} ms", f"+{r['overhead_pct']:.0f}%"])
            csv_rows.append([r["context"], r["contiguous_ms"], r["paged_ms"],
                             r["overhead_pct"]])
        write_csv("paged_latency.csv", header, csv_rows)
        parts.append(md_table(header, body, ["---"] + ["---:"] * (len(header) - 1)))

    md = "\n\n".join(parts) + "\n"
    path = os.path.join(RESULTS, "tables.md")
    with open(path, "w") as f:
        f.write(md)
    print(f"  wrote {path}  ({len(md.splitlines())} lines)")


if __name__ == "__main__":
    main()
