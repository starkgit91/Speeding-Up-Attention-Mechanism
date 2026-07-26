"""
Cross-check the numbers written into REPORT.md against the raw benchmark JSON.

Every quantitative claim in the report should be traceable to a measurement. This
script re-derives the headline claims from `results/*.json` and asserts they match
what the prose says, so a stale number cannot survive a re-run of the benchmarks.

Run:  .venv/bin/python tests/verify_report.py
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
RESULTS = os.path.join(ROOT, "results")
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
checks: list[tuple[str, bool, str]] = []


def load(n):
    with open(os.path.join(RESULTS, n)) as f:
        return json.load(f)


def check(name, claimed, actual, tol=0.02, unit=""):
    """Relative-tolerance comparison; report both values either way."""
    if actual is None:
        ok = False
        detail = "no measurement found"
    else:
        rel = abs(claimed - actual) / max(abs(actual), 1e-30)
        ok = rel <= tol
        detail = f"report says {claimed:g}{unit}, data says {actual:.6g}{unit} ({rel*100:.2f}% off)"
    checks.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}: {detail}")


def get(rows, sweep, kernel, N, field="ms_median"):
    for r in rows:
        if (r.get("sweep") == sweep and r.get("kernel") == kernel
                and r.get("seq_len") == N):
            return r.get(field) if r.get("status") == "ok" else None
    return None


def status(rows, sweep, kernel, N):
    for r in rows:
        if (r.get("sweep") == sweep and r.get("kernel") == kernel
                and r.get("seq_len") == N):
            return r.get("status")
    return None


def main():
    print("=" * 78)
    print("VERIFYING REPORT.md CLAIMS AGAINST RAW MEASUREMENTS")
    print("=" * 78)
    k = load("kernel_benchmarks.json")
    layer = load("layer_benchmarks.json")
    tstep = load("train_step_benchmarks.json")
    dec = load("decode_benchmarks.json")
    tr = load("training_results.json")
    ss = load("seed_study.json")
    peak = load("matmul_peak.json")
    pg = load("paged_benchmarks.json")

    print("\n§5.1 Headline table (sweep A, batch 16, fp16, non-causal)")
    for N, ms_n, ms_f, sp, mem_n, mem_f, saved in [
        (128, 0.044, 0.024, 1.82, 15.7, 3.2, 79.4),
        (256, 0.188, 0.053, 3.52, 56.6, 6.5, 88.5),
        (512, 1.194, 0.200, 5.96, 213.9, 13.0, 93.9),
        (1024, 4.514, 0.786, 5.74, 830.5, 26.0, 96.9),
        (2048, 19.747, 3.139, 6.29, 3271.6, 51.9, 98.4),
    ]:
        a_n = get(k, "A_main", "naive", N)
        a_f = get(k, "A_main", "flash", N)
        check(f"N={N} standard MHA latency", ms_n, a_n, 0.03, " ms")
        check(f"N={N} flash latency", ms_f, a_f, 0.03, " ms")
        check(f"N={N} speed-up", sp, (a_n / a_f) if a_n and a_f else None, 0.03, "x")
        mn = get(k, "A_main", "naive", N, "peak_mem_mb")
        mf = get(k, "A_main", "flash", N, "peak_mem_mb")
        check(f"N={N} standard MHA memory", mem_n, mn, 0.02, " MB")
        check(f"N={N} flash memory", mem_f, mf, 0.02, " MB")
        check(f"N={N} memory saved", saved,
              100 * (1 - mf / mn) if mn and mf else None, 0.01, "%")

    print("\n§5.1 / §5.3 OOM claims")
    for label, sweep, kern, N in [("standard MHA OOMs at N=4096 (batch 16)", "A_main",
                                   "naive", 4096),
                                  ("math backend OOMs at N=4096 (batch 16)", "A_main",
                                   "math", 4096),
                                  ("standard MHA OOMs at N=8192 (batch 4)", "B_longctx",
                                   "naive", 8192)]:
        st = status(k, sweep, kern, N)
        checks.append((label, st == "OOM", f"status={st}"))
        print(f"  [{PASS if st == 'OOM' else FAIL}] {label}: status={st}")
    for label, N, ms, mem in [("flash at N=8192 (batch 4)", 8192, 13.31, 51.9),
                              ("flash at N=16384 (batch 4)", 16384, 53.48, 103.8),
                              ("flash at N=32768 (batch 4)", 32768, 213.79, 207.6)]:
        check(label + " latency", ms, get(k, "B_longctx", "flash", N), 0.03, " ms")
        check(label + " memory", mem, get(k, "B_longctx", "flash", N, "peak_mem_mb"),
              0.02, " MB")

    print("\n§3.1 / §3.4 roofline claims")
    pk = peak["peak_tflops"]
    check("device fp16 matmul peak", 67.2, pk, 0.02, " TFLOP/s")
    check("device HBM bandwidth", 561, peak["hbm_gb_s"], 0.02, " GB/s")
    tf_naive = get(k, "A_main", "naive", 2048, "tflops")
    tf_flash2 = get(k, "A_main", "flash", 2048, "tflops")
    check("naive % of peak at N=2048", 15.5, 100 * tf_naive / pk, 0.06, "%")
    check("flash % of peak at N=2048", 97.8, 100 * tf_flash2 / pk, 0.03, "%")
    check("naive TFLOP/s at N=2048", 10.4, tf_naive, 0.03, " TFLOP/s")

    print("\n§3.5 control experiment (tiling without fusion)")
    for N, tiled, naive in [(128, 0.179, 0.044), (256, 0.710, 0.188),
                            (512, 2.749, 1.194)]:
        check(f"N={N} unfused tiled algorithm", tiled,
              get(k, "A_main", "tiled_flash", N), 0.05, " ms")
    t512 = get(k, "A_main", "tiled_flash", 512)
    n512 = get(k, "A_main", "naive", 512)
    check("unfused algorithm is 2.3x SLOWER than baseline at N=512", 2.3,
          t512 / n512 if t512 and n512 else None, 0.06, "x")
    check("unfused tiled memory at N=512", 69.8,
          get(k, "A_main", "tiled_flash", 512, "peak_mem_mb"), 0.02, " MB")

    print("\n§5.4 accuracy claims")
    check("flash max abs diff vs standard path at N=2048", 2.44e-4,
          get(k, "A_main", "flash", 2048, "max_abs_err_vs_math"), 0.02)
    check("naive rel err vs fp64 at N=2048", 5.56e-4,
          get(k, "A_main", "naive", 2048, "rel_err_vs_fp64"), 0.02)
    check("flash rel err vs fp64 at N=2048", 2.82e-4,
          get(k, "A_main", "flash", 2048, "rel_err_vs_fp64"), 0.02)
    check("math rel err vs fp64 at N=2048", 2.08e-4,
          get(k, "A_main", "math", 2048, "rel_err_vs_fp64"), 0.02)
    rn = get(k, "A_main", "naive", 2048, "rel_err_vs_fp64")
    rf = get(k, "A_main", "flash", 2048, "rel_err_vs_fp64")
    ok = rf < rn
    checks.append(("flash is MORE accurate than the fp16 baseline", ok,
                   f"flash {rf:.3e} < naive {rn:.3e}"))
    print(f"  [{PASS if ok else FAIL}] flash is MORE accurate than the fp16 baseline: "
          f"flash {rf:.3e} < naive {rn:.3e}")
    cos = get(k, "A_main", "flash", 2048, "cosine_vs_math")
    check("cosine similarity vs standard path", 1.0, cos, 1e-6)

    print("\n§5.5 causal sweep")
    for N, sp in [(512, 13.9), (1024, 16.1), (2048, 19.8)]:
        a = get(k, "C_causal", "naive", N)
        b = get(k, "C_causal", "flash", N)
        check(f"causal N={N} speed-up", sp, a / b if a and b else None, 0.03, "x")

    print("\n§5.6 sensitivity sweeps")
    for bs, sp in [(1, 2.86), (8, 6.12), (32, 6.10)]:
        a = get(k, f"E_batch{bs}", "naive", 1024)
        b = get(k, f"E_batch{bs}", "flash", 1024)
        check(f"batch {bs} speed-up", sp, a / b if a and b else None, 0.03, "x")
    for hd, sp in [(32, 11.20), (64, 6.27), (128, 3.32)]:
        a = get(k, f"D_headdim{hd}", "naive", 2048)
        b = get(k, f"D_headdim{hd}", "flash", 2048)
        check(f"head_dim {hd} speed-up", sp, a / b if a and b else None, 0.03, "x")

    print("\n§5.7 layer and training step")
    lm = get(layer, "layer", "math", 2048)
    lf = get(layer, "layer", "flash", 2048)
    check("full layer speed-up at N=2048", 10.85, lm / lf if lm and lf else None,
          0.03, "x")
    check("full layer flash memory at N=2048", 125.8,
          get(layer, "layer", "flash", 2048, "peak_mem_mb"), 0.02, " MB")
    tm = get(tstep, "train_step", "math", 2048)
    tf = get(tstep, "train_step", "flash", 2048)
    check("training step speed-up at N=2048", 19.4, tm / tf if tm and tf else None,
          0.04, "x")
    mm = get(tstep, "train_step", "math", 2048, "peak_mem_mb")
    mf = get(tstep, "train_step", "flash", 2048, "peak_mem_mb")
    check("training step memory ratio at N=2048", 24.0, mm / mf if mm and mf else None,
          0.04, "x")

    print("\n§5.8 kernel equivalence on the trained model")
    ke = [r for r in tr["kernel_equivalence"] if r.get("status") == "ok"]
    losses = [r["val_loss"] for r in ke]
    check("val-loss spread across kernels", 2.43e-4, max(losses) - min(losses), 0.05,
          " nats")
    fast = min(r["fwd_ms"] for r in ke)
    slow = max(r["fwd_ms"] for r in ke)
    check("fastest-to-slowest kernel ratio", 2.92, slow / fast, 0.03, "x")
    worst = max(ke, key=lambda r: r["val_loss"])["kernel"]
    ok = worst == "naive"
    checks.append(("largest loss deviation belongs to the baseline", ok,
                   f"worst kernel = {worst}"))
    print(f"  [{PASS if ok else FAIL}] largest loss deviation belongs to the baseline: "
          f"worst = {worst}")
    trit = [r for r in tr["kernel_equivalence"] if r["kernel"] == "triton_flash"]
    ok = bool(trit) and trit[0].get("status") == "error"
    checks.append(("our Triton kernel failed on head_dim 48 (reported honestly)", ok,
                   trit[0].get("error", "")[:60] if trit else "missing"))
    print(f"  [{PASS if ok else FAIL}] Triton kernel failure recorded: "
          f"{trit[0].get('error','')[:60] if trit else 'missing'}")

    print("\n§7.2 / §7.3 decode and the GQA trap")
    d16 = {r["layout"]: r for r in dec
           if r.get("status") == "ok" and r["context_len"] == 16384}
    mha = d16["MHA"]
    for lay, sp in [("GQA-8", 1.95), ("GQA-4", 3.95), ("GQA-2", 9.25), ("MQA", 18.07)]:
        check(f"{lay} decode speed-up (native)", sp,
              mha["ms_median"] / d16[lay]["ms_median"], 0.03, "x")
    check("MQA KV cache at 16k", 134.2, d16["MQA"]["kv_cache_mb"], 0.01, " MB")
    check("MHA KV cache at 16k", 2147.5, d16["MHA"]["kv_cache_mb"], 0.01, " MB")
    gqa8_rep = mha["ms_median"] / d16["GQA-8"]["ms_median_naive_repeat"]
    ok = gqa8_rep < 1.0
    checks.append(("GQA-8 with repeat_kv is SLOWER than MHA", ok,
                   f"{gqa8_rep:.2f}x"))
    print(f"  [{PASS if ok else FAIL}] GQA-8 with repeat_kv is slower than MHA: "
          f"{gqa8_rep:.2f}x")

    print("\n§7.4 seed study")
    summ = ss["summary"]
    means = [v["mean"] for v in summ.values()]
    between = max(means) - min(means)
    within = sum(v["std"] for v in summ.values()) / len(summ)
    check("between-layout spread", 0.0014, between, 0.06, " nats")
    check("within-layout seed noise", 0.0061, within, 0.03, " nats")
    check("ratio between/within", 0.22, between / within, 0.06, "x")
    ok = between < within
    checks.append(("layout differences are within seed noise", ok,
                   f"between {between:.4f} < within {within:.4f}"))
    print(f"  [{PASS if ok else FAIL}] layout differences are within seed noise")
    mha_span = summ["MHA"]["max"] - summ["MHA"]["min"]
    check("MHA's own seed span", 0.0152, mha_span, 0.03, " nats")

    print("\n§8 PagedAttention (vLLM-style KV cache)")
    fr = {r["mean_gen"]: r for r in pg["fragmentation"]}
    check("contiguous cache flat at 85.9 GB", 85.90, fr[64]["contiguous_gb"], 0.01,
          " GB")
    check("paged @ mean_gen 64: 3.24 GB", 3.24, fr[64]["paged_gb"], 0.02, " GB")
    check("paged reduction @ mean_gen 64", 26.5, fr[64]["reduction_x"], 0.03, "x")
    check("paged utilization @ mean_gen 256", 97.2, 100 * fr[256]["paged_util"], 0.02,
          "%")
    check("contiguous utilization @ mean_gen 64", 3.4, 100 * fr[64]["contiguous_util"],
          0.05, "%")
    check("KV bytes/token for serving model", 160 * 1024,
          2 * pg["model"]["n_layers"] * pg["model"]["n_kv_heads"]
          * pg["model"]["head_dim"] * pg["model"]["dtype_bytes"], 0.001, " B")
    conc = {r["budget_gb"]: r for r in pg["concurrency"]}
    check("paged admits 342 seqs @ 16 GB", 342, conc[16]["paged_seqs"], 0.01)
    check("contiguous admits 47 seqs @ 16 GB", 47, conc[16]["contiguous_seqs"], 0.02)
    check("paged admit ratio @ 16 GB", 7.3, conc[16]["admit_ratio"], 0.03, "x")
    ps = {r["n_seqs"]: r for r in pg["prefix_sharing"]}
    check("prefix sharing @ N=256: 5.45 GB", 5.453, ps[256]["paged_share_gb"], 0.01,
          " GB")
    check("prefix sharing vs contiguous @ N=256", 4.9, ps[256]["vs_contiguous_x"],
          0.03, "x")
    check("prefix sharing @ N=64: 4.7x", 4.7, ps[64]["vs_contiguous_x"], 0.03, "x")
    pl = {r["context"]: r for r in pg["paged_latency"] if r.get("status") == "ok"}
    ok = pl[2048]["paged_ms"] > pl[2048]["contiguous_ms"]
    checks.append(("unfused paged gather is slower than contiguous (honest cost)", ok,
                   f"paged {pl[2048]['paged_ms']:.2f} > contiguous "
                   f"{pl[2048]['contiguous_ms']:.2f} ms"))
    print(f"  [{PASS if ok else FAIL}] unfused paged gather slower than contiguous: "
          f"paged {pl[2048]['paged_ms']:.2f} vs {pl[2048]['contiguous_ms']:.2f} ms")

    n_ok = sum(1 for _, o, _ in checks if o)
    print("\n" + "=" * 78)
    print(f"RESULT: {n_ok}/{len(checks)} report claims verified against raw data")
    if n_ok < len(checks):
        print("\nMismatches:")
        for name, o, detail in checks:
            if not o:
                print(f"  - {name}: {detail}")
    print("=" * 78)
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
