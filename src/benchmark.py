"""
Benchmark harness: latency, peak memory and numerical fidelity for every attention
kernel, over a sweep of configurations.

Measurement protocol
--------------------
* Latency  - paired torch.cuda.Event timers around each individual call, after a
             warm-up phase (first CUDA call to any kernel pays one-off autotuning /
             module-load cost). We report the median plus p5/p95 over N repetitions,
             not the mean, because the distribution is right-skewed by scheduler noise.
* Memory   - torch.cuda.max_memory_allocated() minus the allocation level measured
             *after* q/k/v exist, so the number is the attention workspace itself
             (the O(N^2) term) rather than the inputs.
* Accuracy - every kernel is compared against a float64 evaluation of the textbook
             formula, so the fp16 baseline is held to the same standard as the fused
             kernels rather than being assumed correct.
* Throughput - achieved TFLOP/s from the analytic FLOP count of the two matmuls.

Usage:  python benchmark.py [--quick] [--out ../results]
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, Optional

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from attention import (  # noqa: E402
    MultiHeadSelfAttention,
    available_backends,
    flash_attention_tiled,
    naive_attention,
    sdpa_attention,
)
from flash_triton import HAS_TRITON, flash_attention_triton  # noqa: E402

DEV = "cuda"
OOM_SENTINEL = float("nan")


# ---------------------------------------------------------------------------
# Kernel registry
# ---------------------------------------------------------------------------

@dataclass
class Kernel:
    key: str
    label: str
    family: str          # "baseline" | "fused-exact" | "reference-impl"
    fn: Callable
    exact: bool = True
    note: str = ""


def build_kernels(include_slow: bool = True) -> list[Kernel]:
    ks = [
        Kernel("naive", "Standard MHA (explicit N x N)", "baseline",
               lambda q, k, v, c: naive_attention(q, k, v, causal=c),
               note="materialises S and P in HBM"),
        Kernel("math", "PyTorch SDPA - math backend", "baseline",
               lambda q, k, v, c: sdpa_attention(q, k, v, causal=c, backend="math"),
               note="reference 'unfused' path used by the report"),
    ]
    for b in available_backends():
        if b == "math":
            continue
        label = {"flash": "FlashAttention-2 kernel (SDPA flash)",
                 "mem_efficient": "xFormers-style mem-efficient (SDPA)",
                 "cudnn": "cuDNN fused attention (SDPA)"}.get(b, b)
        ks.append(Kernel(b, label, "fused-exact",
                         lambda q, k, v, c, _b=b: sdpa_attention(q, k, v, causal=c,
                                                                 backend=_b)))
    ks.append(Kernel("auto", "PyTorch SDPA - automatic dispatch", "fused-exact",
                     lambda q, k, v, c: sdpa_attention(q, k, v, causal=c, backend=None)))
    if HAS_TRITON:
        ks.append(Kernel("triton_flash", "FlashAttention (our Triton kernel)",
                         "fused-exact",
                         lambda q, k, v, c: flash_attention_triton(q, k, v, causal=c),
                         note="hand-written, from the paper's Algorithm 1"))
    if include_slow:
        ks.append(Kernel("tiled_flash", "FlashAttention algorithm (pure PyTorch)",
                         "reference-impl",
                         lambda q, k, v, c: flash_attention_tiled(q, k, v, causal=c),
                         note="algorithm without kernel fusion - shows fusion is the point"))
    return ks


# ---------------------------------------------------------------------------
# Timing / memory primitives
# ---------------------------------------------------------------------------

def _sync():
    torch.cuda.synchronize()


def time_op(fn: Callable, warmup: int = 10, iters: int = 50) -> dict:
    """Median/percentile wall-clock latency in ms via CUDA events."""
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        fn()
        ends[i].record()
    _sync()
    times = [s.elapsed_time(e) for s, e in zip(starts, ends)]
    times.sort()
    return {
        "ms_median": statistics.median(times),
        "ms_mean": statistics.fmean(times),
        "ms_min": times[0],
        "ms_p05": times[max(0, int(0.05 * len(times)) - 1)],
        "ms_p95": times[min(len(times) - 1, int(0.95 * len(times)))],
        "ms_std": statistics.pstdev(times) if len(times) > 1 else 0.0,
        "iters": iters,
    }


def peak_memory_of(fn: Callable) -> float:
    """Peak *additional* MB allocated by fn, over the level before it ran."""
    gc.collect()
    torch.cuda.empty_cache()
    _sync()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    out = fn()
    _sync()
    peak = torch.cuda.max_memory_allocated()
    del out
    gc.collect()
    torch.cuda.empty_cache()
    return (peak - base) / 1e6


def attention_flops(B, H, N_q, N_k, D, causal: bool) -> float:
    """2 matmuls: (N_q x D)(D x N_k) and (N_q x N_k)(N_k x D). 2 FLOP per MAC."""
    f = 2.0 * B * H * N_q * N_k * D * 2.0
    return f * 0.5 if causal else f


# ---------------------------------------------------------------------------
# Core sweep
# ---------------------------------------------------------------------------

@dataclass
class Row:
    sweep: str
    kernel: str
    label: str
    family: str
    batch: int
    heads: int
    seq_len: int
    head_dim: int
    dtype: str
    causal: bool
    ms_median: float = OOM_SENTINEL
    ms_p05: float = OOM_SENTINEL
    ms_p95: float = OOM_SENTINEL
    ms_std: float = OOM_SENTINEL
    peak_mem_mb: float = OOM_SENTINEL
    tflops: float = OOM_SENTINEL
    max_abs_err_vs_fp64: float = OOM_SENTINEL
    rel_err_vs_fp64: float = OOM_SENTINEL
    max_abs_err_vs_math: float = OOM_SENTINEL
    cosine_vs_math: float = OOM_SENTINEL
    allclose_vs_math: Optional[bool] = None
    status: str = "ok"
    error: str = ""


HIGH_PREC_BUDGET_BYTES = 512e6   # per score chunk


def high_precision_reference(q, k, v, causal, dtype=torch.float64) -> torch.Tensor:
    """Ground truth in `dtype`, computed in query chunks sized to a memory budget.

    fp64 is used wherever it fits; the caller may fall back to fp32 for the very
    largest shapes. Either way the reference is far more precise than any fp16
    kernel under test, so it is a valid yardstick for all of them.
    """
    B, H, N, D = q.shape
    itemsize = torch.empty(0, dtype=dtype).element_size()
    chunk = int(HIGH_PREC_BUDGET_BYTES / max(1, B * H * N * itemsize))
    chunk = max(8, min(N, chunk))
    out = torch.empty(B, H, N, D, device=q.device, dtype=dtype)
    kd, vd = k.to(dtype), v.to(dtype)
    scale = 1.0 / math.sqrt(D)
    for i in range(0, N, chunk):
        j = min(i + chunk, N)
        s = (q[:, :, i:j].to(dtype) @ kd.transpose(-2, -1)) * scale
        if causal:
            idx_q = torch.arange(i, j, device=q.device).unsqueeze(1)
            idx_k = torch.arange(N, device=q.device).unsqueeze(0)
            s = s.masked_fill(idx_k > idx_q, float("-inf"))
        out[:, :, i:j] = torch.softmax(s, dim=-1) @ vd
        del s
    del kd, vd
    return out


def fp64_reference(q, k, v, causal) -> torch.Tensor:
    """fp64 where affordable, fp32 otherwise; always upcast to fp64 for comparison."""
    torch.cuda.empty_cache()
    B, H, N, D = q.shape
    try:
        if B * H * N * D * 8 * 4 < 3e9:          # q,k,v,out copies in fp64
            return high_precision_reference(q, k, v, causal, torch.float64)
        raise torch.cuda.OutOfMemoryError("fp64 reference too large")
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        return high_precision_reference(q, k, v, causal, torch.float32).double()


def run_sweep(
    sweep_name: str,
    kernels: list[Kernel],
    batch: int,
    heads: int,
    head_dim: int,
    seq_lens: list[int],
    dtype=torch.float16,
    causal: bool = False,
    warmup: int = 10,
    iters: int = 50,
    accuracy: bool = True,
    slow_kernel_max_n: int = 1024,
    seed: int = 0,
    verbose: bool = True,
) -> list[Row]:
    rows: list[Row] = []
    for N in seq_lens:
        if verbose:
            print(f"\n  --- {sweep_name}: B={batch} H={heads} N={N} D={head_dim} "
                  f"dtype={str(dtype).split('.')[-1]} causal={causal} ---")
        torch.manual_seed(seed)
        gc.collect()
        torch.cuda.empty_cache()
        shape = (batch, heads, N, head_dim)
        try:
            q = torch.randn(shape, device=DEV, dtype=dtype)
            k = torch.randn(shape, device=DEV, dtype=dtype)
            v = torch.randn(shape, device=DEV, dtype=dtype)
        except torch.cuda.OutOfMemoryError:
            print(f"      cannot even allocate q/k/v at N={N}; stopping sweep")
            break

        # ground truth + the 'standard path' output for the head-to-head comparison
        ref64 = None
        math_out = None
        if accuracy:
            try:
                ref64 = fp64_reference(q, k, v, causal)
            except torch.cuda.OutOfMemoryError:
                ref64 = None
                gc.collect(); torch.cuda.empty_cache()
            try:
                math_out = sdpa_attention(q, k, v, causal=causal, backend="math").float()
            except torch.cuda.OutOfMemoryError:
                math_out = None
                gc.collect(); torch.cuda.empty_cache()

        flops = attention_flops(batch, heads, N, N, head_dim, causal)

        for kern in kernels:
            row = Row(sweep=sweep_name, kernel=kern.key, label=kern.label,
                      family=kern.family, batch=batch, heads=heads, seq_len=N,
                      head_dim=head_dim, dtype=str(dtype).split(".")[-1], causal=causal)
            if kern.family == "reference-impl" and N > slow_kernel_max_n:
                row.status = "skipped"
                row.error = f"pure-python tiling too slow above N={slow_kernel_max_n}"
                rows.append(row)
                if verbose:
                    print(f"      {kern.key:14s}  skipped ({row.error})")
                continue
            call = lambda kk=kern: kk.fn(q, k, v, causal)  # noqa: E731
            try:
                # accuracy first (cheapest to lose to OOM)
                if accuracy:
                    out = call()
                    _sync()
                    if ref64 is not None:
                        d = out.double() - ref64
                        row.max_abs_err_vs_fp64 = d.abs().max().item()
                        row.rel_err_vs_fp64 = (d.norm() / ref64.norm()).item()
                    if math_out is not None:
                        of = out.float()
                        row.max_abs_err_vs_math = (of - math_out).abs().max().item()
                        row.cosine_vs_math = F.cosine_similarity(
                            of.flatten(), math_out.flatten(), dim=0).item()
                        row.allclose_vs_math = bool(
                            torch.allclose(of, math_out, atol=1e-2, rtol=1e-2))
                    del out
                    gc.collect(); torch.cuda.empty_cache()

                row.peak_mem_mb = peak_memory_of(call)
                t = time_op(call, warmup=warmup, iters=iters)
                row.ms_median, row.ms_p05 = t["ms_median"], t["ms_p05"]
                row.ms_p95, row.ms_std = t["ms_p95"], t["ms_std"]
                row.tflops = flops / (row.ms_median * 1e-3) / 1e12
                if verbose:
                    print(f"      {kern.key:14s}  {row.ms_median:9.3f} ms   "
                          f"{row.peak_mem_mb:9.1f} MB   {row.tflops:7.2f} TFLOP/s   "
                          f"relerr {row.rel_err_vs_fp64:.2e}")
            except torch.cuda.OutOfMemoryError as exc:
                row.status = "OOM"
                row.error = str(exc)[:120]
                gc.collect(); torch.cuda.empty_cache()
                if verbose:
                    print(f"      {kern.key:14s}  OUT OF MEMORY")
            except Exception as exc:  # noqa: BLE001
                row.status = "error"
                row.error = f"{type(exc).__name__}: {str(exc)[:160]}"
                gc.collect(); torch.cuda.empty_cache()
                if verbose:
                    print(f"      {kern.key:14s}  ERROR {row.error}")
            rows.append(row)

        del q, k, v, ref64, math_out
        gc.collect()
        torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Layer-level sweep (projections included) - what a real model actually runs
# ---------------------------------------------------------------------------

def run_layer_sweep(seq_lens, batch=8, embed_dim=768, heads=12, causal=False,
                    kernels=("math", "flash", "mem_efficient", "cudnn", "auto",
                             "triton_flash"),
                    dtype=torch.float16, warmup=10, iters=30, verbose=True) -> list[Row]:
    rows = []
    for N in seq_lens:
        torch.manual_seed(0)
        gc.collect(); torch.cuda.empty_cache()
        layer = MultiHeadSelfAttention(embed_dim, heads, causal=causal,
                                       kernel="math").to(DEV).to(dtype).eval()
        x = torch.randn(batch, N, embed_dim, device=DEV, dtype=dtype)
        if verbose:
            print(f"\n  --- layer sweep: B={batch} N={N} E={embed_dim} H={heads} ---")
        base_out = None
        with torch.no_grad():
            for kk in kernels:
                if kk == "triton_flash" and not HAS_TRITON:
                    continue
                if kk not in ("math", "naive", "tiled_flash", "auto", "triton_flash") \
                        and kk not in available_backends():
                    continue
                row = Row(sweep="layer", kernel=kk, label=f"MHA layer [{kk}]",
                          family="baseline" if kk in ("math", "naive") else "fused-exact",
                          batch=batch, heads=heads, seq_len=N,
                          head_dim=embed_dim // heads,
                          dtype=str(dtype).split(".")[-1], causal=causal)
                call = lambda _k=kk: layer(x, kernel=_k)  # noqa: E731
                try:
                    out = call(); _sync()
                    if kk == "math":
                        base_out = out.float().clone()
                    if base_out is not None:
                        of = out.float()
                        row.max_abs_err_vs_math = (of - base_out).abs().max().item()
                        row.cosine_vs_math = F.cosine_similarity(
                            of.flatten(), base_out.flatten(), dim=0).item()
                        row.allclose_vs_math = bool(
                            torch.allclose(of, base_out, atol=1e-2, rtol=1e-2))
                    del out
                    gc.collect(); torch.cuda.empty_cache()
                    row.peak_mem_mb = peak_memory_of(call)
                    t = time_op(call, warmup=warmup, iters=iters)
                    row.ms_median, row.ms_p05, row.ms_p95 = (t["ms_median"], t["ms_p05"],
                                                             t["ms_p95"])
                    row.tflops = attention_flops(batch, heads, N, N,
                                                 embed_dim // heads, causal) / (
                        row.ms_median * 1e-3) / 1e12
                    if verbose:
                        print(f"      {kk:14s}  {row.ms_median:9.3f} ms   "
                              f"{row.peak_mem_mb:9.1f} MB")
                except torch.cuda.OutOfMemoryError:
                    row.status = "OOM"
                    gc.collect(); torch.cuda.empty_cache()
                    if verbose:
                        print(f"      {kk:14s}  OUT OF MEMORY")
                except Exception as exc:  # noqa: BLE001
                    row.status = "error"; row.error = f"{type(exc).__name__}: {exc}"[:160]
                    gc.collect(); torch.cuda.empty_cache()
                    if verbose:
                        print(f"      {kk:14s}  ERROR {row.error}")
                rows.append(row)
        del layer, x, base_out
        gc.collect(); torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Training-step sweep: forward + backward, which is where memory really bites
# ---------------------------------------------------------------------------

def run_train_step_sweep(seq_lens, batch=4, heads=12, head_dim=64, causal=True,
                         kernels=("math", "flash", "mem_efficient"),
                         dtype=torch.float16, warmup=5, iters=20, verbose=True):
    rows = []
    for N in seq_lens:
        torch.manual_seed(0)
        gc.collect(); torch.cuda.empty_cache()
        if verbose:
            print(f"\n  --- train step (fwd+bwd): B={batch} N={N} causal={causal} ---")
        for kk in kernels:
            if kk not in available_backends() and kk != "auto":
                continue
            row = Row(sweep="train_step", kernel=kk, label=f"fwd+bwd [{kk}]",
                      family="baseline" if kk == "math" else "fused-exact",
                      batch=batch, heads=heads, seq_len=N, head_dim=head_dim,
                      dtype=str(dtype).split(".")[-1], causal=causal)

            def step(_k=kk):
                q = torch.randn(batch, heads, N, head_dim, device=DEV, dtype=dtype,
                                requires_grad=True)
                kt = torch.randn_like(q, requires_grad=True)
                vt = torch.randn_like(q, requires_grad=True)
                o = sdpa_attention(q, kt, vt, causal=causal,
                                   backend=None if _k == "auto" else _k)
                o.sum().backward()
                return o

            try:
                row.peak_mem_mb = peak_memory_of(step)
                t = time_op(step, warmup=warmup, iters=iters)
                row.ms_median, row.ms_p05, row.ms_p95 = (t["ms_median"], t["ms_p05"],
                                                         t["ms_p95"])
                # fwd (2 matmuls) + bwd (~4 matmuls) ~ 3.5x forward FLOPs
                row.tflops = 3.5 * attention_flops(batch, heads, N, N, head_dim,
                                                   causal) / (row.ms_median * 1e-3) / 1e12
                if verbose:
                    print(f"      {kk:14s}  {row.ms_median:9.3f} ms   "
                          f"{row.peak_mem_mb:9.1f} MB")
            except torch.cuda.OutOfMemoryError:
                row.status = "OOM"
                gc.collect(); torch.cuda.empty_cache()
                if verbose:
                    print(f"      {kk:14s}  OUT OF MEMORY")
            except Exception as exc:  # noqa: BLE001
                row.status = "error"; row.error = f"{type(exc).__name__}: {exc}"[:160]
                gc.collect(); torch.cuda.empty_cache()
            rows.append(row)
        gc.collect(); torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# Decode / KV-cache sweep: where MQA and GQA (architecture-level) actually win
# ---------------------------------------------------------------------------

def run_decode_sweep(context_lens, batch=32, embed_dim=1024, heads=16,
                     kv_head_options=(16, 8, 4, 2, 1), dtype=torch.float16,
                     kernel="flash", warmup=5, iters=30, verbose=True) -> list[dict]:
    """Single-token decoding against a pre-filled KV cache, MHA vs GQA vs MQA.

    Each layout is timed twice: once with the kernel consuming the narrow KV cache
    natively (`enable_gqa`), once with the caller expanding K/V to full width first.
    The gap between the two is the difference between GQA helping and GQA hurting.
    """
    out_rows = []
    head_dim = embed_dim // heads
    for T in context_lens:
        for n_kv in kv_head_options:
            torch.manual_seed(0)
            gc.collect(); torch.cuda.empty_cache()
            layout = ("MHA" if n_kv == heads else "MQA" if n_kv == 1 else f"GQA-{n_kv}")
            rec = dict(sweep="decode", layout=layout, n_kv_heads=n_kv, context_len=T,
                       batch=batch, embed_dim=embed_dim, heads=heads, kernel=kernel,
                       status="ok")
            try:
                layer = MultiHeadSelfAttention(embed_dim, heads, n_kv_heads=n_kv,
                                               kernel=kernel).to(DEV).to(dtype).eval()
                x_t = torch.randn(batch, 1, embed_dim, device=DEV, dtype=dtype)
                cache = (torch.randn(batch, n_kv, T, head_dim, device=DEV, dtype=dtype),
                         torch.randn(batch, n_kv, T, head_dim, device=DEV, dtype=dtype))
                cache_mb = 2 * batch * n_kv * T * head_dim * 2 / 1e6

                def step(native=True):
                    with torch.no_grad():
                        y, _ = layer.decode_step(x_t, cache, native_gqa=native)
                    return y

                t = time_op(lambda: step(True), warmup=warmup, iters=iters)
                t_rep = time_op(lambda: step(False), warmup=warmup, iters=iters)
                rec.update(ms_median=t["ms_median"], ms_p05=t["ms_p05"],
                           ms_p95=t["ms_p95"],
                           ms_median_naive_repeat=t_rep["ms_median"],
                           kv_cache_mb=cache_mb,
                           tokens_per_s=batch / (t["ms_median"] * 1e-3),
                           tokens_per_s_naive_repeat=batch / (t_rep["ms_median"] * 1e-3),
                           params_m=sum(p.numel() for p in layer.parameters()) / 1e6)
                if verbose:
                    print(f"      T={T:6d} {layout:7s}  native {t['ms_median']:7.3f} ms"
                          f"   repeat-kv {t_rep['ms_median']:7.3f} ms   "
                          f"KV {cache_mb:8.1f} MB   "
                          f"{rec['tokens_per_s']:9.0f} tok/s")
                del layer, x_t, cache
            except torch.cuda.OutOfMemoryError:
                rec["status"] = "OOM"
                if verbose:
                    print(f"      T={T:6d} {layout:7s}  OUT OF MEMORY")
            except Exception as exc:  # noqa: BLE001
                rec["status"] = "error"; rec["error"] = f"{type(exc).__name__}: {exc}"[:160]
            out_rows.append(rec)
            gc.collect(); torch.cuda.empty_cache()
    return out_rows


# ---------------------------------------------------------------------------
def env_info() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"sm_{p.major}{p.minor}",
        "total_mem_gb": round(p.total_memory / 1e9, 2),
        "sms": p.multi_processor_count,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "triton": (__import__("triton").__version__ if HAS_TRITON else None),
        "driver": os.popen(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader"
        ).read().strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "sdpa_backends": available_backends(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def save(rows: list, path: str):
    data = [asdict(r) if hasattr(r, "__dataclass_fields__") else r for r in rows]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  wrote {len(data)} rows -> {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "results"))
    ap.add_argument("--quick", action="store_true", help="short sweeps for a smoke test")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    info = env_info()
    print("=" * 78)
    print("ATTENTION BENCHMARK")
    print("=" * 78)
    for kk, vv in info.items():
        print(f"  {kk:20s} {vv}")
    with open(os.path.join(args.out, "environment.json"), "w") as f:
        json.dump(info, f, indent=2)

    kernels = build_kernels(include_slow=True)
    print(f"\n  kernels under test: {[k.key for k in kernels]}")

    it, wu = (10, 3) if args.quick else (args.iters, args.warmup)
    all_rows: list[Row] = []

    # -- Sweep A: the report's headline configuration --------------------
    print("\n" + "=" * 78)
    print("SWEEP A - report configuration: batch 16, embed 768, 12 heads, fp16, "
          "non-causal")
    print("=" * 78)
    seq_a = [128, 256, 512] if args.quick else [128, 256, 512, 1024, 2048, 4096]
    all_rows += run_sweep("A_main", kernels, batch=16, heads=12, head_dim=64,
                          seq_lens=seq_a, causal=False, warmup=wu, iters=it,
                          slow_kernel_max_n=512)

    # -- Sweep B: long context at smaller batch --------------------------
    print("\n" + "=" * 78)
    print("SWEEP B - long context, batch 4 (where the naive path runs out of memory)")
    print("=" * 78)
    seq_b = [1024, 2048] if args.quick else [1024, 2048, 4096, 8192, 16384, 32768]
    all_rows += run_sweep("B_longctx", kernels, batch=4, heads=12, head_dim=64,
                          seq_lens=seq_b, causal=False, warmup=max(3, wu // 2),
                          iters=max(10, it // 2), slow_kernel_max_n=0, accuracy=True)

    # -- Sweep C: causal (decoder-style) ---------------------------------
    print("\n" + "=" * 78)
    print("SWEEP C - causal masking (decoder-style attention)")
    print("=" * 78)
    seq_c = [512, 1024] if args.quick else [512, 1024, 2048, 4096, 8192]
    all_rows += run_sweep("C_causal", kernels, batch=8, heads=12, head_dim=64,
                          seq_lens=seq_c, causal=True, warmup=wu, iters=it,
                          slow_kernel_max_n=0)

    # -- Sweep D: head-dim sensitivity -----------------------------------
    print("\n" + "=" * 78)
    print("SWEEP D - head dimension sensitivity at N=2048")
    print("=" * 78)
    for hd, hh in ([(64, 12)] if args.quick else [(32, 24), (64, 12), (128, 6)]):
        all_rows += run_sweep(f"D_headdim{hd}", kernels, batch=8, heads=hh, head_dim=hd,
                              seq_lens=[2048], causal=False, warmup=wu, iters=it,
                              slow_kernel_max_n=0)

    # -- Sweep E: batch-size sensitivity ---------------------------------
    print("\n" + "=" * 78)
    print("SWEEP E - batch-size sensitivity at N=1024")
    print("=" * 78)
    for bs in ([1, 8] if args.quick else [1, 2, 4, 8, 16, 32]):
        all_rows += run_sweep(f"E_batch{bs}", kernels, batch=bs, heads=12, head_dim=64,
                              seq_lens=[1024], causal=False, warmup=wu, iters=it,
                              slow_kernel_max_n=0)

    # -- Sweep F: dtype (bf16 vs fp16) -----------------------------------
    if not args.quick:
        print("\n" + "=" * 78)
        print("SWEEP F - bfloat16")
        print("=" * 78)
        all_rows += run_sweep("F_bf16", kernels, batch=16, heads=12, head_dim=64,
                              seq_lens=[512, 1024, 2048], dtype=torch.bfloat16,
                              causal=False, warmup=wu, iters=it, slow_kernel_max_n=0)

    save(all_rows, os.path.join(args.out, "kernel_benchmarks.json"))

    # -- Layer level -----------------------------------------------------
    print("\n" + "=" * 78)
    print("SWEEP G - full MHA layer (with Q/K/V/O projections)")
    print("=" * 78)
    layer_rows = run_layer_sweep([256, 512] if args.quick
                                 else [128, 256, 512, 1024, 2048, 4096],
                                 batch=8, embed_dim=768, heads=12,
                                 warmup=wu, iters=max(20, it // 2))
    save(layer_rows, os.path.join(args.out, "layer_benchmarks.json"))

    # -- Training step ---------------------------------------------------
    print("\n" + "=" * 78)
    print("SWEEP H - training step (forward + backward)")
    print("=" * 78)
    train_rows = run_train_step_sweep([512, 1024] if args.quick
                                      else [512, 1024, 2048, 4096, 8192],
                                      batch=4, causal=True,
                                      kernels=("math", "flash", "mem_efficient"),
                                      warmup=3, iters=15)
    save(train_rows, os.path.join(args.out, "train_step_benchmarks.json"))

    # -- Decode / KV cache ----------------------------------------------
    print("\n" + "=" * 78)
    print("SWEEP I - autoregressive decode with KV cache: MHA vs GQA vs MQA")
    print("=" * 78)
    dec_rows = run_decode_sweep([1024, 4096] if args.quick
                                else [512, 2048, 8192, 16384],
                                batch=32, embed_dim=1024, heads=16,
                                warmup=5, iters=30)
    save(dec_rows, os.path.join(args.out, "decode_benchmarks.json"))

    print("\nAll benchmarks complete.")


if __name__ == "__main__":
    main()
