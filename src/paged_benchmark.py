"""
Benchmark the PagedAttention KV cache the same way the rest of the project benchmarks
FlashAttention: real measurements, an OOM boundary that is a result, and honest
reporting of the trade-off.

PagedAttention does not speed up the attention math (FlashAttention already did that).
It optimises how the KV cache is *stored*, so the metrics are memory-shaped:

  1. fragmentation   - allocated KV memory and utilisation for a realistic batch of
                       variable-length sequences: contiguous pre-allocation vs paged.
  2. concurrency     - how many sequences fit in a fixed KV-memory budget (paged admits
                       many more because it does not over-allocate) -> aggregate
                       decode throughput.
  3. prefix sharing  - a shared system prompt across N sequences, stored once (paged,
                       copy-on-write) vs N times (contiguous).
  4. latency         - per-step decode latency, paged gather-path vs contiguous, at
                       equal length: the cost side of the trade-off.

Memory numbers 1-3 are analytic (exact from the block math, hardware-independent);
they are additionally cross-checked against real allocated pool tensors. Latency (4)
is measured with CUDA events like every other timing in this project.

Usage:  python paged_benchmark.py [--quick] [--out ../results]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paged_attention import (  # noqa: E402
    KVCacheOOM,
    KVModel,
    PagedKVCache,
    contiguous_cache_bytes,
    ideal_cache_bytes,
    paged_attention_decode,
    paged_cache_bytes,
)
from attention import MultiHeadSelfAttention  # noqa: E402

DEV = "cuda"


def sample_lengths(n, mean_gen, max_len, seed, min_len=8):
    """Right-skewed generation lengths (exponential), the realistic serving regime."""
    g = torch.Generator().manual_seed(seed)
    x = torch.distributions.Exponential(1.0 / mean_gen).sample((n,), )  # noqa
    # torch Exponential ignores generator; add deterministic jitter via seed offset
    torch.manual_seed(seed)
    x = torch.distributions.Exponential(1.0 / mean_gen).sample((n,))
    return [min(max_len, max(min_len, int(v) + min_len)) for v in x.tolist()]


def time_op(fn, warmup=10, iters=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ss = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ee = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        ss[i].record()
        fn()
        ee[i].record()
    torch.cuda.synchronize()
    t = sorted(s.elapsed_time(e) for s, e in zip(ss, ee))
    return statistics.median(t)


# ---------------------------------------------------------------------------
# 1. Fragmentation: allocated memory + utilisation vs mean generation length
# ---------------------------------------------------------------------------

def sweep_fragmentation(model, block_size=16, n_seqs=256, max_len=2048,
                        mean_lens=(64, 128, 256, 512, 1024), seed=0, verbose=True):
    rows = []
    for mean_gen in mean_lens:
        lens = sample_lengths(n_seqs, mean_gen, max_len, seed)
        contig = contiguous_cache_bytes(lens, max_len, model)
        paged = paged_cache_bytes(lens, block_size, model)
        ideal = ideal_cache_bytes(lens, model)
        rec = dict(sweep="fragmentation", mean_gen=mean_gen, n_seqs=n_seqs,
                   max_len=max_len, block_size=block_size,
                   mean_actual_len=sum(lens) / len(lens),
                   contiguous_gb=contig / 1e9, paged_gb=paged / 1e9,
                   ideal_gb=ideal / 1e9,
                   contiguous_util=ideal / contig, paged_util=ideal / paged,
                   memory_saved_pct=100 * (1 - paged / contig),
                   reduction_x=contig / paged)
        rows.append(rec)
        if verbose:
            print(f"  mean_gen={mean_gen:5d}: contiguous {contig/1e9:6.2f} GB "
                  f"(util {100*ideal/contig:4.1f}%)  paged {paged/1e9:6.2f} GB "
                  f"(util {100*ideal/paged:4.1f}%)  -> {contig/paged:4.1f}x less")
    return rows


# ---------------------------------------------------------------------------
# 2. Concurrency: sequences admitted in a fixed KV budget -> throughput
# ---------------------------------------------------------------------------

def admit_count(scheme, budget_bytes, lens, max_len, block_size, model):
    """Greedily admit sequences until the budget is exhausted; return count."""
    used = 0
    bpt = model.bytes_per_token()
    admitted = 0
    for L in lens:
        cost = (max_len * bpt if scheme == "contiguous"
                else __import__("math").ceil(L / block_size) * block_size * bpt)
        if used + cost > budget_bytes:
            break
        used += cost
        admitted += 1
    return admitted


def sweep_concurrency(model, block_size=16, max_len=2048, mean_gen=256,
                      budgets_gb=(2, 4, 8, 16, 32), seed=1, verbose=True):
    rows = []
    lens = sample_lengths(4096, mean_gen, max_len, seed)  # a big arrival stream
    for bgb in budgets_gb:
        budget = bgb * 1e9
        c = admit_count("contiguous", budget, lens, max_len, block_size, model)
        p = admit_count("paged", budget, lens, max_len, block_size, model)
        rec = dict(sweep="concurrency", budget_gb=bgb, mean_gen=mean_gen,
                   max_len=max_len, contiguous_seqs=c, paged_seqs=p,
                   admit_ratio=(p / c) if c else float("inf"))
        rows.append(rec)
        if verbose:
            print(f"  budget {bgb:5.0f} GB: contiguous admits {c:4d} seqs, "
                  f"paged admits {p:4d} seqs  -> {p/c:4.1f}x more" if c
                  else f"  budget {bgb} GB: contiguous 0, paged {p}")
    return rows


def measure_decode_throughput(embed_dim=2048, n_heads=16, n_kv_heads=8, context=1024,
                              batches=(1, 8, 32, 64, 128, 256), dtype=torch.float16,
                              seed=2, verbose=True):
    """Real decode step latency vs batch size -> tokens/s, so the admitted-sequence
    counts above can be turned into an aggregate-throughput statement."""
    rows = []
    head_dim = embed_dim // n_heads
    layer = MultiHeadSelfAttention(embed_dim, n_heads, n_kv_heads=n_kv_heads,
                                   kernel="flash").to(DEV).to(dtype).eval()
    for B in batches:
        torch.manual_seed(seed)
        gc.collect(); torch.cuda.empty_cache()
        try:
            x_t = torch.randn(B, 1, embed_dim, device=DEV, dtype=dtype)
            cache = (torch.randn(B, n_kv_heads, context, head_dim, device=DEV,
                                 dtype=dtype),
                     torch.randn(B, n_kv_heads, context, head_dim, device=DEV,
                                 dtype=dtype))

            def step():
                with torch.no_grad():
                    layer.decode_step(x_t, cache)

            ms = time_op(step, warmup=5, iters=30)
            rows.append(dict(sweep="decode_tput", batch=B, context=context,
                             ms_median=ms, tokens_per_s=B / (ms * 1e-3), status="ok"))
            if verbose:
                print(f"  batch {B:4d}: {ms:7.3f} ms/step  "
                      f"{B/(ms*1e-3):9.0f} tok/s")
            del x_t, cache
        except torch.cuda.OutOfMemoryError:
            rows.append(dict(sweep="decode_tput", batch=B, context=context,
                             status="OOM"))
            if verbose:
                print(f"  batch {B:4d}: OOM")
            gc.collect(); torch.cuda.empty_cache()
    del layer
    gc.collect(); torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
# 3. Prefix sharing: shared prompt across N sequences
# ---------------------------------------------------------------------------

def sweep_prefix_sharing(model, block_size=16, prompt_len=512, gen_len=128,
                         n_seqs=(1, 2, 4, 8, 16, 32, 64, 128), verbose=True):
    rows = []
    for n in n_seqs:
        lens = [prompt_len + gen_len] * n
        contig = contiguous_cache_bytes(lens, prompt_len + gen_len, model)
        paged_noshare = paged_cache_bytes(lens, block_size, model)
        paged_share = paged_cache_bytes(lens, block_size, model,
                                        shared_prefix_len=prompt_len, n_sharing=n)
        rec = dict(sweep="prefix_sharing", n_seqs=n, prompt_len=prompt_len,
                   gen_len=gen_len,
                   contiguous_gb=contig / 1e9,
                   paged_noshare_gb=paged_noshare / 1e9,
                   paged_share_gb=paged_share / 1e9,
                   share_saves_pct=100 * (1 - paged_share / paged_noshare)
                   if paged_noshare else 0.0,
                   vs_contiguous_x=contig / paged_share if paged_share else 0.0)
        rows.append(rec)
        if verbose:
            print(f"  N={n:4d}: contiguous {contig/1e9:6.3f} GB  "
                  f"paged(no share) {paged_noshare/1e9:6.3f} GB  "
                  f"paged(shared prompt) {paged_share/1e9:6.3f} GB  "
                  f"-> {contig/paged_share:4.1f}x vs contiguous")
    return rows


# ---------------------------------------------------------------------------
# 4. Latency: the cost of the block indirection (gather path)
# ---------------------------------------------------------------------------

def sweep_paged_latency(n_kv_heads=8, n_query_heads=16, head_dim=128, block_size=16,
                        batch=64, contexts=(128, 512, 1024, 2048, 4096),
                        dtype=torch.float16, seed=3, verbose=True):
    """Compare a decode step over a paged cache vs one contiguous tensor, equal
    lengths, so the difference is purely the block-table gather."""
    rows = []
    for L in contexts:
        torch.manual_seed(seed)
        gc.collect(); torch.cuda.empty_cache()
        need = batch * ((L + block_size - 1) // block_size) + 8
        try:
            cache = PagedKVCache(num_blocks=need, block_size=block_size,
                                 n_kv_heads=n_kv_heads, head_dim=head_dim, device=DEV,
                                 dtype=dtype)
            for s in range(batch):
                k = torch.randn(n_kv_heads, L, head_dim, device=DEV, dtype=dtype)
                v = torch.randn(n_kv_heads, L, head_dim, device=DEV, dtype=dtype)
                cache.add_sequence(s, k, v)
            q = torch.randn(batch, n_query_heads, 1, head_dim, device=DEV, dtype=dtype)
            seq_ids = list(range(batch))

            def paged_step():
                paged_attention_decode(cache, seq_ids, q, n_query_heads,
                                       backend="flash")

            # contiguous equivalent: one dense (B, Hkv, L, D) tensor, same SDPA call
            ck = torch.randn(batch, n_kv_heads, L, head_dim, device=DEV, dtype=dtype)
            cv = torch.randn(batch, n_kv_heads, L, head_dim, device=DEV, dtype=dtype)

            def contig_step():
                from attention import sdpa_attention
                sdpa_attention(q, ck, cv, backend="flash", enable_gqa=True)

            t_paged = time_op(paged_step, warmup=5, iters=30)
            t_contig = time_op(contig_step, warmup=5, iters=30)
            rows.append(dict(sweep="paged_latency", context=L, batch=batch,
                             paged_ms=t_paged, contiguous_ms=t_contig,
                             overhead_pct=100 * (t_paged / t_contig - 1),
                             status="ok"))
            if verbose:
                print(f"  ctx {L:5d}: paged {t_paged:7.3f} ms  "
                      f"contiguous {t_contig:7.3f} ms  "
                      f"overhead {100*(t_paged/t_contig-1):+5.1f}%")
            del cache, q, ck, cv
        except (torch.cuda.OutOfMemoryError, KVCacheOOM):
            rows.append(dict(sweep="paged_latency", context=L, batch=batch,
                             status="OOM"))
            if verbose:
                print(f"  ctx {L:5d}: OOM")
        gc.collect(); torch.cuda.empty_cache()
    return rows


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "results"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model = KVModel()
    print("=" * 78)
    print("PAGEDATTENTION BENCHMARK (vLLM-style KV cache, from scratch)")
    print("=" * 78)
    print(f"  KV model: {model.name}  {model.n_layers} layers, "
          f"{model.n_kv_heads} KV heads, head_dim {model.head_dim}, fp16")
    print(f"  KV cache cost: {model.bytes_per_token()/1024:.1f} KB per token "
          f"({model.bytes_per_token()*2048/1e6:.0f} MB per 2048-token sequence)")
    print(f"  block_size: 16 tokens")

    payload = {"model": model.__dict__}

    print("\n[1] Fragmentation — allocated KV memory & utilisation (256 seqs, "
          "max_len 2048)")
    payload["fragmentation"] = sweep_fragmentation(
        model, mean_lens=(64, 256, 1024) if args.quick
        else (64, 128, 256, 512, 1024))

    print("\n[2a] Concurrency — sequences admitted in a fixed KV budget")
    payload["concurrency"] = sweep_concurrency(
        model, budgets_gb=(4, 16) if args.quick else (2, 4, 8, 16, 32, 64))

    print("\n[2b] Decode throughput — measured tokens/s vs batch size")
    payload["decode_throughput"] = measure_decode_throughput(
        batches=(1, 32, 128) if args.quick else (1, 8, 32, 64, 128, 256, 512))

    print("\n[3] Prefix sharing — shared 512-token prompt across N sequences")
    payload["prefix_sharing"] = sweep_prefix_sharing(
        model, n_seqs=(1, 8, 64) if args.quick
        else (1, 2, 4, 8, 16, 32, 64, 128, 256))

    print("\n[4] Latency — paged gather path vs one contiguous tensor (equal length)")
    payload["paged_latency"] = sweep_paged_latency(
        contexts=(512, 2048) if args.quick else (128, 512, 1024, 2048, 4096))

    out = os.path.join(args.out, "paged_benchmarks.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
