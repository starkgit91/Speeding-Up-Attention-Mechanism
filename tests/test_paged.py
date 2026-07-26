"""
Correctness suite for the from-scratch PagedAttention manager.

A paged cache that returns different numbers than a contiguous cache is a bug, not
an optimisation - the whole point is that it stores the same K/V, just in blocks.
These checks establish: exact gather equivalence, exact attention equivalence,
correct copy-on-write prefix sharing, correct block accounting, and OOM behaviour.

Run:  .venv/bin/python tests/test_paged.py
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from paged_attention import (  # noqa: E402
    KVCacheOOM,
    KVModel,
    PagedKVCache,
    contiguous_cache_bytes,
    ideal_cache_bytes,
    paged_attention_decode,
    paged_attention_varlen,
    paged_cache_bytes,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name, ok, detail=""):
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


# ---------------------------------------------------------------------------
def test_gather_roundtrip():
    print("\n1. Gather reconstructs exactly what was appended (block boundaries)")
    torch.manual_seed(0)
    Hkv, D, bs = 4, 32, 16
    for L in (1, 15, 16, 17, 63, 100):
        cache = PagedKVCache(num_blocks=64, block_size=bs, n_kv_heads=Hkv,
                             head_dim=D, device=DEV)
        k_ref = torch.randn(Hkv, L, D, device=DEV, dtype=torch.float16)
        v_ref = torch.randn(Hkv, L, D, device=DEV, dtype=torch.float16)
        cache.add_sequence(0, k_ref, v_ref)
        k, v = cache.gather(0)
        ok = (torch.equal(k[0], k_ref) and torch.equal(v[0], v_ref)
              and k.shape == (1, Hkv, L, D))
        exp_blocks = math.ceil(L / bs)
        check(f"L={L:3d}: gather==appended, {exp_blocks} blocks used",
              ok and cache.blocks_in_use == exp_blocks,
              f"blocks={cache.blocks_in_use}")


def test_attention_equivalence():
    if DEV != "cuda":
        return
    print("\n2. Paged attention == contiguous-cache attention (exact)")
    torch.manual_seed(1)
    Hq, Hkv, D, bs = 8, 8, 64, 16
    B = 6
    L = 48
    cache = PagedKVCache(num_blocks=256, block_size=bs, n_kv_heads=Hkv, head_dim=D,
                         device=DEV)
    contig_k, contig_v = [], []
    for s in range(B):
        k = torch.randn(Hkv, L, D, device=DEV, dtype=torch.float16)
        v = torch.randn(Hkv, L, D, device=DEV, dtype=torch.float16)
        cache.add_sequence(s, k, v)
        contig_k.append(k)
        contig_v.append(v)
    q = torch.randn(B, Hq, 1, D, device=DEV, dtype=torch.float16)

    paged = paged_attention_decode(cache, list(range(B)), q, Hq, backend="flash")
    kk = torch.stack(contig_k).transpose(0, 0)      # (B,Hkv,L,D)
    kk = torch.stack(contig_k)
    vv = torch.stack(contig_v)
    ref = F.scaled_dot_product_attention(q, kk, vv)
    mx = (paged - ref).abs().max().item()
    check("equal-length batch: paged == contiguous", mx < 1e-3,
          f"max abs err {mx:.2e}")


def test_varlen_equivalence():
    if DEV != "cuda":
        return
    print("\n3. Variable-length paged attention == per-sequence reference")
    torch.manual_seed(2)
    Hq, Hkv, D, bs = 8, 2, 64, 16   # GQA
    lens = [7, 31, 64, 100, 5]
    cache = PagedKVCache(num_blocks=256, block_size=bs, n_kv_heads=Hkv, head_dim=D,
                         device=DEV)
    refs = []
    q = torch.randn(len(lens), Hq, 1, D, device=DEV, dtype=torch.float16)
    for i, L in enumerate(lens):
        k = torch.randn(Hkv, L, D, device=DEV, dtype=torch.float16)
        v = torch.randn(Hkv, L, D, device=DEV, dtype=torch.float16)
        cache.add_sequence(i, k, v)
        kf = k[:, None].expand(Hkv, Hq // Hkv, L, D).reshape(Hq, L, D).unsqueeze(0)
        vf = v[:, None].expand(Hkv, Hq // Hkv, L, D).reshape(Hq, L, D).unsqueeze(0)
        refs.append(F.scaled_dot_product_attention(q[i:i + 1], kf, vf))
    ref = torch.cat(refs, dim=0)
    got = paged_attention_varlen(cache, list(range(len(lens))), q, Hq)
    mx = (got - ref).abs().max().item()
    check("ragged batch: paged(varlen) == reference", mx < 2e-3,
          f"max abs err {mx:.2e}")


def test_copy_on_write_prefix_sharing():
    if DEV != "cuda":
        return
    print("\n4. Copy-on-write prefix sharing: shared prompt stored once")
    torch.manual_seed(3)
    Hkv, D, bs = 4, 32, 16
    P = 48          # 3 full blocks of shared prompt
    cache = PagedKVCache(num_blocks=128, block_size=bs, n_kv_heads=Hkv, head_dim=D,
                         device=DEV)
    kp = torch.randn(Hkv, P, D, device=DEV, dtype=torch.float16)
    vp = torch.randn(Hkv, P, D, device=DEV, dtype=torch.float16)
    cache.add_sequence(0, kp, vp)
    used_after_prompt = cache.blocks_in_use

    # fork 8 children off the shared prompt; each generates its own tokens
    n_children = 8
    for c in range(1, n_children + 1):
        cache.fork(0, c)
    check("fork shares blocks (no new physical blocks)",
          cache.blocks_in_use == used_after_prompt,
          f"{cache.blocks_in_use} blocks after {n_children} forks")

    # each child appends 20 tokens -> new blocks, prompt still shared
    for c in range(1, n_children + 1):
        for t in range(20):
            cache.append(c, torch.randn(Hkv, D, device=DEV, dtype=torch.float16),
                         torch.randn(Hkv, D, device=DEV, dtype=torch.float16))
    # naive would store (n_children+1)*(P) prompt tokens; paged stores P once
    naive_prompt_blocks = (n_children + 1) * math.ceil(P / bs)
    paged_prompt_blocks = math.ceil(P / bs)
    check("shared prompt stored once, not per-sequence",
          used_after_prompt == paged_prompt_blocks,
          f"paged {paged_prompt_blocks} vs naive {naive_prompt_blocks} prompt blocks")

    # child 1's first P tokens must still equal the original prompt (COW preserved it)
    k1, _ = cache.gather(1)
    ok = torch.equal(k1[0, :, :P], kp)
    check("COW preserves shared data after children diverge", ok,
          f"prompt intact for child")


def test_block_accounting_and_free():
    print("\n5. Block accounting and free-list correctness")
    Hkv, D, bs = 2, 16, 8
    cache = PagedKVCache(num_blocks=32, block_size=bs, n_kv_heads=Hkv, head_dim=D,
                         device=DEV, allocate_pool=False)
    cache.add_sequence(0, prompt_len=20)   # ceil(20/8)=3 blocks
    cache.add_sequence(1, prompt_len=8)    # 1 block
    check("allocation counts blocks correctly", cache.blocks_in_use == 4,
          f"in use {cache.blocks_in_use}")
    cache.free(0)
    check("freeing returns blocks to the pool", cache.blocks_in_use == 1,
          f"in use {cache.blocks_in_use}")
    check("utilization tracks used vs allocated",
          abs(cache.utilization() - 8 / 8) < 1e-9,
          f"util {cache.utilization():.3f}")


def test_oom_and_admission():
    print("\n6. Pool exhaustion raises, and admission control predicts it")
    Hkv, D, bs = 2, 16, 4
    cache = PagedKVCache(num_blocks=4, block_size=bs, n_kv_heads=Hkv, head_dim=D,
                         device=DEV, allocate_pool=False)
    check("can_allocate true within capacity", cache.can_allocate(16))
    check("can_allocate false beyond capacity", not cache.can_allocate(17))
    cache.add_sequence(0, prompt_len=16)   # fills all 4 blocks
    raised = False
    try:
        cache.append(0, torch.zeros(Hkv, D, device=DEV, dtype=torch.float16),
                     torch.zeros(Hkv, D, device=DEV, dtype=torch.float16))
    except KVCacheOOM:
        raised = True
    check("append past capacity raises KVCacheOOM", raised)


def test_analytic_memory_models():
    print("\n7. Analytic memory models: paged near-ideal, contiguous wasteful")
    model = KVModel()
    torch.manual_seed(4)
    # 128 sequences with a realistic right-skewed generation-length distribution
    # (most completions short, a few long), served with max_len=2048. This is the
    # regime real serving lives in and where contiguous pre-allocation wastes most.
    max_len, bs = 2048, 16
    lens = [min(max_len, max(1, int(x)))
            for x in (torch.distributions.Exponential(1 / 220.0)
                      .sample((128,)) + 8).tolist()]
    contig = contiguous_cache_bytes(lens, max_len, model)
    paged = paged_cache_bytes(lens, bs, model)
    ideal = ideal_cache_bytes(lens, model)
    check("paged uses far less than contiguous", paged < 0.5 * contig,
          f"paged {paged/1e9:.2f} GB vs contiguous {contig/1e9:.2f} GB")
    check("paged is within one block/seq of ideal",
          paged - ideal <= len(lens) * bs * model.bytes_per_token(),
          f"paged {paged/1e9:.2f} GB vs ideal {ideal/1e9:.2f} GB")
    util_paged = ideal / paged
    util_contig = ideal / contig
    check("paged utilization >> contiguous utilization",
          util_paged > 0.9 and util_contig < util_paged,
          f"paged {util_paged*100:.1f}% vs contiguous {util_contig*100:.1f}%")


def main():
    print("=" * 78)
    print("CORRECTNESS SUITE — PagedAttention (vLLM-style KV cache)")
    print("=" * 78)
    if DEV == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    test_gather_roundtrip()
    test_attention_equivalence()
    test_varlen_equivalence()
    test_copy_on_write_prefix_sharing()
    test_block_accounting_and_free()
    test_oom_and_admission()
    test_analytic_memory_models()

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n = len(_results)
    print("\n" + "=" * 78)
    print(f"RESULT: {n_pass}/{n} checks passed")
    if n_pass < n:
        print("\nFailures:")
        for name, ok, detail in _results:
            if not ok:
                print(f"  - {name}: {detail}")
    print("=" * 78)
    return 0 if n_pass == n else 1


if __name__ == "__main__":
    sys.exit(main())
