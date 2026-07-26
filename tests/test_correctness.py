"""
Correctness suite. Everything here must pass before any timing number is trusted:
a fast kernel that computes the wrong thing is not an optimisation.

Run:  .venv/bin/python tests/test_correctness.py
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from attention import (  # noqa: E402
    MultiHeadSelfAttention,
    available_backends,
    flash_attention_tiled,
    naive_attention,
    sdpa_attention,
)
from flash_triton import HAS_TRITON, flash_attention_triton  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
PASS, FAIL = "\033[92mPASS\033[0m", "\033[91mFAIL\033[0m"
_results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = ""):
    _results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  ({detail})" if detail else ""))


def fp64_reference(q, k, v, causal=False):
    """Ground truth in float64 - the yardstick every fp16 kernel is measured against."""
    qd, kd, vd = q.double(), k.double(), v.double()
    s = (qd @ kd.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        m = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril(n_k - n_q)
        s = s.masked_fill(~m, float("-inf"))
    return torch.softmax(s, dim=-1) @ vd


def rel_err(x, ref):
    return ((x.double() - ref).norm() / ref.norm()).item()


# ---------------------------------------------------------------------------
# 1. The online-softmax recurrence is algebraically exact
# ---------------------------------------------------------------------------
def test_tiled_flash_is_exact():
    print("\n1. From-scratch tiled FlashAttention == textbook attention (fp64 math)")
    torch.manual_seed(0)
    for causal in (False, True):
        for (B, H, N, D) in [(2, 3, 128, 32), (1, 2, 200, 64), (2, 2, 512, 64)]:
            q, k, v = (torch.randn(B, H, N, D, device=DEV, dtype=torch.float64)
                       for _ in range(3))
            ref = naive_attention(q, k, v, causal=causal)
            got = flash_attention_tiled(q, k, v, causal=causal, block_q=64, block_k=64,
                                        accum_dtype=torch.float64)
            err = (got - ref).abs().max().item()
            check(f"tiled vs naive  B{B} H{H} N{N} D{D} causal={causal}",
                  err < 1e-12, f"max abs err {err:.2e}")


def test_tiled_flash_block_size_invariance():
    print("\n2. Result is independent of tile size (the tiling is not an approximation)")
    torch.manual_seed(1)
    q, k, v = (torch.randn(2, 4, 384, 64, device=DEV, dtype=torch.float64) for _ in range(3))
    ref = naive_attention(q, k, v)
    for bq, bk in [(16, 16), (32, 128), (128, 32), (384, 384), (100, 77)]:
        got = flash_attention_tiled(q, k, v, block_q=bq, block_k=bk,
                                    accum_dtype=torch.float64)
        err = (got - ref).abs().max().item()
        check(f"block_q={bq:3d} block_k={bk:3d}", err < 1e-12, f"max abs err {err:.2e}")


# ---------------------------------------------------------------------------
# 2. Numerical stability: the running max is what keeps exp() from overflowing
# ---------------------------------------------------------------------------
def test_numerical_stability_large_logits():
    print("\n3. Stability under large logits (running-max rescaling)")
    torch.manual_seed(2)
    q = torch.randn(1, 2, 256, 64, device=DEV, dtype=torch.float32) * 30.0
    k = torch.randn(1, 2, 256, 64, device=DEV, dtype=torch.float32) * 30.0
    v = torch.randn(1, 2, 256, 64, device=DEV, dtype=torch.float32)
    ref = fp64_reference(q, k, v)
    got = flash_attention_tiled(q, k, v, block_q=64, block_k=64)
    check("tiled flash finite with logits ~O(1e3)", torch.isfinite(got).all().item(),
          f"rel err {rel_err(got, ref):.2e}")
    if DEV == "cuda":
        fa = sdpa_attention(q.half(), k.half(), v.half(), backend="flash")
        check("SDPA flash finite with huge logits", torch.isfinite(fa).all().item(),
              f"rel err {rel_err(fa, ref):.2e}")


def test_fully_masked_rows_no_nan():
    print("\n4. Causal masking edge cases produce no NaNs")
    torch.manual_seed(3)
    q, k, v = (torch.randn(1, 1, 64, 64, device=DEV, dtype=torch.float32) for _ in range(3))
    got = flash_attention_tiled(q, k, v, causal=True, block_q=16, block_k=16)
    check("tiled causal: all finite", torch.isfinite(got).all().item())
    ref = fp64_reference(q, k, v, causal=True)
    check("tiled causal matches fp64 ref", rel_err(got, ref) < 1e-6,
          f"rel err {rel_err(got, ref):.2e}")
    # first query row attends to exactly one key -> output must equal v[0]
    check("row 0 of causal output == v[0]",
          torch.allclose(got[0, 0, 0], v[0, 0, 0], atol=1e-5))


# ---------------------------------------------------------------------------
# 3. Every fused kernel agrees with the fp64 ground truth
# ---------------------------------------------------------------------------
def test_all_backends_agree():
    if DEV != "cuda":
        return
    print("\n5. All fused kernels vs float64 ground truth (fp16 inputs)")
    torch.manual_seed(4)
    B, H, N, D = 4, 8, 512, 64
    q, k, v = (torch.randn(B, H, N, D, device=DEV, dtype=torch.float16) for _ in range(3))
    ref = fp64_reference(q, k, v)

    outs = {}
    for name in available_backends():
        outs[f"sdpa:{name}"] = sdpa_attention(q, k, v, backend=name)
    outs["sdpa:auto"] = sdpa_attention(q, k, v, backend=None)
    outs["naive_fp16"] = naive_attention(q, k, v)
    outs["tiled_flash"] = flash_attention_tiled(q, k, v, block_q=128, block_k=128)
    if HAS_TRITON:
        outs["triton_flash"] = flash_attention_triton(q, k, v)

    for name, o in outs.items():
        r = rel_err(o, ref)
        mx = (o.double() - ref).abs().max().item()
        check(f"{name:20s} vs fp64", r < 5e-3, f"rel {r:.2e}  max abs {mx:.2e}")

    # the assignment's actual question: flash vs the standard path
    std, fa = outs["sdpa:math"], outs["sdpa:flash"]
    mx = (std - fa).abs().max().item()
    cos = F.cosine_similarity(std.flatten().float(), fa.flatten().float(), dim=0).item()
    check("math vs flash: rounding-scale difference", mx < 1e-2,
          f"max abs err {mx:.2e}, cosine {cos:.8f}")
    check("math vs flash: allclose(atol=1e-2)", torch.allclose(std, fa, atol=1e-2))


def test_causal_backends():
    if DEV != "cuda":
        return
    print("\n6. Same, with causal masking on")
    torch.manual_seed(5)
    q, k, v = (torch.randn(2, 8, 1024, 64, device=DEV, dtype=torch.float16) for _ in range(3))
    ref = fp64_reference(q, k, v, causal=True)
    for name in available_backends():
        try:
            o = sdpa_attention(q, k, v, causal=True, backend=name)
            check(f"causal sdpa:{name}", rel_err(o, ref) < 5e-3, f"rel {rel_err(o, ref):.2e}")
        except Exception as exc:  # noqa: BLE001
            check(f"causal sdpa:{name}", False, f"{type(exc).__name__}")
    if HAS_TRITON:
        o = flash_attention_triton(q, k, v, causal=True)
        check("causal triton_flash", rel_err(o, ref) < 5e-3, f"rel {rel_err(o, ref):.2e}")


def test_non_multiple_sequence_lengths():
    if DEV != "cuda":
        return
    print("\n7. Sequence lengths that are not multiples of the tile size")
    torch.manual_seed(6)
    for N in (1, 7, 63, 65, 129, 333):
        q, k, v = (torch.randn(1, 4, N, 64, device=DEV, dtype=torch.float16) for _ in range(3))
        ref = fp64_reference(q, k, v)
        ok = True
        detail = []
        for fn, nm in [(lambda: sdpa_attention(q, k, v, backend="flash"), "flash"),
                       (lambda: flash_attention_triton(q, k, v) if HAS_TRITON else None,
                        "triton")]:
            o = fn()
            if o is None:
                continue
            r = rel_err(o, ref)
            ok &= r < 5e-3
            detail.append(f"{nm} {r:.1e}")
        check(f"N={N:4d}", ok, ", ".join(detail))


# ---------------------------------------------------------------------------
# 4. The full layer: kernel swap must not change layer output
# ---------------------------------------------------------------------------
def test_layer_kernel_swap_is_transparent():
    if DEV != "cuda":
        return
    print("\n8. Full MHA layer: swapping the kernel leaves the layer output unchanged")
    torch.manual_seed(7)
    layer = MultiHeadSelfAttention(768, 12, kernel="math").to(DEV).half().eval()
    x = torch.randn(4, 512, 768, device=DEV, dtype=torch.float16)
    with torch.no_grad():
        base = layer(x, kernel="math")
        for kern in ["flash", "mem_efficient", "auto", "tiled_flash"] + (
            ["triton_flash"] if HAS_TRITON else []
        ):
            o = layer(x, kernel=kern)
            mx = (o - base).abs().max().item()
            cos = F.cosine_similarity(o.flatten().float(), base.flatten().float(),
                                      dim=0).item()
            check(f"layer[{kern:13s}] vs layer[math]", mx < 5e-2 and cos > 0.9999,
                  f"max abs {mx:.2e}, cosine {cos:.8f}")


def test_gqa_mqa_shapes_and_cache():
    if DEV != "cuda":
        return
    print("\n9. GQA / MQA head layouts and KV-cache decoding")
    torch.manual_seed(8)
    for n_kv in (12, 4, 2, 1):
        layer = MultiHeadSelfAttention(768, 12, n_kv_heads=n_kv, kernel="flash",
                                       causal=True).to(DEV).half().eval()
        x = torch.randn(2, 64, 768, device=DEV, dtype=torch.float16)
        with torch.no_grad():
            y = layer(x)
        label = "MHA" if n_kv == 12 else ("MQA" if n_kv == 1 else f"GQA-{n_kv}")
        params = sum(p.numel() for p in layer.parameters())
        check(f"{label:7s} forward shape + finite", y.shape == x.shape
              and torch.isfinite(y).all().item(),
              f"params {params/1e6:.2f}M, kv-cache/token/seq {layer.kv_cache_bytes(1,1)}B")

    # incremental decode must reproduce the parallel forward
    layer = MultiHeadSelfAttention(256, 8, n_kv_heads=2, kernel="math",
                                   causal=True).to(DEV).float().eval()
    x = torch.randn(2, 16, 256, device=DEV)
    with torch.no_grad():
        parallel = layer(x)
        cache, steps = None, []
        for t in range(16):
            y_t, cache = layer.decode_step(x[:, t : t + 1], cache, kernel="math")
            steps.append(y_t)
        incremental = torch.cat(steps, dim=1)
    mx = (parallel - incremental).abs().max().item()
    check("KV-cache decode == parallel causal forward", mx < 1e-4, f"max abs err {mx:.2e}")


# ---------------------------------------------------------------------------
# 5. Memory claim: flash really is O(N) where naive is O(N^2)
# ---------------------------------------------------------------------------
def test_memory_scaling_claim():
    if DEV != "cuda":
        return
    print("\n10. Peak-memory scaling: O(N^2) for naive vs O(N) for flash")
    prev = {}
    for N in (512, 1024, 2048):
        q, k, v = (torch.randn(2, 8, N, 64, device=DEV, dtype=torch.float16)
                   for _ in range(3))
        row = {}
        for nm, fn in [("naive", lambda: naive_attention(q, k, v)),
                       ("flash", lambda: sdpa_attention(q, k, v, backend="flash"))]:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            o = fn()
            torch.cuda.synchronize()
            row[nm] = (torch.cuda.max_memory_allocated() - base) / 1e6
            del o
        if prev:
            g_naive = row["naive"] / prev["naive"]
            g_flash = row["flash"] / prev["flash"]
            check(f"N {N//2}->{N}: naive grows ~4x, flash ~2x",
                  g_naive > 3.0 and g_flash < 2.6,
                  f"naive x{g_naive:.2f} ({row['naive']:.1f}MB), "
                  f"flash x{g_flash:.2f} ({row['flash']:.1f}MB)")
        prev = row
        del q, k, v


# ---------------------------------------------------------------------------
def main():
    print("=" * 78)
    print("CORRECTNESS SUITE — attention kernels")
    print("=" * 78)
    if DEV == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}  "
              f"sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}  "
              f"torch {torch.__version__}")
    test_tiled_flash_is_exact()
    test_tiled_flash_block_size_invariance()
    test_numerical_stability_large_logits()
    test_fully_masked_rows_no_nan()
    test_all_backends_agree()
    test_causal_backends()
    test_non_multiple_sequence_lengths()
    test_layer_kernel_swap_is_transparent()
    test_gqa_mqa_shapes_and_cache()
    test_memory_scaling_claim()

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
