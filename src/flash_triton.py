"""
A hand-written FlashAttention-2 style forward kernel in Triton.

This is the *fused kernel* counterpart to `attention.flash_attention_tiled`:
same algorithm (tiling + online softmax), but the inner loop runs inside a single
GPU kernel so the Br x Bc score tile lives in registers/SRAM and is never written
to HBM. That is the entire source of the speed-up.

Parallelisation follows FlashAttention-2: one program (CTA) per (query block, head,
batch) so the sequence dimension is parallelised too, not just batch x heads.

Falls back gracefully (raises RuntimeError) if Triton is unavailable or the GPU
architecture is unsupported; the benchmark treats that as a missing backend.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    HAS_TRITON = False
    triton = None
    tl = None


if HAS_TRITON:

    @triton.jit
    def _flash_fwd_kernel(
        Q, K, V, O,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_ob, stride_oh, stride_om, stride_od,
        H: tl.constexpr,
        N_CTX_Q, N_CTX_K,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        # ---- which query block / which (batch, head) am I? --------------------
        start_m = tl.program_id(0)          # query-block index  (FA-2: seq is parallel)
        off_bh = tl.program_id(1)
        off_b = off_bh // H
        off_h = off_bh % H

        offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q_ptrs = (Q + off_b * stride_qb + off_h * stride_qh
                  + offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        k_ptrs = (K + off_b * stride_kb + off_h * stride_kh
                  + offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
        v_ptrs = (V + off_b * stride_vb + off_h * stride_vh
                  + offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

        # ---- load Q block once; it stays resident for the whole inner loop ----
        m_mask = offs_m < N_CTX_Q
        q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0)

        # ---- running softmax state (fp32 accumulators) ------------------------
        m_i = tl.full([BLOCK_M], value=-float("inf"), dtype=tl.float32)   # running max
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)                       # running denom
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)              # running output

        # causal: only key blocks up to the diagonal matter
        hi = N_CTX_K
        if IS_CAUSAL:
            hi = tl.minimum(N_CTX_K, (start_m + 1) * BLOCK_M + (N_CTX_K - N_CTX_Q))

        for start_n in range(0, hi, BLOCK_N):
            start_n = tl.multiple_of(start_n, BLOCK_N)
            n_idx = start_n + offs_n
            n_mask = n_idx < N_CTX_K

            k = tl.load(k_ptrs + start_n * stride_kn, mask=n_mask[:, None], other=0.0)

            # S_ij = Q_i K_j^T / sqrt(d)   -- BLOCK_M x BLOCK_N, stays in registers
            qk = tl.dot(q, tl.trans(k)).to(tl.float32) * sm_scale
            qk = tl.where(n_mask[None, :], qk, -float("inf"))
            if IS_CAUSAL:
                causal_ok = (offs_m[:, None] + (N_CTX_K - N_CTX_Q)) >= n_idx[None, :]
                qk = tl.where(causal_ok, qk, -float("inf"))

            # ---- online softmax update (Dao et al. Alg. 1 / FA-2 variant) -----
            m_new = tl.maximum(m_i, tl.max(qk, 1))
            m_safe = tl.where(m_new == -float("inf"), 0.0, m_new)
            p = tl.exp(qk - m_safe[:, None])
            p = tl.where(n_mask[None, :], p, 0.0)
            alpha = tl.exp(m_i - m_safe)
            alpha = tl.where(m_i == -float("inf"), 0.0, alpha)

            l_i = l_i * alpha + tl.sum(p, 1)
            acc = acc * alpha[:, None]

            vblk = tl.load(v_ptrs + start_n * stride_vn, mask=n_mask[:, None], other=0.0)
            acc += tl.dot(p.to(vblk.dtype), vblk).to(tl.float32)
            m_i = m_new

        # ---- normalise once, then the single HBM write-back ------------------
        acc = acc / tl.where(l_i == 0.0, 1.0, l_i)[:, None]
        o_ptrs = (O + off_b * stride_ob + off_h * stride_oh
                  + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od)
        tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=m_mask[:, None])


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def flash_attention_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_m: int = 64,
    block_n: int = 64,
    num_warps: int = 4,
    num_stages: int = 2,
) -> torch.Tensor:
    """FlashAttention forward via the Triton kernel above. q,k,v: (B, H, N, D)."""
    if not HAS_TRITON:
        raise RuntimeError("Triton is not available in this environment")
    assert q.is_cuda and k.is_cuda and v.is_cuda, "Triton kernel needs CUDA tensors"
    B, H, N_q, D = q.shape
    N_k = k.shape[2]
    assert k.shape[1] == H and v.shape[1] == H, "expand GQA/MQA kv heads before calling"
    block_d = _next_pow2(D)
    assert block_d == D, f"head_dim must be a power of two (got {D})"

    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    o = torch.empty_like(q)
    sm_scale = 1.0 / math.sqrt(D)

    grid = (triton.cdiv(N_q, block_m), B * H)
    _flash_fwd_kernel[grid](
        q, k, v, o,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        H=H,
        N_CTX_Q=N_q, N_CTX_K=N_k,
        BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_D=block_d,
        IS_CAUSAL=causal,
        num_warps=num_warps, num_stages=num_stages,
    )
    return o


def probe(device="cuda") -> tuple[bool, str]:
    """Check the Triton kernel compiles and is numerically right on this GPU."""
    if not HAS_TRITON:
        return False, "triton not installed"
    try:
        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 256, 64, device=device, dtype=torch.float16)
                   for _ in range(3))
        out = flash_attention_triton(q, k, v, causal=False)
        ref = torch.softmax(
            (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(64), dim=-1
        ) @ v.float()
        err = (out.float() - ref).abs().max().item()
        if not torch.isfinite(out).all():
            return False, "non-finite output"
        if err > 2e-2:
            return False, f"numerically wrong (max abs err {err:.3e})"
        return True, f"ok (max abs err vs fp32 ref {err:.2e})"
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {str(exc)[:200]}"
