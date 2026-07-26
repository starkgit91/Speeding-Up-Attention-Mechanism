"""
Attention implementations for the FlashAttention benchmark study.

Contents
--------
1. `naive_attention`          - textbook scaled-dot-product attention. Explicitly
                               materialises the N x N score matrix S and the N x N
                               probability matrix P. This is the O(N^2)-memory baseline.
2. `flash_attention_tiled`    - a *from-scratch* implementation of the FlashAttention
                               algorithm (tiling + online/running softmax) written in
                               pure PyTorch. It is not fast (Python loop over tiles),
                               but it is the literal Algorithm 1 of Dao et al. 2022 and
                               is used to prove that the online-softmax recurrence is
                               mathematically exact.
3. `sdpa_attention`           - torch.nn.functional.scaled_dot_product_attention forced
                               onto a specific fused backend (MATH / FLASH / EFFICIENT /
                               CUDNN) or left on automatic dispatch.
4. `MultiHeadSelfAttention`   - a complete MHA *layer* (W_q, W_k, W_v, W_o) whose inner
                               attention kernel is selectable, and which additionally
                               supports Grouped-Query (GQA) and Multi-Query (MQA) head
                               layouts plus an incremental KV cache for decoding.

Shape convention throughout: q, k, v are (B, H, N, D) - batch, heads, sequence, head_dim.
"""

from __future__ import annotations

import contextlib
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Backend registry / availability probing
# ---------------------------------------------------------------------------

try:  # PyTorch >= 2.2 current API
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _HAS_SDPA_KERNEL = True
except ImportError:  # pragma: no cover - very old torch
    SDPBackend = None
    sdpa_kernel = None
    _HAS_SDPA_KERNEL = False


_BACKEND_BY_NAME = {}
if _HAS_SDPA_KERNEL:
    _BACKEND_BY_NAME = {
        "math": SDPBackend.MATH,
        "flash": SDPBackend.FLASH_ATTENTION,
        "mem_efficient": SDPBackend.EFFICIENT_ATTENTION,
    }
    # CUDNN_ATTENTION only exists in newer releases
    if hasattr(SDPBackend, "CUDNN_ATTENTION"):
        _BACKEND_BY_NAME["cudnn"] = SDPBackend.CUDNN_ATTENTION


@contextlib.contextmanager
def force_backend(name: Optional[str]):
    """Context manager pinning SDPA to one backend. `None` == automatic dispatch."""
    if name is None or name == "auto" or not _HAS_SDPA_KERNEL:
        yield
        return
    backend = _BACKEND_BY_NAME.get(name)
    if backend is None:
        raise ValueError(f"unknown SDPA backend {name!r}; have {sorted(_BACKEND_BY_NAME)}")
    with sdpa_kernel(backend):
        yield


def available_backends() -> list[str]:
    return sorted(_BACKEND_BY_NAME)


# ---------------------------------------------------------------------------
# 1. Baseline: textbook attention that materialises the N x N matrices
# ---------------------------------------------------------------------------


def naive_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    return_probs: bool = False,
):
    """Attention(Q,K,V) = softmax(Q K^T / sqrt(d)) V, written literally.

    Both S = QK^T/sqrt(d) and P = softmax(S) are full (B, H, N, N) tensors that live
    in GPU HBM. This is the O(N^2) memory term that FlashAttention removes.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)

    s = torch.matmul(q, k.transpose(-2, -1)) * scale          # (B,H,N,N)  <-- HBM write
    if causal:
        n_q, n_k = s.shape[-2], s.shape[-1]
        mask = torch.ones(n_q, n_k, dtype=torch.bool, device=s.device).tril(
            diagonal=n_k - n_q
        )
        s = s.masked_fill(~mask, float("-inf"))
    p = torch.softmax(s, dim=-1)                              # (B,H,N,N)  <-- HBM write
    out = torch.matmul(p, v)                                  # (B,H,N,D)
    if return_probs:
        return out, p
    return out


def naive_attention_fp32_accum(q, k, v, causal: bool = False):
    """Same as `naive_attention` but the softmax/matmuls run in fp32.

    Used as a fairer accuracy yardstick: it isolates 'different summation order'
    from 'different accumulation precision'.
    """
    out = naive_attention(q.float(), k.float(), v.float(), causal=causal)
    return out.to(q.dtype)


# ---------------------------------------------------------------------------
# 2. FlashAttention from scratch: tiling + online softmax (Dao et al., Alg. 1)
# ---------------------------------------------------------------------------


def flash_attention_tiled(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    block_q: int = 128,
    block_k: int = 128,
    accum_dtype: torch.dtype = torch.float32,
):
    """Reference implementation of the FlashAttention forward algorithm.

    The full N x N score matrix is never formed. Q is split into row blocks of
    `block_q`; K/V into column blocks of `block_k`. For every query block we keep a
    running row-max `m`, a running softmax denominator `l`, and a running unnormalised
    output accumulator `acc`, rescaling all three whenever a new block shifts the max:

        m_new = max(m, rowmax(S_ij))
        alpha = exp(m - m_new)                 # correction factor for what we already have
        P_ij  = exp(S_ij - m_new)
        l     = alpha * l   + rowsum(P_ij)
        acc   = alpha * acc + P_ij @ V_j
        m     = m_new

    After the last block, `acc / l` equals the textbook softmax result exactly (in
    exact arithmetic), because every partial sum has been rescaled by the same factor
    that the final normaliser would have applied.

    NOTE: this is the *algorithm*, not the *kernel*. In Python each tile still round-
    trips through HBM, so it is slower than the naive version; the real speed-up needs
    a fused CUDA/Triton kernel (see `flash_triton.py` and the SDPA flash backend).
    Its purpose here is to demonstrate exactness and the O(N) memory profile.
    """
    B, H, N_q, D = q.shape
    N_k = k.shape[2]
    scale = 1.0 / math.sqrt(D)

    out = torch.empty_like(q)

    for i0 in range(0, N_q, block_q):
        i1 = min(i0 + block_q, N_q)
        q_i = q[:, :, i0:i1, :]                                   # (B,H,Br,D) - in "SRAM"

        m_i = torch.full((B, H, i1 - i0, 1), float("-inf"), device=q.device, dtype=accum_dtype)
        l_i = torch.zeros((B, H, i1 - i0, 1), device=q.device, dtype=accum_dtype)
        acc = torch.zeros((B, H, i1 - i0, D), device=q.device, dtype=accum_dtype)

        for j0 in range(0, N_k, block_k):
            j1 = min(j0 + block_k, N_k)
            if causal and j0 > (i1 - 1) + (N_k - N_q):
                break  # whole tile is masked out -> skip it entirely (this is why
                       # causal FlashAttention is ~2x cheaper than a masked dense matmul)

            k_j = k[:, :, j0:j1, :]
            v_j = v[:, :, j0:j1, :]

            # ---- small Br x Bc tile, would live entirely in SRAM ------------------
            s_ij = torch.matmul(q_i, k_j.transpose(-2, -1)).to(accum_dtype) * scale
            if causal:
                qi = torch.arange(i0, i1, device=q.device).unsqueeze(1) + (N_k - N_q)
                kj = torch.arange(j0, j1, device=q.device).unsqueeze(0)
                s_ij = s_ij.masked_fill(kj > qi, float("-inf"))

            # ---- online softmax update -------------------------------------------
            m_ij = s_ij.amax(dim=-1, keepdim=True)
            m_new = torch.maximum(m_i, m_ij)
            # guard the all -inf row (fully masked tile) so exp(-inf - -inf) != NaN
            m_safe = torch.where(torch.isinf(m_new), torch.zeros_like(m_new), m_new)

            p_ij = torch.exp(s_ij - m_safe)
            alpha = torch.exp(m_i - m_safe)
            alpha = torch.where(torch.isinf(m_i), torch.zeros_like(alpha), alpha)

            l_i = alpha * l_i + p_ij.sum(dim=-1, keepdim=True)
            acc = alpha * acc + torch.matmul(p_ij.to(v.dtype), v_j).to(accum_dtype)
            m_i = m_new

        out[:, :, i0:i1, :] = (acc / l_i.clamp_min(1e-30)).to(q.dtype)

    return out


# ---------------------------------------------------------------------------
# 3. Fused kernels via PyTorch SDPA
# ---------------------------------------------------------------------------


def sdpa_attention(q, k, v, causal: bool = False, backend: Optional[str] = None,
                   enable_gqa: bool = False):
    """scaled_dot_product_attention pinned to `backend` ('math'|'flash'|'mem_efficient'|
    'cudnn'|None for auto).

    `enable_gqa=True` lets the kernel consume fewer KV heads than Q heads directly,
    instead of the caller materialising a repeated copy of K/V. This matters a great
    deal at decode time - see `MultiHeadSelfAttention.decode_step`.
    """
    with force_backend(backend):
        if enable_gqa:
            return F.scaled_dot_product_attention(q, k, v, is_causal=causal,
                                                  enable_gqa=True)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal)


def probe_backend(backend: str, device="cuda", dtype=torch.float16,
                  shape=(2, 4, 256, 64), causal=False) -> tuple[bool, str]:
    """Return (works, message) for a backend on this machine + these shapes."""
    try:
        q, k, v = (torch.randn(shape, device=device, dtype=dtype) for _ in range(3))
        with force_backend(backend):
            out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        torch.cuda.synchronize()
        if not torch.isfinite(out).all():
            return False, "produced non-finite output"
        return True, "ok"
    except Exception as exc:  # noqa: BLE001 - we genuinely want any failure
        return False, f"{type(exc).__name__}: {str(exc)[:160]}"


# ---------------------------------------------------------------------------
# 4. The full multi-head self-attention *layer*
# ---------------------------------------------------------------------------

KERNELS = ("naive", "tiled_flash", "math", "flash", "mem_efficient", "cudnn", "auto")


def _dispatch(kernel: str, q, k, v, causal: bool):
    if kernel == "naive":
        return naive_attention(q, k, v, causal=causal)
    if kernel == "naive_fp32":
        return naive_attention_fp32_accum(q, k, v, causal=causal)
    if kernel == "tiled_flash":
        return flash_attention_tiled(q, k, v, causal=causal)
    if kernel == "triton_flash":
        from flash_triton import flash_attention_triton

        return flash_attention_triton(q, k, v, causal=causal)
    if kernel == "auto":
        return sdpa_attention(q, k, v, causal=causal, backend=None)
    return sdpa_attention(q, k, v, causal=causal, backend=kernel)


class MultiHeadSelfAttention(nn.Module):
    """Standard MHA layer with a swappable inner kernel and optional GQA/MQA layout.

    n_kv_heads == n_heads  -> standard Multi-Head Attention (MHA)
    1 < n_kv_heads < n_heads -> Grouped-Query Attention (GQA)
    n_kv_heads == 1        -> Multi-Query Attention  (MQA)

    Note the head layout (`n_kv_heads`) changes *what* is computed and therefore needs
    training; the kernel choice (`kernel`) changes only *how* it is computed and is a
    drop-in swap on a trained checkpoint.
    """

    def __init__(
        self,
        embed_dim: int,
        n_heads: int,
        n_kv_heads: Optional[int] = None,
        kernel: str = "auto",
        causal: bool = False,
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must divide n_heads"
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_heads if n_kv_heads is None else n_kv_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = embed_dim // n_heads
        self.kernel = kernel
        self.causal = causal
        self.dropout = dropout

        kv_dim = self.n_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, kv_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
        """(B, H_kv, N, D) -> (B, H_kv*n_rep, N, D) by repeating each kv head."""
        if n_rep == 1:
            return x
        B, H, N, D = x.shape
        return x[:, :, None].expand(B, H, n_rep, N, D).reshape(B, H * n_rep, N, D)

    def project(self, x: torch.Tensor):
        B, N, _ = x.shape
        q = self.q_proj(x).view(B, N, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, N, self.n_kv_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor, kernel: Optional[str] = None) -> torch.Tensor:
        B, N, _ = x.shape
        q, k, v = self.project(x)
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)
        out = _dispatch(kernel or self.kernel, q, k, v, self.causal)
        out = out.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        return self.out_proj(out)

    # -- incremental decoding (where MQA/GQA actually pay off) --------------
    @torch.no_grad()
    def decode_step(self, x_t: torch.Tensor, cache: Optional[tuple] = None,
                    kernel: Optional[str] = None, native_gqa: bool = True):
        """One autoregressive step. `x_t` is (B, 1, E). Returns (y_t, new_cache).

        The KV cache is (B, n_kv_heads, T, D) - note it grows with `n_kv_heads`, not
        `n_heads`, which is precisely the memory-bandwidth argument for MQA/GQA.

        `native_gqa=True` hands the narrow K/V straight to the kernel. With
        `native_gqa=False` the caller expands K/V to `n_heads` first - the obvious
        implementation, and a trap: it re-materialises the full-width cache on every
        step, which throws away the bandwidth saving that motivated GQA in the
        first place. Both paths are benchmarked in the report.
        """
        B = x_t.shape[0]
        q, k, v = self.project(x_t)
        if cache is not None:
            k = torch.cat([cache[0], k], dim=2)
            v = torch.cat([cache[1], v], dim=2)
        new_cache = (k, v)
        kern = kernel or self.kernel
        use_native = native_gqa and self.n_rep > 1 and kern in (
            "flash", "mem_efficient", "cudnn", "math", "auto")
        if use_native:
            out = sdpa_attention(q, k, v, causal=False,
                                 backend=None if kern == "auto" else kern,
                                 enable_gqa=True)
        else:
            out = _dispatch(kern, q, self._repeat_kv(k, self.n_rep),
                            self._repeat_kv(v, self.n_rep), causal=False)
        out = out.transpose(1, 2).contiguous().view(B, 1, self.embed_dim)
        return self.out_proj(out), new_cache

    def kv_cache_bytes(self, batch: int, seq_len: int, dtype_size: int = 2) -> int:
        return 2 * batch * self.n_kv_heads * seq_len * self.head_dim * dtype_size

    def extra_repr(self) -> str:
        layout = ("MHA" if self.n_kv_heads == self.n_heads
                  else "MQA" if self.n_kv_heads == 1 else "GQA")
        return (f"embed_dim={self.embed_dim}, n_heads={self.n_heads}, "
                f"n_kv_heads={self.n_kv_heads} [{layout}], kernel={self.kernel}, "
                f"causal={self.causal}")
