"""
vLLM-style PagedAttention, implemented from scratch in PyTorch.

Reference: Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, ... Ion Stoica,
"Efficient Memory Management for Large Language Model Serving with PagedAttention",
SOSP 2023 (the paper behind vLLM).

------------------------------------------------------------------------------
What problem this solves (and what it does NOT)
------------------------------------------------------------------------------
FlashAttention (the rest of this project) makes the attention *computation* cheap.
It says nothing about how the KV cache is *stored* across many concurrent sequences
during generation. That is a memory-management problem, and it is the one vLLM's
PagedAttention targets.

The naive KV cache used by `MultiHeadSelfAttention.decode_step` stores each
sequence's K/V in one contiguous tensor. To serve a batch you must pre-allocate
that tensor to the *maximum* possible length, because you cannot know in advance
how many tokens a sequence will generate. The result is massive internal
fragmentation: a request that stops after 50 tokens still holds `max_len` slots.
Reported waste in real serving is 60-80% of KV memory.

PagedAttention borrows virtual memory's idea. The KV cache is cut into fixed-size
**blocks** (pages). A shared pool holds the physical blocks; each sequence owns a
**block table** mapping its logical token positions to physical block ids. Blocks
are allocated on demand as a sequence grows, so the only waste is the last, partly
filled block of each sequence (< block_size tokens). Blocks can also be shared
copy-on-write between sequences, which makes a shared prompt (system prompt, few-shot
prefix, beam search) cost memory once instead of once per sequence.

This file implements that manager and an attention step that reads through the block
table. It is the *mechanism*, in pure PyTorch. Production vLLM adds a fused CUDA
kernel that reads the block table inside the kernel (so no gather) plus a scheduler;
here the compute path gathers the blocks and calls SDPA, which computes the identical
result and lets us measure the memory story cleanly. This mirrors how the FlashAttention
part of the project pairs a from-scratch algorithm with the production fused kernel.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from attention import sdpa_attention


# ---------------------------------------------------------------------------
# Model KV-cache sizing (used by the analytic memory model)
# ---------------------------------------------------------------------------

@dataclass
class KVModel:
    """Just enough of a model to size a KV cache. Defaults ≈ a Llama-2-13B-ish
    GQA config, chosen so the megabyte numbers are realistic for serving."""
    name: str = "gqa-13b-like"
    n_layers: int = 40
    n_kv_heads: int = 8          # GQA; set == n_heads for MHA
    head_dim: int = 128
    dtype_bytes: int = 2         # fp16 / bf16

    def bytes_per_token(self) -> int:
        # K and V, every layer.
        return 2 * self.n_layers * self.n_kv_heads * self.head_dim * self.dtype_bytes


# ---------------------------------------------------------------------------
# The paged KV cache manager
# ---------------------------------------------------------------------------

class KVCacheOOM(RuntimeError):
    """Raised when the block pool cannot satisfy an allocation."""


class PagedKVCache:
    """A single-layer physical KV pool addressed through per-sequence block tables.

    Physical storage is two tensors of shape (num_blocks, block_size, n_kv_heads,
    head_dim). `block_table[seq_id]` is a list of physical block ids; `ref_count`
    tracks sharing so blocks freed by one sequence are not pulled from under another
    that still references them (copy-on-write).
    """

    def __init__(self, num_blocks: int, block_size: int, n_kv_heads: int,
                 head_dim: int, device="cuda", dtype=torch.float16,
                 allocate_pool: bool = True):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.device = device
        self.dtype = dtype
        if allocate_pool:
            shape = (num_blocks, block_size, n_kv_heads, head_dim)
            self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
            self.v_pool = torch.zeros(shape, device=device, dtype=dtype)
        else:  # accounting-only mode: skip the (possibly huge) allocation
            self.k_pool = self.v_pool = None
        self.free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))
        self.ref_count = [0] * num_blocks
        self.block_table: dict[int, list[int]] = {}
        self.seq_len: dict[int, int] = {}

    # -- pool bookkeeping ---------------------------------------------------
    @property
    def blocks_in_use(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def _alloc_block(self) -> int:
        if not self.free_blocks:
            raise KVCacheOOM(f"block pool exhausted ({self.num_blocks} blocks)")
        b = self.free_blocks.pop()
        self.ref_count[b] = 1
        return b

    def _free_block(self, b: int):
        self.ref_count[b] -= 1
        if self.ref_count[b] == 0:
            self.free_blocks.append(b)

    def can_allocate(self, num_tokens: int) -> bool:
        return math.ceil(num_tokens / self.block_size) <= len(self.free_blocks)

    # -- sequence lifecycle -------------------------------------------------
    def add_sequence(self, seq_id: int, k: Optional[torch.Tensor] = None,
                     v: Optional[torch.Tensor] = None, prompt_len: int = 0):
        """Register a new sequence. If k/v (shape (H, T, D)) are given they are the
        prompt's KV and are written in; otherwise `prompt_len` empty slots are
        reserved."""
        assert seq_id not in self.block_table, f"seq {seq_id} already exists"
        self.block_table[seq_id] = []
        self.seq_len[seq_id] = 0
        if k is not None:
            T = k.shape[1]
            for t in range(T):
                self.append(seq_id, k[:, t], v[:, t])
        elif prompt_len:
            for _ in range(prompt_len):
                self._grow_one(seq_id)

    def _grow_one(self, seq_id: int) -> tuple[int, int]:
        """Ensure a slot exists for the next token; return (block_id, offset)."""
        L = self.seq_len[seq_id]
        table = self.block_table[seq_id]
        block_idx, offset = divmod(L, self.block_size)
        if block_idx == len(table):
            table.append(self._alloc_block())
        pid = table[block_idx]
        # copy-on-write: never write into a block another sequence still shares
        if self.ref_count[pid] > 1:
            new = self._alloc_block()
            if self.k_pool is not None:
                self.k_pool[new].copy_(self.k_pool[pid])
                self.v_pool[new].copy_(self.v_pool[pid])
            self.ref_count[pid] -= 1
            table[block_idx] = new
            pid = new
        self.seq_len[seq_id] = L + 1
        return pid, offset

    def append(self, seq_id: int, k_t: torch.Tensor, v_t: torch.Tensor):
        """Append one token's KV. k_t, v_t have shape (n_kv_heads, head_dim)."""
        pid, offset = self._grow_one(seq_id)
        if self.k_pool is not None:
            self.k_pool[pid, offset] = k_t
            self.v_pool[pid, offset] = v_t

    def fork(self, parent_id: int, child_id: int):
        """Create a child sharing the parent's blocks (copy-on-write prefix sharing).

        This is how a shared system prompt / few-shot prefix / beam-search fan-out
        costs memory once rather than once per sequence."""
        assert child_id not in self.block_table
        parent_tbl = self.block_table[parent_id]
        self.block_table[child_id] = list(parent_tbl)   # same physical ids
        self.seq_len[child_id] = self.seq_len[parent_id]
        for b in parent_tbl:
            self.ref_count[b] += 1

    def free(self, seq_id: int):
        for b in self.block_table.pop(seq_id):
            self._free_block(b)
        self.seq_len.pop(seq_id)

    # -- reading the cache back --------------------------------------------
    def gather(self, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Reconstruct contiguous K, V of shape (1, n_kv_heads, L, head_dim)."""
        L = self.seq_len[seq_id]
        table = self.block_table[seq_id]
        nb = math.ceil(L / self.block_size) if L else 0
        if nb == 0:
            empty = torch.empty(1, self.n_kv_heads, 0, self.head_dim,
                                device=self.device, dtype=self.dtype)
            return empty, empty.clone()
        idx = torch.tensor(table[:nb], device=self.device)
        k = self.k_pool[idx].reshape(-1, self.n_kv_heads, self.head_dim)[:L]
        v = self.v_pool[idx].reshape(-1, self.n_kv_heads, self.head_dim)[:L]
        return (k.permute(1, 0, 2).unsqueeze(0).contiguous(),
                v.permute(1, 0, 2).unsqueeze(0).contiguous())

    def gather_batch_equal(self, seq_ids: list[int]) -> tuple[torch.Tensor,
                                                               torch.Tensor]:
        """Vectorised gather for a batch of EQUAL-length sequences.

        Builds the whole (B, Hkv, L, D) K and V with a single block-table indexing
        op instead of a Python loop, so a latency measurement reflects the cost of
        the indirection itself rather than per-sequence Python overhead. (A fused
        paged kernel folds even this indexing into the attention kernel.)
        """
        L = self.seq_len[seq_ids[0]]
        assert all(self.seq_len[s] == L for s in seq_ids), "lengths must be equal"
        nb = math.ceil(L / self.block_size)
        tables = torch.tensor([self.block_table[s][:nb] for s in seq_ids],
                              device=self.device)               # (B, nb)
        B = len(seq_ids)
        k = self.k_pool[tables]                                 # (B,nb,bs,Hkv,D)
        v = self.v_pool[tables]
        k = k.reshape(B, nb * self.block_size, self.n_kv_heads, self.head_dim)[:, :L]
        v = v.reshape(B, nb * self.block_size, self.n_kv_heads, self.head_dim)[:, :L]
        return (k.permute(0, 2, 1, 3).contiguous(),
                v.permute(0, 2, 1, 3).contiguous())

    # -- memory accounting --------------------------------------------------
    def physical_tokens_capacity(self) -> int:
        return self.num_blocks * self.block_size

    def logical_tokens(self) -> int:
        """Sum of sequence lengths (counts a shared prefix once per sequence)."""
        return sum(self.seq_len.values())

    def physical_tokens_used(self) -> int:
        """block_size * blocks physically held (shared blocks counted once)."""
        return self.blocks_in_use * self.block_size

    def utilization(self) -> float:
        used = self.physical_tokens_used()
        return self.logical_tokens() / used if used else 0.0


# ---------------------------------------------------------------------------
# Attention over a paged cache (gather-then-SDPA compute path)
# ---------------------------------------------------------------------------

def paged_attention_decode(cache: PagedKVCache, seq_ids: list[int],
                           q: torch.Tensor, n_query_heads: int,
                           backend: str = "flash") -> torch.Tensor:
    """One decode step for a batch of sequences of EQUAL current length.

    q: (B, n_query_heads, 1, head_dim), one per seq_id (same order).
    Returns (B, n_query_heads, 1, head_dim). Used for the latency benchmark, where
    holding lengths equal isolates the block-gather overhead from everything else.
    """
    k, v = cache.gather_batch_equal(seq_ids)   # (B, Hkv, L, D), one indexing op
    n_rep = n_query_heads // cache.n_kv_heads
    if n_rep > 1:                         # GQA/MQA: let SDPA consume the narrow cache
        return sdpa_attention(q, k, v, causal=False,
                              backend=None if backend == "auto" else backend,
                              enable_gqa=True)
    return sdpa_attention(q, k, v, causal=False,
                          backend=None if backend == "auto" else backend)


def paged_attention_varlen(cache: PagedKVCache, seq_ids: list[int],
                           q: torch.Tensor, n_query_heads: int) -> torch.Tensor:
    """Decode step for sequences of DIFFERENT lengths, via padding + a key mask.

    Correctness path: mirrors what a fused paged kernel computes, but with an
    explicit gather+pad+mask so it can be checked against a contiguous reference.
    q: (B, n_query_heads, 1, head_dim). Returns the same shape.
    """
    lens = [cache.seq_len[s] for s in seq_ids]
    Lmax = max(lens)
    B = len(seq_ids)
    Hkv, D = cache.n_kv_heads, cache.head_dim
    k = torch.zeros(B, Hkv, Lmax, D, device=cache.device, dtype=cache.dtype)
    v = torch.zeros_like(k)
    key_mask = torch.zeros(B, Lmax, dtype=torch.bool, device=cache.device)
    for i, sid in enumerate(seq_ids):
        kk, vv = cache.gather(sid)
        L = kk.shape[2]
        k[i, :, :L] = kk[0]
        v[i, :, :L] = vv[0]
        key_mask[i, :L] = True
    n_rep = n_query_heads // Hkv
    if n_rep > 1:
        k = k[:, :, None].expand(B, Hkv, n_rep, Lmax, D).reshape(B, n_query_heads,
                                                                 Lmax, D)
        v = v[:, :, None].expand(B, Hkv, n_rep, Lmax, D).reshape(B, n_query_heads,
                                                                 Lmax, D)
    attn_mask = key_mask[:, None, None, :]          # (B,1,1,Lmax)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)


# ---------------------------------------------------------------------------
# Analytic memory models (hardware-independent; exact from the block math)
# ---------------------------------------------------------------------------

def contiguous_cache_bytes(seq_lens: list[int], max_len: int, model: KVModel) -> int:
    """Naive cache: every sequence pre-allocates `max_len` slots."""
    return len(seq_lens) * max_len * model.bytes_per_token()


def paged_cache_bytes(seq_lens: list[int], block_size: int, model: KVModel,
                      shared_prefix_len: int = 0, n_sharing: int = 0) -> int:
    """Paged cache: each sequence costs ceil(L/block_size) blocks; an optional shared
    prefix is counted once across `n_sharing` sequences (copy-on-write)."""
    bpt = block_size * model.bytes_per_token()
    if shared_prefix_len and n_sharing:
        shared_blocks = math.ceil(shared_prefix_len / block_size)
        gen_blocks = sum(math.ceil(max(0, L - shared_prefix_len) / block_size)
                         for L in seq_lens)
        return (shared_blocks + gen_blocks) * bpt
    return sum(math.ceil(L / block_size) for L in seq_lens) * bpt


def ideal_cache_bytes(seq_lens: list[int], model: KVModel) -> int:
    """Lower bound: exactly one slot per real token, no blocks, no padding."""
    return sum(seq_lens) * model.bytes_per_token()
