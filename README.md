# Speeding up multi-head self-attention: FlashAttention + PagedAttention, benchmarked end to end

A complete, reproducible implementation and measurement study of attention-inference
optimisation, built around the assignment brief in
`FlashAttention_Benchmark_Report.docx` and extended with a vLLM-style PagedAttention
KV-cache manager. Every number in `REPORT.md` was measured on this machine — nothing is
quoted from a paper or transcribed by hand — and `tests/verify_report.py` re-derives
each one from the raw JSON.

**The finished report is [`REPORT.md`](REPORT.md).**

Three optimisations at three layers of the stack, each implemented and measured here:

- **FlashAttention** — the attention *kernel* (IO-aware tiling + online softmax).
- **GQA / MQA** — the model *architecture* (fewer KV heads).
- **PagedAttention** — the KV-cache *memory manager* (vLLM-style paging + copy-on-write).

## What is in here

| Path | What it is |
|---|---|
| `src/attention.py` | The layer under study. Textbook MHA that materialises the N×N matrices, the FlashAttention algorithm written from scratch in pure PyTorch, wrappers for every fused SDPA backend, and an MHA/GQA/MQA layer with a KV cache. |
| `src/flash_triton.py` | A hand-written FlashAttention-2 style forward kernel in Triton — the actual fused kernel, not just the algorithm. |
| `src/paged_attention.py` | A **vLLM-style PagedAttention KV cache from scratch**: block pool, per-sequence block tables, on-demand allocation, copy-on-write prefix sharing, admission control, and a gather-then-attend compute path. |
| `tests/test_correctness.py` | 49 checks on the FlashAttention kernels: exactness vs fp64, tile-size invariance, stability, causal edge cases, ragged lengths, layer kernel-swap transparency, KV-cache decode equivalence, O(N²)-vs-O(N) memory. |
| `tests/test_paged.py` | 20 checks on PagedAttention: exact gather/attention equivalence to a contiguous cache, copy-on-write sharing, block accounting, OOM/admission, and the analytic memory models. |
| `tests/verify_report.py` | ~100 checks that re-derive every quantitative claim in `REPORT.md` from the raw JSON, so a stale number cannot survive a re-run. |
| `src/benchmark.py` | FlashAttention harness. Nine sweeps: sequence length, long context, causal, head dim, batch size, dtype, full layer, training step, autoregressive decode. |
| `src/paged_benchmark.py` | PagedAttention harness: fragmentation/utilisation, concurrent sequences per memory budget, prefix sharing, and unfused-gather latency overhead. |
| `src/train_gpt.py` | Trains a small character-level GPT four times (MHA, GQA-4, GQA-2, MQA) to measure the quality cost of the architecture-level methods, then re-evaluates one trained checkpoint under every kernel to verify the kernel-level ones are free. |
| `src/seed_study.py` | Retrains three layouts across three seeds to establish the noise floor — the architecture sweep's ranking turns out not to survive it. |
| `src/plots.py`, `src/tables.py` | Figures (`figures/`) and markdown/CSV tables (`results/`). |
| `results/` | Raw JSON, generated tables, CSVs, and the full benchmark/training/seed logs. |
| `figures/` | 23 figures. |

## Reproducing

```bash
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/bin/pip install matplotlib pandas numpy

.venv/bin/python tests/test_correctness.py          # 49/49 FlashAttention checks
.venv/bin/python tests/test_paged.py                # 20/20 PagedAttention checks
cd src
../.venv/bin/python benchmark.py       --out ../results   # ~25 min on an RTX 5070
../.venv/bin/python train_gpt.py       --steps 3000       # ~6 min
../.venv/bin/python seed_study.py      --seeds 1337 7 2024
../.venv/bin/python paged_benchmark.py --out ../results
../.venv/bin/python tables.py && ../.venv/bin/python plots.py
cd .. && .venv/bin/python tests/verify_report.py    # every report number vs raw data
```

`benchmark.py --quick` and `paged_benchmark.py --quick` run short smoke-test versions.

The CUDA 12.8 wheel is required for Blackwell (sm_120); on an older GPU any recent
PyTorch build works and the `--index-url` can be dropped.

## Headline result

On an RTX 5070 at batch 16, 12 heads, head_dim 64, fp16, N = 2048:

- **5.3× faster** than standard MHA in the forward pass, **18.3×** with causal masking,
  **18.0×** for a full training step, **10.6×** for the complete layer including projections.
- **98.4% less** attention workspace memory (3,272 MB → 52 MB), which is what lets the
  fused kernel reach 32,768 tokens on a 12 GB card where the baseline dies at 8,192.
- Arithmetic utilisation rises from **16% to 97%** of the card's dense-matmul peak — the
  memory-bound-to-compute-bound conversion, measured.
- **Exact**: max absolute difference from the standard path is 2.4e-4, one fp16 ulp.
  Measured against a float64 ground truth, FlashAttention is in fact *more* accurate
  than the fp16 baseline it replaces.

And with the PagedAttention KV cache (serving-scale model, realistic generation lengths):

- **Up to 26× less** KV-cache memory and **~97% utilisation vs ~12%** for contiguous
  pre-allocation, by allocating fixed-size blocks on demand instead of booking `max_len`.
- **~7× more concurrent sequences** served in the same KV memory budget.
- **Up to ~5× again** on top when a prompt is shared across sequences (copy-on-write),
  and it returns exactly the same attention output as a contiguous cache (max abs err 0).

Four findings worth more than the headline numbers:

1. The same tiled algorithm **without** kernel fusion is *2.3× slower* than the naive
   baseline while still capturing most of the memory saving — tiling buys memory,
   fusion buys speed. The identical pattern reappears in PagedAttention: the paging
   *mechanism* saves memory, but the unfused gather is ~4× slower until a fused kernel
   removes the copy.
2. The speed-up **saturates** once the kernel hits the hardware's matmul peak, so the
   common claim that it keeps growing with sequence length does not hold here.
3. GQA implemented the obvious way (`repeat_kv` before an ordinary kernel) is **slower
   than the MHA it replaced** — architecture-level optimisations are contingent on
   kernel support, not merely compatible with it.
4. **Most KV memory is lost to management, not computation** — contiguous caches run at
   3–42% utilisation on realistic workloads; PagedAttention recovers it to ~97%.

The three optimisations are orthogonal and compose: a production stack runs GQA inside a
PagedAttention cache computed by a FlashAttention kernel.

> Note on the headline numbers: exact latency/throughput figures are re-measured on the
> target hardware and vary a few percent run-to-run; the figures above are representative
> of the committed `results/` snapshot. `tests/verify_report.py` keeps the report and the
> raw JSON in agreement.
