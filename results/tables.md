# Measured results


All numbers measured on the machine described below. Every table here is generated directly from the benchmark JSON in `results/` — nothing is transcribed by hand.


## Environment


| Item | Value |
| --- | --- |
| gpu | NVIDIA GeForce RTX 5070 |
| compute_capability | sm_120 |
| total_mem_gb | 12.35 |
| sms | 48 |
| torch | 2.11.0+cu128 |
| cuda | 12.8 |
| cudnn | 91900 |
| triton | 3.6.0 |
| driver | 595.71.05 |
| python | 3.12.3 |
| platform | Linux-7.0.0-28-generic-x86_64-with-glibc2.39 |
| sdpa_backends | ['cudnn', 'flash', 'math', 'mem_efficient'] |
| timestamp | 2026-07-26 02:39:30 |


Measured device ceilings: **67.2 TFLOP/s** dense fp16 matmul (N=2048), **561 GB/s** HBM copy bandwidth.



## 1. Headline: standard MHA vs FlashAttention


Configuration: batch 16, embed 768, 12 heads, head_dim 64, fp16, non-causal. Median of 50 timed runs after 10 warm-up iterations.


| Seq. length N | Standard MHA (ms) | FlashAttention (ms) | Speed-up | Peak workspace: standard | Peak workspace: flash | Memory saved | Max abs. difference | Cosine similarity |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 0.044 | 0.024 | **1.82x** | 15.7 MB | 3.2 MB | 79.4% | 9.77e-04 | 0.99999988 |
| 256 | 0.188 | 0.053 | **3.52x** | 56.6 MB | 6.5 MB | 88.5% | 4.88e-04 | 1.00000000 |
| 512 | 1.194 | 0.200 | **5.96x** | 213.9 MB | 13.0 MB | 93.9% | 2.44e-04 | 1.00000000 |
| 1,024 | 4.514 | 0.786 | **5.74x** | 830.5 MB | 26.0 MB | 96.9% | 2.44e-04 | 0.99999994 |
| 2,048 | 19.747 | 3.139 | **6.29x** | 3,271.6 MB | 51.9 MB | 98.4% | 2.44e-04 | 1.00000000 |
| 4,096 | **OOM** | 13.414 | — (baseline OOM) | **OOM** | 103.8 MB | — | — | — |


## 2. All kernels, forward latency (ms)


| Kernel | N=128 | N=256 | N=512 | N=1,024 | N=2,048 | N=4,096 | peak mem @ largest N | TFLOP/s @ largest N | rel. err vs fp64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard MHA (explicit N x N) | 0.044 | 0.188 | 1.194 | 4.514 | 19.747 | **OOM** | 3,271.6 MB | 10.4 | 5.56e-04 |
| PyTorch SDPA - math backend | 0.187 | 0.983 | 4.098 | 15.404 | 59.314 | **OOM** | 7,650.8 MB | 3.5 | 2.08e-04 |
| Flash algorithm (pure PyTorch) | 0.179 | 0.710 | 2.749 | skipped | skipped | skipped | 69.8 MB | 4.7 | 5.74e-04 |
| Memory-efficient (xFormers-style) | 0.031 | 0.105 | 0.401 | 1.594 | 6.754 | 27.008 | 100.7 MB | 30.5 | 2.84e-04 |
| cuDNN fused attention | 0.033 | 0.059 | 0.215 | 0.834 | 3.444 | 13.421 | 100.7 MB | 61.4 | 2.84e-04 |
| FlashAttention (our Triton kernel) | 0.021 | 0.067 | 0.248 | 1.030 | 4.241 | 16.420 | 100.7 MB | 50.2 | 2.84e-04 |
| PyTorch SDPA - auto dispatch | 0.017 | 0.054 | 0.201 | 0.794 | 3.147 | 13.421 | 103.8 MB | 61.4 | 2.84e-04 |
| FlashAttention (SDPA flash) | 0.024 | 0.053 | 0.200 | 0.786 | 3.139 | 13.414 | 103.8 MB | 61.5 | 2.84e-04 |


## 3. Long context (batch 4) — where the baseline stops fitting


| Kernel | N=1,024 | N=2,048 | N=4,096 | N=8,192 | N=16,384 | N=32,768 | peak mem @ largest N | TFLOP/s @ largest N | rel. err vs fp64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard MHA (explicit N x N) | 1.255 | 4.606 | 22.384 | **OOM** | **OOM** | **OOM** | 3,246.4 MB | 9.2 | 5.61e-04 |
| PyTorch SDPA - math backend | 3.548 | 13.797 | 57.896 | **OOM** | **OOM** | **OOM** | 7,449.3 MB | 3.6 | 2.07e-04 |
| Flash algorithm (pure PyTorch) | skipped | skipped | skipped | skipped | skipped | skipped | — | — | — |
| Memory-efficient (xFormers-style) | 0.405 | 1.585 | 6.329 | 26.841 | 107.625 | 429.617 | 201.3 MB | 30.7 | 3.00e-04 |
| cuDNN fused attention | 0.209 | 0.784 | 3.138 | 13.212 | 52.863 | 210.647 | 201.3 MB | 62.6 | 2.92e-04 |
| FlashAttention (our Triton kernel) | 0.245 | 1.001 | 4.148 | 16.253 | 64.826 | 258.084 | 201.3 MB | 51.1 | 2.92e-04 |
| PyTorch SDPA - auto dispatch | 0.200 | 0.778 | 3.128 | 13.383 | 53.520 | 213.440 | 207.6 MB | 61.8 | 2.92e-04 |
| FlashAttention (SDPA flash) | 0.200 | 0.778 | 3.128 | 13.313 | 53.484 | 213.788 | 207.6 MB | 61.7 | 2.92e-04 |


## 4. Causal / decoder-style attention (batch 8)


| Kernel | N=512 | N=1,024 | N=2,048 | N=4,096 | N=8,192 | peak mem @ largest N | TFLOP/s @ largest N | rel. err vs fp64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard MHA (explicit N x N) | 1.080 | 4.028 | 17.418 | 73.018 | **OOM** | 6,509.6 MB | 2.8 | 4.69e-04 |
| PyTorch SDPA - math backend | 2.194 | 9.202 | 37.403 | **OOM** | **OOM** | 3,842.2 MB | 1.4 | 2.01e-04 |
| Flash algorithm (pure PyTorch) | skipped | skipped | skipped | skipped | skipped | — | — | — |
| Memory-efficient (xFormers-style) | 0.134 | 0.466 | 1.839 | 6.599 | 27.482 | 100.7 MB | 30.0 | 2.59e-04 |
| cuDNN fused attention | 0.098 | 0.297 | 0.911 | 3.396 | 13.835 | 100.7 MB | 59.6 | 2.59e-04 |
| FlashAttention (our Triton kernel) | 0.073 | 0.250 | 0.919 | 3.566 | 14.882 | 100.7 MB | 55.4 | 2.59e-04 |
| PyTorch SDPA - auto dispatch | 0.077 | 0.248 | 0.878 | 3.333 | 13.793 | 103.8 MB | 59.8 | 2.59e-04 |
| FlashAttention (SDPA flash) | 0.078 | 0.250 | 0.879 | 3.346 | 13.804 | 103.8 MB | 59.7 | 2.59e-04 |


## 5. Batch-size sensitivity at N=1024


| Kernel | batch 1 | batch 2 | batch 4 | batch 8 | batch 16 | batch 32 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard MHA (explicit N x N) | 0.157 | 0.529 | 1.115 | 2.410 | 4.814 | 9.630 |
| PyTorch SDPA - math backend | 0.831 | 1.713 | 3.855 | 7.160 | 15.420 | 30.878 |
| Flash algorithm (pure PyTorch) | skipped | skipped | skipped | skipped | skipped | skipped |
| Memory-efficient (xFormers-style) | 0.112 | 0.218 | 0.405 | 0.796 | 1.732 | 3.414 |
| cuDNN fused attention | 0.053 | 0.106 | 0.209 | 0.414 | 0.840 | 1.676 |
| FlashAttention (our Triton kernel) | 0.066 | 0.127 | 0.245 | 0.486 | 1.034 | 2.202 |
| PyTorch SDPA - auto dispatch | 0.055 | 0.102 | 0.202 | 0.400 | 0.792 | 1.583 |
| FlashAttention (SDPA flash) | 0.055 | 0.102 | 0.202 | 0.394 | 0.790 | 1.579 |


## 6. Head-dimension sensitivity at N=2048 (embed 768)


| Kernel | head_dim 32 | head_dim 64 | head_dim 128 |
| --- | ---: | ---: | ---: |
| Standard MHA (explicit N x N) | 18.403 | 9.850 | 5.656 |
| PyTorch SDPA - math backend | 55.213 | 29.669 | 20.948 |
| Flash algorithm (pure PyTorch) | skipped | skipped | skipped |
| Memory-efficient (xFormers-style) | 4.858 | 3.189 | 3.910 |
| cuDNN fused attention | 3.183 | 1.597 | 1.700 |
| FlashAttention (our Triton kernel) | 1.764 | 2.002 | 2.713 |
| PyTorch SDPA - auto dispatch | 1.644 | 1.717 | 1.705 |
| FlashAttention (SDPA flash) | 1.643 | 1.571 | 1.704 |


## 7. bfloat16


| Kernel | N=512 | N=1,024 | N=2,048 | peak mem @ largest N | TFLOP/s @ largest N | rel. err vs fp64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Standard MHA (explicit N x N) | 1.258 | 4.698 | 20.563 | 3,271.6 MB | 10.0 | 4.45e-03 |
| PyTorch SDPA - math backend | 4.099 | 15.406 | 59.345 | 7,650.8 MB | 3.5 | 1.66e-03 |
| Flash algorithm (pure PyTorch) | skipped | skipped | skipped | — | — | — |
| Memory-efficient (xFormers-style) | 0.404 | 1.586 | 6.782 | 50.3 MB | 30.4 | 2.25e-03 |
| cuDNN fused attention | 0.217 | 0.834 | 3.210 | 50.3 MB | 64.2 | 2.25e-03 |
| FlashAttention (our Triton kernel) | 0.250 | 1.027 | 3.977 | 50.3 MB | 51.8 | 2.25e-03 |
| PyTorch SDPA - auto dispatch | 0.200 | 0.790 | 3.429 | 51.9 MB | 60.1 | 2.25e-03 |
| FlashAttention (SDPA flash) | 0.201 | 0.790 | 3.147 | 51.9 MB | 65.5 | 2.25e-03 |


## 8. Complete MHA layer (Q/K/V/O projections included)


| Kernel | N=128 | N=256 | N=512 | N=1,024 | N=2,048 | N=4,096 | peak mem @ largest N | TFLOP/s @ largest N | rel. err vs fp64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch SDPA - math backend | 0.231 | 0.544 | 2.260 | 8.124 | 31.890 | **OOM** | 3,900.9 MB | 3.2 | — |
| Memory-efficient (xFormers-style) | 0.143 | 0.209 | 0.509 | 1.546 | 4.398 | 15.117 | 251.7 MB | 27.3 | — |
| cuDNN fused attention | 0.142 | 0.185 | 0.406 | 1.117 | 2.819 | 9.297 | 251.7 MB | 44.4 | — |
| FlashAttention (our Triton kernel) | 0.158 | 0.222 | 0.489 | 1.278 | 3.894 | 11.720 | 352.3 MB | 35.2 | — |
| PyTorch SDPA - auto dispatch | 0.128 | 0.181 | 0.399 | 1.022 | 2.785 | 9.307 | 251.7 MB | 44.3 | — |
| FlashAttention (SDPA flash) | 0.131 | 0.181 | 0.399 | 0.993 | 2.938 | 8.711 | 251.7 MB | 47.3 | — |


## 9. Training step (forward + backward), causal, batch 4


| Kernel | N=512 | N=1,024 | N=2,048 | N=4,096 | N=8,192 | peak mem @ largest N | TFLOP/s @ largest N | rel. err vs fp64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch SDPA - math backend | 2.022 | 8.830 | 35.022 | **OOM** | **OOM** | 3,347.1 MB | 2.6 | — |
| Memory-efficient (xFormers-style) | 0.369 | 1.140 | 4.252 | 15.796 | 61.829 | 506.6 MB | 23.3 | — |
| FlashAttention (SDPA flash) | 0.211 | 0.557 | 1.802 | 6.489 | 25.746 | 556.8 MB | 56.1 | — |


## 10. Autoregressive decode with a KV cache (batch 32, embed 1024, 16 heads)


| Layout | KV heads | KV cache @ 16k ctx | params | tok/s @ 512 | tok/s @ 2,048 | tok/s @ 8,192 | tok/s @ 16,384 | speed-up vs MHA @ longest | same, with repeat_kv |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MHA | 16 | 2,147.5 MB | 4.20 M | 64,016 | 20,039 | 4,886 | 2,461 | **1.00x** | 1.00x |
| GQA-8 | 8 | 1,073.7 MB | 3.15 M | 124,054 | 39,270 | 10,244 | 4,804 | **1.95x** | 0.85x |
| GQA-4 | 4 | 536.9 MB | 2.62 M | 239,006 | 74,000 | 20,378 | 9,712 | **3.95x** | 1.06x |
| GQA-2 | 2 | 268.4 MB | 2.36 M | 324,834 | 152,894 | 44,503 | 22,774 | **9.25x** | 1.21x |
| MQA | 1 | 134.2 MB | 2.23 M | 316,256 | 322,009 | 85,273 | 44,479 | **18.07x** | 2.68x |


## 11. Architecture-level: models trained from scratch


| Layout | KV heads | Params | KV cache / token | Final val loss | Val perplexity | Δ loss vs MHA | Train tokens/s | Peak train mem |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MHA | 8 | 10.75 M | 9,216 B | 1.4878 | 4.427 | +0.0000 | 486,666 | 1,043 MB |
| GQA-4 | 4 | 9.87 M | 4,608 B | 1.4979 | 4.472 | +0.0101 | 520,074 | 1,117 MB |
| GQA-2 | 2 | 9.42 M | 2,304 B | 1.4934 | 4.452 | +0.0056 | 532,994 | 1,104 MB |
| MQA | 1 | 9.20 M | 1,152 B | 1.4911 | 4.442 | +0.0033 | 553,778 | 1,034 MB |


## 12. Kernel-level: one trained checkpoint, every kernel


| Kernel | Val loss (same weights) | Val perplexity | Δ vs first kernel | Forward (ms) | Relative speed | Peak memory |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch SDPA - auto dispatch | 1.487835 | 4.4275 | -1.10e-05 | 4.828 | 1.00x | 110.9 MB |
| FlashAttention (SDPA flash) | 1.487835 | 4.4275 | -1.10e-05 | 4.849 | 1.00x | 110.9 MB |
| cuDNN fused attention | 1.487811 | 4.4274 | -3.41e-05 | 4.855 | 1.01x | 110.9 MB |
| Memory-efficient (xFormers-style) | 1.487821 | 4.4274 | -2.49e-05 | 5.016 | 1.04x | 110.9 MB |
| Flash algorithm (pure PyTorch) | 1.487990 | 4.4282 | +1.45e-04 | 10.231 | 2.12x | 145.9 MB |
| Standard MHA (explicit N x N) | 1.488055 | 4.4285 | +2.09e-04 | 10.629 | 2.20x | 233.2 MB |
| PyTorch SDPA - math backend | 1.487846 | 4.4275 | +0.00e+00 | 14.115 | 2.92x | 266.4 MB |


## 13. Seed-variance study — is the layout ranking real?


| Layout | KV cache / token | Mean val loss | Std dev | Min | Max | seed 1337 | seed 7 | seed 2024 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| MHA | 9,216 B | 1.4943 | 0.0088 | 1.4892 | 1.5045 | 1.4892 | 1.4893 | 1.5045 |
| GQA-4 | 4,608 B | 1.4937 | 0.0049 | 1.4893 | 1.4990 | 1.4990 | 1.4927 | 1.4893 |
| MQA | 1,152 B | 1.4930 | 0.0047 | 1.4902 | 1.4985 | 1.4902 | 1.4903 | 1.4985 |


Spread **between** layouts (max mean − min mean): **0.0014** nats. Average **within**-layout seed standard deviation: **0.0061** nats. Ratio **0.22×**.



## 14. PagedAttention — KV-cache fragmentation (model: 40 layers, 8 KV heads, head_dim 128, 160 KB/token)


| Mean gen length | Contiguous cache | Paged cache | Ideal | Contiguous util | Paged util | Memory saved | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 85.90 GB | 3.24 GB | 2.93 GB | 3.4% | 90.3% | 96.2% | 26.5× |
| 128 | 85.90 GB | 5.86 GB | 5.54 GB | 6.5% | 94.5% | 93.2% | 14.6× |
| 256 | 85.90 GB | 11.09 GB | 10.77 GB | 12.5% | 97.2% | 87.1% | 7.7× |
| 512 | 85.90 GB | 21.02 GB | 20.71 GB | 24.1% | 98.5% | 75.5% | 4.1× |
| 1,024 | 85.90 GB | 36.46 GB | 36.18 GB | 42.1% | 99.2% | 57.6% | 2.4× |


## 15. PagedAttention — concurrent sequences in a fixed KV budget (mean gen 256, max_len 2048)


| KV budget | Contiguous seqs | Paged seqs | More |
| --- | ---: | ---: | ---: |
| 2 GB | 5 | 36 | 7.2× |
| 4 GB | 11 | 87 | 7.9× |
| 8 GB | 23 | 165 | 7.2× |
| 16 GB | 47 | 342 | 7.3× |
| 32 GB | 95 | 685 | 7.2× |
| 64 GB | 190 | 1,410 | 7.4× |


## 16. PagedAttention — prefix sharing (shared 512-token prompt + 128 generated tokens each)


| Sequences | Contiguous | Paged (no sharing) | Paged (shared prompt) | Sharing saves | vs contiguous |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.105 GB | 0.105 GB | 0.105 GB | 0% | 1.0× |
| 2 | 0.210 GB | 0.210 GB | 0.126 GB | 40% | 1.7× |
| 4 | 0.419 GB | 0.419 GB | 0.168 GB | 60% | 2.5× |
| 8 | 0.839 GB | 0.839 GB | 0.252 GB | 70% | 3.3× |
| 16 | 1.678 GB | 1.678 GB | 0.419 GB | 75% | 4.0× |
| 32 | 3.355 GB | 3.355 GB | 0.755 GB | 78% | 4.4× |
| 64 | 6.711 GB | 6.711 GB | 1.426 GB | 79% | 4.7× |
| 128 | 13.422 GB | 13.422 GB | 2.768 GB | 79% | 4.8× |
| 256 | 26.844 GB | 26.844 GB | 5.453 GB | 80% | 4.9× |


## 17. PagedAttention — unfused gather latency overhead (batch 64, equal length)


| Context length | Contiguous + SDPA | Paged gather + SDPA | Overhead |
| --- | ---: | ---: | ---: |
| 128 | 0.145 ms | 0.340 ms | +135% |
| 512 | 0.307 ms | 1.288 ms | +320% |
| 1,024 | 0.587 ms | 2.558 ms | +335% |
| 2,048 | 1.154 ms | 5.110 ms | +343% |
| 4,096 | 2.288 ms | 10.227 ms | +347% |
