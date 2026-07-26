"""
End-to-end training experiment: does the optimisation change the model?

Two distinct questions are answered here, and they have different answers.

Q1 (kernel-level, FlashAttention): if we swap the attention *kernel*, do we get a
    different model? Answer: no. We train one model, then evaluate the identical
    weights under every kernel and compare loss to 6 decimal places. This is what
    "exact, drop-in, no retraining" actually means, verified rather than asserted.

Q2 (architecture-level, GQA/MQA): if we shrink the number of KV heads, do we get a
    different model? Answer: yes - so we have to train each layout from scratch and
    measure the quality cost against the memory/latency benefit.

Model: a small character-level GPT on Tiny Shakespeare. Small enough to train several
variants in minutes, large enough for the loss differences to be meaningful.

Usage:  python train_gpt.py [--steps 3000] [--out ../results]
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import statistics
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from attention import MultiHeadSelfAttention, sdpa_attention  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class CharData:
    def __init__(self, path: str, split: float = 0.9):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        self.chars = sorted(set(text))
        self.vocab_size = len(self.chars)
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        ids = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)
        n = int(split * len(ids))
        self.train, self.val = ids[:n], ids[n:]
        print(f"  data: {len(ids):,} chars, vocab {self.vocab_size}, "
              f"train {len(self.train):,} / val {len(self.val):,}")

    def batch(self, split: str, batch_size: int, block: int, generator=None):
        src = self.train if split == "train" else self.val
        ix = torch.randint(len(src) - block - 1, (batch_size,), generator=generator)
        x = torch.stack([src[i : i + block] for i in ix])
        y = torch.stack([src[i + 1 : i + 1 + block] for i in ix])
        return x.to(DEV, non_blocking=True), y.to(DEV, non_blocking=True)

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    vocab_size: int = 65
    block_size: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_kv_head: int = 8
    n_embd: int = 384
    dropout: float = 0.1
    kernel: str = "flash"


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = MultiHeadSelfAttention(cfg.n_embd, cfg.n_head,
                                           n_kv_heads=cfg.n_kv_head,
                                           kernel=cfg.kernel, causal=True, bias=False)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=False),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=False),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, kernel=None):
        x = x + self.attn(self.ln1(x), kernel=kernel)
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight            # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx, targets=None, kernel=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x, kernel=kernel)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 0.8,
                 top_k: int = 40, kernel=None, generator=None):
        for _ in range(max_new_tokens):
            idx_c = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_c, kernel=kernel)
            logits = logits[:, -1, :] / temperature
            if top_k:
                vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < vals[:, [-1]]] = -float("inf")
            probs = F.softmax(logits.float(), dim=-1)
            nxt = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def kv_cache_bytes_per_token(self, dtype_size=2) -> int:
        c = self.cfg
        return (2 * c.n_layer * c.n_kv_head * (c.n_embd // c.n_head) * dtype_size)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, data, batch_size, block, iters=40, kernel=None, seed=1234):
    model.eval()
    g = torch.Generator().manual_seed(seed)
    out = {}
    for split in ("train", "val"):
        losses = []
        for _ in range(iters):
            x, y = data.batch(split, batch_size, block, generator=g)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, loss = model(x, y, kernel=kernel)
            losses.append(loss.item())
        out[split] = statistics.fmean(losses)
    model.train()
    return out


def train_one(name: str, cfg: GPTConfig, data: CharData, steps: int, batch_size: int,
              lr: float = 3e-4, eval_every: int = 250, eval_iters: int = 40,
              seed: int = 1337, log=print) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gc.collect(); torch.cuda.empty_cache()

    model = GPT(cfg).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    g = torch.Generator().manual_seed(seed)

    torch.cuda.reset_peak_memory_stats()
    hist = {"step": [], "train_loss": [], "val_loss": [], "wall_s": []}
    t0 = time.perf_counter()
    step_times = []

    for step in range(1, steps + 1):
        x, y = data.batch("train", batch_size, cfg.block_size, generator=g)
        torch.cuda.synchronize()
        ts = time.perf_counter()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        torch.cuda.synchronize()
        if step > 20:                     # skip warm-up steps in the timing statistic
            step_times.append(time.perf_counter() - ts)

        if step % eval_every == 0 or step == 1 or step == steps:
            ev = evaluate(model, data, batch_size, cfg.block_size, iters=eval_iters)
            hist["step"].append(step)
            hist["train_loss"].append(ev["train"])
            hist["val_loss"].append(ev["val"])
            hist["wall_s"].append(time.perf_counter() - t0)
            log(f"    [{name}] step {step:5d}/{steps}  train {ev['train']:.4f}  "
                f"val {ev['val']:.4f}  ({time.perf_counter()-t0:6.1f}s)")

    total_s = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1e6
    final = evaluate(model, data, batch_size, cfg.block_size, iters=100)
    med_step = statistics.median(step_times)
    tok_per_s = batch_size * cfg.block_size / med_step

    return {
        "name": name,
        "config": cfg.__dict__.copy(),
        "params_m": model.n_params() / 1e6,
        "kv_cache_bytes_per_token": model.kv_cache_bytes_per_token(),
        "final_train_loss": final["train"],
        "final_val_loss": final["val"],
        "final_val_ppl": math.exp(final["val"]),
        "median_step_ms": med_step * 1e3,
        "tokens_per_s": tok_per_s,
        "total_train_s": total_s,
        "peak_mem_mb": peak_mb,
        "history": hist,
        "_model": model,
    }


# ---------------------------------------------------------------------------
# Q1: kernel swap on a trained checkpoint must be a no-op
# ---------------------------------------------------------------------------

def kernel_equivalence_study(model, data, batch_size, block, kernels, log=print) -> dict:
    """Evaluate identical weights under every kernel; also time generation."""
    log("\n  Evaluating one trained checkpoint under each attention kernel:")
    rows = []
    base_val = None
    for kern in kernels:
        try:
            gc.collect(); torch.cuda.empty_cache()
            ev = evaluate(model, data, batch_size, block, iters=100, kernel=kern)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base_alloc = torch.cuda.memory_allocated()
            # timed forward pass
            g = torch.Generator().manual_seed(7)
            x, y = data.batch("val", batch_size, block, generator=g)
            for _ in range(5):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x, y, kernel=kern)
            torch.cuda.synchronize()
            ts = time.perf_counter()
            for _ in range(30):
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    model(x, y, kernel=kern)
            torch.cuda.synchronize()
            fwd_ms = (time.perf_counter() - ts) / 30 * 1e3
            peak = (torch.cuda.max_memory_allocated() - base_alloc) / 1e6
            if base_val is None:
                base_val = ev["val"]
            rows.append({"kernel": kern, "val_loss": ev["val"],
                         "val_ppl": math.exp(ev["val"]),
                         "delta_vs_first": ev["val"] - base_val,
                         "fwd_ms": fwd_ms, "peak_mem_mb": peak, "status": "ok"})
            log(f"    {kern:14s} val loss {ev['val']:.6f}  ppl {math.exp(ev['val']):7.4f}"
                f"   fwd {fwd_ms:7.3f} ms   peak {peak:8.1f} MB")
        except Exception as exc:  # noqa: BLE001
            rows.append({"kernel": kern, "status": "error",
                         "error": f"{type(exc).__name__}: {exc}"[:200]})
            log(f"    {kern:14s} ERROR {type(exc).__name__}")
    return rows


def sample_text(model, data, kernel="flash", n_tokens=400, seed=0) -> str:
    g = torch.Generator(device=DEV).manual_seed(seed)
    idx = torch.zeros((1, 1), dtype=torch.long, device=DEV)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model.generate(idx, n_tokens, kernel=kernel, generator=g)
    return data.decode(out[0].tolist())


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..",
                                                   "data", "tinyshakespeare.txt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "results"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log_path = os.path.join(args.out, "training_log.txt")
    logf = open(log_path, "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log("=" * 78)
    log("TRAINING STUDY - kernel swap vs architecture change")
    log("=" * 78)
    log(f"  device {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    data = CharData(args.data)

    base = dict(vocab_size=data.vocab_size, block_size=args.block, n_layer=6,
                n_head=8, n_embd=384, dropout=0.1)

    # ---------------- Q2: architecture variants, trained from scratch -------
    log("\n" + "-" * 78)
    log("PART 1 - architecture-level: train MHA / GQA / MQA from scratch")
    log("        (these change what is computed, so each needs its own training run)")
    log("-" * 78)
    layouts = [("MHA", 8), ("GQA-4", 4), ("GQA-2", 2), ("MQA", 1)]
    arch_results = []
    trained_mha = None
    for label, n_kv in layouts:
        cfg = GPTConfig(**base, n_kv_head=n_kv, kernel="flash")
        log(f"\n  --- {label}  (n_kv_head={n_kv}) ---")
        r = train_one(label, cfg, data, args.steps, args.batch_size, log=log)
        model = r.pop("_model")
        r["sample"] = sample_text(model, data)
        log(f"    -> val loss {r['final_val_loss']:.4f}  ppl {r['final_val_ppl']:.3f}  "
            f"params {r['params_m']:.2f}M  "
            f"KV/token {r['kv_cache_bytes_per_token']}B  "
            f"{r['tokens_per_s']:,.0f} tok/s  peak {r['peak_mem_mb']:.0f} MB")
        if label == "MHA":
            trained_mha = model
        else:
            del model
        arch_results.append(r)
        gc.collect(); torch.cuda.empty_cache()

    # ---------------- Q1: kernel equivalence on the trained MHA -------------
    log("\n" + "-" * 78)
    log("PART 2 - kernel-level: same trained MHA weights, different attention kernels")
    log("        (these change only how it is computed - the model must not change)")
    log("-" * 78)
    kernels = ["math", "flash", "mem_efficient", "cudnn", "auto", "triton_flash",
               "naive", "tiled_flash"]
    kern_rows = kernel_equivalence_study(trained_mha, data, args.batch_size,
                                         args.block, kernels, log=log)

    ok = [r for r in kern_rows if r.get("status") == "ok"]
    if ok:
        spread = max(r["val_loss"] for r in ok) - min(r["val_loss"] for r in ok)
        log(f"\n    val-loss spread across all kernels: {spread:.2e} nats "
            f"({100*spread/ok[0]['val_loss']:.4f}% of the loss)")
        fastest = min(ok, key=lambda r: r["fwd_ms"])
        slowest = max(ok, key=lambda r: r["fwd_ms"])
        log(f"    fastest kernel {fastest['kernel']} {fastest['fwd_ms']:.3f} ms vs "
            f"slowest {slowest['kernel']} {slowest['fwd_ms']:.3f} ms "
            f"-> {slowest['fwd_ms']/fastest['fwd_ms']:.2f}x")

    log("\n  Sample generation from the trained MHA model (flash kernel):")
    log("  " + "-" * 60)
    for line in sample_text(trained_mha, data, n_tokens=300).split("\n"):
        log("  | " + line)
    log("  " + "-" * 60)

    payload = {
        "architecture_variants": arch_results,
        "kernel_equivalence": kern_rows,
        "train_config": {"steps": args.steps, "batch_size": args.batch_size,
                         "block": args.block, **base},
    }
    with open(os.path.join(args.out, "training_results.json"), "w") as f:
        json.dump(payload, f, indent=2)
    log(f"\n  wrote {os.path.join(args.out, 'training_results.json')}")
    logf.close()


if __name__ == "__main__":
    main()
