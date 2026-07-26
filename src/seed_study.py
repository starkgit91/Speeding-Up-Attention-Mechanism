"""
How big is run-to-run noise in the training study?

The architecture sweep in train_gpt.py produced a non-monotonic ordering (GQA-4 came
out worse than GQA-2 and MQA), which is a warning sign: at this model scale the
KV-head count may simply not be the binding constraint, and the apparent differences
may be seed noise rather than signal.

This script measures the noise floor directly - the same MHA and MQA configurations
trained under several seeds - so the report can state whether the observed spread
between layouts is resolvable at all. Answering "we cannot distinguish these" is a
result; pretending to a quality ranking that the data does not support is not.

Usage:  python seed_study.py [--seeds 1337 7 2024] [--steps 3000]
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
from train_gpt import CharData, GPTConfig, train_one  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1337, 7, 2024])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--layouts", nargs="+", default=["MHA", "GQA-4", "MQA"])
    ap.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "..",
                                                   "data", "tinyshakespeare.txt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..",
                                                  "results"))
    args = ap.parse_args()

    log_path = os.path.join(args.out, "seed_study_log.txt")
    logf = open(log_path, "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    log("=" * 78)
    log("SEED STUDY - is the MHA/GQA/MQA quality gap above the noise floor?")
    log("=" * 78)
    data = CharData(args.data)
    base = dict(vocab_size=data.vocab_size, block_size=args.block, n_layer=6,
                n_head=8, n_embd=384, dropout=0.1)
    kv_of = {"MHA": 8, "GQA-4": 4, "GQA-2": 2, "MQA": 1}

    results = {}
    for layout in args.layouts:
        results[layout] = []
        for seed in args.seeds:
            cfg = GPTConfig(**base, n_kv_head=kv_of[layout], kernel="flash")
            log(f"\n  --- {layout}, seed {seed} ---")
            r = train_one(f"{layout}/s{seed}", cfg, data, args.steps, args.batch_size,
                          seed=seed, eval_every=args.steps, log=log)
            r.pop("_model")
            r["seed"] = seed
            r["layout"] = layout
            results[layout].append(r)
            log(f"    -> val loss {r['final_val_loss']:.4f}  "
                f"ppl {r['final_val_ppl']:.3f}")
            gc.collect(); torch.cuda.empty_cache()

    log("\n" + "=" * 78)
    log("SUMMARY")
    log("=" * 78)
    summary = {}
    for layout, rs in results.items():
        losses = [r["final_val_loss"] for r in rs]
        m = statistics.fmean(losses)
        sd = statistics.stdev(losses) if len(losses) > 1 else 0.0
        summary[layout] = {"mean": m, "std": sd, "min": min(losses), "max": max(losses),
                           "n": len(losses), "losses": losses,
                           "kv_bytes_per_token": rs[0]["kv_cache_bytes_per_token"]}
        log(f"  {layout:7s} val loss {m:.4f} +/- {sd:.4f}  "
            f"(range {min(losses):.4f}-{max(losses):.4f}, n={len(losses)})  "
            f"seeds: {['%.4f' % x for x in losses]}")

    if len(summary) > 1:
        means = [v["mean"] for v in summary.values()]
        between = max(means) - min(means)
        within = statistics.fmean([v["std"] for v in summary.values()])
        log(f"\n  spread BETWEEN layouts (max mean - min mean): {between:.4f} nats")
        log(f"  average WITHIN-layout seed std:                 {within:.4f} nats")
        if within > 0:
            log(f"  ratio between/within: {between/within:.2f}")
            verdict = ("the layouts are NOT distinguishable at this scale - the "
                       "between-layout spread is within seed noise"
                       if between < 2 * within else
                       "the between-layout difference exceeds the seed noise")
            log(f"  verdict: {verdict}")

    with open(os.path.join(args.out, "seed_study.json"), "w") as f:
        json.dump({"summary": summary, "runs": results,
                   "seeds": args.seeds, "steps": args.steps}, f, indent=2)
    log(f"\n  wrote {os.path.join(args.out, 'seed_study.json')}")
    logf.close()


if __name__ == "__main__":
    main()
