#!/usr/bin/env python3
"""Phase 1b: knowledge-distillation student — fly topology vs controls, soft targets.

Student = flylm.py's LIF connectome LM (frozen graph + adapters). Teacher =
teacher.py's byte transformer. Loss per byte:
    L = (1-alpha) * CE(student, true byte) + alpha * T^2 * CE(teacher_probs_T, student)
where teacher_probs_T = softmax of the teacher's next-byte distribution over the
preceding ctx bytes, tempered by T (Hinton-style). The teacher qualifies every
answer: which alternatives were plausible, which it rules out.

40M train bytes/condition (2x Phase 1). Same val protocol (hard-CE bpb) so
numbers are directly comparable to Phase 0/1. ckpt auto-resume + config guard.
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flylm import (FlyLM, degree_preserving_shuffle, erdos_renyi, load_graph,
                   to_sparse_wt)

BUF_ROWS = 65536          # target positions of probs held in RAM (~32MB fp16)


def find_probs(train_bytes, teacher_ckpt):
    """Locate a valid probs cache covering train_bytes, teacher-mtime matched."""
    if not os.path.exists(teacher_ckpt):
        return None, None
    mt = os.path.getmtime(teacher_ckpt)
    for f in sorted(os.listdir(".")):
        if f.startswith("probs_") and f.endswith(".npy"):
            try:
                m = json.load(open(f + ".meta.json"))
            except Exception:
                continue
            if (abs(m.get("teacher_mtime", -1) - mt) < 2
                    and m.get("train_bytes", 0) >= train_bytes
                    and m.get("rows", 0) >= train_bytes - m.get("ctx", 512)):
                return f, m
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["fly", "shuffled", "er", "fly-uniform"])
    ap.add_argument("--teacher-ckpt", default="ckpt_teacher.pt")
    ap.add_argument("--kd-alpha", type=float, default=0.5)
    ap.add_argument("--kd-temp", type=float, default=2.0)
    ap.add_argument("--train-bytes", type=int, default=40_000_000)
    ap.add_argument("--val-bytes", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="tinystories_slice.txt")
    ap.add_argument("--graph", default="larval_graph.npz")
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--tag", default="kd40")
    ap.add_argument("--out", default=None)
    ap.add_argument("--ckpt-every", type=int, default=5000)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    data = np.fromfile(args.data, dtype=np.uint8)

    pfile, pmeta = find_probs(args.train_bytes, args.teacher_ckpt)
    if pfile is None:
        print(f"no valid probs cache for {args.train_bytes}B + teacher mtime; "
              f"run probs_cache.py first", flush=True)
        sys.exit(2)
    ctx = pmeta.get("ctx", 512)
    probs_mm = np.load(pfile, mmap_mode="r")
    print(f"teacher val_bpb={pmeta.get('teacher_val_bpb', -1)} ctx={ctx} | "
          f"probs cache: {pfile} rows={probs_mm.shape[0]:,} | "
          f"data {len(data):,} bytes", flush=True)

    u, v, w, N, sensory, output = load_graph(args.graph)
    if args.condition == "fly":
        uu, vv = u.astype(np.int64), v.astype(np.int64)
        ww = w[: len(uu)]
    elif args.condition == "fly-uniform":
        uu, vv = u.astype(np.int64), v.astype(np.int64)
        ww = np.ones(len(uu), dtype=np.float32)
    elif args.condition == "shuffled":
        uu, vv = degree_preserving_shuffle(u, v, N, seed=args.seed)
        ww = w[np.random.default_rng(args.seed).permutation(len(w))][: len(uu)]
    else:
        uu, vv = erdos_renyi(u, v, N, seed=args.seed)
        ww = w[np.random.default_rng(args.seed).permutation(len(w))][: len(uu)]
    WT = to_sparse_wt(uu, vv, ww, N, device)
    model = FlyLM(N, sensory, output, WT, T=args.T, width=args.width).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    label = f"{args.tag}_{args.condition}" if args.tag else args.condition
    print(f"{label}: N={N} edges={len(uu):,} trainable={n_params:,} "
          f"T={args.T} width={args.width} alpha={args.kd_alpha}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.train_bytes // args.batch

    # ---- teacher probs: RAM buffer over the disk cache ----
    # COORDINATE SYSTEMS: row r of the cache = distribution for TARGET p = r+ctx.
    # All buffer math below is in ROW space; k0 arrives as a target position.
    probs_buf = {"start": -1, "t": None}

    def ensure_buf(k0):
        row0 = k0 - ctx                      # row of the first target
        if (probs_buf["t"] is not None and probs_buf["start"] <= row0
                and row0 + args.batch <= probs_buf["start"] + probs_buf["t"].shape[0]):
            return
        start = (row0 // BUF_ROWS) * BUF_ROWS
        hi = min(start + BUF_ROWS, probs_mm.shape[0])
        blk = np.asarray(probs_mm[start:hi], dtype=np.float16)
        if (blk < 0).any():                  # sentinel = missing coverage
            print(f"CORRUPT probs cache rows [{start:,}, {hi:,}) (sentinel -1); "
                  f"aborting", flush=True)
            sys.exit(3)
        probs_buf["t"] = torch.from_numpy(blk)
        probs_buf["start"] = start
        print(f"  [probs] buffer rows [{start:,}, {hi:,}) "
              f"(targets [{start + ctx:,}, {hi + ctx:,}))", flush=True)

    # ---- checkpointing ----
    ckpt_path = f"ckpt_{label}_s{args.seed}.pt"
    start_step, running_h, running_k, t0 = 0, 0.0, 0.0, time.time()
    log = []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sa = ck.get("args", {})
        mism = [k for k in ("T", "train_bytes", "batch", "lr", "width",
                            "kd_alpha", "kd_temp")
                if sa.get(k) != getattr(args, k)]
        if mism:
            os.replace(ckpt_path, ckpt_path + ".stale")
            print(f"args mismatch {mism} -> stale; fresh start", flush=True)
        else:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_step = ck["step"]; running_h = ck.get("running_h", 0.0)
            running_k = ck.get("running_k", 0.0); log = ck.get("log", [])
            print(f"resumed {label} seed {args.seed} at step {start_step}", flush=True)
    steps = args.train_bytes // args.batch

    tau, alpha = args.kd_temp, args.kd_alpha
    for step in range(start_step, steps):
        k0 = (step * args.batch) % (args.train_bytes - ctx - args.batch) + ctx
        ensure_buf(int(k0))
        rel = int(k0) - ctx - probs_buf["start"]
        rows = probs_buf["t"][rel:rel + args.batch].to(device).float()  # [B,256]
        rows = rows ** (1.0 / tau)
        p_t = rows / rows.sum(-1, keepdim=True)                     # tempered
        cur = torch.as_tensor(data[int(k0) - 1 + np.arange(args.batch)],
                              dtype=torch.long, device=device)
        nxt = torch.as_tensor(data[int(k0) + np.arange(args.batch)],
                              dtype=torch.long, device=device)
        if step == start_step:
            state = model.init_state(args.batch, device)
        logits, state2 = model(cur, state)
        l_hard = F.cross_entropy(logits, nxt)
        l_kd = -(p_t * F.log_softmax(logits / tau, -1)).sum(-1).mean() * (tau * tau)
        loss = (1 - alpha) * l_hard + alpha * l_kd
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state = state2.detach()
        running_h += l_hard.item(); running_k += l_kd.item()
        if (step + 1) % 500 == 0:
            bh = running_h / 500 / math.log(2)
            bk = running_k / 500 / math.log(2)
            log.append({"step": step + 1, "hard_bpb": round(bh, 4),
                        "kd_term": round(bk, 4)})
            print(f"  step {step+1}/{steps} hard_bpb={bh:.4f} kd={bk:.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            running_h = running_k = 0.0
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "running_h": running_h,
                        "running_k": running_k, "log": log,
                        "args": vars(args)}, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path} @ step {step+1}", flush=True)

    # validation: hard-CE bpb, same protocol as flylm.py (comparable numbers)
    v0 = model.init_state(args.batch, device)
    ce, n = 0.0, 0
    vb = args.val_bytes
    offs = np.arange(args.batch) + args.train_bytes
    for k in range(vb - 1):
        cur = torch.as_tensor(data[offs + k], dtype=torch.long, device=device)
        nxt = torch.as_tensor(data[offs + k + 1], dtype=torch.long, device=device)
        with torch.no_grad():
            logits, v0 = model(cur, v0)
        ce += F.cross_entropy(logits, nxt).item(); n += 1
    val_bpb = ce / n / math.log(2)
    dt = time.time() - t0
    print(f"RESULT {label}: val_bpb={val_bpb:.4f} time={dt:.0f}s steps={steps}", flush=True)
    out = args.out or f"result_{label}.json"
    if os.path.exists(out):
        prev = json.load(open(out))
        prev.setdefault("runs", []).append({"seed": args.seed, "val_bpb": val_bpb,
                                            "time_s": dt, "train_bytes": args.train_bytes,
                                            "val_bytes": vb, "T": args.T,
                                            "width": args.width, "tag": args.tag,
                                            "kd_alpha": args.kd_alpha})
        json.dump(prev, open(out, "w"), indent=2)
    else:
        json.dump({"condition": args.condition, "seed": args.seed, "N": N,
                   "edges": int(len(uu)), "trainable_params": n_params,
                   "T": args.T, "width": args.width, "tag": args.tag,
                   "kd_alpha": args.kd_alpha, "kd_temp": args.kd_temp,
                   "batch": args.batch, "train_bytes": args.train_bytes,
                   "val_bytes": vb, "steps": steps, "val_bpb": val_bpb,
                   "time_s": dt, "curve": log, "lr": args.lr,
                   "device": device}, open(out, "w"), indent=2)
    print("saved", out, flush=True)


if __name__ == "__main__":
    main()