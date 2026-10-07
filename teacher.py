#!/usr/bin/env python3
"""Teacher for Phase 1b: byte-level nano-transformer on the same TinyStories slice.

Same data conventions as flylm.py (bpb = CE/ln2, val region after train region).
Trainable everything (it is the quality reference, not the fly). ~2.5M params.
Checkpoint auto-resume, Windows-reboot safe (same pattern as flylm.py).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.h = h

    def forward(self, x, ca):
        a = self.ln1(x)
        z, _ = self.attn(a, a, a, attn_mask=ca, need_weights=False)
        x = x + z
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    def __init__(self, d=256, layers=4, heads=8, ctx=512):
        super().__init__()
        self.ctx = ctx
        self.emb = nn.Embedding(256, d)
        self.pos = nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, 256, bias=False)
        self.head.weight = self.emb.weight  # weight tying

    def forward(self, idx):
        B, L = idx.shape
        x = self.emb(idx) + self.pos(torch.arange(L, device=idx.device))
        ca = torch.triu(torch.full((L, L), float("-inf"), device=idx.device), 1)
        for b in self.blocks:
            x = b(x, ca)
        return self.head(self.lnf(x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bytes", type=int, default=40_000_000)
    ap.add_argument("--val-bytes", type=int, default=150_000)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="tinystories_slice.txt")
    ap.add_argument("--ckpt", default="ckpt_teacher.pt")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    args = ap.parse_args()

    torch.manual_seed(0); np.random.seed(0)
    device = args.device
    data = np.fromfile(args.data, dtype=np.uint8)
    print(f"data bytes: {len(data):,} | device: {device}", flush=True)

    model = TinyTransformer().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"teacher: trainable={n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    # steps = byte-windows of (ctx+1) consumed sequentially
    win = args.ctx + 1
    total_windows = args.train_bytes // win

    ckpt_path = args.ckpt
    start_w, log = 0, []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        if ck.get("args", {}).get("train_bytes") == args.train_bytes:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_w = ck["window"]; log = ck.get("log", [])
            print(f"resumed teacher at window {start_w}", flush=True)
        else:
            os.replace(ckpt_path, ckpt_path + ".stale")
            print("teacher args mismatch -> stale; fresh start", flush=True)

    t0 = time.time()
    running, nb = 0.0, 0
    for w in range(start_w, total_windows):
        k0 = w * win % (len(data) - win - 1)   # wrap around the slice
        block = torch.as_tensor(data[k0:k0 + win].astype(np.int64), device=device)
        # 512-token window (matches pos table): outputs 0..511 predict
        # targets s+1..s+512 — the same conditioning the probs cache uses.
        xb = block[:-1].unsqueeze(0)
        yb = block[1:].unsqueeze(0)
        logits = model(xb)
        loss = F.cross_entropy(logits.reshape(-1, 256), yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        running += loss.item(); nb += 1
        # cosine decay to 10% by end
        for g in opt.param_groups:
            g["lr"] = args.lr * (0.1 + 0.45 * (1 + math.cos(math.pi * w / total_windows)))
        if (w + 1) % 500 == 0:
            bpb = running / nb / math.log(2)
            log.append({"w": w + 1, "train_bpb": round(bpb, 4)})
            print(f"  win {w+1}/{total_windows} train_bpb={bpb:.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
            running, nb = 0.0, 0
        if (w + 1) % args.ckpt_every == 0 or (w + 1) == total_windows:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "window": w + 1, "log": log, "args": vars(args)}, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path} @ win {w+1}", flush=True)

    # validation: fresh contiguous windows from held-out region
    model.eval()
    ce, n = 0.0, 0
    v0 = args.train_bytes
    with torch.no_grad():
        for k0 in range(v0, v0 + args.val_bytes - win, win):
            block = torch.as_tensor(data[k0:k0 + win].astype(np.int64), device=device)
            logits = model(block[:-1].unsqueeze(0))
            ce += F.cross_entropy(logits.reshape(-1, 256),
                                  block[1:].reshape(-1),
                                  reduction="sum").item()
            n += win - 1
    val_bpb = ce / n / math.log(2)
    print(f"RESULT teacher: val_bpb={val_bpb:.4f} time={time.time()-t0:.0f}s", flush=True)
    stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    out_name = "result_teacher.json" if ckpt_path == "ckpt_teacher.pt" else f"result_{stem}.json"
    json.dump({"teacher_val_bpb": val_bpb, "n_params": n_params,
               "ctx": args.ctx, "train_bytes": args.train_bytes,
               "log": log[-50:]}, open(out_name, "w"), indent=2)
    print("saved", out_name, flush=True)


if __name__ == "__main__":
    main()