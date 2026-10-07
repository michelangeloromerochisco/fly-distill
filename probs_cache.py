#!/usr/bin/env python3
"""One-time teacher-probs cache for Phase 1b KD (topology-independent).

Computes the teacher's next-byte probability distribution for every target
position p in [ctx, train_bytes) and stores fp16 rows (p - ctx) in a .npy
memmap (~20.5 GB at 40M). KD jobs memmap-read this file; the teacher never
runs during student training.

Efficient scheme: non-overlapping 512-token blocks (exactly the teacher's
training window). Prediction at block index j -> target s+j+1 with context
length j+1 bytes, matching training-time conditioning exactly. Stride 512
tiles coverage contiguously: block s covers targets [s+1, s+512].
"""
import argparse
import json
import os
import time

import numpy as np
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def meta_path(path):
    return path + ".meta.json"


def valid(path, train_bytes, t_mtime):
    if not os.path.exists(path) or not os.path.exists(meta_path(path)):
        return False
    try:
        m = json.load(open(meta_path(path)))
        return (m.get("rows", 0) >= train_bytes - m.get("ctx", 512)
                and m.get("teacher_mtime") == t_mtime)
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-bytes", type=int, default=40_000_000)
    ap.add_argument("--teacher-ckpt", default="ckpt_teacher.pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--g", type=int, default=32, help="blocks per forward")
    args = ap.parse_args()

    import teacher
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = np.fromfile("tinystories_slice.txt", dtype=np.uint8)
    tpath = args.teacher_ckpt
    ck = torch.load(tpath, map_location="cpu", weights_only=False)
    net = teacher.TinyTransformer(ctx=ck.get("args", {}).get("ctx", 512))
    net.load_state_dict(ck["model"])
    net.to(device).eval()
    ctx = net.ctx                       # 512 = block length AND window size

    out = args.out or f"probs_{args.train_bytes}_{int(os.path.getmtime(tpath))}.npy"
    rows = args.train_bytes - ctx       # rows for targets p in [ctx, train_bytes)
    print(f"teacher val_bpb={ck.get('val_bpb', -1)} | rows={rows:,} -> {out}", flush=True)

    if os.path.exists(out) and os.path.getsize(out) == rows * 256 * 2 \
            and valid(out, args.train_bytes, os.path.getmtime(tpath)):
        print("probs cache already valid -> nothing to do", flush=True)
        return
    if os.path.exists(out):
        os.remove(out)

    mm = np.lib.format.open_memmap(out, mode="w+", dtype=np.float16,
                                   shape=(rows, 256))
    mm[:] = np.float16(-1.0)            # sentinel: unwritten rows
    G = args.g
    t0 = time.time()
    n_blocks = (args.train_bytes - 2) // ctx + 1   # cover targets up to T-1
    with torch.no_grad():
        for b0 in range(0, n_blocks, G):
            g = min(G, n_blocks - b0)
            starts = np.arange(b0, b0 + g) * ctx
            blocks = np.stack([data[s:s + ctx] for s in starts]).astype(np.int64)
            x = torch.from_numpy(blocks).to(device)      # [g, 512]
            probs = torch.softmax(net(x), -1)            # [g, 512, 256]
            for bi, s in enumerate(starts):
                # targets p in [s+1, s+512]; rows r = p-ctx; probs idx j = p-s-1
                r_lo = max(s + 1, ctx) - ctx
                r_hi = min(s + ctx, args.train_bytes - 1) + 1 - ctx
                if r_hi <= r_lo:
                    continue
                j_lo = r_lo + ctx - s - 1
                mm[r_lo:r_hi] = probs[bi, j_lo:j_lo + (r_hi - r_lo)] \
                    .half().cpu().numpy()
            if (b0 // G) % 200 == 0:
                done_p = min((b0 + g) * ctx + 1, args.train_bytes)
                el = time.time() - t0
                print(f"  ~rows {done_p:,}/{args.train_bytes:,} elapsed={el:.0f}s "
                      f"eta={el * (args.train_bytes - done_p) / max(done_p, 1):.0f}s",
                      flush=True)

    # full sentinel scan (missing coverage = corrupt cache)
    bad = 0
    CH = 4_000_000
    for i in range(0, rows, CH):
        if (np.asarray(mm[i:min(i + CH, rows)]) < 0).any():
            bad += 1
            print(f"  sentinel rows found in [{i:,}, {min(i+CH, rows):,})", flush=True)
    if bad:
        print("corrupt cache -> NOT writing meta; delete file to retry", flush=True)
        mm.flush()
        return

    json.dump({"rows": rows, "ctx": ctx, "train_bytes": args.train_bytes,
               "teacher_mtime": os.path.getmtime(tpath),
               "teacher_val_bpb": ck.get("val_bpb", -1),
               "built": time.strftime("%d/%m/%Y %H:%M:%S")},
              open(meta_path(out), "w"), indent=2)
    print(f"DONE {out} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()