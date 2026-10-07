#!/usr/bin/env python3
"""Phase 0: larval-connectome byte-level language model vs controls.

Conditions (identical dynamics/encoder/readout/optimizer/data, only topology differs):
  fly      = real Winding-2023 larval graph (frozen weights = synapse counts)
  shuffled = degree-preserving random rewire of the same graph
  er       = Erdos-Renyi random graph, same N and edge count

Trainable (adapters only): byte->drive embedding, per-neuron synaptic gains, readout head.
Dynamics: LIF (Shiu et al. 2024 params), T ticks per byte, surrogate-gradient BPTT
within the byte window only; membrane state carries across bytes (forward-only).
"""
import argparse, json, math, os, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

V_REST, V_THRESH, TAU, W_MV = -52.0, -45.0, 20.0, 0.275
DT_FAC = 1.0 / TAU          # leak factor per 1ms tick
I_DRIVE = 12.0              # mV/tick max sensory injection
W_SCALE = 0.10              # recurrent current scale
SURROGATE_ALPHA = 2.0

def load_graph(npz_path):
    d = np.load(npz_path)
    return d["u"], d["v"], d["w"], int(d["ids"].shape[0]), d["sensory"], d["output"]

def to_sparse_wt(u, v, w, N, device):
    """WT[j,i] = weight of edge i->j, as CSR sparse tensor (constant buffer)."""
    idx = torch.stack([torch.from_numpy(v.astype(np.int64)),
                       torch.from_numpy(u.astype(np.int64))])   # (row=j, col=i)
    vals = torch.from_numpy(w.astype(np.float32))
    WT = torch.sparse_coo_tensor(idx, vals, (N, N)).coalesce().to(device)
    return WT

def degree_preserving_shuffle(u, v, N, seed=0):
    rng = np.random.default_rng(seed)
    indeg = np.bincount(v, minlength=N)
    pool = np.repeat(np.arange(N), indeg)
    rng.shuffle(pool)
    # fix self-loops by swapping with random safe positions
    for _ in range(50):
        bad = np.nonzero(pool == u)[0]
        if len(bad) == 0:
            break
        for i in bad:
            for _ in range(50):
                j = rng.integers(0, len(pool))
                if pool[j] != u[j] and pool[j] != u[i] and pool[i] != u[j]:
                    pool[i], pool[j] = pool[j], pool[i]
                    break
    return u, pool.astype(np.int64)

def erdos_renyi(u, v, N, seed=0):
    rng = np.random.default_rng(seed)
    E = len(u)
    a, b = rng.integers(0, N, E), rng.integers(0, N, E)
    keep = a != b
    return a[keep].astype(np.int64), b[keep].astype(np.int64)

def fly_hubcap(u, v, N, cap=40, seed=0):
    """Fly topology with hub in-degrees capped: any postsynaptic neuron with
    in-degree > cap keeps a random subset of `cap` of its incoming edges; the
    trimmed edges are re-attached to random strictly-below-cap recipients
    (keeping the original presynaptic sources and weights, never forming
    self-loops). Isolates the effect of biological hub concentration (fly max
    in-deg 99 vs ER max 29)."""
    rng = np.random.default_rng(seed)
    uu = u.astype(np.int64)
    vv = v.astype(np.int64).copy()
    for _ in range(200):
        vin = np.bincount(vv, minlength=N)
        hubs = np.nonzero(vin > cap)[0]
        if len(hubs) == 0:
            break
        h = hubs[np.argmax(vin[hubs])]                    # worst hub first
        idx = np.nonzero(vv == h)[0]                      # incoming edges of h
        drop = rng.choice(idx, size=vin[h] - cap, replace=False)
        below = np.nonzero(vin < cap)[0]
        # avoid self-loops: resample destinations for edges whose source == dst
        new_dst = rng.choice(below, size=len(drop), replace=True)
        clash = np.nonzero(uu[drop] == new_dst)[0]
        while len(clash):
            new_dst[clash] = rng.choice(below, size=len(clash), replace=True)
            clash = np.nonzero(uu[drop] == new_dst)[0]
        vv[drop] = new_dst
    return uu, vv

class SpikeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g / (1.0 + math.pi * SURROGATE_ALPHA * x.abs())

class FlyLM(nn.Module):
    def __init__(self, N, sensory, output, WT, T=4, width=0):
        super().__init__()
        self.N, self.T = N, T
        self.register_buffer("WT", WT)                    # frozen [N,N] CSR
        self.register_buffer("sens_mask",
            torch.eye(N, device=WT.device)[torch.as_tensor(sensory, dtype=torch.long)].sum(0))
        self.out_idx = torch.as_tensor(output, dtype=torch.long)
        n_o = len(output)
        self.emb = nn.Embedding(256, N)                   # byte -> full-N drive
        self.gain = nn.Parameter(torch.ones(N))           # postsynaptic gains
        if width and width > 0:                           # capacity arm: MLP readout
            self.head = nn.Sequential(
                nn.Linear(2 * n_o, width), nn.GELU(), nn.Linear(width, 256))
            nn.init.zeros_(self.head[2].bias)
        else:
            self.head = nn.Linear(2 * n_o, 256)           # [spike_mean, v_offset] -> logits
            nn.init.zeros_(self.head.bias)
        nn.init.normal_(self.emb.weight, 0.0, 0.5)

    def init_state(self, B, device):
        return torch.full((B, self.N), V_REST, device=device)

    def forward(self, bytes_in, v):
        """bytes_in: [B] long. v: [B,N] carried state (detached). One byte window."""
        drive_full = torch.sigmoid(self.emb(bytes_in)) * I_DRIVE     # [B,N]
        drive = drive_full * self.sens_mask                          # inject at sensory only
        spikes_sum, v_last = 0.0, v
        for t in range(self.T):
            # sensory injection as membrane bump
            bump = drive                                             # [B,N]
            x = v_last + bump - V_THRESH
            s = SpikeFn.apply(x)                                     # [B,N]
            current = torch.sparse.mm(self.WT, s.t()).t()            # [B,N]
            current = current * (self.gain.clamp(0.0, 4.0) * W_MV * W_SCALE).unsqueeze(0)
            dv = (-(v_last - V_REST) + bump + current) * DT_FAC
            v_next = v_last + dv
            v_next = torch.where(s.bool(), torch.full_like(v_next, V_REST - 2.0), v_next)
            spikes_sum = spikes_sum + s[:, self.out_idx]
            v_last = v_next
        spike_mean = spikes_sum / self.T
        feats = torch.cat([spike_mean, (v_last[:, self.out_idx] - V_REST) / 7.0], dim=1)
        return self.head(feats), v_last

def stream_CE(model, data, start, length, B, device, opt=None, train=False):
    """Process `length` bytes as B parallel streams from random offsets."""
    rng = np.random.default_rng(12345 if not train else int(time.time()) % 2**31)
    offs = rng.integers(start, start + length - 2, size=B)
    v = model.init_state(B, device)
    ce_sum, n = 0.0, 0
    L = length - 1
    for k in range(L):
        cur = torch.as_tensor(data[offs + k], dtype=torch.long, device=device)
        nxt = torch.as_tensor(data[offs + k + 1], dtype=torch.long, device=device)
        logits, v2 = model(cur, v)
        loss = F.cross_entropy(logits, nxt)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            v = v2.detach()
        else:
            v = v2
        ce_sum += loss.item(); n += 1
        if train and k % 4096 == 0 and k > 0:
            v = v.detach()
    return ce_sum / max(n, 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True,
                    choices=["fly", "shuffled", "er", "fly-uniform", "fly-hubcap"])
    ap.add_argument("--train-bytes", type=int, default=3_000_000)
    ap.add_argument("--val-bytes", type=int, default=100_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--data", default="tinystories_slice.txt")
    ap.add_argument("--graph", default="larval_graph.npz")
    ap.add_argument("--out", default=None)
    ap.add_argument("--width", type=int, default=0,
                    help="if >0, MLP readout hidden width (capacity arm)")
    ap.add_argument("--tag", default="",
                    help="arm tag; prefixes ckpt/result filenames (e.g. t8, w1024)")
    ap.add_argument("--ckpt-every", type=int, default=5000, help="steps between checkpoint saves")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = args.device
    data = np.fromfile(args.data, dtype=np.uint8)
    print(f"data bytes: {len(data):,} | device: {device}", flush=True)

    u, v, w, N, sensory, output = load_graph(args.graph)
    if args.condition == "fly":
        uu, vv = u.astype(np.int64), v.astype(np.int64)
        ww = w[: len(uu)]
    elif args.condition == "fly-hubcap":
        uu, vv = fly_hubcap(u, v, N, cap=40, seed=args.seed)
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
          f"T={args.T} width={args.width}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    steps = args.train_bytes // args.batch

    # ---- checkpointing (survives Windows reboot) ----
    ckpt_path = f"ckpt_{label}_s{args.seed}.pt"
    start_step, running, t0 = 0, 0.0, time.time()
    val_history, log = [], []
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        sa = ck.get("args", {})
        mism = [k for k in ("T", "train_bytes", "batch", "lr")
                if sa.get(k) != getattr(args, k)]
        if sa.get("width", 0) != args.width:
            mism.append("width")
        if mism:
            stale = ckpt_path + ".stale"
            os.replace(ckpt_path, stale)
            print(f"args mismatch {mism} -> old ckpt -> {os.path.basename(stale)}; "
                  f"fresh start", flush=True)
        else:
            model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
            start_step = ck["step"]; running = ck.get("running", 0.0)
            val_history = ck.get("val_history", [])
            log = ck.get("log", [])
            print(f"resumed {label} seed {args.seed} at step {start_step}", flush=True)
    steps = args.train_bytes // args.batch

    for step in range(start_step, steps):
        offs = np.random.randint(0, args.train_bytes - 2, size=args.batch)
        cur = torch.as_tensor(data[offs], dtype=torch.long, device=device)
        nxt = torch.as_tensor(data[offs + 1], dtype=torch.long, device=device)
        if step == start_step:
            state = model.init_state(args.batch, device)
        logits, state2 = model(cur, state)
        loss = F.cross_entropy(logits, nxt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        state = state2.detach()
        running += loss.item()
        if (step + 1) % 500 == 0:
            bpb = running / 500 / math.log(2)
            log.append({"step": step + 1, "train_bpb": round(bpb, 4)})
            print(f"  step {step+1}/{steps} train_bpb={bpb:.4f} elapsed={time.time()-t0:.0f}s", flush=True)
            running = 0.0
        if (step + 1) % args.ckpt_every == 0 or (step + 1) == steps:
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step + 1, "running": running, "log": log,
                        "val_history": val_history, "args": vars(args)}, ckpt_path)
            print(f"  [ckpt] saved {ckpt_path} @ step {step+1}", flush=True)

    # validation: batch of parallel streams from held-out region, state carried
    v0 = model.init_state(args.batch, device)
    ce = 0.0; n = 0
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
                                            "width": args.width, "tag": args.tag})
        val_bpb_final = prev  # multi-seed accumulation
        json.dump(prev, open(out, "w"), indent=2)
    else:
        json.dump({"condition": args.condition, "seed": args.seed, "N": N,
                   "edges": int(len(uu)), "trainable_params": n_params,
                   "T": args.T, "width": args.width, "tag": args.tag,
                   "batch": args.batch, "train_bytes": args.train_bytes,
                   "val_bytes": vb, "steps": steps, "val_bpb": val_bpb,
                   "time_s": dt, "curve": log, "val_history": val_history,
                   "lr": args.lr, "device": device}, open(out, "w"), indent=2)
    print("saved", out, flush=True)

if __name__ == "__main__":
    main()