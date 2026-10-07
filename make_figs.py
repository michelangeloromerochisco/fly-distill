#!/usr/bin/env python3
"""Paper figures from frozen result JSONs. Regenerable: reads result_*.json only.
Each config uses an EXPLICIT file whitelist so quarantined KD arms can never
contaminate base cells (kd20g fly s0 would otherwise overwrite base fly s0)."""
import glob
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.chdir(os.path.dirname(os.path.abspath(__file__)))
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

COL = {"fly": "#1a1a1a", "shuffled": "#e07b39", "er": "#3d7ea6",
       "fly-uniform": "#888888", "fly-hubcap": "#6a3d9a"}
LBL = {"fly": "fly (real connectome)", "shuffled": "shuffled (degree-matched)",
       "er": "ER random graph", "fly-uniform": "fly-uniform (weights erased)",
       "fly-hubcap": "fly-hubcap (hubs capped, cap=40)"}


def cells_from(files, tb, T, w, conds=COL.keys()):
    merged = {}
    for f in files:
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("condition") not in conds:
            continue
        for e in [d] + list(d.get("runs", [])):
            if ((e.get("train_bytes") or d.get("train_bytes")) == tb
                    and e.get("T", 4) == T and e.get("width", 0) == w):
                merged.setdefault(d["condition"], {})[
                    e.get("seed", d.get("seed"))] = e["val_bpb"]
    return merged


BASE20 = ["result_fly.json", "result_shuffled.json", "result_er.json",
          "result_fly-uniform.json"]
T820 = [f"result_t8_{c}.json" for c in COL if c != "fly-hubcap"]
W1024 = [f"result_w1024_{c}.json" for c in COL if c != "fly-hubcap"]
P150 = sorted(glob.glob("result_p2_*.json")) + sorted(glob.glob("result_p3_*.json"))


def mean(v):
    return sum(v) / len(v)


def sem(gaps):
    if len(gaps) < 2:
        return 0.0
    m = mean(gaps)
    return math.sqrt(sum((g - m) ** 2 for g in gaps) / (len(gaps) - 1)
                     / len(gaps))


# ---------------- Figure 1: scale ladder of the fly-shuffled gap ----------------
configs = [
    ("20M\nT=4 lin", cells_from(BASE20, 20_000_000, 4, 0)),
    ("20M\nT=8", cells_from(T820, 20_000_000, 8, 0)),
    ("20M\nw1024", cells_from(W1024, 20_000_000, 4, 1024)),
    ("150M\nT=8 w1024", cells_from(P150, 150_000_000, 8, 1024)),
]
fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=200)
xs, ys, errs = [], [], []
for i, (lbl, c) in enumerate(configs):
    if "fly" not in c or "shuffled" not in c:
        continue
    seeds = sorted(set(c["fly"]) & set(c["shuffled"]))
    gaps = [c["fly"][s] - c["shuffled"][s] for s in seeds]
    xs.append(i)
    ys.append(mean(gaps))
    errs.append(sem(gaps))
ax.bar(xs, ys, yerr=errs, capsize=4,
       color=["#c44e52" if y > 0 else "#2a7f4f" for y in ys], alpha=0.9, width=0.55)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(xs)
ax.set_xticklabels([c[0] for c, _ in zip(configs, range(len(xs)))][: len(xs)])
ax.set_ylabel("fly − shuffled val bpb\n(lower = fly better)")
ax.set_title("The fly\u2212shuffled gap across scale: no reliable sign change\n(pre-registered four-seed rule: H0 at 150M)")
ns = [len(sorted(set(c["fly"]) & set(c["shuffled"]))) for _, c in configs if "fly" in c and "shuffled" in c]
for x, y, e, n in zip(xs, ys, errs, ns):
    ax.annotate(f"{y:+.4f}", (x, y + (e + 0.003) if y > 0 else y - (e + 0.003)),
                ha="center", va="bottom" if y > 0 else "top", fontsize=9)
    ax.annotate(f"n={n}", (x, 0), textcoords="offset points", xytext=(14, 4),
                fontsize=7, color="#555555")
plt.tight_layout()
plt.savefig(f"{FIG}/fig1_scale_ladder.png")
plt.close()

# ---------------- Figure 2: per-seed gaps + controls vs fly ----------------
c150 = cells_from(P150, 150_000_000, 8, 1024)
c20 = cells_from(BASE20, 20_000_000, 4, 0)
seeds150 = sorted(set(c150.get("fly", {})) & set(c150.get("shuffled", {})))
fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), dpi=200)
ax = axes[0]
g20 = [c20["fly"][s] - c20["shuffled"][s]
       for s in sorted(set(c20.get("fly", {})) & set(c20.get("shuffled", {})))]
g150 = [c150["fly"][s] - c150["shuffled"][s] for s in seeds150]
ax.scatter([0.0] * len(g20), g20, color="#c44e52", s=46, label=f"20M (n={len(g20)})")
ax.scatter([1.0] * len(g150), g150, color="#2a7f4f", s=46, label=f"150M (n={len(g150)})")
for x, gv, col in ((0.0, g20, "#c44e52"), (1.0, g150, "#2a7f4f")):
    ax.hlines(mean(gv), x - 0.22, x + 0.22, color=col, ls="--", lw=1.4)
    ax.annotate(f"mean {mean(gv):+.4f}", (x, mean(gv)), textcoords="offset points",
                xytext=(10, -3), fontsize=8, color=col)
ax.axhline(0, color="black", lw=1)
ax.set_xticks([0, 1])
ax.set_xticklabels(["20M", "150M"])
ax.set_ylabel("per-seed fly − shuffled gap (bpb)")
ax.set_title("Paired per-seed gaps: seed-dominated at 150M")
ax.legend(fontsize=8, loc="lower left")

ax = axes[1]
for cond, mk in (("er", "o"), ("fly-uniform", "s"), ("fly-hubcap", "^")):
    g = [c150[cond][s] - c150["fly"][s]
         for s in seeds150 if s in c150.get(cond, {})]
    if g:
        ax.scatter([0.15] * len(g), g, s=46, color=COL[cond], marker=mk,
                   label=LBL[cond] + " @150M")
    if cond in c20:
        g20c = [c20[cond][s] - c20["fly"][s]
                for s in sorted(set(c20[cond]) & set(c20.get("fly", {})))]
        if g20c:
            ax.scatter([-0.15] * len(g20c), g20c, s=46, color=COL[cond], marker=mk,
                       alpha=0.45, label=LBL[cond] + " @20M")
ax.axhline(0, color="black", lw=1)
ax.set_xticks([-0.15, 0.15])
ax.set_xticklabels(["20M", "150M"])
ax.set_ylabel("control − fly gap (bpb)")
ax.set_title("Control \u2212 fly gaps: ER deficit widens with scale;\nhub capping improves the connectome (150M)")
ax.legend(fontsize=7, loc="center left")
plt.tight_layout()
plt.savefig(f"{FIG}/fig2_seeds_and_controls.png")
plt.close()

# ---------------- Figure 3: training curves ----------------
fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=200)
for f, key in [("result_fly.json", "20M fly"),
               ("result_shuffled.json", "20M shuffled"),
               ("result_er.json", "20M er"),
               ("result_p2_fly.json", "150M fly"),
               ("result_p2_shuffled.json", "150M shuffled")]:
    try:
        d = json.load(open(f))
    except Exception:
        continue
    curve = d.get("curve", [])
    if not curve:
        continue
    st = [c["step"] for c in curve]
    bp = [c.get("train_bpb") for c in curve]
    big = key.startswith("150M")
    ax.plot(st, bp, lw=1.2 if big else 1.0,
            color=COL[key.split()[-1]],
            alpha=0.95 if big else 0.4,
            ls="-" if big else "--", label=key)
ax.set_xlabel("training step")
ax.set_ylabel("train bits/byte")
ax.set_title("Training curves at 20M and 150M bytes")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{FIG}/fig3_curves.png")
plt.close()

print("figures written:", sorted(os.listdir(FIG)))