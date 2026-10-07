#!/usr/bin/env python3
"""Parse Winding et al. 2023 larval connectome -> edge list + manifest.
Inputs: larval_raw/Supplementary-Data-S1/{all-all_connectivity_matrix.csv,annotations.csv}
Outputs: larval_graph.npz (u,v,w arrays), larval_manifest.json
"""
import json
import numpy as np
import pandas as pd

RAW = "larval_raw/Supplementary-Data-S1"
mat = pd.read_csv(f"{RAW}/all-all_connectivity_matrix.csv", index_col=0)
ids = mat.columns.astype(np.int64).to_numpy()  # column ids == row ids
assert (mat.index.astype(np.int64).to_numpy() == ids).all(), "row/col id mismatch"
M = mat.to_numpy(dtype=np.float32)
N = M.shape[0]
total_syn = int(M.sum())

ann = pd.read_csv(f"{RAW}/annotations.csv")
print("annotations rows:", len(ann))
print("celltype counts:")
print(ann["celltype"].value_counts())

# Map ids -> position
pos = {int(i): k for k, i in enumerate(ids)}

def ann_positions(celltype_val):
    sub = ann[ann["celltype"] == celltype_val]
    out = []
    for _, r in sub.iterrows():
        for c in ("left_id", "right_id"):
            v = str(r[c])
            if v not in ("no pair", "nan", "None") and v.isdigit() and int(v) in pos:
                out.append(pos[int(v)])
    return sorted(set(out))

sensory_pos = ann_positions("sensory")
# output-ish classes in Winding annotations: 'descending', 'ascending', 'secretory', 'motor'
output_pos = []
for ct in ("descending", "ascending", "secretory", "motor"):
    output_pos += ann_positions(ct)
output_pos = sorted(set(output_pos))
print("sensory (input) neurons:", len(sensory_pos))
print("output (descending/ascending/secretory/motor):", len(output_pos))
print("matrix N:", N, "total synapse count:", total_syn)

# Edge lists at thresholds
stats = {}
for thr in (1, 2, 3, 5):
    stats[thr] = int((M >= thr).sum())
print("edge counts at thresholds:", stats)

THR = 3  # Winding et al. 'strong connection' convention
u, v = np.nonzero(M >= THR)
w = M[u, v].astype(np.float32)
order = np.argsort(-w)
u, v, w = u[order], v[order], w[order]
print(f"kept edges (>= {THR} syn): {len(u)}")

np.savez_compressed(
    "larval_graph.npz",
    u=u.astype(np.int32), v=v.astype(np.int32), w=w,
    ids=ids,
    sensory=np.array(sensory_pos, dtype=np.int32),
    output=np.array(output_pos, dtype=np.int32),
)

manifest = {
    "source": "Winding et al. 2023 (Science), Supplementary Data S1, brain-networks/larval-drosophila-connectome",
    "N_neurons": int(N),
    "total_synapse_count": total_syn,
    "threshold": THR,
    "edges_kept": int(len(u)),
    "edges_by_threshold": stats,
    "sensory_count": len(sensory_pos),
    "output_count": len(output_pos),
    "annotations_rows": int(len(ann)),
    "params": {"v_rest_mV": -52.0, "v_thresh_mV": -45.0, "tau_ms": 20.0, "w_mV": 0.275,
                "dt_ms": 1.0, "T_ticks_per_byte": 16},
}
with open("larval_manifest.json", "w") as f:
    json.dump(manifest, f, indent=2)
print(json.dumps(manifest, indent=2))