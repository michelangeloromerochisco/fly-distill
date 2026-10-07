#!/usr/bin/env python3
"""Graph statistics explaining WHY conditions differ in LM performance.
Computes, for fly / degree-shuffled / ER: degree stats, reciprocity,
strongly-connected structure (Tarjan), spectral radius (power iteration),
and sensory->output reachability.
"""
import numpy as np
import sys
sys.path.insert(0, ".")
from flylm import load_graph, degree_preserving_shuffle, erdos_renyi

u, v, w, N, sensory, output = load_graph("larval_graph.npz")

def stats(uu, vv, ww, label):
    E = len(uu)
    outdeg = np.bincount(uu, minlength=N)
    indeg = np.bincount(vv, minlength=N)
    # reciprocity: fraction of edges whose reverse exists
    eset = set(zip(uu.tolist(), vv.tolist()))
    recip = np.mean([1.0 if (j, i) in eset else 0.0 for i, j in zip(uu.tolist(), vv.tolist())])
    # SCC via iterative Tarjan (python, 2952 nodes ok)
    adj = [[] for _ in range(N)]
    for i, j in zip(uu.tolist(), vv.tolist()):
        adj[i].append(j)
    index_counter = [0]; stack = []; lowlink = [0]*N; idx = [0]*N; on = [False]*N
    comp = [-1]*N; cid = [0]
    def strongconnect(node):
        work = [(node, 0)]
        while work:
            nd, pi = work[-1]
            if pi == 0:
                idx[nd] = index_counter[0]; lowlink[nd] = index_counter[0]
                index_counter[0] += 1; stack.append(nd); on[nd] = True
            recurse = False
            for k in range(pi, len(adj[nd])):
                s = adj[nd][k]
                if idx[s] == 0:
                    work[-1] = (nd, k+1); work.append((s, 0)); recurse = True; break
                elif on[s]:
                    lowlink[nd] = min(lowlink[nd], idx[s])
            if recurse:
                continue
            if lowlink[nd] == idx[nd]:
                while True:
                    s = stack.pop(); on[s] = False; comp[s] = cid[0]
                    if s == nd: break
                cid[0] += 1
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[nd])
    for nd in range(N):
        if idx[nd] == 0:
            strongconnect(nd)
    sizes = np.bincount(np.array([c for c in comp if c >= 0]))
    big_scc = int(sizes.max()) if len(sizes) else 0
    in_scc = np.mean([1.0 if comp[i] != -1 and sizes[comp[i]] > 1 else 0.0
                      for i, j in zip(uu.tolist(), vv.tolist())])
    # spectral radius via power iteration on dense W
    W = np.zeros((N, N), dtype=np.float32)
    W[uu, vv] = np.asarray(ww, dtype=np.float32)
    x = np.random.default_rng(0).random(N).astype(np.float32)
    lam = 0.0
    for _ in range(60):
        x = W @ x
        nrm = np.linalg.norm(x) + 1e-12
        lam = float(nrm); x /= nrm
    # sensory -> output reachability (BFS, unweighted)
    from collections import deque
    dist = [-1]*N
    dq = deque()
    for s in sensory:
        dist[s] = 0; dq.append(s)
    while dq:
        nd = dq.popleft()
        for nb in adj[nd]:
            if dist[nb] == -1:
                dist[nb] = dist[nd] + 1; dq.append(nb)
    reached = [dist[o] for o in output if dist[o] >= 0]
    print(f"== {label}")
    print(f"   edges {E:,} | mean out-deg {outdeg.mean():.1f} | max in-deg {indeg.max()} "
          f"| reciprocity {recip:.3f}")
    print(f"   largest SCC {big_scc} ({100*big_scc/N:.1f}% of N) | edges inside recurrent SCCs {100*in_scc:.1f}%")
    print(f"   spectral radius {lam:.2f}")
    if reached:
        print(f"   sensory->output reachable {len(reached)}/{len(output)} | mean hops {np.mean(reached):.1f} | max {max(reached)}")
    print()

stats(u.astype(np.int64), v.astype(np.int64), w, "FLY (real Winding-2023, >=3 syn)")
uu, vv = degree_preserving_shuffle(u, v, N, seed=0)
stats(uu, vv, np.ones(len(uu), dtype=np.float32), "SHUFFLED (degree-preserving, seed 0)")
uu, vv = erdos_renyi(u, v, N, seed=0)
stats(uu, vv, np.ones(len(uu), dtype=np.float32), "ER (same density, seed 0)")