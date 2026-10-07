# Does Connectome Structure Help a Language Model? A Controlled Test with the Drosophila Larval Connectome

**Project:** fly-distill | **Status:** Phase 1 complete; Phase 1b in progress; Phase 2 pre-registered
**Last updated:** 15/09/2026 (Phase 2 pre-registration written before any Phase 2 data exists)

---

## 1. Abstract (draft)

We test whether biological connectome structure improves a language model at very small scale. Using the Winding et al. (2023) larval *Drosophila* connectome (2,952 neurons, 38,401 edges above a 3-synapse threshold) as the frozen recurrent substrate of a leaky integrate-and-fire (LIF) byte-level language model, we train identical models that differ **only** in graph topology: the real fly graph, a degree-preserving shuffle, an Erdős–Rényi random graph, and the fly graph with erased (uniform) synapse weights. Under matched training (20M bytes of TinyStories) the real connectome is *not* better than any control: it is slightly worse than the degree-preserving shuffle (+0.029 bits/byte, t=2.02, n=3–4 seeds, not significant), clearly worse than a random graph (+0.245, t=27), and its synapse weights carry real information (erasing them costs +0.115, t=7.2). Knowledge distillation from a 3.4M-parameter transformer teacher does not rescue the fly topology. Two capacity levers — deeper per-byte dynamics (T=8) and a wider readout — shrink the fly's disadvantage toward zero but never reverse it. Phase 2 (pre-registered below) tests the remaining hypothesis at maximum feasible scale on the same hardware: 7.5× more data with both capacity levers combined.

---

## 2. Background and motivation

- **Question.** Biological neural connectivity is shaped by evolution for behavior, not for language. But if connectome structure encodes general computational primitives (recurrence, hub structure, sensory→motor dimensionality reduction), a frozen connectome substrate might help even on a non-natural task. Prior work (Shiu et al. 2024, "connectome-constrained networks") shows fly-inspired architectures can match neurodata; whether they help on *language* is untested.
- **Why this matters.** A positive result would suggest brain-derived priors as a cheap inductive bias for tiny models (edge-compute / democratization angle). A rigorous negative result is equally valuable: it bounds the "free lunch from biology" claim with controls that most connectome-ML papers lack (degree-preserving shuffle, ER graph, weight-erased graph).
- **Design principle.** Everything is held fixed except the graph: same dynamics, same adapters, same optimizer, same data order statistics. Any difference in validation loss is attributable to topology (or its interaction with the one lever being varied).

## 3. Methods

### 3.1 Connectome
- Source: Winding et al. 2023 (Science), Supplementary Data S1, larval *Drosophila* brain connectome.
- N = 2,952 neurons; 352,611 total synapses; edges kept where synapse count ≥ 3 → 38,401 edges. Edge weight = synapse count (frozen).
- 430 sensory neurons (byte input sites), 46 output neurons (readout sites).

### 3.2 Model (flylm.py)
- LIF byte-LM: per byte, T ticks of LIF dynamics (V_rest=-52 mV, V_thresh=-45 mV, τ=20 ms, dt=1 ms, W=0.275 mV, recurrent current scale 0.10); surrogate-gradient BPTT within the byte window; membrane state carries across bytes (forward-only).
- Frozen: sparse graph W^T [N×N]. Trainable adapters only: byte→drive embedding (256×2952), per-neuron synaptic gains (2952), linear readout ([spike_mean, v_offset] → 256 logits). ~782k trainable parameters.
- Capacity arms: T=8 (deeper per-byte dynamics); width=1024 (MLP readout, +334k params).

### 3.3 Controls (topology-only differences)
- **fly** — real graph, real weights.
- **shuffled** — degree-preserving random rewire (in-degree sequence preserved), weights permuted.
- **er** — Erdős–Rényi random graph, same N and edge count, weights permuted.
- **fly-uniform** — real topology, all weights set to 1 (isolates the contribution of synapse weights).

### 3.4 Training protocol (Phase 1)
- TinyStories slice, 157,286,401 bytes on disk; 20,000,000 train bytes; 150,000 validation bytes from the held-out region [20M, 20.15M).
- batch 64, AdamW lr 3e-3, grad clip 1.0, 312,500 steps, seeds {0,1,2,3} for fly / {0,1,2} for controls.
- Metric: validation bits-per-byte (hard CE), identical protocol across all arms and phases.

### 3.5 Phase 1b: knowledge distillation (kd_flylm.py)
- Teacher: 3.4M-param byte transformer, ctx 512, trained on 40M bytes; teacher val 1.344 bpb (≈ 1.9× lower than any student — distillation gap is real).
- Soft targets: teacher next-byte probabilities precomputed once over the 40M region (probs cache, 39,999,488 rows × 256 fp16, ~20 GB on disk, 32 MB rolling RAM buffer).
- Student loss: L = (1−α)·CE(true) + α·τ²·CE(teacher_probs_τ). Two settings: α=0.5, τ=2 (first pass — **quarantined** due to a +512-byte reader shift bug) and α=0.25, τ=1 (gentle re-run with the aligned reader). 20M bytes, seed 0, T=4 — directly comparable to Phase 1 baselines.

## 4. Results — Phase 1 (complete, 14/09/2026)

### 4.1 Multi-seed @ 20M, T=4, linear readout
| condition | n | mean bpb | sd | sem |
|---|---|---|---|---|
| fly | 4 | 4.2728 | 0.0081 | 0.0040 |
| shuffled | 3 | 4.2442 | 0.0291 | 0.0168 |
| er | 3 | 4.0288 | 0.0067 | 0.0038 |
| fly-uniform | 3 | 4.3590 | 0.0016 | 0.0009 |

Pairwise (over matched seeds):
- fly − shuffled = **+0.0291** (sem 0.0144, t=2.02) — suggestive, not significant at n=3–4.
- fly − er = **+0.2445** (sem 0.0090, t=27.0) — the real connectome is substantially worse than a random graph.
- er − shuffled = −0.2154 (t=−12.5) — ER's unconstrained degree concentration (hubs) *helps*; the fly's degree distribution is not the helpful kind.
- fly-uniform − shuffled = **+0.1148** (sem 0.0159, t=7.2) — **synapse weights carry real information** (+0.09 bpb when erased); the topology alone does not.

### 4.2 Capacity arms (seed 0, 20M)
| arm | fly | shuffled | er | fly-uniform | fly−shuf gap |
|---|---|---|---|---|---|
| T=4 linear (base) | 4.2621 | 4.2256 | 4.0352 | 4.3582 | +0.0365 |
| T=8 | 4.0579 | 4.0543 | 3.7376 | 4.1919 | **+0.0036** |
| w1024 (T=4) | 4.1841 | 4.1777 | 3.9783 | 4.1770 | **+0.0063** |

- T=8 is the dominant lever for every topology (−0.17 to −0.30 bpb).
- Both capacity arms shrink the fly−shuffled gap (+0.0365 → +0.0036 / +0.0063) — **directionally, capacity favors the fly** — but no arm flips the sign. fly−er stays large (+0.21 to +0.32) in every arm.
- Per-arm deltas: fly gains more from T=8 (−0.2042) than shuffled (−0.1713) and from w1024 (−0.0780 vs −0.0479). Consistent direction, sub-threshold magnitude.

### 4.3 Phase 1 verdict
H0 ("connectome topology provides no language-modeling benefit at this scale") survives all arms: multi-seed, capacity scale-up, and the weight-erasure control. The only robust positive finding is about **weights**, not **wiring**: real synapse-strength distributions improve the model (+0.115 vs uniform weights, t=7.2), and hub-y random topology (ER) beats the fly's degree distribution.

## 5. Results — Phase 1b (knowledge distillation, COMPLETE 15/09/2026)

Teacher: 3,356,160-param byte transformer, ctx 512, val 1.344 bpb (≈2.9× better than any student — the distillation gap is real).

kd20 (α=0.5, τ=2): quarantined (reader +512-byte shift bug). Record only: fly 4.4242 (invalid).
kd40 (40M): all cells failed (CUDA OOM / paging on 4 GB GPU) — MISSING.

**kd20g (α=0.25, τ=1, aligned reader), seed 0, 20M — complete quartet:**

| condition | kd20g | hard-CE (Phase 1) | delta |
|---|---|---|---|
| fly | 4.2564 | 4.2621 | −0.0057 |
| shuffled | 4.2263 | 4.2256 | +0.0007 |
| er | 4.0214 | 4.0352 | −0.0138 |

Topology gap under KD: hard +0.0365 → kd20g **+0.0301** (KD moved the gap −0.0063 in fly's favor).

**Verdict (per pre-committed rule §8):** kd-gap +0.0301 > 0 → **KD does NOT rescue the fly topology.** Directionally, every KD/capacity lever nudges the gap toward the fly and KD helps the weakest topologies most (ER −0.0138, fly −0.0057, shuffled +0.0007) — consistent with distillation acting as a capacity equalizer, not a topology equalizer — but the sign test never fires.

## 6. Phase 2 — PRE-REGISTRATION (written before any Phase 2 data)

**Motivation.** Phase 1's capacity arms monotonically shrank the fly−shuffled gap and fly gained more per lever than shuffled in both arms. The remaining untested hypothesis: at sufficiently large scale (data × capacity), the fly topology's disadvantage reverses.

**Config (all jobs identical).** 150,000,000 train bytes (7.5× Phase 1), T=8, width=1024 (both levers combined — the two best Phase-1 arms), batch 64, lr 3e-3, 150k val bytes from [150M, 150.15M), adapters-only, ckpt every 5,000 steps. Val region necessarily differs from Phase 1 ([150M,150.15M) vs [20M,20.15M)); cross-phase comparisons carry that caveat, same-phase (seed-0 quartet) comparisons do not.

**Jobs.** {fly, shuffled, er, fly-uniform} × seeds {0, 1}. Seed-0 quartet first (~58 h total on the RTX 3050 Laptop 4 GB); seed 1 adds a second quartet (~116 h total) for a two-seed replication. Est. ~14.5 h/job.

**Primary endpoint (decided before launch).**
Δ_gap = (fly − shuffled)|150M,T8,w1024 − (fly − shuffled)|20M,T4,linear, with seed-0 cells.
- Δ_gap ≤ 0 AND fly−shuffled ≤ 0 at 150M → **scale rescues the fly topology** (H1).
- Δ_gap < −0.01 but gap still > 0 → trend toward rescue; report as partial.
- |Δ_gap| ≤ 0.01 → scale does not change the topology effect (H0 holds).
- Δ_gap > +0.01 → scale widens the disadvantage; H0 strengthened.

**Secondary endpoints.** fly−er at 150M (does the ER advantage persist at scale?); fly−fly-uniform at 150M (does the weight contribution persist?); per-lever attribution via Phase 1 refs.

**Stopping/robustness.** Deadline guard 122 h with persistent t0 (reboot-safe); runner idempotent with ckpt auto-resume; failure gate (4 strikes per job); marker prevents watchdog livelock (the Phase 1b failure mode). Report auto-generated at cap.

## 6.1 Phase 2 results (COMPLETE 20/09/2026, 05:48)

All 8 jobs completed, 0 failures, ~12.8 h/job, total run 3.3 days.

| condition | n | mean bpb | sd | seeds |
|---|---|---|---|---|
| fly | 2 | **3.9546** | 0.0277 | [3.9741, 3.9350] |
| shuffled | 2 | 3.9871 | 0.0247 | [3.9696, 4.0046] |
| er | 2 | 3.5385 | 0.0409 | [3.5675, 3.5096] |
| fly-uniform | 2 | 4.0074 | 0.0069 | [4.0123, 4.0025] |

**Primary endpoint — the topology gap across scale:**

| config | fly | shuffled | gap |
|---|---|---|---|
| 20M T=4 linear | 4.2728 | 4.2442 | +0.0286 |
| 20M T=8 | 4.0579 | 4.0543 | +0.0036 |
| 20M w1024 (T=4) | 4.1841 | 4.1777 | +0.0063 |
| 150M T=8 w1024 | 3.9546 | 3.9871 | **−0.0325** |

Δ_gap = −0.0325 − (+0.0286) = **−0.061** → the sign flipped. Per-seed: s0 +0.0045 (ambiguous alone), s1 **−0.0696** (fly wins outright). Every scale point moved the gap in fly's favor.

**Verdict (pre-registered rule §6): H1 at n=2 — SUPERSEDED by Phase 3 seed 2.** At the time of completion (20/09) the rule fired: the real connectome, a liability at 20M bytes (+0.029), became an asset at 150M (−0.033). **Phase 3's seed 2 (arrived 22/09) reversed this: fly 3.9911 vs shuffled 3.8707 → gap +0.1204.** Pooled n=3: gaps +0.0045 / −0.0696 / +0.1204, mean **+0.0185**, 1/3 seeds negative, gap sd ≈ 0.095. The sign of the 150M fly−shuffled gap is **seed-dominated** — the effect (either direction) is within seed noise (mean ± 2·sem ≈ ±0.11). Honest reading as of 22/09: **no reliable topology difference between fly and shuffled at 150M**; the robust claims are (a) ER > fly at every scale with a growing deficit, (b) real weights > uniform weights. Seed 3 + the hubcap arm (§7) complete the pre-registered protocol; the final paper claim will be whatever the full n=4 delivers under the §7 rules — the §6 H1 language above is retained as the pre-registration trail, not as the current belief.

**Secondary endpoints:**
- **fly−er grows with scale**: +0.245 (20M T4) → +0.320 (20M T8) → **+0.416** (150M). The hub-y random graph's advantage *increases* with training — whatever ER's degree concentration provides, the fly graph lacks, and scale amplifies the difference. The rescue is real but partial: fly beats its degree-matched control, not the unconstrained random graph.
- **Weight contribution shrinks but persists**: fly−fly-uniform = −0.086 (20M T4) → −0.053 (150M). Real synapse strengths remain worth ~0.05 bpb at 150M (non-monotonic across Phase 1 arms: w1024@20M had it at +0.007).
- Seed stability: fly and shuffled have comparable spreads (sd 0.028 vs 0.025); no evidence the fly graph stabilizes training.

## 7. Phase 3 — PRE-REGISTRATION (written 20/09/2026 before any Phase 3 data)

**Two questions open after Phase 2.** (a) The primary sign flip rests on n=2 seeds (mean gap −0.0325, sem ≈ 0.037 — directionally consistent everywhere but not individually significant). (b) The mechanism of the rescue is unidentified: WHY does the fly graph lose to shuffled at 20M but win at 150M, and why does ER keep winning at every scale?

**Jobs (all at the Phase 2 config: 150M bytes, T=8, w1024, batch 64, lr 3e-3):**
1. **Seeds 2–3 of the Phase 2 quartet** — fly, shuffled, er, fly-uniform × seeds {2, 3} = 8 jobs. With Phase 2's seeds 0–1 this gives n=4 per cell for the primary endpoint (paired over seeds; sign test on the per-seed gaps +0.0045, −0.0696, +g2, +g3).
2. **fly-hubcap mechanism arm** (new control, cap=40) × seeds {0, 1} = 2 jobs. Construction: real fly graph, every postsynaptic neuron's in-degree capped at 40 (ER's max is 29; fly's is 99 — 137 neurons exceed 40); trimmed edges re-attached to random strictly-below-cap recipients, preserving presynaptic sources, synapse weights, edge count (38,401), and mean in-degree (13.01). Same frozen LIF dynamics.

**Primary endpoint (seeds 0–3 pooled):** fly−shuffled gap mean and a sign test on the 4 per-seed gaps (each seed pair shares data order RNG per seed). Pre-committed: claim "scale-rescue is robust" requires ≥3 of 4 seeds with gap < 0 (or mean gap < 0 with 3/4 sign agreement); 2/4 with mean > 0 → "flip not robust, ambiguous"; else H0 reinstated for the gap.

**Mechanism endpoints (hubcap):**
- H-mech1 (hub liability): fly-hubcap ≥ fly at BOTH 20M and 150M (capping hubs removes something the fly needs) → hubs are beneficial, contradiction with ER-advantage.
- H-mech2 (hub drag at 150M): fly-hubcap < fly at 150M (capping helps once trained long) → hub concentration is the data-hungry part of the fly disadvantage; predicts hubcap ≈ shuffled at 150M.
- H-mech3 (hubs irrelevant): |fly-hubcap − fly| < 0.01 at 150M → rescue mechanism is NOT hub concentration; look elsewhere (e.g., motif structure).
- 20M hubcap pilot is NOT included (Phase 3 runs at 150M only; the 20M comparison comes free later if needed).

**Ops.** 10 jobs × ~12.8 h ≈ 5.3 days; deadline guard 150 h persistent; same idempotent runner pattern with the Phase 2 livelock fixes; results in result_p3_*.json; report_phase3.txt auto-generated.

### 7.1 Phase 3 RESULTS (final, 26/09/2026 — complete, 10/10 jobs, 0 failures)

**Cells (150M, T=8, w1024), seeds pooled (P2 s0–1 + P3 s2–3):**

| condition | n | mean bpb | sd | per-seed |
|---|---|---|---|---|
| fly | 4 | 3.9572 | 0.0303 | 3.9741 / 3.9350 / 3.9911 / 3.9284 |
| shuffled | 4 | 3.9171 | 0.0843 | 3.9696 / 4.0046 / 3.8707 / 3.8234 |
| er | 4 | 3.5234 | 0.0294 | 3.5675 / 3.5096 / 3.5093 / 3.5073 |
| fly-uniform | 4 | 4.0049 | 0.0050 | 4.0123 / 4.0025 / 4.0024 / 4.0022 |
| fly-hubcap | 2 | 3.8381 | 0.0538 | 3.8001 / 3.8761 |

**Primary endpoint (pre-committed rule, decided):** per-seed fly−shuffled gaps: +0.0045 / −0.0696 / +0.1204 / +0.1049 → mean **+0.0401**, 1/4 seeds negative, gap sd 0.089, paired t≈0.9 (df=3, ns). → **H0 reinstated: no reliable fly-vs-shuffled difference at 150M.** The Phase 2 n=2 sign flip was a seed-1 fluke. The pre-registered "scale rescue" claim is formally dead; the §6/§6.1 H1 language remains only as the pre-registration trail.

**Mechanism endpoint (hubcap, H-mech2 confirmed, n=2, both seeds, both controls):**

| seed | hubcap | fly | hubcap−fly | shuffled | hubcap−shuffled |
|---|---|---|---|---|---|
| 0 | 3.8001 | 3.9741 | **−0.1741** | 3.9696 | **−0.1695** |
| 1 | 3.8761 | 3.9350 | **−0.0589** | 4.0046 | **−0.1285** |

Capping the fly's hub in-degrees at 40 (same edges, same weights, only hub concentration removed) beats both the raw connectome and its degree-matched shuffle in **every pair tested** (mean diff vs fly −0.1165, vs shuffled −0.1490). **H-mech2 supported: hub concentration is a bottleneck in the real connectome; removing it unmasked a genuine biological-wiring advantage over random wiring.** Note the ordering: ER (3.523) < fly-hubcap (3.838) < shuffled (3.917) < fly (3.957) < fly-uniform (4.005) — hubcap beats its matched controls but NOT ER; the abstract must claim "unmasked benefit vs degree-matched random wiring," never "beats all random graphs."

**Robustness notes (n=4):** ER > fly at every scale; deficit grows with training (+0.245 @20M → +0.434 @150M mean). ER is eerily seed-stable (sd 0.029, range 3.507–3.568); shuffled is wildly seed-dependent (sd 0.084, range 3.823–4.005) — the shuffled control, not the fly, carries the seed variance in this comparison. fly-uniform is pinned at 4.0022–4.0123 across all seeds/scales: real synapse weights > uniform weights, always.

**Paper implication:** title candidate "Hub Concentration Masks the Benefit of Biological Connectome Structure in Small Language Models" is supported as stated (n=2 hubcap, both controls, both seeds). Three-legged story: (1) raw connectome ≈ degree-matched random at 150M (n=4, ns) — with the seed-variance lesson as a methods contribution; (2) hub concentration masks a real wiring benefit (H-mech2, intervention-based); (3) synapse weights carry stable signal at every scale.

## 8. Limitations (honest)

1. Scale ceiling: 4 GB laptop GPU. 150M bytes / 782k adapters is still tiny-LM territory; a cloud-scale replication (≥1B bytes) remains untested.
2. The fly's sensory/output populations (430/46) are wired for reflexes, not text; the byte→sensory injection is a strong domain mismatch by design.
3. Val regions differ across phases (documented); within-phase comparisons are clean.
4. n=3–4 seeds per cell at Phase 1; the fly−shuffled effect (if any) is below detection power at this n. Phase 2's two-seed quartets improve but do not fully solve this.
5. Single dataset (TinyStories). The topology effect could be dataset-dependent.
6. The kd40 (40M KD) cells failed on hardware limits; the KD conclusion rests on the 20M gentle arm only.

## 8. Interpretation rules (pre-committed, binding)

- **KD rescue claim** requires: kd20g fly ≤ kd20g shuffled (gap ≤ 0). Otherwise "KD does not rescue the fly topology" — regardless of how fly's absolute delta looks.
- **Scale-rescue claim** requires the Phase 2 primary endpoint per §6.
- Claims of "scale benefits the fly" must beat the shuffled control at matched budget, never just improve fly's absolute loss (everything improves with scale).

## 9. Artifacts

- Code: flylm.py (model+controls), teacher.py, probs_cache.py, kd_flylm.py, run_phase1.py, run_phase1b.py, run_phase2.py, watchdogs, report_phase1.py/txt, report_phase1b.py/txt, report_phase2.py/txt.
- Data: larval_graph.npz (parsed connectome), tinystories_slice.txt (157MB), probs cache (20 GB).
- Results: result_*.json (one per cell, curves included), checkpoints ckpt_*.pt.
- Full Phase 1 report: report_phase1.txt. Phase 1b: report_phase1b.txt. Phase 2: report_phase2.txt (after completion).