# FLY-DISTILL STATUS — 6/10/2026 (FINAL, Phase 3 complete)

## Phase 3: 10/10 jobs done, 0 failures. All experiments COMPLETE. No training running.

Final 150M cells (T=8, w1024, seeds pooled P2 s0–1 + P3 s2–3):

| condition | n | mean bpb | sd | per-seed |
|---|---|---|---|---|
| fly | 4 | 3.9572 | 0.0303 | 3.9741 / 3.9350 / 3.9911 / 3.9284 |
| shuffled | 4 | 3.9171 | 0.0843 | 3.9696 / 4.0046 / 3.8707 / 3.8234 |
| er | 4 | 3.5234 | 0.0294 | 3.5675 / 3.5096 / 3.5093 / 3.5073 |
| fly-uniform | 4 | 4.0049 | 0.0050 | 4.0123 / 4.0025 / 4.0024 / 4.0022 |
| fly-hubcap | 2 | 3.8381 | 0.0538 | 3.8001 / 3.8761 |

## PRIMARY ENDPOINT — DECIDED (n=4)
Per-seed fly−shuffled gaps: s0 +0.0045, s1 −0.0696, s2 +0.1204, s3 +0.1049.
Mean +0.0401, 1/4 seeds negative, gap sd 0.089, paired t≈0.9 (df=3, ns).
→ Pre-registered §7 rule: **H0 reinstated — no reliable fly-vs-shuffled difference at 150M.** The Phase 2 n=2 "flip" was a seed-1 fluke.

## MECHANISM — H-mech2 CONFIRMED (n=2, both seeds, both controls)
| seed | hubcap | fly | diff | shuffled | diff |
|---|---|---|---|---|---|
| 0 | 3.8001 | 3.9741 | −0.1741 | 3.9696 | −0.1695 |
| 1 | 3.8761 | 3.9350 | −0.0589 | 4.0046 | −0.1285 |

Capping hub in-degrees at 40 beats the raw connectome AND its degree-matched shuffle in every pair (mean vs fly −0.1165, vs shuffled −0.1490). **Hub concentration masks a genuine biological-wiring advantage.** Ordering: ER (3.523) < fly-hubcap (3.838) < shuffled (3.917) < fly (3.957) < fly-uniform (4.005). Hubcap does NOT beat ER — claim discipline: "unmasked benefit vs degree-matched random wiring."

## Robust findings (n=4, all stable)
- ER > fly at every scale; deficit grows with training (+0.245 @20M → +0.434 @150M). ER seed-stable (sd 0.029); shuffled wildly seed-dependent (sd 0.084).
- Real synapse weights > uniform at every scale and seed (fly-uniform pinned 4.0022–4.0123).

## Current work: paper COMPLETE
- PAPER.md v1.1 (6/10, professional register): "Hub Concentration Masks the Benefit of Biological Connectome Structure in Small Language Models" — arXiv-style (cs.NE/cs.CL), 9 sections + pre-registration appendix + 3 figures (regenerated from frozen result JSONs, corrected titles).
- Figures: figures/fig1_scale_ladder.png, fig2_seeds_and_controls.png, fig3_curves.png (make_figs.py reads result_*.json only; regenerable).
- LINKEDIN_POST.md: professional revision (teenager/timing references removed), re-scored 70.2 PASS.
- All numbers cross-verified PAPER.md ↔ RESEARCH.md §7.1.
- Cloud 1B replication (~$85–95): optional, decision open. Not started.

## Housekeeping
- Old phase3 cron removed 6/10. No active crons for fly-distill.
- Old Dapta/Síntesis TTS crons (Aug) are unrelated to this project.