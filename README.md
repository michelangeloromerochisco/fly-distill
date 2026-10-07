# fly-distill

Code, results, and paper for a controlled study of the larval *Drosophila*
connectome as the frozen recurrent substrate of a byte-level language model.

**Paper:** "Hub Concentration Masks the Benefit of Biological Connectome
Structure in Small Language Models" — see `PAPER.md` (source) / `PAPER.pdf`
(typeset). Pre-registered decision rules, control-matched conditions, four
seeds at 150M training bytes plus a two-seed hub-capping mechanism arm.
All experiments ran on a single consumer laptop GPU (NVIDIA RTX 3050, 4 GB),
~277 GPU-hours total.

## What is in this repository

| Path | Contents |
|---|---|
| `PAPER.md` / `PAPER.pdf` | The paper (Markdown source and typeset PDF, figures in `figures/`) |
| `RESEARCH.md` | Full lab notebook: methods, pre-registrations (written before data), phase reports, final results (§7.1) |
| `flylm.py` | Model + control graph construction (LIF dynamics, frozen substrate, degree-preserving shuffle, ER, weight-erase, hub cap) |
| `parse_larval.py` | Connectome parsing (Winding et al. 2023 → `larval_graph.npz`) |
| `analyze_graph.py` | Graph statistics (degree distributions, spectral radius, reachability) |
| `kd_flylm.py`, `teacher.py`, `probs_cache.py` | Knowledge-distillation arm |
| `run_phase1.py` … `run_phase3.py` | Idempotent runners with checkpoint/resume (each phase pre-registered in `RESEARCH.md` before launch) |
| `watchdog*.py` | Self-relaunching watchdogs (reboot-safe) |
| `make_figs.py`, `make_pdf.py` | Regenerate figures and the PDF from frozen results |
| `report_phase*.txt` | Auto-generated phase reports |
| `result_*.json` | Per-cell results with full training curves (every number in the paper is traceable to these files) |
| `larval_graph.npz`, `larval_manifest.json` | Parsed connectome (2,952 neurons, 38,401 edges at the ≥3-synapse threshold) |

Not included (regenerable, or too large / not results): training checkpoints
(`ckpt_*.pt`), the 20 GB teacher-probability cache, raw connectome
supplementary data, job logs, and the TinyStories training slice. See
"Reproducing" below.

## Reproducing

1. **Data.** Download `Supplementary-Data-S1.zip` from
   [brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome)
   and extract it so that `larval_raw/Supplementary-Data-S1/` exists, then run
   `python parse_larval.py`. Download the TinyStories training slice
   (Eldan & Li, 2023) as `tinystories_slice.txt` (~157 MB).
2. **Results.** `python run_phase1.py` (20M multi-seed + capacity arms),
   `run_phase2.py` (150M, seeds 0-1), `run_phase3.py` (seeds 2-3 + hubcap).
   Each runner is idempotent with checkpoint/resume; a single 4 GB GPU is
   sufficient but slow (~13 h per 150M job).
3. **Figures / PDF.** `python make_figs.py`, then `python make_pdf.py`.

## Data sources

- Connectome: Winding, Pedigo, Barnes, et al. (2023). *The connectome of an
  insect brain.* Science 379(6636). doi:10.1126/science.add9330.
  Supplementary Data S1, distributed via
  [brain-networks/larval-drosophila-connectome](https://github.com/brain-networks/larval-drosophila-connectome).
- Language data: Eldan, R. & Li, Y. (2023). *TinyStories: How Small Can
  Language Models Be and Still Speak Coherent English?* arXiv:2305.07759.

## License

Code and results: MIT. The parsed connectome in `larval_graph.npz` derives
from Supplementary Data S1 of Winding et al. (2023) and is redistributed for
reproducibility in accordance with the source repository's terms; the
original data and paper are cited above.