# Final Main and Supplementary Paper Figures

This is the standalone plotting bundle for the formal developmental-stage and
long-range-loop case-study figures. It reads copied, SHA-256-verified plotting
data and does not rerun UMAP, silhouette, imputation, loop calling, APA, or
held-out support calculations.

## Main figure

- **A:** Raw, scHiCluster, Higashi-nbr5, and scHiC-Diff UMAPs across E7.0–E11.5.
  Numbers are the stored stage-specific SVD15 Red/Blue silhouettes; the compact
  main panel omits the UMAP orientation glyph and Red/Blue legend.
- **B:** Raw, scHiCluster, Higashi-nbr5, scVI-3D, and scHiC-Diff contact maps
  and seed42 loop summits for 10, 100, and 476 aggregated early-neuron cells.
- **C:** held-out normalized APA for scHiCluster, Higashi-nbr5, scVI-3D, and
  scHiC-Diff at ≥600 kb, Top10/Top20/Top50. Each heat map displays the stored
  P2LL center value without the split SD; its shared color bar is vertical at
  the right edge.
- **D:** held-out raw-supported fraction for all six imputation methods.
  It is positioned below panel C; the scHiCluster series ends at Top100, with
  the limited available-N context reserved for the manuscript text.

## Supplementary figures

1. Complete seven-method × seven-stage UMAP grid.
2. Complete seven-method × four-depth contact-map/summit grid.
3. Complete three-Top-N × six-method held-out APA grid with center-only P2LL
   annotations.
4. Distinct summit and significant loop-pixel quantity diagnostics.
5. Held-out raw-supported counts among each method's all-eligible loop set.

## Reproduce all figures

```bash
cd /Users/wuhaoliu/Downloads/02_First_Review/results/7_caseData/nature-plot-new
MPLCONFIGDIR=/tmp/nature-plot-new-mpl \
/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python \
  src/plot_final_figures.py --data-dir data --out-dir outputs --dpi 600
```

Selective modes are `--main-only` and `--supplementary-only`. Output DPI below
300 is rejected.

## Tests

```bash
MPLCONFIGDIR=/tmp/nature-plot-new-mpl \
PYTHONPYCACHEPREFIX=/tmp/nature-plot-new-pycache \
PYTHONPATH=src:. \
/Users/wuhaoliu/mamba/envs/10_snaphic_env/bin/python \
  -m unittest discover -s tests -p 'test_*.py' -v
```

## Data and source-code provenance

- `data/` contains 16 copied formal plotting inputs.
- `data/copied_data_manifest.csv` records original absolute paths, sizes, and
  SHA-256 digests.
- `scripts/copy_frozen_data.py --verify-only` rechecks the copied data.
- `source_generation/` preserves 29 relevant generator/calculation scripts and
  configurations.
- `source_generation/source_code_manifest.csv` records their original paths and
  SHA-256 digests.
- `source_generation/SOURCE_CODE_MAP.md` maps every frozen plotting input to its
  direct generator and upstream calculation lineage.

The archived generators still require their original large model matrices,
cell subsets, Juicer Tools installation, and path-corrected configurations when
re-executed. Those external inputs are not needed to redraw the figures from
the frozen `data/` bundle.

## Output tree

```text
outputs/
├── main/             # composite A–D figure and metadata
├── panels/           # standalone A, B, C, and D
└── supplementary/    # Supplementary Figures 1–5
```

Every figure is exported as vector PDF, 600 dpi PNG, and SVG.

## Interpretation boundaries

- Stage-specific UMAP separation is not a claim of universal superiority across
  all developmental stages.
- Distinct summits and significant pixels measure output quantity, not loop
  accuracy.
- APA P2LL is aggregate center enrichment, not per-loop truth.
- Held-out raw support is internal split reproducibility, not an external bulk
  Hi-C ground truth or independent biological replication.
- The loop analysis covers one 2 Mb region, one cell type, and three cell
  partitions; SD is not a confidence interval.
- FLAMINGO loop results retain the documented canonical-row-order assumption
  because the source archive lacks cell names.
