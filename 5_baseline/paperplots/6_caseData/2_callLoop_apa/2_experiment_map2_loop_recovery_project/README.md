# Map2 Known Loop Recovery

This project measures recovery of a known Map2 enhancer-promoter loop in scHi-C contact matrices.

## Inputs

Expected matrix layout:

```text
data/
├── raw/
│   ├── Map2_raw_n3.npy
│   └── ...
├── scHiCluster/
│   ├── Map2_scHiCluster_n3.npy
│   └── ...
└── scHiCDiff/
    ├── Map2_scHiCDiff_n3.npy
    └── ...
```

Known loop BEDPE:

```text
chr1 65800000 65820000 chr1 66020000 66040000
```

Region file:

```text
chr1
65000000
67000000
```

## Run

```bash
python scripts/main.py \
  --data-dir data \
  --loop-bedpe data/Map2_known_loop.bedpe \
  --region-file data/region.txt \
  --resolution 20000 \
  --output-dir results
```

## Outputs

```text
results/
├── analysis.log
├── loop_bin_pair.csv
├── P2LL_results.csv
├── enrichment_results.csv
├── Map2_known_loop_P2LL.pdf
└── Map2_known_loop_enrichment.pdf
```

## Notes

- Matrices with `max(matrix) < 20` are treated as already log-transformed.
- Other matrices are transformed with `log1p(matrix)` before scoring.
- `center_signal` is the mean of a centered 5x5 window.
- `background_signal` is the mean of a surrounding 20x20 window with the center window removed.
- This tool intentionally does not infer a known enhancer-promoter loop automatically. If no known loop is available, provide a candidate BEDPE from a separate loop-calling workflow.
