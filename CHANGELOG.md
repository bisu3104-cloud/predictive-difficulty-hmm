# Changelog

## v0.5.0 release metadata — 2026-08-19

- Added GitHub repository URL `https://github.com/bisu3104-cloud/predictive-difficulty-hmm`.
- Added reserved Zenodo DOI `10.5281/zenodo.22004946`.
- Updated `CITATION.cff`, README, and Data and Code Availability text for the public release.
- Scientific data, run-level results, and analysis code were not changed.

## v0.5 release-candidate cleanup — 2026-08-19

- Removed notebook outputs and Colab account/execution metadata before public release.
- Added notebook provenance guidance and a public-release checklist.
- Added `.gitignore` for repository hygiene.
- Removed invalid placeholder DOI/repository fields and the premature release date from `CITATION.cff`; real values will be added after GitHub/Zenodo setup.
- Scientific data, code cells, run-level results, and analysis outputs were not changed.

## v0.5 — 2026-08-05

- Added explicit mixed licensing: MIT for software/code and CC BY 4.0 for data, derived outputs, figures, and documentation.
- Added the complete dataset-realization robustness evidence base used in manuscript Section 5.4 and Appendix H.
- Added the deterministic 16-HMM subset reconstruction, public selection lock, and protocol.
- Added 1,152 run-level results across three dataset realizations, plus width-level and HMM-level summaries.
- Added dataset split seeds and SHA-256 checksums for the two regenerated dataset replicates per HMM.
- Added HMM-level process ICC, dataset-pair rank stability, PCA stability, frozen-profile performance, and supplementary variance-component tables.
- Added manuscript Tables 24 and 25 under `outputs/paper_tables/`.
- Added resumable notebook `06_dataset_realization_robustness.ipynb`, public runner, and standalone reconstruction script.
- Extended artifact validation, data dictionary, schema, reproducibility matrix, source-archive notes, and manifest.
- Updated citation metadata to version 0.5.0; release identifiers were added in the 2026-08-19 release-metadata update above.

## v0.4 — 2026-08-04

- Added `code/scripts/reconstruct_seed_aware_variance.py`.
- Added point estimates, variance components, condition scaling, 10,000 ordinary and K-stratified bootstrap replicates, metadata, and a result summary under `outputs/variance_components/`.
- Reproduced interaction shares of 0.187929% for Shape RMSE and 16.502929% for terminal excess CE.
- Defined a fully reproducible HMM-cluster bootstrap yielding 95% intervals of 0.0%–8.5% and 2.7%–28.4%, with K-stratified sensitivity intervals also archived.
- Replaced the fixed-alpha Table 4 ridge reconstruction with the nested LOHO penalty-selection procedure in the manuscript.
- Reproduced the updated full-ridge Table 4 values: standardized RMSE 0.720, pooled R² 0.482, and mean Spearman 0.733.
- Updated README, citation version, reproducibility matrix, data documentation, and artifact manifest.


## v0.3 — 2026-08-03

- Added `code/scripts/reconstruct_confirmatory_table4_loho.py`.
- Added the five confirmatory LOHO reconstruction outputs under `outputs/reconstructed_table4/`.
- Reproduced all three main Table 4 metrics for categorical \(K\)-only, PLS-1, PLS-2, and full ridge from the saved confirmatory data.
- Confirmed that all predictor preprocessing, response standardization, and model fitting are performed within each 63-HMM training fold.
- Documented `alpha=16.0` as the explicit reconstructed full-ridge penalty that reproduces the reported rounded values, and included an alpha-sensitivity table.
- Updated README, citation version, reproducibility matrix, and artifact manifest.

## v0.2 — 2026-08-02

- Added `code/scripts/reconstruct_external_pls_transfer.py`.
- Added `outputs/paper_tables/external_pls_composite_transfer.csv`.
- Added `outputs/analysis_metadata/external_pls_composite_transfer_metadata.json`.
- Confirmed that PLS is trained on eight confirmatory neural Shape conditions and external RMSE is evaluated on the six architecture-by-width conditions shared by the confirmatory and external cohorts.
- Reproduced categorical K-only RMSE 0.011918, PLS-1 RMSE 0.010346, and PLS-2 RMSE 0.010291.
- Updated README, citation version, and reproducibility matrix.
