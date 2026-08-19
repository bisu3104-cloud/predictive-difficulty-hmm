# Predictive Difficulty Beyond Hidden-State Cardinality — Public Artifact v0.5.0

This package is the **curated source-data and code artifact (version 0.5.0)** for the manuscript
*Predictive Difficulty Beyond Hidden-State Cardinality: A Shared Process Component and Learner-Specific Responses in Controlled Hidden Markov Models*.

## What is included

- Selected HMM parameters \(T,O,\pi\) for the development, confirmatory, predictor-specific, strict-replication, and external-generator cohorts.
- Run-level and aggregate CSV files from the completed neural and count-predictor experiments.
- Selection records, protocol files, and frozen prediction outputs.
- Six principal Colab notebooks, cleaned of execution outputs and account metadata, including the dataset-realization robustness experiment.
- Validation and deterministic reconstruction scripts for development HMMs, confirmatory Table 4, external PLS transfer, seed-aware variance components, and dataset-realization robustness.

## Cohorts and supplementary robustness experiment

| Cohort / analysis | HMMs | Neural runs | Included |
|---|---:|---:|---|
| Development | 24 | 2,688 | HMMs reconstructed from fixed generator records; run-level and aggregate results |
| Confirmatory | 64 | 2,560 | Selected HMMs, lock, frozen prediction tables, cross-\(K\) results, nested-LOHO composite reconstruction, and seed-aware variance analysis |
| Predictor-specific | 48 | 1,440 | Selected HMMs, 24 matched pairs, neural and count results |
| Strict replication | 16 | 480 | Selected HMMs, 8 pairs, strict contrast results |
| External generators | 32 | 576 | Selected HMMs, generator protocol, frozen transfer results |
| Dataset-realization robustness | 16 | 1,152 total (768 newly trained) | Three dataset realizations, selected-HMM lock, split seeds/checksums, run-level results, ICC/rank/PCA/frozen-prediction analyses, and manuscript Tables 24–25 |

## Version 0.5 additions

Version 0.5 adds the complete public evidence base for Appendix H and Section 5.4:

- deterministic reconstruction of the outcome-blind 16-HMM subset;
- the public selection lock and experiment protocol;
- 1,152 run-level neural results across three dataset realizations;
- split seeds and checksums for the two regenerated datasets per HMM;
- HMM-level rank stability, process ICC, PCA stability, and frozen-profile results;
- manuscript Tables 24 and 25;
- the resumable Colab runner and a standalone analysis reconstruction script.

The regenerated sequence arrays themselves are not included. They can be reproduced exactly from the included HMM parameters, recorded split seeds, and public runner. The original replicate-0 dataset cache was unavailable, but its completed run summaries are included.

## Quick validation

```bash
python code/scripts/validate_artifact.py
```

Expected result: zero validation problems, 184 selected HMM parameter files checked, and 1,152 complete dataset-realization runs.

## Dataset-realization analysis reconstruction

```bash
python code/scripts/reconstruct_dataset_realization_analysis.py
```

Expected rounded primary results:

| Architecture | Outcome | Process ICC | Dataset-pair Spearman range |
|---|---|---:|---:|
| GRU | \(A_{\mathrm{width}}\)-Shape | 0.794 | 0.765–0.874 |
| Transformer | \(A_{\mathrm{width}}\)-Shape | 0.799 | 0.750–0.838 |
| GRU | Mean terminal excess CE | 0.550 | 0.479–0.500 |
| Transformer | Mean terminal excess CE | 0.807 | 0.615–0.756 |

The frozen augmented profile reduces Shape RMSE relative to \(K\)-only in all six architecture-by-dataset comparisons, by 27.0%–46.4%.

## Other reconstructions

```bash
python code/scripts/reconstruct_development_hmms.py
python code/scripts/reconstruct_confirmatory_table4_loho.py
python code/scripts/reconstruct_seed_aware_variance.py
python code/scripts/reconstruct_external_pls_transfer.py
```

## Data path sanitization

Original Google Drive paths in published CSV files have been replaced by `project://`, `drive://`, `regenerated://`, or `not-archived://` provenance markers. They are not executable paths.

## Licensing

This repository uses a mixed-license arrangement:

- **Software and code** under `code/` are licensed under the **MIT License**; see `LICENSE`.
- **Data, derived tables, figures, and documentation** under `data/`, `outputs/`, and `docs/` are licensed under **Creative Commons Attribution 4.0 International (CC BY 4.0)**; see `LICENSE-DATA`.
- The manuscript itself is **not** included under these repository licenses; its publication license is governed separately by the journal/publisher.

If a particular third-party file states its own license, that file-specific license takes precedence.

## Release identifiers

- GitHub repository: `https://github.com/bisu3104-cloud/predictive-difficulty-hmm`
- Zenodo DOI: `10.5281/zenodo.22004946`
- Version: `0.5.0`

The scientific artifact is complete, passes validation, and has explicit code/data licenses.
`CITATION.cff` contains the repository URL and reserved Zenodo DOI for this release.
The DOI becomes registered when the corresponding Zenodo draft is published.

The Colab notebooks are preserved as experiment-provenance source code. Outputs and user-specific
execution metadata have been removed; see `code/notebooks/README.md`.
