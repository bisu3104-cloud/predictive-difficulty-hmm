# Experiment notebooks

These notebooks preserve the source code used for the principal Colab experiment workflows.
For public release, cell outputs, execution counts, and Colab account/execution metadata have
been removed; the scientific code cells are unchanged.

Notebooks 01--05 retain the original Google Drive project-path conventions because they are
provenance records of the experiment workflows. Edit the path constants before rerunning them
in a different Drive or local environment. The portable, data-to-result reconstruction scripts
under `../scripts/` are the recommended entry point for reproducing the reported statistical
tables from the archived public CSV/NPZ inputs.

`06_dataset_realization_robustness.ipynb` is already parameterized through environment variables
(`PREDICTIVE_CONFIRMATORY_ROOT` and `DATASET_REALIZATION_PROJECT_ROOT`) and can be resumed
without changing the scientific protocol.

The archived numerical results are stored under `data/` and `outputs/`; notebook output cells
are intentionally not used as the evidence record.
