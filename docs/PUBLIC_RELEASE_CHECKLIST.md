# Public release checklist

## Completed before final public release

- [x] Artifact validation passes.
- [x] 184 selected HMM NPZ files pass probability/stationarity checks.
- [x] Dataset-realization grid contains 1,152 unique completed neural runs.
- [x] No deprecated experimental archives are mixed into the evidence base.
- [x] No GitHub-size-problem files are present (largest tracked file is below 5 MB).
- [x] Notebook outputs, execution counts, Colab user IDs, display names, authorship tags, and mount-file IDs are removed.
- [x] `CITATION.cff` contains no invalid placeholder DOI or repository URL.
- [x] Git hygiene file (`.gitignore`) is present.

## Must be completed before making the repository public / publishing Zenodo

- [x] Choose the code license: MIT.
- [x] Choose the data/results license: CC BY 4.0.
- [x] Add explicit license files and a licensing scope statement to the README.
- [x] Create the GitHub repository and record its URL (`https://github.com/bisu3104-cloud/predictive-difficulty-hmm`).
- [x] Create a Zenodo draft and reserve DOI `10.5281/zenodo.22004946`.
- [x] Add the GitHub URL and reserved Zenodo DOI to `CITATION.cff`.
- [x] Replace DOI/repository placeholders in README and data-availability documentation.
- [ ] Replace the DOI placeholder in the manuscript with `10.5281/zenodo.22004946` and add the GitHub URL if desired.
- [x] Re-run `python code/scripts/validate_artifact.py` after release-metadata updates.
- [ ] Create the final GitHub tag/release.
- [ ] Upload the exact final release archive to the reserved Zenodo record and publish it.

## Recommended DOI workflow for this paper

Use a **manual Zenodo draft** for the first archival release: create the GitHub repository first,
then create a Zenodo draft and use **Get a DOI now** to reserve a DOI. Insert that reserved DOI
into the repository files and manuscript before creating the final GitHub release. Upload that
exact release archive to the same Zenodo draft and publish it. This avoids a circular metadata
problem and avoids accidentally creating a second Zenodo DOI through automatic GitHub-release
archiving.
