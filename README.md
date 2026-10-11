# Normalization-aware kernel-selection analysis

Reproducibility materials for **Functional Meaning of Kernel-Selection Weights:
Normalization, Branch Interventions, and Static Simplification in 3D Segmentation**.

**Authors:** Jaewon Cheon and Hee Kuk (shared first authorship), Zeeshan Abbas
and Seung Won Lee (co-corresponding authors).

This is a research manuscript prepared for submission to *Mathematics*, not an
accepted publication. No journal acceptance or DOI is asserted.

## Start here

- **Read the manuscript:** [paper PDF](manuscript/mathematics/main.pdf).
- **Download the complete frozen reproduction package:**
  [v1.0.0 release](https://github.com/jwon112/kernel-selection-normalization/releases/tag/v1.0.0).
- Download both `mathematics-reproducibility-v1.0.0.zip` and its `.sha256` sidecar.
- Verify the ZIP checksum, extract into a new directory, then run there:

```bash
python verify_package.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_package.py --scientific --recalculate
PYTHONPATH=src:manuscript/mathematics/tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

Install the recorded versions in `requirements-paper.txt` in your own environment
before the numerical commands. See the detailed guide below for hardware and
PyTorch wheel selection. Do not install the entire historical pip freeze.

The Git repository contains the paper-specific code, settings, summaries, and
manuscript. **The full numerical records and package manifest are in the release
ZIP**, so run the package verifier on a clean extracted ZIP, not on this Git clone.
The archive is the frozen reference for the paper; the default branch may evolve.

## What can be reproduced

CPU reanalysis reconstructs recorded patient-cluster bootstrap intervals,
selector associations, stage sensitivities, normalization comparisons and
channel distributions. It does not independently reproduce image inference or
training. GPU regeneration instructions are provided separately below.

BraTS images, preprocessed images, trained checkpoints and full feature tensors
are not redistributed. Obtain BraTS under the provider's terms. De-identified
challenge study IDs and the recorded patient-grouped split are included in the
full reproduction package. Existing third-party notices remain applicable.

## Detailed reproduction guide

# Mathematics paper-specific reproducibility package

Paper: **Functional Meaning of Kernel-Selection Weights: Normalization,
Branch Interventions, and Static Simplification in 3D Segmentation**.
Authors: Jaewon Cheon and Hee Kuk (shared first authorship), Zeeshan Abbas
and Seung Won Lee (co-corresponding authors).

This paper-specific package supports peer review and numerical reproduction.
The exporter creates an archive locally; running it does not publish files.
The maintained code repository is
`https://github.com/jwon112/kernel-selection-normalization`. Download the complete
numerical package and checksum from
`https://github.com/jwon112/kernel-selection-normalization/releases/tag/v1.0.0`.
Deposition and download verification are recorded separately. The full working research
repository is unnecessary for the supported reproduction steps.

## Included material and verification levels

- Paper PDF, LaTeX source, bibliography and required MDPI template assets.
- Dated claim-to-source reference audit, review-response source checks and their
  reviewed-retrieval query plans;
  these record checked support and scope, not an exhaustive novelty search.
- Exact training, augmentation, model, profiling and intervention source files;
  paper-specific configs and the original dated protocol snapshots.
- Recorded Python environment and a six-package numerical/training/profiling requirement
  selection. CUDA wheel availability and hardware compatibility must be checked
  on the receiving host; this package does not install anything automatically.
- Recorded de-identified BraTS study split membership and cohort-level metadata.
  These are not images, patient identities or permission to access the dataset.
- Frozen analysis summaries, stage/normalization cells, tables, figures and source
  hashes. The summaries retain both favorable and unfavorable outcomes.
- Original study-level metric/intervention/selector CSVs used by the paper,
  paired normalization case scores, and losslessly packed channel measurements.
  The numeric records preserve longitudinal patient linkage through the
  existing de-identified BraTS challenge IDs, not names or new clinical data.
- SHA256 manifest and a CPU-only verification/rebuild entry point.

No raw or preprocessed images, large trained checkpoints, full channel JSONL
logs or feature tensors, third-party paper PDFs, credentials, Git metadata, other manuscripts
or ongoing unrelated experiments are included. Generic shared modules retain
optional constructors/helpers, but only the paper-configured kernel models and
their dependencies are supported here. In particular, other architectures'
vendor implementations and datasets are not bundled.

The manifest distinguishes original source hashes from package-file hashes.
Two shared reporting modules are exported using exact source spans of their
required definitions only; their original and exported hashes are both recorded.
Source paths in the evidence JSON identify original provenance, not a promise
that every referenced file is distributed. Historical protocol `status` fields
are preserved; the supplied completed evidence, not an old protocol status, is
the reporting snapshot. The complete pip freeze documents the original broader
environment; it is not the minimal installation command.

The manifest also records exact agreement of the supplied trainer, selector,
model, augmentation, reporting and stage-intervention code with the primary
study's archived source snapshot. This does not imply that historical
independently fitted checkpoints are bitwise reproducible on a new machine.

## 1. Verify file identity (Python standard library only)

Extract the ZIP into an empty directory, then run from its root:

```bash
python3 verify_package.py
```

The adjacent `.zip.sha256` file verifies the ZIP before extraction. SHA256 detects
changes against a supplied manifest; it is not a digital signature or independent
validation of scientific correctness. Keep any locally generated outputs outside
the extracted package so that unlisted-file detection remains useful.

## 2. Verify mathematics, summary analyses and table generation (CPU)

Use an environment with the versions in `requirements-paper.txt`. The study used
Python 3.11 and PyTorch 2.8.0 with CUDA 12.8; the CPU verification does not require
a GPU. Do not install the unrelated editable packages in the complete freeze.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_package.py --scientific
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_package.py --recalculate
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_package.py --rebuild-to ../math-rebuilt
PYTHONPATH=src:manuscript/mathematics/tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

These commands rerun finite-epsilon/derivative/selector probes, stage-rank
sensitivities, normalization exceedance frequencies and split-isolation checks,
and compare every generated table/summary JSON with the supplied snapshot.
Rebuild writes tables and figures into a new directory without changing the
delivered evidence. Figure PDF bytes may differ with PDF metadata/font versions.

`--scientific` is the quick summary-to-table check. **`--recalculate` instead
starts with the supplied numerical records**, recomputing the ten primary and
all-case patient-cluster bootstrap intervals (4,000 shared draws), all primary
and historical stage associations, stage sensitivities, pooling results,
fivefold model scores, 540 normalization endpoints and nine channel-distribution
groups, including case-linked extrema and the finite-epsilon ratio identity.
It checks every reconstructed value against the manuscript evidence with
explicit numerical tolerances and rejects duplicate/missing identities.
This is recorded-output reanalysis, not independent image inference or training.
It needs only NumPy/SciPy for numerical calculations, uses one process, and
does not initialize CUDA. Allow a few minutes and approximately 1 GB of RAM.

`analysis_data/manifest.json` describes source/output hashes, study indices,
CSV mappings and the little-endian packed binary schema. Each `.bin` holds at
most 65,536 records, with unsigned identity/flag fields and float64 measurements;
the field layout is `recalculate.CHANNEL_DTYPE`. JSON null is preserved as NaN
and omitted from the corresponding quantile exactly as in the original builder.
There are no object arrays or pickled objects. A file can be read directly using
`numpy.fromfile(path, dtype=recalculate.CHANNEL_DTYPE)`; stage, seed, fold and
precision mode are identified by its manifest entry. Only columns needed for
reported numerical analyses are exported from the full channel logs.

`tools/build.py` remains the
original complete-artifact builder and requires the separately retained original
result tree, checkpoints and literature sources; running it on this package alone
is not supported. `--check` on that original builder is not the package verifier.

## 3. Recreate images, train models and rerun interventions (separate GPU work)

Obtain the official BraTS 2023 GLI training archive under its provider's terms.
The paper uses union-nonzero anatomical cropping with margin 4, then padding only
to 192 × 224 × 192, nonzero per-modality z-score clipped to [-5,5], and channel
order t2f/t1n/t1c/t2w. Labels never determine crop bounds.

```bash
python prepare_gli.py --archive /path/to/asnr-miccai-brats2023-gli-challenge-trainingdata.zip --cache-out data/rebuilt/gli --benchmark-out data/processed/brats2023_gli_crop4
```

The GLI-only adapter reuses unchanged crop, normalization, loading, writing and
patient-grouped split functions; it avoids requiring MEN/PED/2024 datasets that
were present in the historical preprocessing launcher. It refuses existing
outputs and compares regenerated split membership with the recorded snapshot.
This adapter has not itself reprocessed the full image archive as a new study;
CPU synthetic crop and split checks are separate from full-data replication.

Example for one of the registered fifteen primary stage banks:

```bash
CUDA_VISIBLE_DEVICES=0 SEG3D_CPU_THREADS=2 python src/train_run.py --config configs/kernel_stage_fivefold_twoseed_v1/seed2026.json --fold 0 --attempt-id independent_replication
```

Repeat folds 0–4 for base seeds 2026, 20261001 and 20261011 using their supplied
configs. The trainer adds the fold to the base seed. Preserve terminal epoch
99 (100 epochs), bf16, batch size 4, augmentation and the other scientific
settings. Use distinct attempts/results directories; never substitute a best
checkpoint. A new hardware/runtime execution is not expected to match checkpoint
bytes. Static-schedule and pooling comparisons use their individual supplied
configs and all five folds, but remain single-base-seed studies.
Some historical configs retain absolute data/results paths. Adapt only these
filesystem locations to the receiving host and record the path changes;
do not change scientific settings, split membership or the base seed.

For a completed fold directory containing config.json, run.json and latest.pt:

```bash
python src/evaluate_stage_selector_interventions.py /path/to/fold_0 --intervention drop --output-dir /new/drop
python src/evaluate_stage_selector_interventions.py /path/to/fold_0 --intervention drop --renormalize --output-dir /new/renormalized-drop
python src/evaluate_stage_selector_interventions.py /path/to/fold_0 --intervention keep --output-dir /new/keep
python src/evaluate_selector_interventions.py --help
python src/evaluate_kernel_normalization.py --help
```

The normalization evaluator additionally requires original paired drop/reroute
case CSVs and the recorded outcome-independent sensitivity selection. Its
selection rule and precision/batch protocol are in the supplied protocol and
analysis source. The exact selection JSON is supplied at
`data_contract/normalization_sensitivity_selection.json`; the executed plans
under `protocols/` record the original bank/attempt grid. The CPU verifier
checks the sensitivity selection against the outcome-independent hash rule.
Full image-level raw-artifact reconstruction needs those external
artifacts; their hashes are recorded but their delivery is not promised by this
package. Contact the corresponding authors if additional artifacts are
needed, subject to availability and applicable data-use conditions.

## Build/update the package (authors, working checkout only)

First regenerate/check the Mathematics evidence and PDF in the working checkout.
Export numerical records into the ignored `analysis_data/` folder (exclusive
create; preserve an older export separately before making a new one). Changes
only to manuscript provenance hashes do not invalidate an otherwise identical
scientific snapshot. Then run its exporter with a NEW archive filename:

```bash
python manuscript/mathematics/tools/export_analysis_data.py --out manuscript/mathematics/reproducibility/analysis_data
python manuscript/mathematics/tools/package_reproducibility.py --out manuscript/mathematics/reproducibility/dist/mathematics-reproducibility-20261011-review.zip
```

The exporter rejects changed hash-tracked sources, symlinked/oversized files,
unsafe paths, duplicate entries, and checkpoint/image/JSONL extensions. It only
reads explicitly selected sources; it never scans the whole repository for
arbitrary files. ZIP entries have fixed timestamps and sorted paths. Preserve
the archive and checksum together, run all three verification levels after extraction,
and upload only the final author-approved paper package through the chosen
review/supplementary channel. Until that upload, the manuscript states that the
package is prepared and held locally. After upload, replace that status sentence
with the actual supplementary-file designation and access location, and check
that the supplied archive name and checksum match the uploaded file.

Third-party MDPI template assets retain their original notices. This package
does not grant rights to BraTS or third-party software beyond their own terms,
and no new blanket open-source license is assigned to the research project.
