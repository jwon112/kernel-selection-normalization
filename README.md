# Normalization-aware kernel-selection analysis

Reproducibility materials for **Functional Meaning of Kernel-Selection Weights:
Normalization, Branch Interventions, and Static Simplification in 3D Segmentation**.

**Jaewon Cheon†, Hee Kuk†, Zeeshan Abbas*, Seung Won Lee*.**
† Shared first authorship. * Co-corresponding authors.
Manuscript prepared for *Mathematics*; no acceptance or publication DOI is asserted.

[Paper PDF](manuscript/mathematics/main.pdf) ·
[Versioned package](https://github.com/jwon112/kernel-selection-normalization/releases/tag/v1.0.0) ·
[Detailed reproduction guide](docs/reproduction.md) ·
[Verification report](docs/verification.md) ·
[Citation metadata](CITATION.cff)

## What this study examines

Learned kernel-selection weights have different meanings under branch removal,
renormalized removal, forced single-path inference, and independently trained
compact models. The study connects finite-epsilon normalization conditions to
controlled numerical probes and stage-level observations in 3D segmentation.
It does not establish a generally optimal compression schedule or a new
state-of-the-art segmentation model.

## Download and verify

The Git clone provides code, configs, manuscript and summary evidence. The
**full numerical records and package manifest are in the release ZIP**.
Run the verifier in a clean extracted archive, not in the Git clone.

```bash
mkdir kernel-reproduction
cd kernel-reproduction
curl -fL -O https://github.com/jwon112/kernel-selection-normalization/releases/download/v1.0.0/mathematics-reproducibility-v1.0.0.zip
curl -fL -O https://github.com/jwon112/kernel-selection-normalization/releases/download/v1.0.0/mathematics-reproducibility-v1.0.0.zip.sha256
sha256sum -c mathematics-reproducibility-v1.0.0.zip.sha256
python3 -m zipfile -e mathematics-reproducibility-v1.0.0.zip package
cd package
python3 verify_package.py
```

`sha256sum` is a Linux command; on macOS use `shasum -a 256 -c`.
The archive contains **543 files / 299,489,782 bytes**. Expected SHA256:

```text
b21afe68c438b7749263ff0be7601b69bdf2dfe125cbd03ad3af0d658cdcbb21
```

### CPU reanalysis without PyTorch or a GPU

From the extracted `package/` directory, using Python 3.11:

```bash
python3 -m venv ../reanalysis-env
. ../reanalysis-env/bin/activate
python -m pip install numpy==2.4.4 scipy==1.17.1
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_package.py --recalculate
```

This recomputes recorded-output bootstrap intervals, selector associations and
normalization/channel statistics. Allow a few minutes and approximately 1 GB
of RAM for reanalysis, excluding installation/download overhead. The environment
is outside the package so that manifest verification can detect unlisted files.

### Full CPU probes, table checks and packaged tests

In that same environment, install the CPU PyTorch wheel and plotting packages:

```bash
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install matplotlib==3.11.1 nibabel==5.4.2 thop==0.1.1.post2209072238
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python verify_package.py --scientific
PYTHONPATH=src:manuscript/mathematics/tools OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -p 'test_*.py'
```

These shell examples target Linux/macOS; they have been exercised on Linux.
`requirements-paper.txt` records the original CUDA training environment and is
not the CPU-only installation command. The optional CPU requirement selections
in this Git repository are also documented in the [installation guide](docs/reproduction.md).

## Find the evidence

| Question/result | Supplied evidence | Reproduction entry point |
|---|---|---|
| Finite-epsilon identity and selector probes | `generated/evidence.json`, `normalization.pdf`, `interventions.pdf` | `verify_package.py --scientific` |
| Weight/removal versus forced-path associations | `stage.tex`, `within_stage.tex`, `stage_sensitivity.tex` | `verify_package.py --recalculate` |
| Paired-removal differences and low-variance exceptions | `normalization_checkpoint.tex`, `normalization_frequency.tex`, `normalization_channels.tex` | `verify_package.py --recalculate` |
| Static schedules, uncertainty and denominators | `schedules.tex`, `contrasts.tex`, `fp_contrasts.tex`, `denominators.tex` | `verify_package.py --recalculate` |
| Numerical-record schemas and source hashes | `analysis_data/manifest.json` in the ZIP | `recalculate.py` |
| Recorded patient-grouped split | `data_contract/splits.json` | `verify_package.py --scientific` |

The `generated/` files above are under `manuscript/mathematics/generated/`.
See the [verification report](docs/verification.md) for counts and boundaries.

## Image-level training and interventions

GPU reruns require authorized access to BraTS 2023 GLI. Images, preprocessed
images, trained checkpoints and full feature tensors are not redistributed.
The supplied study has 1,251 studies / 1,133 patient groups and a five-fold,
patient-grouped split. Primary stage replication uses three training seed groups;
static-schedule comparisons remain single-base-seed exploratory evidence.

[Preprocessing, training, intervention commands and path adaptation](docs/reproduction.md#3-recreate-images-train-models-and-rerun-interventions-separate-gpu-work).
Historical configs retain original paths; adapt filesystem locations while
preserving split membership, budget, seeds and scientific settings.
CPU reanalysis is not independent image inference or retraining, and a full
image-level rerun has not been demonstrated on a new external machine.

## Versioning, citation and questions

Use the [v1.0.0 release](https://github.com/jwon112/kernel-selection-normalization/releases/tag/v1.0.0)
for the frozen manuscript package. Later README/guide improvements on `main`
do not replace that archive. Cite the manuscript title/authors and release
version; machine-readable metadata is in [CITATION.cff](CITATION.cff).
For reproduction problems, [open an issue](https://github.com/jwon112/kernel-selection-normalization/issues).

## Reuse and third-party material

No blanket open-source license has been assigned to the research code. Public
availability does not itself grant unrestricted reuse or redistribution rights.
MDPI template assets and third-party software retain their existing terms;
BraTS access remains subject to the provider's terms. See the original notices
under `manuscript/mathematics/Definitions/`.
