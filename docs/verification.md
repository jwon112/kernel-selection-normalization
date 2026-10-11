# Reproduction verification

## Frozen release v1.0.0

- Public code commit: `fe3e0780ae1ff18aace6ce794660e83c3fe58e34`.
- ZIP: `mathematics-reproducibility-v1.0.0.zip`.
- Size: 299,489,782 bytes; 543 ZIP entries.
- SHA256: `b21afe68c438b7749263ff0be7601b69bdf2dfe125cbd03ad3af0d658cdcbb21`.
- Public anonymous re-download matches the uploaded/local ZIP checksum.
- Full working-artifact build/check: passed.
- Working Mathematics tests: 48 passed.
- Tests in a clean extracted ZIP: 33 passed.
- Package manifest verification: 542 payload files matched.
- Numerical reanalysis: 10 patient-cluster intervals with 4,000 draws,
  180 primary stage cells, 540 normalization cells and 3,421,824 channel records
  reconstructed successfully.

[Machine-readable numerical report](2026-10-11-public-package-verification.json) ·
[Public deposit/download audit](2026-10-11-github-public-release.json).

These checks were initially run in the recorded working Python environment
against a clean extraction. They verify supplied mathematical probes and
recorded numerical outputs, not a new image-level inference or training run.
The scientific evidence is identical to the prior review snapshot; author and
delivery metadata are updated. Later documentation commits on `main` do not
change the v1.0.0 archive.

## Independent environment check

Completed on Linux with a **new Python 3.11 virtual environment without system
site packages**, using publicly installed NumPy 2.4.4, SciPy 1.17.1 and
**PyTorch 2.8.0+cpu**. A fresh extraction of the anonymous public ZIP passed:

- installation and `pip check`;
- all 542 payload hashes;
- mathematical probes, table checks, split checks and recorded numerical reanalysis;
- all 33 packaged tests.

[Environment metadata](clean-cpu-environment.json) ·
[CPU verification output](clean-cpu-verification.json) ·
[Installed dependency versions](clean-cpu-freeze.txt).

This is a separate environment on the same Linux host, not an independently
operated external machine. No image-level inference or training was rerun.
These additional verification reports are on `main`; the frozen archive is unchanged.

## Scope of reproduction

| Level | Status |
|---|---|
| Archive identity and anonymous download | Verified |
| Mathematical probes, generated tables and split checks | Verified |
| Reanalysis from recorded study and channel measurements | Verified |
| Full image preprocessing on a new host | Not independently rerun |
| Complete 100-epoch training and image inference on a new host | Not independently rerun |

Checkpoints, images and full feature tensors are intentionally not distributed.
Original source hashes identify provenance; they are not a claim that every
original working-tree artifact is present in this code repository.
