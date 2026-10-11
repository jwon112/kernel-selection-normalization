# Mathematics reference recheck — 2026-10-10

## Scope and outcome

Rechecked the current Mathematics manuscript after completion of the fifteen
normalization banks, five-fold extensions, effect-frequency and stage-sensitivity
analyses. All 20 bibliography entries are cited and rendered; no missing,
duplicate-key or orphaned entries were found. No checked citation contradicts
the revised results. The current evidence does not require adding references
solely to increase their count.

Three precision edits were made:

1. Byra's approximately 0.7 coefficients refer to the early expansion-path SK
   blocks, not a single pooled statistic over the entire expansion path.
2. BraTS source citations identify the challenge/benchmark; the study/patient
   counts are explicitly attributed to our data manifest.
3. ROAR's legacy PDF URL is replaced by the current official proceedings record.

No equation, experimental endpoint, numerical result or contribution claim is
changed. The finite-epsilon deductions and measured decoder-specific effects
remain this study's analysis, not results attributed to the cited originals.

## Claim-to-source checks

Locators below refer to the original articles, not to this manuscript's page
numbers. The primary-source reviews dated October 1 and October 5 remain part
of the audit trail; they were not overwritten.

| BibTeX key | Original locator / source | Supported use and relevant boundary |
| --- | --- | --- |
| li2019sknet | SKNet, §3 and §4.3–4.4, Tables 7–8 | Channel-wise branch selection and scale response; not evidence for our post-fusion InstanceNorm or patient-level optimality. |
| ronneberger2015unet | U-Net, §2, Figure 1; DOI 10.1007/978-3-319-24574-4_28 | Encoder–decoder architectural precedent; our 3D implementation and settings are independently specified. |
| jain2019attention | §4.1–4.2 and Discussion; DOI 10.18653/v1/N19-1357 | Attribution comparisons and alternative attention distributions; token-attention findings do not determine kernel-mixture behavior. |
| wiegreffe2019attention | §3–4; DOI 10.18653/v1/D19-1002 | Explanation targets, seed calibration and model-consistent controls; not a universal endorsement of attention interpretation. |
| serrano2019attention | Intervention experiments and Discussion; DOI 10.18653/v1/P19-1282 | Attention magnitude versus prediction changes under manipulation; not medical convolutional replication. |
| ulyanov2016instance | §2; arXiv:1607.08022 | Per-instance/channel spatial normalization and contrast removal; our finite-epsilon paired-removal identity is a deduction from the definition. |
| bakas2023cluster | Zenodo record, DOI 10.5281/zenodo.7837974 | BraTS 2023 challenge provenance; not the source of our locally verified repeated-study counts. |
| menze2015brats | Benchmark article; DOI 10.1109/TMI.2014.2377694 | Foundational BraTS benchmark; not the 2023 cohort release or its exact counts. |
| kobayashi2020norm | §3–4; DOI 10.18653/v1/2020.emnlp-main.574 | Attention together with transformed-vector norms; distinct from a normalized spatial convolutional mixture. |
| kobayashi2021normalization | Block analysis incorporating residuals and normalization; DOI 10.18653/v1/2021.emnlp-main.373 | Operator-aware Transformer attribution; not our InstanceNorm identity or its measured Dice effects. |
| byra2020skunet | §2.4; p. 6, Figure 7 and following paragraph; DOI 10.1016/j.bspc.2020.102027 | Sample attention–mass-size association, approximately 0.7 for early expansion blocks; not necessity, forced-path competence or independent compact-model performance. |
| chen2020dynamic | §3; p. 5 / printed p. 11034, §5.2, Table 3 | Simplex kernel aggregation; averaging, maximum-attention kernel and shuffled-attention interventions. These are precedents, not identical single-stage candidate edits. |
| yang2019condconv | §2, Figure 2, kernel-assembly equation | Input-conditioned kernel assembly and sigmoid routing; simplex assumptions cannot automatically be transferred to it. |
| chen2024esknet | Architecture/method sections; DOI 10.1016/j.eswa.2024.123265 | Adaptive kernel selection with recalibration, residual learning and deep supervision; application evidence is not this study's removal analysis. |
| shi2026sgdc | Method sections; PMLR 315:989–1003 | Supervised structural conditioning and spatial dynamic kernels/gates; operator differs from spatially constant per-channel mixtures. |
| vanlaarhoven2017normalization | pp. 1–3, §1–3; arXiv:1706.05350 | Parameter rescaling and effective learning rates under normalization; not the finite-epsilon fixed-checkpoint removal relation. |
| michel2019heads | pp. 3–4, §2.3, §3.2–3.3 | Single-head retention, masking and pruning; frozen-path performance is not independent compact-model competence. |
| hooker2019roar | pp. 1–4, §1 and §3 | Input-pixel removal followed by retraining addresses altered-input confounding; our compact schedules alter internal architecture and are not implementations of ROAR. |
| adebayo2018sanity | pp. 1–2, §1 | Parameter/label randomization as explanation checks; these tests were not performed here and are not claimed as results. |
| wu2018groupnorm | §3.1, Equations (1), (2), (7); arXiv:1803.08494 | Spatial/channel-group statistics and the one-channel-per-group IN limit; common/unequal-scale algebra and Jacobian are our deductions. |

## Source and metadata provenance

Local original PDFs/text were checked alongside the dated source-review records:

- `literature/MATHEMATICS_KERNEL_SOURCE_REVIEW_2026-10-01.md`
- `literature/MATHEMATICS_REFERENCE_EXTENSION_2026-10-05.md`
- `generated/metadata_reference_audit_20261006.json` (historical 19-reference audit)

Official ACL Anthology, NeurIPS and PMLR records, arXiv originals and current
Crossref/DataCite metadata were used for online corroboration. Current metadata
checks confirmed Byra, ESKNet, Menze, U-Net and the BraTS challenge record;
the five ACL records and SGDC record confirmed their bibliographic details.
Several CVF/publisher webpage requests were blocked; those checks rely on the
available original documents and dated metadata audit, not a claimed fresh
successful publisher-page fetch. Published ESKNet title/year are retained rather
than replaced by its differently titled, earlier arXiv preprint. Instance
Normalization's first-posting year 2016 is retained despite later revisions.
The corrected SGDC author initial `Zhu, W.-P.` remains intact.

Useful primary records:

- Byra: https://doi.org/10.1016/j.bspc.2020.102027
- BraTS: https://doi.org/10.5281/zenodo.7837974
- GN: https://arxiv.org/pdf/1803.08494
- Normalization/rescaling: https://arxiv.org/pdf/1706.05350
- ROAR: https://proceedings.neurips.cc/paper_files/paper/2019/hash/fe4b8556000d0f0cae99daa5c5c5a410-Abstract.html
- SGDC: https://proceedings.mlr.press/v315/shi26a.html

## Reviewed retrieval and limits

Query plan: `audits/2026-10-10-reference-query.json`.
The strict reviewed RAG returned INDEX_STATUS=PASS and SOURCE_STATUS=PASS.
Evidence bundle:
`results/rag_query_bundles/a0943f23693520e285298d9b675bd308dec4831571e1f5e45c50a288121dce81.json`.
Relevant retrieved atoms were a040 (Jain, qualified proxy-validation evidence,
§4.1–4.2/Discussion), a072 (SKNet architecture utility, §4.3/Table 7) and a073
(SKNet scale-response observation, §4.4). Their limitations exclude universal
attention failure and patient-level best-path identification. Nonmatching
retrievals were not promoted to support for finite-epsilon normalization;
that question was checked directly against originals.

This is a focused check of cited claims and bibliographic consistency, not a
systematic search proving absence of prior derivations. It does not assess every
uncited paper or re-execute the original authors' experiments. No new reviewed
RAG atom is claimed. Build/package validation is recorded in the current
manuscript README after completion.
