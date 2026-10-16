# Confidence-aware, bit-budgeted reporting of maintenance diagnoses over SMS

Code to reproduce the experiments of a methodological study on **SMS-based maintenance diagnosis in low-connectivity
settings**. A model-agnostic *reporting layer* is placed after any probabilistic classifier and tested on three data
types, each used **separately (no fusion)**:

1. a calibrated classifier returning class probabilities and a confidence `c = max p(y|x)`;
2. **selective reporting**: report only if `c ≥ τ`, with `τ` chosen by **nested validation** inside each training fold
   (target accuracy α = 0.90, minimum support 20), never on test data;
3. a **bit-budgeted code**: the top-*k* classes, `k·⌈log₂ C⌉` bits, to be sent within the 1 120 bits of one SMS.

Benchmarks (all public, see [DATA.md](DATA.md)): text (FMC-MWO2KG, 22 ISO 14224 failure modes), thermal images of a
motor and of a transformer, and CWRU bearing vibration. Six model families are compared under identical,
leakage-controlled protocols, with negative controls that qualify the image and vibration scores.

> **What is NOT in this repository:** the article and the datasets. Data are downloaded from their original sources
> (links and pinned versions in [DATA.md](DATA.md)). **Scope:** public benchmarks only; no field data from Cameroon/Gabon,
> no language model, no multimodal fusion. These results validate a reporting methodology, not fault recognition in the field.

## Repository layout

```
src/smsreport/          paths.py (locations), common.py (nested-threshold evaluation, ECE, AURC, top-k), datasets.py (loaders + protocols)
experiments/
  run_models.py         MAIN: 6 model families x 4 benchmarks, nested selective reporting  -> results/all_models.json
  text_study.py         text-only analyses: official split, noise robustness, group codebooks, confusion -> results/text_study/
  controls_images.py    negative controls for thermal images                                -> results/controls_images.json
  controls_vibration.py negative controls / feature-group ablation for CWRU                 -> results/controls_vibration.json
  stats_text_models.py  paired Wilcoxon tests, dimensions, versions                         -> results/extra.json
  make_figures.py       figures                                                              -> results/figures/
  smoke_test.py         fast end-to-end check (a few minutes)
scripts/download_data.sh  fetches the datasets from the original repositories (pinned commits)
results/reference/      reference outputs of the study (JSON) to compare with your own run
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_data.sh        # ~1.1 GB; needs git and bsdtar (libarchive-tools) or unrar for the .rar archives
python experiments/smoke_test.py     # checks data + code in a few minutes

# full reproduction (~1 h on a laptop CPU; no GPU needed)
make all
```

You can run one benchmark at a time (and in parallel): `python experiments/run_models.py --only vibration_CWRU`
(`text_FMC`, `image_motor`, `image_transformer`, `vibration_CWRU`). The script resumes from the JSON already written.
Data locations can be overridden with `FMC_DIR`, `THERMAL_DIR`, `CWRU_DIR`, `SMSREPORT_RESULTS`.

## Validation protocols (leakage control)

| Data | Outer protocol | Inner protocol (threshold selection) |
|---|---|---|
| Text | 5-fold stratified CV × 3 repeats (15 folds) | stratified 5-fold |
| Thermal images | 5 contiguous blocks per class (neighbouring captures stay together) | leave-one-block-out |
| Vibration | leave-one-load-out (4 folds); all windows of a recording share its load | leave-one-load-out |

## Reference results (logistic regression; mean ± SD over folds)

With the pinned versions in `requirements.txt` the numbers are reproduced exactly (checked for the logistic-regression runs).

| Benchmark | Classes | Accuracy | ECE | Nested selective reporting (coverage → accuracy) |
|---|---|---|---|---|
| Text (FMC-MWO2KG) | 22 | 0.716 ± 0.024 | 0.060 | 0.600 → 0.904 |
| Thermal image, motor | 11 | 0.973 ± 0.024 | 0.020 | 0.997 → 0.976 |
| Thermal image, transformer | 9 | 0.988 ± 0.027 | 0.010 | 0.996 → 0.992 |
| Vibration (CWRU) | 10 | 0.999 ± 0.001 | 0.009 | 1.000 → 0.999 |

**Interpret with care.** Negative controls show that (i) permuted labels give chance-level accuracy (no label leakage),
(ii) for the motor images, the four corner patches alone reach 0.80 and five global colour statistics reach 0.77, so the
image scores are largely explained by global heating/acquisition conditions, and (iii) CWRU is saturated. Only the text
task is informative about reporting behaviour. The expected calibration error is strongly model dependent
(0.048–0.203 on text across model families), so the confidence threshold must be validated for the exact model deployed.

## Reproducibility notes

* Fixed seeds throughout; scikit-learn 1.8.0, NumPy 2.4.4, SciPy 1.17.1, Python 3.12; CPU only.
* Logistic-regression `C = 100` for text was selected on the official validation split; `C = 1` for image/vibration was not tuned.
  The target accuracy α = 0.90 and support 20 are fixed a priori.
* Text TF–IDF dimension 5 064; image features 3 224 (HSV histogram 512 + 32×24 grayscale 768 + HOG 1 944);
  vibration features 106 (10 time-domain + 64 spectral bands + 32 envelope-spectrum bands).
* Fold-based p-values (Wilcoxon) are optimistic because repeated folds overlap; image/vibration use only 4–5 folds
  and are not tested.

## Citation and licence

Code: MIT (see `LICENSE`; the datasets keep their own terms, see `DATA.md`). Please cite the associated article
[reference to be added] and the original dataset papers listed in `DATA.md`. `CITATION.cff` is a template to complete.
