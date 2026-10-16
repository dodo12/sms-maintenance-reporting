# Data sources, provenance and licences

**This repository does not contain any dataset.** All data are public and must be downloaded from their original
sources (`bash scripts/download_data.sh` does it, pinned to the exact commits used in the study).
Respect each dataset's licence/terms and cite the original authors when you publish results.

| # | Dataset | Content | Original source / DOI | Pinned version | Terms (as stated by the source) |
|---|---|---|---|---|---|
| 1 | **FMC-MWO2KG** (failure-mode classification) | 626 short maintenance texts (502 train / 62 dev / 62 test), 22 ISO 14224 failure modes, English | https://github.com/nlp-tlp/mwo2kg-and-echidna (folder `mwo2kg/failure_mode_classification/input_data/`). Associated paper: M. Stewart, M. Hodkiewicz, W. Liu, T. French, *MWO2KG and Echidna: Constructing and exploring knowledge graphs from maintenance data*, Proc. IMechE Part O 238(5) (2024) 920–932, doi:10.1177/1748006X221131128 | commit `96fc4477ea43a4bfd768430da386008b8803f1d1` (2023-02-13) | Repository licence: Apache-2.0 |
| 2 | **Thermal images, induction motor** | 369 thermographic images (320×240), 11 conditions (healthy, stuck rotor, cooling-fan failure, 8 stator short-circuit configurations) | Mendeley Data: https://data.mendeley.com/datasets/m4sbt8hbvk/3 (doi:10.17632/m4sbt8hbvk.3). GitHub copy used here: https://github.com/mohnaj-nit/thermal-images-equip. Related paper: Najafi et al., ICSPIS 2020, doi:10.1109/ICSPIS51611.2020.9349599 | commit `8ac89a0b08f3f21f59a5922107164cca98d7e841` (2020-11-12), archive `IR-Motor-bmp.rar` | Authors' README: use allowed with citation of the creators (check the README and the Mendeley page) |
| 3 | **Thermal images, 1-phase dry-type transformer** | 255 thermographic images (320×240), 9 conditions (healthy + 8 levels of short-circuited turns) | Mendeley Data: https://data.mendeley.com/datasets/8mg8mkc7k5/3 (doi:10.17632/8mg8mkc7k5.3). Same GitHub copy as above (archive `IR_trans_bmp.rar`) | same commit as #2 | Same as #2 |
| 4 | **CWRU bearing data** | Bearing vibration signals (we use drive-end, 12 kHz, normal + 0.007/0.014/0.021 in faults; 56 recordings → 3 897 windows of 2 048 samples; loads 0–3 hp) | Case Western Reserve University Bearing Data Center: https://engineering.case.edu/bearingdatacenter . Reference study: W. A. Smith, R. B. Randall, *Mech. Syst. Signal Process.* 64–65 (2015) 100–131, doi:10.1016/j.ymssp.2015.04.021. GitHub mirror used here: https://github.com/s-whynot/CWRU-dataset | commit `0d07cb1caf0a5a68a955c2f43cfc21b9a208406a` (2021-02-17) | The mirror has **no explicit licence**; prefer the original site and cite it |

## Integrity checks (SHA-256)

```
train.txt          cf535a461c07e29555d8ad829a910bba6ceebaf7b2934eba231134967c58c4bc
dev.txt            83fe89df0f97ceca21eca6c32a3306e7ec138ddbbfffbf7770ab872a4909a567
test.txt           42227739b30a0e25ed5c36aaa630e1403dcd5cd3f1f8515aa43910e388ca6565
IR-Motor-bmp.rar   443815d41f96baf971008687eb2a80e7bc21ca7bfc3dc61cb8fa7ad9295020ea
IR_trans_bmp.rar   2d20e4b5dbc61b9ec05e814e87cca4d3b304d2937d3ade8a1c068f8e94e9ade0
```

`python experiments/smoke_test.py` also asserts the expected counts (626 texts/22 classes; 369 motor images/11 classes;
255 transformer images/9 classes; 3 897 × 106 vibration feature matrix).

## Known limitations of these data (read before interpreting results)

* **Thermal images** were captured sequentially on a laboratory bench; images of a class are near-duplicates of their
  neighbours, hence the *blocked* validation. Negative controls (`experiments/controls_images.py`) show that corner patches
  alone and five global colour statistics classify the conditions well above chance: the scores reflect global heating
  and acquisition conditions, not only local fault signatures.
* **CWRU** is saturated and uses the same bearings under all loads; leave-one-load-out tests load transfer only.
* **FMC-MWO2KG** is small, English-language and imbalanced (three classes have a single training example).
* No field data from Cameroon or Gabon are used. Nothing here is a field validation.
