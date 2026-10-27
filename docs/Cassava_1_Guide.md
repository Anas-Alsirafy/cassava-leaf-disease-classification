# Cassava 1 — Guide

Oct 26, 2026 · Anas Alsirafy

The cassava track built a classifier for the five-class Kaggle cassava leaf-disease set under the CERERE project at AUC, and timed it on edge devices. It is also where the fine-tuning pipeline later used across all datasets was developed. Best result: EfficientNet-B0 plus a ConvNeXt-Base specialist, combined by per-class weights, at 90.32% (5-fold). This set is for a new team member with basic AI knowledge.

## Where this sits

The handover has three parts. This set covers cassava. Maize has its own eight docs starting at [Cereal 1 — Guide](https://github.com/Anas-Alsirafy/maize-multilabel-foliar-disease/blob/main/docs/Cereal_1_Guide.md). The five-dataset comparison behind Paper 2, which reuses this pipeline on cassava and four other datasets, is one doc: [Cross-Dataset Study — Five Datasets (Paper 2)](https://github.com/Anas-Alsirafy/leaf-disease-cross-dataset-study/blob/main/docs/Cross_Dataset_Study.md).

## Status

| Piece | State (Oct 2026) |
| --- | --- |
| Classifier | Done. Best single model 87.88% (Exp 10), best ensemble 90.32% (Exp 11), 5-fold. |
| Edge timing | Done. Ensemble inference: RTX 5090 89 ms, Jetson Xavier NX 65 ms, Raspberry Pi 4 254 ms per image. |
| Field pipeline (FIWARE) | Architecture designed; how much was built beyond Jetson inference is to confirm. |
| Clean re-evaluation | Pending: the Exp 11 specialist was trained on folds that don't match the evaluation folds. |

## How to read this set

1. [Cassava 2 — Environment Setup](Cassava_2_Environment_Setup.md) — hardware, packages, Kaggle and edge-device notes.
2. [Cassava 3 — Data](Cassava_3_Data.md) — the dataset, its five classes and their imbalance.
3. [Cassava 4 — Configs](Cassava_4_Configs.md) — settings for the best models.
4. [Cassava 5 — Experiments](Cassava_5_Experiments.md) — the 11-experiment log, from 67% to 90%.
5. [Cassava 6 — Results and Findings](Cassava_6_Results_and_Findings.md) — what the numbers mean, including how cassava did in the cross-dataset study.
6. [Cassava 7 — Known Issues and Gotchas](Cassava_7_Known_Issues_and_Gotchas.md) — bugs and traps. Read before touching code.
7. [Cassava 8 — Future Work](Cassava_8_Future_Work.md) — what to do next.

## Glossary

| Term | Meaning here |
| --- | --- |
| Backbone | A pretrained image network (MobileNetV2, EfficientNet-B0, ConvNeXt-Base) that turns a photo into features. |
| Frozen vs fine-tuned | Frozen: backbone weights locked, only a small head is trained. Fine-tuned: some or all backbone weights are trained too. |
| Three-phase training | Frozen backbone → top 30 layers unfrozen → whole network unfrozen at a lower learning rate. |
| Noisy Student | A set of EfficientNet weights trained with self-training on extra unlabelled images; usually transfers better. |
| CLAHE | Local contrast boost on the lightness channel; makes lesions easier to see. |
| MixUp / CutMix | Blend two training images (MixUp) or paste a patch of one onto another (CutMix), mixing labels too. |
| Focal loss | Down-weights easy examples so training focuses on hard, usually minority-class, ones. Per-class focal loss uses a different strength (gamma) per class. |
| Specialist | A model trained only on the four minority classes, combined with the general model. |
| Weighted ensemble | Average two models' class probabilities, weighting each class by how well each model scores on it (per-class F1). |
| Fold / k-fold CV | Split data into 5 parts; train on 4, test on 1; repeat so every image is tested once. |
| Leakage | Information about test images reaching training, which inflates accuracy. |
| Balanced accuracy / macro-F1 | Scores that weight each class equally, so ignoring the four small classes is penalised. |

## People and files

| Who / what | Detail |
| --- | --- |
| Anas Alsirafy | Built this work; leaving AUC (anas.samy@aucegypt.edu) |
| Prof. Maki K. Habib | Supervisor; sets priorities after handover |
| Successor | Not yet chosen |
| Code and notebooks | Google Drive folder, link to be added |
| Dataset | Kaggle cassava leaf-disease competition (see Data) |

## Open questions for Anas

- [ ] Add the Google Drive link here and in Environment Setup.
- [ ] Confirm the training GPU (RTX 5090) and the Kaggle GPU model (written as P100).
- [ ] Confirm the TensorFlow version (cassava code ran on 2.19).
- [ ] Point to the records for Experiments 6–9.
- [ ] Confirm where the trained `.keras` model files are stored.
- [ ] Confirm which FIWARE links beyond Jetson inference were built.
