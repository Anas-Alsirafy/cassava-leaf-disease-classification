# Cassava 6 — Results and Findings

Oct 26, 2026 · Anas Alsirafy

The best cassava system reaches 90.32 ± 0.78% (5-fold), within about a point of the 2020 Kaggle winner's 91.3% on its hidden test set. Nearly all of the gain over the 67% baseline came from adapting the network to field images, not from the classifier on top.

## Final results

| System | Accuracy (%) | Protocol | Caveat |
| --- | --- | --- | --- |
| Exp 11: EfficientNet-B0 + ConvNeXt-Base specialist, per-class weighted | 90.32 ± 0.78 | 5-fold StratifiedKFold | Specialist folds differ from evaluation folds: optimistic |
| Exp 10: Noisy-Student EfficientNet-B0, MixUp/CutMix, focal loss | 87.88 ± 0.90 | Same | Not group-aware; check whether the held-out fold also drove callbacks |
| Exp 1: frozen MobileNetV2 baseline | 67.32 ± 0.96 | Same | — |
| Always guess CMD | 61.5 | — | Reference floor |

## Findings

- **Fine-tuning and resolution matter most.** Fine-tuning added 5 points (Exp 3); moving from 224 to 384 pixels added 6 (Exp 5). Augmentation on a frozen backbone added nothing (Exp 2).
- **The minority classes are the battleground.** CMD is easy; the other four classes are where focal loss, MixUp/CutMix and the specialist earned their 7–10 points.
- **Edge deployment is feasible.** The full ensemble runs at 65 ms/image on a Jetson Xavier NX and 254 ms on a Raspberry Pi 4.

## Cassava in the cross-dataset study (grouped folds)

| System | Accuracy (%) | Balanced acc. (%) | Macro-F1 |
| --- | --- | --- | --- |
| ConvNeXt-Base, fine-tuned, 384×384 (seed 17) | 86.32 | 76.7 | 0.762 |
| ConvNeXt-Base, frozen, 384×384 (seed 17) | 81.88 | — | — |
| Frozen MobileNetV2 + EfficientNet-B0 + DenseNet121, logistic regression | 70.61 ± 0.16 | 53.6 | 0.534 |

- Fine-tuned beats frozen small backbones by 15.7 points (95% CI 14.5–17.0, p < 10⁻⁵).
- The frozen ConvNeXt-Base closes about three-quarters of that gap; fine-tuning adds the last 4.4 points (95% CI 2.1–6.7, p = 0.006).
- With frozen small backbones, CMD reaches F1 0.86 but the other four classes only 0.38–0.50 — just 9.1 points above always guessing CMD.
- Cost: a fine-tuned fold took about 262 minutes; the frozen-backbone control 170 minutes; the logistic regression 78 seconds.

## Against published results

| Work | Method | Protocol | Accuracy (%) |
| --- | --- | --- | --- |
| Riaz et al. | EfficientNet-B3 | Own test split | 83.0 |
| Fang et al. | ConvNeXt-T, fine-tuned | 80/20 | 88.9 |
| Fang et al. | VGG-13 BN, fine-tuned | 80/20 | 89.0 |
| Kaggle 2020 winner | 4-model ensemble (ResNeXt50, ViT-B/16, EfficientNet-B4 Noisy Student, CropNet MobileNetV3) | Hidden test set | 91.3 |
| This work, Exp 11 | Weighted 2-model ensemble | 5-fold | 90.3 |
| This work, cross-dataset | ConvNeXt-Base, fine-tuned | Grouped 5-fold, 1 seed | 86.3 |

Full references are in Paper 2, Table IV.
