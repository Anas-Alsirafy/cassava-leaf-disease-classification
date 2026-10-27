# Cassava 4 — Configs

Oct 26, 2026 · Anas Alsirafy

The best cassava system is a Noisy-Student EfficientNet-B0 trained with MixUp/CutMix and per-class focal loss, combined with a ConvNeXt-Base minority-class specialist through a per-class weighted average. Values not listed here are in each notebook's `CONFIG` cell.

## Best general model (Exp 10) and specialist (Exp 11)

| Setting | Exp 10 — general model | Exp 11 — minority specialist |
| --- | --- | --- |
| Backbone | EfficientNet-B0, Noisy-Student weights | ConvNeXt-Base |
| Classes | All 5 | CBB, CBSD, CGM, Healthy (CMD excluded) |
| Input | 384×384, CLAHE | Same |
| Augmentation | RandomResizedCrop (scale 0.9–1.0), horizontal flip p 0.5, vertical flip p 0.3, transpose p 0.3 | Same |
| Batch mixing | Per batch: 33% CutMix, 33% MixUp, 34% unchanged | Same |
| Loss | Per-class focal loss (gamma per class in notebook) | Per-class focal loss, gamma = [3.2, 2.2, 1.8, 2.0] |
| Phase 1 | Backbone frozen, `learning_rate` | Same |
| Phase 2A | Top 30 layers unfrozen, `fine_tune_lr` | Same, half the phase-2 epochs |
| Phase 2B | Whole network, 0.3 × `fine_tune_lr` | Same, other half |
| LR schedule | ReduceLROnPlateau on val_loss, factor 0.2, min lr 1e-7 | Same; patience 3 / 2 / 2 by phase |
| Folds | 5-fold StratifiedKFold | 5-fold StratifiedKFold on the specialist subset (different folds) |

**Ensemble (Exp 11):** per-class weighted average of EfficientNet-B0 (Exp 10) and the specialist: combined[c] = (p_EB0[c] × w_EB0[c] + p_spec[c] × w_spec[c]) / (w_EB0[c] + w_spec[c]). The weights are each model's per-class F1 from training — EfficientNet-B0: 0.63, 0.79, 0.80, 0.96, 0.72; specialist: 0.80, 0.89, 0.93, 0.00, 0.85 (CBB, CBSD, CGM, CMD, Healthy). The specialist's CMD weight is 0, so EfficientNet-B0 alone decides CMD. The 4-class specialist output is expanded to 5 classes with CMD = 0 before combining.

## Defaults set by earlier experiments

| From | Setting |
| --- | --- |
| Exp 1 | Class weights = N / (5 × class count); early stopping |
| Exp 3 | Two-phase training: warm-up lr 1e-4 for 15 epochs, then full fine-tune lr 1e-5 for 30 epochs |
| Exp 4 | Offline CLAHE (clip 2.0, 8×8); phase 1 extended to 25 epochs; cosine-decay learning rate over 40 phase-2 epochs, patience 15 |
| Exp 5 | Input 224 → 384 |

## Cassava in the cross-dataset study

When the pipeline was rerun on the shared grouped folds for Paper 2, cassava used: ConvNeXt-Base, 384×384, CLAHE, three phases (0.13 M → 28.6 M → 87.7 M trainable parameters), mixed precision, group-aware 10% validation split from the training fold, class weights from the training fold, seed 17 only (about 4 GPU-hours per fold). A control run kept the backbone frozen in all three phases. Full details are in the cross-dataset doc.
