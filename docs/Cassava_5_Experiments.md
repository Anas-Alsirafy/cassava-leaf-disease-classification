# Cassava 5 — Experiments

Oct 26, 2026 · Anas Alsirafy

Accuracy rose from 67% to 90% over 11 numbered experiments. The big jumps came from fine-tuning the backbone, raising the input to 384×384, focal loss with MixUp/CutMix, and a minority-class specialist. Augmentation alone did nothing.

## Experiment log

All runs: 5-fold StratifiedKFold, mean ± s.d. over folds.

| Exp | What changed | Accuracy (%) | Why it was tried / what it showed |
| --- | --- | --- | --- |
| 1 | Frozen MobileNetV2, head only, class weights, early stopping, no augmentation | 67.32 ± 0.96 | Baseline. Barely above always guessing CMD (61.5%). |
| 2 | + flips, brightness ±0.2, contrast 0.8–1.2 | 66.45 ± 1.46 (3 of 5 folds) | Augmentation alone does nothing for a frozen backbone. |
| 3 | Two-phase: warm-up (lr 1e-4, 15 ep), then full MobileNetV2 fine-tune (lr 1e-5, 30 ep) | 72.37 ± 0.59 | Adapting the backbone is what helps on field images. |
| 4 | + offline CLAHE, phase 1 to 25 ep, cosine decay over 40 phase-2 ep (patience 15) | 74.12 ± 0.30 | Better contrast and longer, smoother training. |
| 5 | Same as 4, input 224 → 384 | 80.45 ± 1.05 | Largest single jump: lesions are small and need resolution. |
| 6–9 | Not recorded in this handover | — | Ask Anas for the log. A soft-voting ensemble reached 82.60 ± 1.00 in this range. |
| 10 | Noisy-Student EfficientNet-B0 + MixUp/CutMix + per-class focal loss | 87.88 ± 0.90 | Best single model. Targets the minority classes directly. |
| 11 | + ConvNeXt-Base specialist on the 4 minority classes, combined with Exp 10 by per-class F1 weights | 90.32 ± 0.78 | Best overall. Specialist folds do not match evaluation folds (see Known Issues). |

Values follow Paper 2 Table VI where it reports them. The original lab notes give slightly different spreads for Exp 1 (±0.88), 2 (66.60, all folds), 3 (±0.87) and 5 (±0.98); other notes record Exp 11 as 90.28%.

## Edge benchmarks

The Exp 11 ensemble was timed per image on three devices: RTX 5090 workstation (WSL2) 89 ms, Jetson Xavier NX 65 ms, Raspberry Pi 4 254 ms. On the Jetson the camera could not be used, so pre-captured images were fed in.

## Re-run in the cross-dataset study

For Paper 2, cassava was re-evaluated on grouped folds with a fine-tuned ConvNeXt-Base (86.32%, seed 17), the same backbone kept frozen (81.88%), and frozen three-backbone features with logistic regression (70.61 ± 0.16%). Results are in the Results doc; the full study is in the cross-dataset doc.

## Rules for the next experiment

1. Set a global seed (Python, NumPy, TensorFlow) at the top of the notebook.
2. Carve a validation slice from the training fold for callbacks; report on the held-out fold only.
3. Train every ensemble member on the same folds.
4. Change one thing per experiment, number it, and write one line on why before running.
5. Report balanced accuracy and macro-F1 next to accuracy — accuracy hides minority-class failure here.
6. Log training time per fold and save a checkpoint after every fold.
