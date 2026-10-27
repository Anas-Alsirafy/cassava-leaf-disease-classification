# Cassava 7 — Known Issues and Gotchas

Oct 26, 2026 · Anas Alsirafy

The 90.32% headline is somewhat optimistic because the specialist was trained on different folds from the ones it was scored on. The notebooks also carry a few code bugs; check each against the version uploaded to Drive, which may already fix them.

## Evaluation weak spots

| Issue | Effect | What to do |
| --- | --- | --- |
| Exp 11 specialist trained on different folds from the evaluation folds | Some evaluation images can have been in the specialist's training data, inflating 90.32% | Retrain the specialist on the same folds as the other ensemble members |
| The specialist loop reduces the learning rate on, and reports accuracy from, the same held-out fold | Mildly optimistic per-fold accuracy | Carve a validation slice from the training fold |
| Plain StratifiedKFold, not grouped | Safe for cassava (no duplicates found), but inconsistent with the cross-dataset protocol | Use StratifiedGroupKFold for new runs |
| Accuracy as the headline metric | Always guessing CMD scores 61.5%, so accuracy hides minority-class failure | Report balanced accuracy and macro-F1 alongside |
| One seed in the cross-dataset rerun (fold s.d. 0.56) | Uncertainty not fully measured | Add seeds 42 and 1337 |

## Code bugs and weak spots

| Where | Problem | Effect | Fix |
| --- | --- | --- | --- |
| Specialist training loop | Phase 2A unfreezes `model.layers[-30:]` — the last 30 layers of the whole model, not of the backbone | If the backbone is nested as one layer, it is fully unfrozen in 2A, not partly | Use a `get_backbone(model)` helper and unfreeze `backbone.layers[-30:]`; check how `build_specialist_model` nests the backbone |
| Data-pipeline cell | `mixup_batch` is defined twice with different signatures; the second replaces the first | `build_dataset_mixup` calls it with four arguments and will raise a TypeError | Rename one of them |
| Data-pipeline cell | `build_dataset(..., img_size='img_size')` defaults to the string `'img_size'` | `set_shape` fails when `augment=True` unless `img_size` is passed | Default to `CONFIG['img_size']` |
| Training loops | `if fold < N: continue` lines left from resuming interrupted runs | A fresh run silently skips folds | Reset before running |
| All pipelines | `preprocess_input` must run after augmentation | Before, colour augmentation acts on already-scaled values | Keep the current order |

## Environment errors already solved

| Error | Cause | Fix |
| --- | --- | --- |
| Saved weights would not reload (TF 2.19) | Old `.h5` save format | Save and load as `.keras` |
| Offline augmentation loop stopped early | The augmentation iterator ran out instead of repeating | Wrap it in `itertools.cycle` |
| Raspberry Pi: numpy ABI conflict | Camera stack and TensorFlow need different numpy builds | Run camera capture in a separate subprocess |
| Jetson: Pi Camera v2 fails on both CSI ports (I2C error -121) | Hardware or driver issue, unresolved | Time inference on pre-captured images |
| Kaggle session ends mid-run | Session time limit | Checkpoint per fold; save out-of-fold predictions after each fold |
