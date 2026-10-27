# Cassava 3 — Data

Oct 26, 2026 · Anas Alsirafy

One dataset: the 2020 Kaggle Cassava Leaf Disease Classification set, 21,397 field photos in five classes. It is dominated by one class — cassava mosaic disease is 61.5% of images — so always guessing CMD already scores 61.5%. The duplicate screen found no duplicates.

## Source

- [Kaggle competition page](https://www.kaggle.com/competitions/cassava-leaf-disease-classification); download with the Kaggle API after accepting the competition rules.
- Photos taken by farmers in Uganda on phones: cluttered backgrounds, mixed lighting, several leaves per frame. Treat it as field data, not lab data.
- The competition's own test set is hidden; all our numbers come from cross-validation on the 21,397 training images.

## Classes

Counts match the class weights logged in training (weight = N / (5 × count)).

| Index | Class | Images | Share | Class weight |
| --- | --- | --- | --- | --- |
| 0 | Cassava Bacterial Blight (CBB) | 1,087 | 5.1% | 3.94 |
| 1 | Cassava Brown Streak Disease (CBSD) | 2,189 | 10.2% | 1.95 |
| 2 | Cassava Green Mottle (CGM) | 2,386 | 11.2% | 1.79 |
| 3 | Cassava Mosaic Disease (CMD) | 13,158 | 61.5% | 0.33 |
| 4 | Healthy | 2,577 | 12.0% | 1.66 |

The four classes other than CMD are the "minority classes" that the Exp 11 specialist targets.

## Integrity screening

Every image was resized to 224×224 and hashed (MD5 for exact copies, 64-bit dHash for near copies, Hamming distance ≤ 5). Cassava had 0 exact and 0 near duplicates, so 21,397 images form 21,397 groups. Plain StratifiedKFold is therefore safe here, though the cross-dataset study used StratifiedGroupKFold for consistency.

## Pre-processing and splits

- **CLAHE (from Exp 4 on):** LAB colour space, CLAHE on the L channel only (clipLimit 2.0, 8×8 tiles), back to RGB. Done once, offline, for all 21,397 images; the processed path is stored in a `clahe_path` column.
- **Resize:** 224×224 up to Exp 4, 384×384 from Exp 5 on.
- **Normalisation:** the backbone's own `preprocess_input` (e.g. EfficientNet), applied after augmentation.
- **Splits:** 5-fold StratifiedKFold, shuffled, fixed seed. Validation for callbacks is the held-out fold itself (see Known Issues).
- **Augmentation:** only on training batches, never on the validation fold.
