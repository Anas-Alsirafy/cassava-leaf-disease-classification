# Cassava Leaf Disease Classification

Five-class leaf-disease classifier for the [Kaggle Cassava Leaf Disease Classification](https://www.kaggle.com/competitions/cassava-leaf-disease-classification) dataset, built under the CERERE project at the American University in Cairo. The best system combines a Noisy-Student EfficientNet-B0 with a ConvNeXt-Base minority-class specialist and reaches **90.32 ± 0.78% accuracy (5-fold CV)**.

This is also where the three-phase fine-tuning pipeline used across the other datasets was first developed.

## Results

| Model | Accuracy (5-fold) | Notes |
| --- | --- | --- |
| Exp 1 — frozen MobileNetV2 baseline | 67.32 ± 0.96% | Always predicting CMD gives 61.5% |
| Exp 5 — fine-tuned, CLAHE, 384×384 | 80.45 ± 1.05% | Input resolution was the largest single gain |
| **Exp 10** — Noisy-Student EfficientNet-B0 + CutMix/MixUp + per-class focal loss | **87.88 ± 0.90%** | Best single model |
| **Exp 11** — Exp 10 + ConvNeXt-Base specialist, per-class weighted ensemble | **90.32 ± 0.78%** | Best overall (optimistic, see [Known issues](#known-issues)) |

Ensemble inference time per image: RTX 5090 (WSL2) 89 ms · Jetson Xavier NX 65 ms · Raspberry Pi 4 254 ms.

## Repository layout

```
cassava-leaf-disease-classification/
├── results.csv   # every reported number, one row per metric
├── cassava_exp10_exp11.ipynb   # main notebook: Exp 10 + Exp 11, evaluation, inference
├── cassava_exp11.py            # builder script that generates the notebook (source of truth)
├── binary_cbb_healthy.ipynb    # side experiment: binary CBB vs Healthy
└── README.md
```

Edit `cassava_exp11.py`, then run `python cassava_exp11.py` to regenerate the notebook. Don't hand-edit the `.ipynb`.

## Dataset

21,397 field photos, 5 classes, heavily imbalanced:

| Label | Class | Images | Share |
| --- | --- | --- | --- |
| 0 | Cassava Bacterial Blight (CBB) | 1,087 | 5.1% |
| 1 | Cassava Brown Streak Disease (CBSD) | 2,189 | 10.2% |
| 2 | Cassava Green Mottle (CGM) | 2,386 | 11.2% |
| 3 | Cassava Mosaic Disease (CMD) | 13,158 | 61.5% |
| 4 | Healthy | 2,577 | 12.0% |

A duplicate screen (MD5 + dHash, Hamming distance ≤ 5) found no exact or near duplicates.

## Setup

```bash
# Windows users: run inside WSL2 (native Windows TensorFlow has no GPU support after 2.10)
python -m venv .venv && source .venv/bin/activate
pip install tensorflow scikit-learn albumentations opencv-python pandas numpy matplotlib seaborn tqdm

# verify the GPU is visible
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# download the data (needs a Kaggle API token in ~/.kaggle/kaggle.json)
kaggle competitions download -c cassava-leaf-disease-classification
```

TensorFlow version used: 2.19 (to confirm). Save models as `.keras`; `.h5` weights would not reload under TF 2.19.

## Usage

1. Open `cassava_exp10_exp11.ipynb` and set the paths in the configuration cell:

   ```python
   DATA_DIR  = '/kaggle/input/cassava-leaf-disease-classification'
   TRAIN_CSV = os.path.join(DATA_DIR, 'train.csv')
   MODEL_DIR = '/kaggle/working'          # .keras files are saved / loaded here
   IMAGE_DIR = '/path/to/your/new/images' # only for the inference section
   ```

2. Run the sections in order:

   | Section | What it does |
   | --- | --- |
   | 1–2 | Imports and `CONFIG` |
   | 3 | Loads `train.csv` and writes CLAHE-processed copies of every image |
   | 4–5 | Dataset/augmentation functions and model builders |
   | 6 | **Exp 10**: trains EfficientNet-B0, 5 folds → `fold_{1-5}_exp10_eb0_model.keras` |
   | 7 | **Exp 11**: trains the ConvNeXt-Base specialist on CBB/CBSD/CGM/Healthy → `fold_{1-5}_specialist_model.keras` |
   | 8 | Loads the saved models and runs the weighted ensemble |
   | 9 | Per-class metrics and confusion matrix → `exp11_confusion_matrix.png`, `exp11_fold_accuracy.png` |
   | 10 | Inference on new images → `inference_results.csv`, `sample_predictions.png` |
   | 11 | Exp 10 vs Exp 11 summary |

   Sections 8–10 only need the saved `.keras` files, so you can skip training if the models exist.

3. To predict a single image, use `predict_single(path)` from section 10.

## Pipeline

**Pre-processing.** CLAHE on the L channel in LAB colour space (`clipLimit=2.0`, 8×8 tiles), resize to 384×384, then the backbone's own `preprocess_input`. Augmentation is applied before normalisation.

**Augmentation.** `RandomResizedCrop` (scale 0.9–1.0), horizontal flip (p 0.5), vertical flip (p 0.3), transpose (p 0.3). Each batch then gets CutMix (33%), MixUp (33%) or nothing (34%).

**Loss.** Per-class focal loss (`focal_loss_per_class`):

```python
GAMMA_EB0        = [3.0, 2.14, 2.04, 1.0, 1.92]  # CBB, CBSD, CGM, CMD, Healthy
GAMMA_SPECIALIST = [3.2, 2.2, 1.8, 2.0]          # CBB, CBSD, CGM, Healthy
```

Loading a saved model needs the loss passed back in, e.g. `custom_objects={'loss': focal_loss_per_class(GAMMA_EB0)}`.

**Training.** Three phases with ReduceLROnPlateau on `val_loss` (factor 0.2, patience 3 in phase 1 and 2 after, min lr 1e-7):

| Phase | Trainable | Learning rate | Epochs |
| --- | --- | --- | --- |
| 1 | Head only (backbone frozen) | `learning_rate` = 1e-4 | `epochs_phase1` = 15 |
| 2A | Top 30 backbone layers | `fine_tune_lr` = 1e-5 | `epochs_phase2 // 2` |
| 2B | Whole network | 0.3 × `fine_tune_lr` | `epochs_phase2 // 2` |

`CONFIG`: `img_size=384`, `batch_size=16`, `num_classes=5`, `epochs_phase2=20`. Folds: 5-fold `StratifiedKFold`, shuffled, fixed seed.

**Ensemble (Exp 11).** The specialist's 4-class output is expanded to 5 classes with CMD = 0, then combined per class:

```
combined[c] = (p_eb0[c] * w_eb0[c] + p_spec[c] * w_spec[c]) / (w_eb0[c] + w_spec[c])

w_eb0  = [0.63, 0.79, 0.80, 0.96, 0.72]   # per-class F1 of EfficientNet-B0
w_spec = [0.80, 0.89, 0.93, 0.00, 0.85]   # per-class F1 of the specialist (CMD = 0)
```

The specialist's CMD weight is 0, so EfficientNet-B0 alone decides CMD.

## Side experiment: `binary_cbb_healthy.ipynb`

Binary CBB vs Healthy classifier. Choose the backbone with `CONFIG['backbone']` (`'efficientnet'` or `'convnext'`). CBB is oversampled ×3, input is 384×384 with CLAHE, and training runs three phases (25 / 20 / 10 epochs at 1e-4 / 1e-5 / 5e-6) over 5 folds. This notebook is exploratory and is not part of the reported results.

## Known issues

- **Exp 11 is optimistic.** The specialist uses its own 5 folds on the 4-class subset, which don't match Exp 10's evaluation folds. Its ensemble weights are per-class F1 values from training rather than from a held-out split.
- **The held-out fold also drives callbacks** (LR reduction), so per-fold accuracy is mildly optimistic.
- **Folds are not group-aware.** That is safe here because no duplicates were found, but it is inconsistent with the cross-dataset protocol.
- **Report balanced accuracy and macro-F1 alongside accuracy.** Accuracy hides minority-class failure on this dataset.
- **Bugs from the original Kaggle notebooks** (duplicate `mixup_batch`, a string `img_size` default, `model.layers[-30:]`) are **not** in this consolidated notebook.

## Roadmap

- [ ] Retrain the specialist on the same folds as Exp 10 and re-score Exp 11
- [ ] Use grouped folds with a validation slice carved from each training fold
- [ ] Rebuild a 15% held-out test set
- [ ] INT8 quantisation (TFLite / TensorRT) for the Raspberry Pi and Jetson
- [ ] End-to-end FIWARE field pipeline (ESP32-CAM → MQTT → Orion → Jetson → database)

## Contact

Anas Alsirafy, American University in Cairo (anas.samy@aucegypt.edu). Supervisor: Prof. Maki K. Habib.

## Related repositories

- [leaf-disease-cross-dataset-study](https://github.com/Anas-Alsirafy/leaf-disease-cross-dataset-study) — Paper 2 five-dataset leakage-controlled comparison
- [maize-multilabel-foliar-disease](https://github.com/Anas-Alsirafy/maize-multilabel-foliar-disease) — multi-label maize (cereal) work
