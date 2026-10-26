import json

nb = {
 "nbformat": 4,
 "nbformat_minor": 5,
 "metadata": {
  "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
  "language_info": {"name": "python", "version": "3.10.0"}
 },
 "cells": []
}

def md(src):
    return {"cell_type":"markdown","metadata":{},"source":src,"id":f"md{len(nb['cells'])}"}

def code(src, outputs=None):
    return {
        "cell_type":"code","metadata":{},"source":src,
        "outputs": outputs or [],"execution_count":None,
        "id":f"cc{len(nb['cells'])}"
    }

# ── TITLE ──────────────────────────────────────────────────────────────────────
nb["cells"].append(md("""# Cassava Leaf Disease — Exp 10 / Exp 11 Full Pipeline
**Pipeline B · ConvNeXtBase + EfficientNetB0 NoisyStudent Ensemble**

This notebook covers:
1. **Setup & CLAHE preprocessing** — reproduce the exact preprocessing used in training
2. **Exp 10 — EfficientNetB0 (NoisyStudent) 5-fold CV** — generalist model, CutMix/Mixup/Focal Loss
3. **Exp 11 — ConvNeXtBase Specialist + Weighted Ensemble** — specialist on 4 non-CMD classes
4. **Evaluation** — per-fold accuracy, per-class F1, confusion matrix (all folds combined)
5. **Inference on new images** — load saved `.keras` models, predict with confidence

> **Results:** Exp 10 = 87.88% · Exp 11 = **90.32% ± 0.0078**
> **Saved models on Kaggle:** `fold_{1-5}_exp10_eb0_model.keras` · `fold_{1-5}_specialist_model.keras`
"""))

# ── CELL 1: INSTALL ────────────────────────────────────────────────────────────
nb["cells"].append(md("## 1. Install & Imports"))
nb["cells"].append(code("""\
# Run once — restart kernel after if installing fresh
# !pip install tensorflow==2.15.0 albumentations opencv-python-headless scikit-learn matplotlib seaborn tqdm

import os, gc, cv2, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import albumentations as A
from tqdm import tqdm

warnings.filterwarnings('ignore')
print("TensorFlow:", tf.__version__)
print("GPU:", tf.config.list_physical_devices('GPU'))
"""))

# ── CELL 2: CONFIG ─────────────────────────────────────────────────────────────
nb["cells"].append(md("## 2. Configuration"))
nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  PATH — update to your local dataset location
# ─────────────────────────────────────────────
DATA_DIR   = '/kaggle/input/cassava-leaf-disease-classification'   # Kaggle path
# DATA_DIR = '/path/to/cassava-leaf-disease-classification'        # Local path

TRAIN_CSV  = os.path.join(DATA_DIR, 'train.csv')
TRAIN_DIR  = os.path.join(DATA_DIR, 'train_images')
MODEL_DIR  = '/kaggle/working'       # where .keras files are saved / loaded from

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ─────────────────────────────────────────────
#  MODEL CONFIG  (must match training exactly)
# ─────────────────────────────────────────────
CONFIG = {
    'img_size'       : 384,
    'batch_size'     : 16,
    'num_classes'    : 5,
    'learning_rate'  : 1e-4,
    'fine_tune_lr'   : 1e-5,
    'epochs_phase1'  : 15,
    'epochs_phase2'  : 20,
}

# Class labels
LABEL_MAP = {0: 'CBB', 1: 'CBSD', 2: 'CGM', 3: 'CMD', 4: 'Healthy'}
CLASS_NAMES = [LABEL_MAP[i] for i in range(5)]

# Specialist class mapping  (CMD=3 excluded, labels remapped 0→0 1→1 2→2 4→3)
SPECIALIST_LABEL_MAP = {0: 'CBB', 1: 'CBSD', 2: 'CGM', 3: 'Healthy'}
SPECIALIST_CLASSES   = [SPECIALIST_LABEL_MAP[i] for i in range(4)]

print("Config loaded ✓")
print(f"Image size: {CONFIG['img_size']}x{CONFIG['img_size']}")
print(f"Classes: {CLASS_NAMES}")
"""))

# ── CELL 3: DATA LOADING ───────────────────────────────────────────────────────
nb["cells"].append(md("## 3. Data Loading & CLAHE Preprocessing"))
nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  Load CSV
# ─────────────────────────────────────────────
df = pd.read_csv(TRAIN_CSV)
df['path'] = df['image_id'].apply(lambda x: os.path.join(TRAIN_DIR, x))
print(f"Total images: {len(df)}")
print(df['label'].value_counts().sort_index().rename(LABEL_MAP))
"""))

nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  CLAHE preprocessing  (must match training)
#  Applied on LAB L-channel: clipLimit=2.0, tileGridSize=8x8
# ─────────────────────────────────────────────
CLAHE_DIR = '/kaggle/working/clahe_images'
os.makedirs(CLAHE_DIR, exist_ok=True)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

def apply_clahe(img_path, out_dir):
    fname = os.path.basename(img_path)
    out_path = os.path.join(out_dir, fname)
    if os.path.exists(out_path):
        return out_path
    img = cv2.imread(img_path)
    if img is None:
        return img_path  # fallback: use original
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(out_path, out)
    return out_path

print("Applying CLAHE to all images (skip if already done)...")
clahe_paths = []
for path in tqdm(df['path'].values):
    clahe_paths.append(apply_clahe(path, CLAHE_DIR))

df['clahe_path'] = clahe_paths
print("CLAHE done ✓")
print(f"Sample path: {df['clahe_path'].iloc[0]}")
"""))

# ── CELL 4: PREPROCESSING FUNCTIONS ───────────────────────────────────────────
nb["cells"].append(md("## 4. Dataset & Augmentation Functions"))
nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  Albumentations pipeline (used in training)
# ─────────────────────────────────────────────
albumentations_aug = A.Compose([
    A.RandomResizedCrop(size=(CONFIG['img_size'], CONFIG['img_size']), scale=(0.9, 1.0), p=1.0),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.3),
    A.Transpose(p=0.3),
])

def preprocess_image_eb0(image_path, label):
    \"\"\"EfficientNetB0 preprocessing — used for Exp 10 generalist\"\"\"
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [CONFIG['img_size'], CONFIG['img_size']])
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

def preprocess_image_convnext(image_path, label):
    \"\"\"ConvNeXt preprocessing — used for specialist\"\"\"
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [CONFIG['img_size'], CONFIG['img_size']])
    img = tf.cast(img, tf.float32) / 255.0
    # ImageNet normalisation
    mean = tf.constant([0.485, 0.456, 0.406])
    std  = tf.constant([0.229, 0.224, 0.225])
    img  = (img - mean) / std
    return img, label

def augment_image_fn(path, label):
    path_str = path.numpy().decode()
    img = cv2.imread(path_str)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    augmented = albumentations_aug(image=img)
    img = augmented['image'].astype(np.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img, label

def build_val_ds_eb0(image_paths, labels):
    \"\"\"Validation dataset for EB0 (no augmentation)\"\"\"
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess_image_eb0, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    return ds

def build_val_ds_convnext(image_paths, labels):
    \"\"\"Validation dataset for ConvNeXt specialist (no augmentation)\"\"\"
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess_image_convnext, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(CONFIG['batch_size']).prefetch(tf.data.AUTOTUNE)
    return ds

print("Preprocessing functions defined ✓")
"""))

# ── CELL 5: FOCAL LOSS ─────────────────────────────────────────────────────────
nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  Per-class focal loss  (required to load models)
# ─────────────────────────────────────────────
def focal_loss_per_class(gamma_vec):
    gamma_vec = tf.constant(gamma_vec, dtype=tf.float32)
    def loss(y_true, y_pred):
        ce    = tf.keras.losses.categorical_crossentropy(y_true, y_pred)
        p_t   = tf.reduce_sum(y_true * y_pred, axis=-1)
        gamma = tf.reduce_sum(y_true * gamma_vec, axis=-1)
        return tf.pow(1.0 - p_t, gamma) * ce
    return loss

# Gamma values used in training
GAMMA_EB0        = [3.0, 2.14, 2.04, 1.0, 1.92]   # CBB, CBSD, CGM, CMD, Healthy
GAMMA_SPECIALIST = [3.2, 2.2, 1.8, 2.0]            # CBB, CBSD, CGM, Healthy (no CMD)

print("Focal loss defined ✓")
"""))

# ── CELL 6: CUTMIX / MIXUP ─────────────────────────────────────────────────────
nb["cells"].append(code("""\
# ─────────────────────────────────────────────
#  CutMix + Mixup  (needed for training cells only)
#  Not needed for inference / evaluation
# ─────────────────────────────────────────────
def cutmix_batch(images, labels, alpha=1.0):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0, 1)
    lam = tf.maximum(lam, 1.0 - lam)
    indices = tf.random.shuffle(tf.range(batch_size))
    H = tf.shape(images)[1]; W = tf.shape(images)[2]
    cut_ratio = tf.sqrt(1.0 - lam)
    cut_h = tf.cast(tf.cast(H, tf.float32) * cut_ratio, tf.int32)
    cut_w = tf.cast(tf.cast(W, tf.float32) * cut_ratio, tf.int32)
    cx = tf.random.uniform([], 0, W, dtype=tf.int32)
    cy = tf.random.uniform([], 0, H, dtype=tf.int32)
    x1 = tf.clip_by_value(cx - cut_w // 2, 0, W)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, H)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, W)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, H)
    # Build mask
    mask = tf.ones([y1, W, 3])
    mid  = tf.concat([tf.ones([y2 - y1, x1, 3]),
                      tf.zeros([y2 - y1, x2 - x1, 3]),
                      tf.ones([y2 - y1, W - x2, 3])], axis=1)
    bot  = tf.ones([H - y2, W, 3])
    mask = tf.concat([mask, mid, bot], axis=0)
    shuffled = tf.gather(images, indices)
    mixed    = images * mask + shuffled * (1.0 - mask)
    lam_adj  = 1.0 - tf.cast((y2 - y1) * (x2 - x1), tf.float32) / tf.cast(H * W, tf.float32)
    mixed_labels = lam_adj * labels + (1.0 - lam_adj) * tf.gather(labels, indices)
    return mixed, mixed_labels

def mixup_batch(images, labels, alpha=0.2):
    batch_size = tf.shape(images)[0]
    lam = tf.random.uniform([], 0, 1)
    lam = tf.maximum(lam, 1.0 - lam)
    indices = tf.random.shuffle(tf.range(batch_size))
    mixed_images = lam * images + (1.0 - lam) * tf.gather(images, indices)
    mixed_labels = lam * labels + (1.0 - lam) * tf.gather(labels, indices)
    return mixed_images, mixed_labels

def apply_augments(images, labels):
    r = tf.random.uniform([])
    return tf.cond(
        r < 0.33,
        lambda: cutmix_batch(images, labels),
        lambda: tf.cond(
            r < 0.66,
            lambda: mixup_batch(images, labels),
            lambda: (images, labels)
        )
    )

print("CutMix / Mixup defined ✓")
"""))

# ── CELL 7: MODEL BUILDERS ─────────────────────────────────────────────────────
nb["cells"].append(md("## 5. Model Builders"))
nb["cells"].append(code("""\
def build_eb0_model(trainable_base=False):
    \"\"\"EfficientNetB0 NoisyStudent generalist (5 classes)\"\"\"
    inputs = tf.keras.Input(shape=(CONFIG['img_size'], CONFIG['img_size'], 3))
    base = tf.keras.applications.EfficientNetB0(
        include_top=False, weights='noisy-student',
        input_tensor=inputs
    )
    base.trainable = trainable_base
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(CONFIG['num_classes'], dtype='float32')(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32')(outputs)
    return tf.keras.Model(inputs, outputs)


def build_specialist_model(trainable_base=False, num_classes=4):
    \"\"\"ConvNeXtBase specialist (4 classes — CBB, CBSD, CGM, Healthy; no CMD)\"\"\"
    inputs = tf.keras.Input(shape=(CONFIG['img_size'], CONFIG['img_size'], 3))
    base = tf.keras.applications.ConvNeXtBase(
        include_top=False, weights='imagenet',
        input_tensor=inputs
    )
    base.trainable = trainable_base
    x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, dtype='float32')(x)
    outputs = tf.keras.layers.Activation('softmax', dtype='float32')(outputs)
    return tf.keras.Model(inputs, outputs)

print("Model builders defined ✓")
"""))

# ── CELL 8: EXP 10 TRAINING ────────────────────────────────────────────────────
nb["cells"].append(md("""## 6. Exp 10 — EfficientNetB0 NoisyStudent Training (5-Fold CV)
> **Skip this section if you already have saved `fold_{1-5}_exp10_eb0_model.keras` files.**
> Jump straight to Section 8 (Evaluation) to load and evaluate.
"""))
nb["cells"].append(code("""\
# ── Exp 10: Training loop ──────────────────────────────────────────────────────
# Builds train_ds with CutMix/Mixup, trains 3-phase, saves per fold

def build_train_ds_exp10(image_paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda p, l: tf.py_function(
        func=augment_image_fn, inp=[p, l], Tout=[tf.float32, tf.int64]
    ), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (
        tf.ensure_shape(x, [CONFIG['img_size'], CONFIG['img_size'], 3]),
        tf.ensure_shape(tf.cast(y, tf.int32), [])
    ))
    # One-hot for mixup/cutmix
    ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), CONFIG['num_classes'])))
    ds = ds.shuffle(1024).batch(CONFIG['batch_size'])
    ds = ds.map(apply_augments, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
exp10_fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df['clahe_path'], df['label'])):
    print(f"\\n{'='*40}\\nFOLD {fold+1}/5\\n{'='*40}")
    tf.keras.backend.clear_session(); gc.collect()

    train_df = df.iloc[train_idx]; val_df = df.iloc[val_idx]

    train_ds = build_train_ds_exp10(train_df['clahe_path'].values, train_df['label'].values)
    val_ds   = build_val_ds_eb0(val_df['clahe_path'].values, val_df['label'].values)

    # Class weights (softened for focal loss)
    cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weight_dict = {i: float(w) for i, w in enumerate(cw)}

    model = build_eb0_model(trainable_base=False)

    # ── PHASE 1: frozen backbone ──────────────────────────────────────────────
    print("[Phase 1] Frozen backbone")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),
        loss=focal_loss_per_class(GAMMA_EB0),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs_phase1'],
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)],
              verbose=1)

    # ── PHASE 2A: unfreeze top 30 layers ─────────────────────────────────────
    print("[Phase 2A] Partial unfreeze (top 30 layers)")
    for layer in model.layers[1].layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['fine_tune_lr'], clipnorm=1.0),
        loss=focal_loss_per_class(GAMMA_EB0),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs_phase2']//2,
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)],
              verbose=1)

    # ── PHASE 2B: full fine-tune ──────────────────────────────────────────────
    print("[Phase 2B] Full fine-tune")
    for layer in model.layers[1].layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-6, clipnorm=1.0),
        loss=focal_loss_per_class(GAMMA_EB0),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)],
              verbose=1)

    save_path = os.path.join(MODEL_DIR, f'fold_{fold+1}_exp10_eb0_model.keras')
    model.save(save_path)
    print(f"Saved: {save_path}")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    exp10_fold_accs.append(val_acc)
    print(f"Fold {fold+1} Accuracy: {val_acc:.4f}")

print(f"\\nExp 10 Mean: {np.mean(exp10_fold_accs):.4f} (+/- {np.std(exp10_fold_accs):.4f})")
# Expected: 0.8788 (+/- ~0.0090)
"""))

# ── CELL 9: SPECIALIST TRAINING ────────────────────────────────────────────────
nb["cells"].append(md("""## 7. Exp 11 — ConvNeXtBase Specialist Training
> **Skip if you already have `fold_{1-5}_specialist_model.keras`.**
> The specialist is trained only on **CBB, CBSD, CGM, Healthy** (4 classes, CMD excluded).
> Labels are remapped: CBB→0, CBSD→1, CGM→2, Healthy→3.
"""))
nb["cells"].append(code("""\
# ── Build specialist dataframe (no CMD = label 3) ─────────────────────────────
remap = {0: 0, 1: 1, 2: 2, 4: 3}   # original → specialist label

df_specialist = df[df['label'] != 3].copy()
df_specialist['label'] = df_specialist['label'].map(remap)
df_specialist = df_specialist.reset_index(drop=True)

print(f"Specialist images: {len(df_specialist)}")
print(df_specialist['label'].value_counts().sort_index().rename(SPECIALIST_LABEL_MAP))
"""))

nb["cells"].append(code("""\
# ── Specialist training loop ──────────────────────────────────────────────────
def build_train_ds_specialist(image_paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(preprocess_image_convnext, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (x, tf.one_hot(tf.cast(y, tf.int32), 4)))
    ds = ds.shuffle(1024).batch(CONFIG['batch_size'])
    ds = ds.map(apply_augments, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
spec_fold_accs = []

for fold, (train_idx, val_idx) in enumerate(
    skf.split(df_specialist['clahe_path'], df_specialist['label'])
):
    print(f"\\n{'='*40}\\nFOLD {fold+1}/5  [Specialist]\\n{'='*40}")
    tf.keras.backend.clear_session(); gc.collect()

    train_df = df_specialist.iloc[train_idx]
    val_df   = df_specialist.iloc[val_idx]

    train_ds = build_train_ds_specialist(train_df['clahe_path'].values, train_df['label'].values)
    val_ds   = build_val_ds_convnext(val_df['clahe_path'].values, val_df['label'].values)

    cw = compute_class_weight('balanced', classes=np.unique(train_df['label']), y=train_df['label'])
    class_weight_dict = {i: float(w) for i, w in enumerate(cw)}

    model = build_specialist_model(trainable_base=False, num_classes=4)

    # Phase 1
    print("[Phase 1] Frozen backbone")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['learning_rate']),
        loss=focal_loss_per_class(GAMMA_SPECIALIST),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs_phase1'],
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=3, min_lr=1e-7, verbose=1)],
              verbose=1)

    # Phase 2A
    print("[Phase 2A] Partial unfreeze (top 30)")
    for layer in model.layers[1].layers[-30:]:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['fine_tune_lr']),
        loss=focal_loss_per_class(GAMMA_SPECIALIST),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=CONFIG['epochs_phase2']//2,
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)],
              verbose=1)

    # Phase 2B
    print("[Phase 2B] Full fine-tune")
    for layer in model.layers[1].layers:
        layer.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(CONFIG['fine_tune_lr'] * 0.3),
        loss=focal_loss_per_class(GAMMA_SPECIALIST),
        metrics=['accuracy']
    )
    model.fit(train_ds, validation_data=val_ds, epochs=10,
              class_weight=class_weight_dict,
              callbacks=[tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.2, patience=2, min_lr=1e-7, verbose=1)],
              verbose=1)

    save_path = os.path.join(MODEL_DIR, f'fold_{fold+1}_specialist_model.keras')
    model.save(save_path)
    print(f"Saved: {save_path}")

    val_loss, val_acc = model.evaluate(val_ds, verbose=0)
    spec_fold_accs.append(val_acc)
    print(f"Fold {fold+1} Accuracy: {val_acc:.4f}")

print(f"\\nSpecialist Mean: {np.mean(spec_fold_accs):.4f} (+/- {np.std(spec_fold_accs):.4f})")
"""))

# ── CELL 10: EVALUATION ────────────────────────────────────────────────────────
nb["cells"].append(md("""## 8. Exp 11 Evaluation — Load Saved Models & Run Ensemble
Run this section to reproduce the 90.32% result using saved `.keras` models.
No retraining needed — just point `MODEL_DIR` to where the `.keras` files are.
"""))

nb["cells"].append(code("""\
# ── Ensemble weights (per-class F1 from training) ─────────────────────────────
# EB0 generalist F1:  CBB=0.63, CBSD=0.79, CGM=0.80, CMD=0.96, Healthy=0.72
# Specialist F1:      CBB=0.80, CBSD=0.89, CGM=0.93, CMD=0.00, Healthy=0.85
# CMD specialist weight = 0.0 → generalist handles CMD alone

EB0_WEIGHTS        = np.array([0.63, 0.79, 0.80, 0.96, 0.72])  # 5 classes
SPECIALIST_WEIGHTS = np.array([0.80, 0.89, 0.93, 0.00, 0.85])  # 5 classes (CMD=0)

# Specialist outputs 4 classes — need to expand to 5 with CMD=0
# Specialist class order: CBB=0, CBSD=1, CGM=2, Healthy=3
# Full order:             CBB=0, CBSD=1, CGM=2, CMD=3, Healthy=4
SPECIALIST_TO_FULL = [0, 1, 2, 4]  # indices in the full 5-class output

def expand_specialist_probs(spec_probs):
    \"\"\"Expand 4-class specialist probs to 5-class (insert 0 for CMD at index 3)\"\"\"
    batch = spec_probs.shape[0]
    full  = np.zeros((batch, 5), dtype=np.float32)
    full[:, SPECIALIST_TO_FULL] = spec_probs
    return full

def weighted_ensemble(eb0_probs, spec_probs_full):
    \"\"\"
    Per-class weighted combination:
      combined[c] = (eb0[c]*w_eb0[c] + spec[c]*w_spec[c]) / (w_eb0[c] + w_spec[c])
    CMD: spec weight=0 → only EB0 contributes
    \"\"\"
    w_sum = EB0_WEIGHTS + SPECIALIST_WEIGHTS
    # avoid div by zero (shouldn't happen but safe)
    w_sum = np.where(w_sum == 0, 1.0, w_sum)
    combined = (eb0_probs * EB0_WEIGHTS + spec_probs_full * SPECIALIST_WEIGHTS) / w_sum
    return combined

print("Ensemble config loaded ✓")
print(f"EB0 weights:        {EB0_WEIGHTS}")
print(f"Specialist weights: {SPECIALIST_WEIGHTS}")
"""))

nb["cells"].append(code("""\
# ── Run 5-fold evaluation with Exp 11 ensemble ────────────────────────────────
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

all_true   = []
all_pred   = []
all_probs  = []
exp11_fold_accs = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df['clahe_path'], df['label'])):
    print(f"\\n--- Fold {fold+1}/5 ---")

    val_df = df.iloc[val_idx]

    # Load saved models
    eb0_path  = os.path.join(MODEL_DIR, f'fold_{fold+1}_exp10_eb0_model.keras')
    spec_path = os.path.join(MODEL_DIR, f'fold_{fold+1}_specialist_model.keras')

    print(f"  Loading EB0:        {eb0_path}")
    print(f"  Loading Specialist: {spec_path}")

    eb0_model  = tf.keras.models.load_model(eb0_path,
        custom_objects={'loss': focal_loss_per_class(GAMMA_EB0)})
    spec_model = tf.keras.models.load_model(spec_path,
        custom_objects={'loss': focal_loss_per_class(GAMMA_SPECIALIST)})

    # Build validation datasets
    val_ds_eb0  = build_val_ds_eb0(val_df['clahe_path'].values, val_df['label'].values)
    val_ds_spec = build_val_ds_convnext(val_df['clahe_path'].values, val_df['label'].values)

    # Get probabilities
    eb0_probs  = eb0_model.predict(val_ds_eb0, verbose=0)       # (N, 5)
    spec_probs = spec_model.predict(val_ds_spec, verbose=0)     # (N, 4)
    spec_full  = expand_specialist_probs(spec_probs)            # (N, 5)

    # Ensemble
    combined   = weighted_ensemble(eb0_probs, spec_full)
    preds      = np.argmax(combined, axis=1)
    true_labels = val_df['label'].values

    fold_acc = np.mean(preds == true_labels)
    exp11_fold_accs.append(fold_acc)
    all_true.extend(true_labels)
    all_pred.extend(preds)
    all_probs.extend(combined)

    print(f"  Fold {fold+1} Acc: {fold_acc:.4f}")

    del eb0_model, spec_model
    tf.keras.backend.clear_session(); gc.collect()

print(f"\\n{'='*50}")
print(f"Exp 11 Mean: {np.mean(exp11_fold_accs):.4f} (+/- {np.std(exp11_fold_accs):.4f})")
print(f"Per-fold:    {[f'{a:.4f}' for a in exp11_fold_accs]}")
print(f"Expected:    0.9032 (+/- 0.0078)")
"""))

# ── CELL 11: CLASSIFICATION REPORT ────────────────────────────────────────────
nb["cells"].append(md("## 9. Per-Class Metrics & Confusion Matrix"))
nb["cells"].append(code("""\
all_true = np.array(all_true)
all_pred = np.array(all_pred)

print("\\n=== Exp 11 — Classification Report (All Folds Combined) ===")
print(classification_report(all_true, all_pred, target_names=CLASS_NAMES, digits=2))
"""))

nb["cells"].append(code("""\
# ── Confusion Matrix ──────────────────────────────────────────────────────────
cm = confusion_matrix(all_true, all_pred)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax)
ax.set_xlabel('Predicted', fontsize=12)
ax.set_ylabel('Actual', fontsize=12)
ax.set_title('Exp 11 Weighted Ensemble — Confusion Matrix (All Folds)', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'exp11_confusion_matrix.png'), dpi=150)
plt.show()
print("Saved: exp11_confusion_matrix.png")
"""))

nb["cells"].append(code("""\
# ── Per-fold accuracy bar chart ───────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 6), exp11_fold_accs, color='steelblue', alpha=0.8)
ax.axhline(np.mean(exp11_fold_accs), color='red', linestyle='--',
           label=f'Mean = {np.mean(exp11_fold_accs):.4f}')
ax.set_xlabel('Fold'); ax.set_ylabel('Accuracy')
ax.set_title('Exp 11 — Per-Fold Accuracy')
ax.set_ylim(0.85, 0.95)
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'exp11_fold_accuracy.png'), dpi=150)
plt.show()
"""))

# ── CELL 12: INFERENCE ─────────────────────────────────────────────────────────
nb["cells"].append(md("""## 10. Inference on New Images
Use this section to run the Exp 11 ensemble on images **not in the training set**.
Point `IMAGE_DIR` to a folder of `.jpg` images (no labels needed).
"""))
nb["cells"].append(code("""\
# ── Inference config ──────────────────────────────────────────────────────────
IMAGE_DIR = '/path/to/your/new/images'   # <-- change this
CONFIDENCE_THRESHOLD = 0.60              # below this → 'Uncertain'

# Load all 5 folds of each model
print("Loading all fold models for inference...")

eb0_models  = []
spec_models = []

for fold in range(1, 6):
    eb0_path  = os.path.join(MODEL_DIR, f'fold_{fold}_exp10_eb0_model.keras')
    spec_path = os.path.join(MODEL_DIR, f'fold_{fold}_specialist_model.keras')
    eb0_models.append(tf.keras.models.load_model(eb0_path,
        custom_objects={'loss': focal_loss_per_class(GAMMA_EB0)}))
    spec_models.append(tf.keras.models.load_model(spec_path,
        custom_objects={'loss': focal_loss_per_class(GAMMA_SPECIALIST)}))

print(f"Loaded {len(eb0_models)} EB0 models and {len(spec_models)} specialist models ✓")
"""))

nb["cells"].append(code("""\
def preprocess_single_eb0(img_path):
    \"\"\"Preprocess a single image for EB0 inference (CLAHE + EB0 norm)\"\"\"
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    # CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size']))
    img = tf.keras.applications.efficientnet.preprocess_input(img.astype(np.float32))
    return img[np.newaxis, ...]   # (1, H, W, 3)

def preprocess_single_convnext(img_path):
    \"\"\"Preprocess a single image for ConvNeXt inference (CLAHE + ImageNet norm)\"\"\"
    img = cv2.imread(img_path)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(2.0, (8, 8)).apply(l)
    lab = cv2.merge([l, a, b])
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (CONFIG['img_size'], CONFIG['img_size'])).astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406]); std = np.array([0.229, 0.224, 0.225])
    img = (img - mean) / std
    return img[np.newaxis, ...]

def predict_single(img_path):
    \"\"\"Run Exp 11 ensemble on one image. Returns predicted class, confidence, all probs.\"\"\"
    x_eb0  = preprocess_single_eb0(img_path)
    x_spec = preprocess_single_convnext(img_path)

    # Average across all 5 folds
    eb0_avg  = np.mean([m.predict(x_eb0,  verbose=0) for m in eb0_models],  axis=0)
    spec_avg = np.mean([m.predict(x_spec, verbose=0) for m in spec_models], axis=0)

    spec_full  = expand_specialist_probs(spec_avg)
    combined   = weighted_ensemble(eb0_avg, spec_full)[0]  # (5,)

    pred_idx   = int(np.argmax(combined))
    confidence = float(combined[pred_idx])

    label = LABEL_MAP[pred_idx] if confidence >= CONFIDENCE_THRESHOLD else 'Uncertain'
    return {
        'predicted_class': LABEL_MAP[pred_idx],
        'label'          : label,
        'confidence'     : confidence,
        'probabilities'  : {LABEL_MAP[i]: float(combined[i]) for i in range(5)}
    }

print("predict_single() ready ✓")
print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
"""))

nb["cells"].append(code("""\
# ── Batch inference on IMAGE_DIR ──────────────────────────────────────────────
image_files = [f for f in Path(IMAGE_DIR).glob('*.jpg')]
print(f"Found {len(image_files)} images in {IMAGE_DIR}")

results = []
for img_path in tqdm(image_files):
    try:
        r = predict_single(str(img_path))
        r['image'] = img_path.name
        results.append(r)
    except Exception as e:
        results.append({'image': img_path.name, 'error': str(e)})

results_df = pd.DataFrame(results)
print("\\n=== Inference Results ===")
print(results_df[['image', 'predicted_class', 'confidence', 'label']].to_string())

# Save to CSV
out_csv = os.path.join(MODEL_DIR, 'inference_results.csv')
results_df.to_csv(out_csv, index=False)
print(f"\\nSaved: {out_csv}")
"""))

nb["cells"].append(code("""\
# ── Visualise predictions on a sample ─────────────────────────────────────────
sample = results_df.dropna(subset=['predicted_class']).head(9)
n = min(len(sample), 9)
fig, axes = plt.subplots(3, 3, figsize=(12, 12))

for i, (_, row) in enumerate(sample.iterrows()):
    ax = axes[i // 3][i % 3]
    img_path = os.path.join(IMAGE_DIR, row['image'])
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax.imshow(img)
    color = 'green' if row['confidence'] >= CONFIDENCE_THRESHOLD else 'orange'
    ax.set_title(f"{row['label']}\\n{row['confidence']:.1%}", color=color, fontsize=10)
    ax.axis('off')

plt.suptitle('Exp 11 Ensemble — Sample Predictions', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'sample_predictions.png'), dpi=150)
plt.show()
"""))

# ── CELL 13: EXP 10 VS 11 COMPARISON ──────────────────────────────────────────
nb["cells"].append(md("## 11. Exp 10 vs Exp 11 Summary"))
nb["cells"].append(code("""\
# ── Compare Exp 10 solo vs Exp 11 ensemble ────────────────────────────────────
# (Run after both evaluation sections above)

summary = {
    'Model'   : ['Exp 10 (EB0 Solo)', 'Exp 11 (Ensemble)'],
    'Mean Acc': [np.mean(exp10_fold_accs) if 'exp10_fold_accs' in dir() else 0.8788,
                 np.mean(exp11_fold_accs)],
    'Std'     : [np.std(exp10_fold_accs)  if 'exp10_fold_accs' in dir() else 0.0090,
                 np.std(exp11_fold_accs)],
}
summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Per-class F1 from paper (hardcoded for reference)
print("\\n=== Per-class F1 comparison (from paper) ===")
comparison = pd.DataFrame({
    'Class'   : CLASS_NAMES,
    'Exp10 F1': [0.63, 0.79, 0.80, 0.96, 0.72],
    'Exp11 F1': [0.69, 0.89, 0.84, 0.95, 0.81],
    'Delta'   : ['+0.06', '+0.10', '+0.04', '-0.01', '+0.09'],
})
print(comparison.to_string(index=False))
"""))

# Write notebook
with open('cassava_exp10_exp11.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)

print("Notebook written ✓")
