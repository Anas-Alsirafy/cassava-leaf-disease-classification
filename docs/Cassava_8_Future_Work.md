# Cassava 8 — Future Work

Oct 26, 2026 · Anas Alsirafy

First make the 90.32% number clean by retraining the specialist on matching folds; then add a held-out test and finish the edge pipeline. The order is a suggestion; Prof. Habib sets the final one.

## High priority

- [ ] Retrain the ConvNeXt-Base specialist on the same folds as the other ensemble members and re-score Exp 11.
- [ ] Fix the code bugs in Known Issues (backbone unfreezing, duplicate `mixup_batch`, `img_size` default, leftover `if fold < N` lines).
- [ ] Re-score the ensemble on grouped folds with a validation slice carved from each training fold, matching the cross-dataset protocol.

## Medium priority

- [ ] Rebuild a 15% held-out test partition and report it next to the cross-validation numbers.
- [ ] Add seeds 42 and 1337 to the fine-tuned ConvNeXt-Base cassava run (about 4 GPU-hours per fold × 10 folds).
- [ ] Report per-class F1 and confusion matrices for Exp 10 and Exp 11.
- [ ] Recover and document Experiments 6–9.

## Low priority

- [ ] Quantise to INT8 (TensorFlow Lite for the Pi, TensorRT for the Jetson); measure latency and accuracy loss against 254 ms and 65 ms.
- [ ] Compare smaller backbones (MobileNetV3, EfficientNet-B0 alone) on the accuracy-vs-latency trade-off.
- [ ] Run the FIWARE chain end to end (ESP32-CAM → MQTT → IoT Agent → Orion → Jetson → Cygnus → database) on live images.
- [ ] Try frozen foundation-model features (DINOv2, CLIP) as a cheaper alternative to fine-tuning.
