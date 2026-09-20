# Project Milestones & Action Items (`ver1`)

## ✅ Completed
- [x] Analyze `round2` dataset statistics (62,707 images, 72 classes, resolution, imbalance).
- [x] Analyze file naming structure for data leakage (`doc_id`, `sg`/`tg` counterparts, `page_id`).
- [x] Design multi-tier deterministic group-aware 80/20 train/test splitting algorithm.
- [x] Create project workspace in `ver1/` and migrate exploration files.
- [x] Create system context and architecture blueprint `AGENT.md`.
- [x] Create `requirements.txt` with required dependencies.
- [x] Create dedicated Python virtual environment (`.venv`) and install dependencies.
- [x] Implement `src/dataset.py` (TIS-620 mapping, deterministic split, letterboxing, balanced dataloader).
- [x] Implement `src/transforms.py` (Morphological dilation/erosion, bounded geometric augmentations).
- [x] Implement `src/models.py` (CustomGlyphCNN, AdaptedResNet18, AdaptedMobileNetV3).
- [x] Implement `src/losses.py` (Class-Balanced Focal Loss, Label Smoothing).
- [x] Implement `src/trainer.py` (AMP mixed precision, Warmup + Cosine Annealing, checkpointing).
- [x] Implement `src/evaluate.py` (Confusion matrix, Top-1/Top-5 accuracy, per-class F1, visual inspection).
- [x] Create Jupyter experiment notebooks (`01_data_split_and_leakage_check.ipynb`, `02_model_training_benchmark.ipynb`, `03_evaluation_and_error_analysis.ipynb`).
- [x] Implement and execute unit test suite `tests/test_pipeline.py` (100% PASS).
- [x] Verify end-to-end integration and backward pass on full 62,707 real image dataset.

## 🚀 Next Steps / Experiments
- [ ] Run benchmark training runs for `CustomGlyphCNN` vs `AdaptedResNet18` vs `AdaptedMobileNetV3`.
- [ ] Generate comparative evaluation metrics and confusion matrices.
