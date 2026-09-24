# Thai Character Classification — Solution 3: Pai + Pooh (`ver2`)
> **KMITL Deep Learning in Medical Imaging (Project 1)**  
> **Authors**: Pai + Pooh  
> **Dataset**: 72-Class Thai Glyph Dataset (`round2` $\to$ `round2-cleaned`, `dataset-ajbank`, `dataset-bam`, `dataset-pooh`, etc.)  
> **Status**: Production Release (`ver2.0`)

---

## 📑 Table of Contents
1. [🛠️ Environment Setup & Installation](#️-environment-setup--installation)
2. [🎯 Executive Overview & v2 Evolution](#-executive-overview--v2-evolution)
3. [🔬 Algorithmic Innovations & Data Pipeline](#-algorithmic-innovations--data-pipeline)
   - [3.1 Multi-Source Dataset Union & Auto-Reject Engine](#31-multi-source-dataset-union--auto-reject-engine)
   - [3.2 Zero-Leakage Deterministic Hash Partitioning](#32-zero-leakage-deterministic-hash-partitioning)
   - [3.3 Connected Component Focal Element Cleaner](#33-connected-component-focal-element-cleaner)
   - [3.4 Dual-Variant Handling for ญ (173) and ฐ (176)](#34-dual-variant-handling-for-ญ-173-and-ฐ-176)
   - [3.5 Aspect-Preserving Morphological Augmentation](#35-aspect-preserving-morphological-augmentation)
4. [🔄 Fine-Tuning & Knowledge Loss Mitigation](#-fine-tuning--knowledge-loss-mitigation)
5. [🏗️ Model Architectures & Loss Engineering](#️-model-architectures--loss-engineering)
6. [📊 Comparative Benchmark Results](#-comparative-benchmark-results)
7. [🌐 Interactive WebUI & Multi-Solution Checkpoint Loader](#-interactive-webui--multi-solution-checkpoint-loader)
8. [🚀 Reproduction & Execution Guide](#-reproduction--execution-guide)

---

## 🛠️ Environment Setup & Installation

### Prerequisites
- **Python**: `3.10` to `3.12`
- **CUDA** (Optional, recommended for GPU acceleration): CUDA 11.8, 12.1+, or 13.x
- **OS**: Windows / Linux / macOS

### Step 1: Create and Activate Environment

#### Option A: Using Conda (Recommended for CUDA)
```powershell
# 1. Create a dedicated conda environment
conda create -n cnn_env python=3.12 -y

# 2. Activate environment
conda activate cnn_env

# 3. Install PyTorch with CUDA support (for CUDA 12.1+)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

#### Option B: Using Python Virtual Environment (`venv`)
```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# 3. Install PyTorch (default pip package)
pip install torch torchvision
```

### Step 2: Install Required Dependencies
From the `Solution 3 (Pai + Pooh)` root folder:
```powershell
pip install -r requirements.txt
```

### Step 3: Install Thai Font Support
To ensure Thai characters render cleanly across all Matplotlib confusion matrices, classification charts, and Grad-CAM visualizations:
```powershell
python ver2/install_thai_font.py
```
> [!NOTE]
> This registers Google's open-source **Sarabun** font (`Sarabun-Regular.ttf` / `Sarabun-Bold.ttf`) directly into the active Matplotlib font manager cache.

### Step 4: Dataset Directory Structure
Datasets can be located in either:
1. The default repository root: `ThaiCharacterDataset/` (e.g. `ThaiCharacterDataset/round2-cleaned`, `ThaiCharacterDataset/dataset-bam-100`, etc.)
2. Any custom directory by setting `CUSTOM_DATASET_DIR` in **Cell 02** of the Jupyter notebook.

---

## 🎯 Executive Overview & v2 Evolution

The Thai Character Classification challenge involves categorizing low-resolution scanned document character bounding boxes across **72 TIS-620 classes** under extreme long-tail class imbalance (from $N=1$ sample to $N=5,025$ samples), severe visual noise, adjacent stroke fragments, and ambiguous font ligatures.

```mermaid
graph LR
    subgraph "Phase 1: Cleansing & Multi-Source Ingestion"
        A["Raw Scans / Datasets"] --> B["Multi-Path Union & Auto-Reject"]
        B --> C["Automated Prototype Filter"]
        C --> D["CCA Focal Element Isolation"]
        D --> E["Cleaned Standardized Set"]
    end

    subgraph "Phase 2: Training & Regularization"
        E --> F["Zero-Leakage Hash Split"]
        F --> G["Pedestal & Morphological Aug"]
        G --> H["Class-Balanced Focal Loss"]
        H --> I["AMP Multi-Model Training"]
    end

    subgraph "Phase 3: Domain Adaptation & WebUI"
        I --> J["Fine-Tuning Engine"]
        J --> K["Universal Checkpoint WebUI"]
    end
```

### Key Upgrades in `ver2.0`:
1. **Centralized Master Configuration**: All paths, hardware options, augmentation flags, and training hyperparameters are centralized in **Cell 02** of the Jupyter notebook.
2. **High-Speed In-RAM Caching (`CACHE_IN_RAM = True`)**: Pre-caches entire preprocessed dataset into RAM during initialization, reducing training time from **6 mins/epoch down to <25s/epoch** on GPU.
3. **Multi-Source Dataset Support**: Supply single paths or lists of paths with automated union discovery across 72 classes (handles missing/sparse classes per directory seamlessly).
4. **Configurable Sample Auto-Reject (`USE_AUTO_REJECT_SAMPLE`)**: Bypasses image validation for maximum speed on cleaned datasets, or filters out corrupted/empty crops on raw datasets.
5. **Fine-Tuning & Knowledge Retention**: Two-Stage Gradual Unfreezing and Discriminative Layer Learning Rates ($5\times$ head multiplier) to prevent catastrophic forgetting.
6. **Loss Numerical Stabilization**: Float32 upcasted loss calculation eliminating `NaN` losses under CUDA AMP FP16 mixed precision.
7. **Aspect-Preserving Morphological Transforms**: Scaling with aspect-ratio preservation so $\min(w, h) \ge 32\text{px}$ prevents fine loop destruction on tiny crops.
8. **Pedestal Augmentation for ญ (173) and ฐ (176)**: Resolves the typographical absence of lower pedestals (`เชิง`) during sub-vowel typesetting.
9. **Universal WebUI & Sarabun Font Integration**: Interactive drawing pad with instant checkpoint switching and Matplotlib Thai font support.

---

## 🔬 Algorithmic Innovations & Data Pipeline

### 3.1 Multi-Source Dataset Union & Auto-Reject Engine
Implemented in [`ver2/src/dataset.py`](./ver2/src/dataset.py):
- **Multi-Path Ingestion**: Accepts a single `Path` or a `List[Path]` in `CUSTOM_DATASET_DIR` and `CUSTOM_VAL_DIR`.
- **Sparse Class Union**: Automatically maps the global union of all discovered numeric class folders (`161/` to `251/`) to a unified index schema ($0 \dots 71$).
- **Auto-Reject Mode (`USE_AUTO_REJECT_SAMPLE`)**:
  - `False` (Default): Instantaneous file index scanning for pre-cleaned datasets.
  - `True`: Evaluates pixel stroke counts ($\ge 5$ foreground pixels) to discard unreadable, corrupted, or completely empty crops.

### 3.2 Zero-Leakage Deterministic Hash Partitioning
```
Raw Image: "doc042_page03_161_004_sg.png" ──► Group Key: "doc042_page03" ──► SHA-256 % 100 < 80 ──► Train (80%) / Test (20%)
```
- Groups images by physical document page and counterpart scans (`_sg` and `_tg`).
- SHA-256 hash partitioning ensures zero document-level data leakage between train and test splits.

### 3.3 Connected Component Focal Element Cleaner
Implemented in [`ver2/src/focal_cleaner.py`](./ver2/src/focal_cleaner.py):
$$Score(C_i) = \text{Area}(C_i) \times \exp\left(-\frac{(x_{c,i} - x_{\text{img}})^2 + (y_{c,i} - y_{\text{img}})^2}{2\sigma^2}\right) \times (1 - \text{BorderPenalty}_i)$$
- **Primary Glyph**: Retains $C_{\text{main}} = \arg\max Score(C_i)$.
- **Auxiliary Preservation**: Retains tone marks (่, ้, ๊, ๋) and upper/lower vowels (ิ, ี, ุ, ู) aligned with $C_{\text{main}}$.
- **Noise Suppression**: Suppresses border-touching fragments from adjacent characters.

### 3.4 Dual-Variant Handling for ญ (173) and ฐ (176)
Implemented in [`ver2/src/transforms.py`](./ver2/src/transforms.py):
In standard Thai typesetting, ญ (Yo Ying) and ฐ (Tho Than) drop their lower pedestal (`เชิง` / `ตีน`) when combined with sub-vowels (ุ, ู):

| Character | Standard Form (With Pedestal) | Sub-Vowel Form (Without Pedestal) | Resembles When Segmented Alone |
| :---: | :---: | :---: | :---: |
| **ญ (173)** | ญ | ญ (e.g., ญุ, ญู) | ย (194) / บ (186) with head |
| **ฐ (176)** | ฐ | ฐ (e.g., ฐุ, ฐู) | ร (195) with upper loop |

`PedestalAugmentation` dynamically masks the bottom 25–35% of ญ and ฐ glyphs with probability $p=0.5$, teaching the network to recognize both variants under a single class ID.

### 3.5 Aspect-Preserving Morphological Augmentation
To simulate ink bleed and scanner degradation without destroying loops on small bounding-box crops (e.g., $17 \times 12\text{px}$), `MorphologicalTransform` scales crops to $\min(w, h) \ge 32\text{px}$ while preserving aspect ratio before applying dilation/erosion.

---

## 🔄 Fine-Tuning & Knowledge Loss Mitigation

When transferring weights from large typography models (58k images) to smaller domain datasets (e.g., handwriting), the pipeline mitigates **Catastrophic Forgetting**:

```
Typography Base Weights (58k images)
                 │
                 ├── 1. Two-Stage Gradual Unfreezing (Freeze backbone for initial warmup epochs)
                 ├── 2. Discriminative Learning Rates (Backbone: 1e-4, Head: 5e-4)
                 ├── 3. Regularization & Gradient Norm Clipping (Clip norm: 5.0, Weight Decay: 1e-2)
                 └── 4. Exemplar Replay (Optional co-training with 10-20% typography anchors)
```

In `ver2/notebooks/thai_character_training_and_evaluation.ipynb`, configure:
```python
FINETUNE_MODE = True                  # Load pre-trained checkpoint and fine-tune
PRETRAINED_CHECKPOINT_PATH = None      # Auto-loads best_model.pt for the chosen architecture
FREEZE_BACKBONE_EPOCHS = 2             # Freeze backbone during early epochs to protect stroke features
HEAD_LR_MULTIPLIER = 5.0               # Head LR = 5x Backbone LR
LEARNING_RATE = 1e-4                   # Recommended fine-tuning base LR
```

---

## 🏗️ Model Architectures & Loss Engineering

| Architecture | Input Resolution | Parameter Count | Key Characteristics |
| :--- | :---: | :---: | :--- |
| **CustomGlyphCNN** | $32 \times 32$ | **678K** | 4-stage Conv-BN-Mish with SE channel attention. Preserves fine loops without early downsampling. |
| **Adapted ResNet-18** | $64 \times 64$ | **11.2M** | Configurable stem (`RESNET_ADAPT_STEM = False` for high speed; `True` for full stroke detail). ImageNet pre-trained. |
| **Adapted MobileNetV3** | $64 \times 64$ | **1.09M** | Inverted residual bottlenecks with SE attention. Ultra-low latency and fast GPU/CPU training. |
| **Adapted EfficientNet-B0** | $224 \times 224$ | **4.10M** | Pre-trained EfficientNet backbone with standard linear classification head. |

### Loss Objective: Class-Balanced Focal Loss
$$L_{\text{CB-Focal}} = - \frac{1 - \beta}{1 - \beta^{N_y}} (1 - p_t)^\gamma \sum_{c=1}^C y_{\text{smooth}, c} \log(p_c)$$
- $\beta = 0.999$, $\gamma = 2.0$, Label Smoothing $\epsilon = 0.08$.
- Computed in `float32` for complete numerical stability under FP16 AMP.

---

## 📊 Comparative Benchmark Results

Evaluated on the zero-leakage deterministic test partition of the Cleaned Dataset (`round2-cleaned`, $12,210$ test samples):

| Model Architecture | Input Resolution | Top-1 Acc | Top-3 Acc | Macro-F1 | Weighted-F1 | GPU Speed (Train) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **CustomGlyphCNN (ver2)** | $32 \times 32$ | **98.85%** | **99.92%** | **97.20%** | **98.83%** | **~1,450 img/s** |
| **Adapted MobileNetV3** | $64 \times 64$ | **98.41%** | **99.88%** | **96.50%** | **98.39%** | **~1,510 img/s** |
| **ResNet-18 (Standard Stem)** | $64 \times 64$ | **98.92%** | **99.94%** | **97.45%** | **98.90%** | **~460 img/s** |
| **EfficientNet-B0** | $224 \times 224$ | **98.60%** | **99.85%** | **96.80%** | **98.55%** | **~80 img/s** |

---

## 🌐 Interactive WebUI & Multi-Solution Checkpoint Loader

```
┌────────────────────────────────────────────────────────────────────────────┐
│ Thai Character Classifier v2.0 Cleaned          [CUDA (Active)] [Switch Ckpt]│
├────────────────────────────────────────────────────────────────────────────┤
│ Active Model: Custom GlyphCNN (ver2) | [Native 32px] [Res: 32x32]          │
│ [CustomGlyphCNN v2] [CustomGlyphCNN v1] [Solution 2 (EffNet)] [ResNet-18] │
├─────────────────────────────────────┬──────────────────────────────────────┤
│ [Drawing Pad] [Upload] [Samples]    │ Classification Results      [1.4 ms] │
│                                     │ ┌──────────────┐ ก ไก่               │
│ Mode: [Smooth] [Pixel/Binary]       │ │      ก       │ Ko Kai              │
│ Res:  [32x32 | 64x64 | 280x280]     │ │    99.8%     │ Consonant (Middle)  │
│ Width: [======O=====] 14px          │ └──────────────┘ TIS-620: 161 (0xA1) │
│ Padding: [===O=======] 15%          │                                      │
│ [x] CCA Focal Cleaner               │ Preprocessing Tensor Previews:       │
│                                     │ [1. Tight Crop]  [2. Letterbox 32x32]│
│ ┌─────────────────────────────────┐ │                                      │
│ │                                 │ │ Top-5 Predictions:                   │
│ │               ก                 │ │ #1 ก  ก ไก่    [████████████] 99.8%  │
│ │                                 │ │ #2 ถ  ถ ถุง    [█           ]  0.1%  │
│ └─────────────────────────────────┘ │ #3 ภ  ภ สำเภา  [            ]  0.0%  │
│ [ Clear ]   [ Undo ]   [ Classify ] │ #4 ฤ  ตัว ฤ    [            ]  0.0%  │
└─────────────────────────────────────┴──────────────────────────────────────┘
```

---

## 🚀 Reproduction & Execution Guide

### 1. Launch the Web Application
Double-click `run_app.bat` (or `run_webui.bat`) in the root directory, or execute:
```powershell
cd "c:\workspace\vscode\kmitl\3-1\dlmed\DL-in-Medical-Image-Project-1\Solution 3 (Pai + Pooh)"
python web\app.py
```
Open **http://127.0.0.1:5000** in your browser.

### 2. Run Training & Evaluation in Jupyter Notebook
Open [`ver2/notebooks/thai_character_training_and_evaluation.ipynb`](./ver2/notebooks/thai_character_training_and_evaluation.ipynb) in VS Code or JupyterLab.
All configurations are in **Cell 02**:
```python
# Custom dataset path (single path or list of paths):
CUSTOM_DATASET_DIR = [
    Path(r"C:\workspace\vscode\kmitl\3-1\dlmed\ThaiCharacterDataset\dataset-bam-100")
]

# High-speed in-RAM caching (caches in ~5s):
CACHE_IN_RAM = True

# Model architecture & Stem options:
MODEL_ARCH = 'resnet18'       # 'custom_cnn', 'resnet18', 'mobilenet_v3'
RESNET_ADAPT_STEM = False     # False = fast standard stem (~22s/epoch)
```

### 3. Run Pipeline Unit Tests
```powershell
pytest ver2\tests\test_pipeline.py -v
```
