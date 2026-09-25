"""
Configuration module for Thai Character Recognition.
Contains hyperparameters, dataset/model paths, device configuration,
and Thai character mapping dictionary with TIS-620 decoding.
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import numpy as np
import torch


# ==========================================
# 1. Path Configuration
# ==========================================
# โฟลเดอร์ Dataset เป็นตัวแปรกำหนดค่า (Configurable Path) เริ่มต้นที่ Path("./data")
DATA_DIR: Path = Path(os.environ.get("THAI_DATA_DIR", "./data"))

# Path สำหรับ Checkpoint โมเดล (ลำดับแรก: best_thai_character_finetuned.pth)
DEFAULT_CHECKPOINT_PATH: Path = Path("best_thai_character_finetuned.pth")

# รายการค้นหา Checkpoint ตามลำดับความสำคัญ (Fallback candidates)
FALLBACK_CHECKPOINT_PATHS: List[Path] = [
    Path("best_thai_character_finetuned.pth"),
    Path("./outputs/best_thai_character_finetuned.pth"),
    Path("best_thai_character_model_v2.pth"),
    Path("./outputs/best_thai_character_model_v2.pth"),
    Path("D:/best_thai_character_finetuned.pth"),
    Path("D:/best_thai_character_model_v2.pth"),
]

def resolve_checkpoint_path(custom_path: Optional[Union[str, Path]] = None) -> Path:
    """
    ค้นหาไฟล์ Checkpoint ตามลำดับความสำคัญ:
    1. Path ที่ผู้ใช้ระบุโดยตรง (หากมีไฟล์อยู่จริง)
    2. best_thai_character_finetuned.pth ในโฟลเดอร์ปัจจุบัน
    3. ./outputs/best_thai_character_finetuned.pth
    4. best_thai_character_model_v2.pth
    5. ./outputs/ หรือ ไดรฟ์ D:/
    """
    if custom_path is not None:
        p = Path(custom_path)
        if not p.exists():
            raise FileNotFoundError(f"ไม่พบไฟล์ Checkpoint ตามที่ระบุ: {p.resolve()}")
        return p

    for candidate in FALLBACK_CHECKPOINT_PATHS:
        if candidate.exists():
            return candidate

    return DEFAULT_CHECKPOINT_PATH


# Output directory สำหรับบันทึกผลลัพธ์ เช่น curves, logs, csv
OUTPUT_DIR: Path = Path("./outputs")
CURVE_IMAGE_PATH: Path = OUTPUT_DIR / "training_curves_v2.png"
HISTORY_CSV_PATH: Path = OUTPUT_DIR / "training_history.csv"
CONFUSION_MATRIX_PATH: Path = OUTPUT_DIR / "confusion_matrix.png"
CLASSIFICATION_REPORT_PATH: Path = OUTPUT_DIR / "classification_report.csv"
AUGMENTED_TRAIN_DIR: Path = OUTPUT_DIR / "augmented_train"


# ==========================================
# 2. Hardware & Reproducibility Settings
# ==========================================
SEED: int = 42

def set_seed(seed: int = SEED) -> None:
    """กำหนด Seed สำหรับ Random Generators ทั้งหมดเพื่อความเสถียรของผลลัพธ์"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 3. Dataset & DataLoader Settings
# ==========================================
IMAGE_SIZE: int = 224
BATCH_SIZE: int = 256
NUM_WORKERS: int = 2
PIN_MEMORY: bool = True if torch.cuda.is_available() else False
VAL_RATIO: float = 0.20
MIN_TRAIN_SAMPLES_PER_CLASS: int = 80

VALID_EXTENSIONS: Set[str] = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"
}

STYLE_FOLDER_MAP: Dict[str, str] = {
    "handwritten": "handwritten",
    "printed": "printed",
}

# ImageNet normalization
IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]


# ==========================================
# 4. Training Hyperparameters
# ==========================================
# Model Architecture
DROPOUT_RATE: float = 0.35
LABEL_SMOOTHING: float = 0.05
WEIGHT_DECAY: float = 1e-4

# Stage 1: Freeze Backbone (Train Classifier Head Only)
STAGE1_EPOCHS: int = 5
STAGE1_LR: float = 1e-3

# Stage 2: Fine-Tuning (35 Epochs สำหรับ Fine-tuning Model)
STAGE2_EPOCHS: int = 35
STAGE2_BACKBONE_LR: float = 1e-5
STAGE2_CLASSIFIER_LR: float = 1e-4

# Learning Rate Scheduler (ReduceLROnPlateau)
SCHEDULER_MODE: str = "min"
SCHEDULER_FACTOR: float = 0.5
SCHEDULER_PATIENCE: int = 2
SCHEDULER_MIN_LR: float = 1e-6


# ==========================================
# 5. Thai Character Mapping (TIS-620 & Char Dict)
# ==========================================
FOLDER_TO_THAI_CHAR = {
    "161": ("101", "ก"),
    "162": ("102", "ข"),
    "163": ("103", "ฃ"),
    "164": ("104", "ค"),
    "167": ("105", "ง"),
    "168": ("106", "จ"),
    "169": ("107", "ฉ"),
    "170": ("108", "ช"),
    "171": ("109", "ซ"),
    "173": ("110", "ญ"),
    "175": ("111", "ฏ"),
    "176": ("112", "ฐ"),
    "177": ("113", "ฑ"),
    "178": ("114", "ฒ"),
    "179": ("115", "ณ"),
    "180": ("116", "ด"),
    "181": ("117", "ต"),
    "182": ("118", "ถ"),
    "183": ("119", "ท"),
    "184": ("120", "ธ"),
    "185": ("121", "น"),
    "186": ("122", "บ"),
    "187": ("123", "ป"),
    "188": ("124", "ผ"),
    "189": ("125", "ฝ"),
    "190": ("126", "พ"),
    "191": ("127", "ฟ"),
    "192": ("128", "ภ"),
    "193": ("129", "ม"),
    "194": ("130", "ย"),
    "195": ("131", "ร"),
    "196": ("132", "ฤ"),
    "197": ("133", "ล"),
    "199": ("134", "ว"),
    "200": ("135", "ศ"),
    "201": ("136", "ษ"),
    "202": ("137", "ส"),
    "203": ("138", "ห"),
    "204": ("139", "ฬ"),
    "205": ("140", "อ"),
    "206": ("141", "ฮ"),
    "207": ("201", "ฯ"),
    "209": ("202", "◌ั"),
    "210": ("203", "า"),
    "212": ("204", "◌ิ"),
    "213": ("205", "◌ี"),
    "214": ("206", "◌ึ"),
    "215": ("207", "◌ื"),
    "216": ("208", "◌ุ"),
    "217": ("209", "◌ู"),
    "224": ("210", "เ"),
    "225": ("211", "แ"),
    "226": ("212", "โ"),
    "227": ("213", "ใ"),
    "228": ("214", "ไ"),
    "229": ("215", "ๅ"),
    "230": ("216", "ๆ"),
    "231": ("217", "◌็"),
    "232": ("218", "◌่"),
    "233": ("219", "◌้"),
    "234": ("220", "◌๊"),
    "236": ("221", "◌์"),
    "240": ("301", "๐"),
    "241": ("302", "๑"),
    "242": ("303", "๒"),
    "243": ("304", "๓"),
    "244": ("305", "๔"),
    "245": ("306", "๕"),
    "246": ("307", "๖"),
    "247": ("308", "๗"),
    "248": ("309", "๘"),
    "249": ("310", "๙"),
}


def folder_to_info(folder_name: str):
    folder = str(folder_name)
    mapping = FOLDER_TO_THAI_CHAR.get(folder)
    if mapping is not None:
        class_id, thai_char = mapping
    else:
        class_id = None
        thai_char = folder
    return {
        "folder": folder,
        "class_id": class_id,
        "thai_char": thai_char
    }


def folder_to_char(folder_name: Union[str, int]) -> str:
    """
    แปลงชื่อโฟลเดอร์รหัสเป็นตัวอักษรไทยจริง
    """
    s_name = str(folder_name).strip()
    if s_name in FOLDER_TO_THAI_CHAR:
        val = FOLDER_TO_THAI_CHAR[s_name]
        return val[1] if isinstance(val, (tuple, list)) else val

    try:
        code = int(s_name)
        return bytes([code]).decode("tis-620")
    except Exception:
        return s_name


def folder_to_tis620_code(folder_name: Union[str, int]) -> str:
    """
    ส่งคืนรหัส TIS-620 ของโฟลเดอร์
    เช่น '161' (Hex: 0xA1)
    """
    s_name = str(folder_name).strip()
    try:
        code = int(s_name)
        return f"{code} (0x{code:02X})"
    except Exception:
        return s_name


FOLDER_TO_CLASS_ID = {k: v[0] for k, v in FOLDER_TO_THAI_CHAR.items()}


