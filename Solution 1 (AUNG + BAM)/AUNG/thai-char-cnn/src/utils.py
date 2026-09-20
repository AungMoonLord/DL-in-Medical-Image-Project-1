"""Utility helpers ที่ทุกโมดูลเรียกใช้ — ไม่ import โมดูลอื่นใน src เลย (ฐานล่างสุด)."""
from __future__ import annotations

import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# ----------------------------------------------------------------------------
# Reproducibility
# ----------------------------------------------------------------------------
def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """ตรึง seed ทุก library. deterministic=False จะเร็วขึ้น ~10-15%."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def worker_init_fn(worker_id: int) -> None:
    """กัน DataLoader worker ใช้ numpy seed ซ้ำกัน (บั๊กคลาสสิกของ augmentation)."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


# ----------------------------------------------------------------------------
# Filesystem
# ----------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    """สร้างโฟลเดอร์ถ้ายังไม่มี (รองรับกรณีส่ง path ของไฟล์มาก็ได้)."""
    p = Path(path)
    target = p.parent if p.suffix else p
    target.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_images(folder: str | Path) -> list[Path]:
    """คืน list ของไฟล์ภาพใน folder เรียงตามชื่อ (deterministic)."""
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in IMG_EXTS)


# ----------------------------------------------------------------------------
# Device
# ----------------------------------------------------------------------------
def get_device(prefer: str = "auto") -> torch.device:
    """
    เลือก device ที่เหมาะสมที่สุดให้อัตโนมัติ:
    1. ถ้ามี Nvidia GPU (CUDA) -> ใช้ cuda
    2. ถ้าเป็น Mac Apple Silicon (MPS) -> ใช้ mps
    3. ถ้าไม่มี GPU เลย -> fallback เป็น cpu อย่างปลอดภัย
    """
    if prefer == "cpu":
        return torch.device("cpu")

    # 1. เช็ค Nvidia GPU
    if torch.cuda.is_available():
        return torch.device("cuda")

    # 2. เช็ค Apple Silicon GPU (macOS)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    # 3. เครื่องทั่วไป
    return torch.device("cpu")


def describe_device(device: torch.device) -> str:
    if device.type != "cuda":
        return "CPU"
    i = device.index or 0
    name = torch.cuda.get_device_name(i)
    cap = torch.cuda.get_device_capability(i)
    vram = torch.cuda.get_device_properties(i).total_memory / 1024**3
    return f"{name} | sm_{cap[0]}{cap[1]} | {vram:.1f} GB | torch {torch.__version__}"


def check_blackwell_support(device: torch.device) -> None:
    """เตือนถ้า PyTorch build ไม่รองรับ sm_120 (RTX 50-series)."""
    if device.type != "cuda":
        return
    cap = torch.cuda.get_device_capability(device.index or 0)
    arch_list = torch.cuda.get_arch_list()
    tag = f"sm_{cap[0]}{cap[1]}"
    if cap[0] >= 12 and tag not in arch_list:
        raise RuntimeError(
            f"PyTorch build นี้ไม่รองรับ {tag} (รองรับ: {arch_list})\n"
            "RTX 50-series ต้องใช้ build cu128:\n"
            "  pip install torch==2.7.0 torchvision==0.22.0 "
            "--index-url https://download.pytorch.org/whl/cu128"
        )


def amp_dtype_from_str(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
            "float16": torch.float16, "fp16": torch.float16}.get(name, torch.bfloat16)


# ----------------------------------------------------------------------------
# Metrics tracking
# ----------------------------------------------------------------------------
class AverageMeter:
    def __init__(self, name: str = "", fmt: str = ":.4f"):
        self.name, self.fmt = name, fmt
        self.reset()

    def reset(self) -> None:
        self.val = self.avg = self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / max(self.count, 1)

    def __str__(self) -> str:
        return f"{self.name} {self.avg:{self.fmt.lstrip(':')}}"


class Timer:
    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.t0

    @property
    def pretty(self) -> str:
        m, s = divmod(self.elapsed, 60)
        return f"{int(m)}m {s:04.1f}s"


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def setup_logger(name: str = "thaichar", log_file: str | Path | None = None,
                 level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("[%(asctime)s] %(levelname)-7s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    if log_file:
        ensure_dir(log_file)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


# ----------------------------------------------------------------------------
# Checkpoint
# ----------------------------------------------------------------------------
def save_checkpoint(path: str | Path, model, optimizer=None, scheduler=None,
                    epoch: int = 0, stage: int = 1, metrics: dict | None = None,
                    cfg: dict | None = None, class_map: dict | None = None) -> None:
    ensure_dir(path)
    # ปลด compile wrapper (ถ้ามี) ก่อนบันทึก state_dict
    raw_model = getattr(model, "_orig_mod", model)
    torch.save({
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
        "scheduler_state": scheduler.state_dict() if scheduler else None,
        "epoch": epoch, "stage": stage,
        "metrics": metrics or {}, "config": cfg or {},
        "class_map": class_map or {},
        "torch_version": torch.__version__,
    }, path)

def load_checkpoint(path: str | Path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


# ----------------------------------------------------------------------------
# Thai font for matplotlib (กันกราฟขึ้นสี่เหลี่ยม □□□)
# ----------------------------------------------------------------------------

# def setup_thai_font() -> str | None:
#     """หาและตั้งค่าฟอนต์ไทยให้ matplotlib. คืนชื่อฟอนต์ที่ใช้ได้ หรือ None."""
#     import matplotlib
#     from matplotlib import font_manager
#     from pathlib import Path

#     win_font_dir = Path("C:/Windows/Fonts")
#     candidates = [
#         ("Leelawadee UI", "LeelawUI.ttf"),
#         ("Leelawadee UI", "LeelUIsl.ttf"),
#         ("Leelawadee",    "leelawad.ttf"),
#     ]

#     for font_name, filename in candidates:
#         font_path = win_font_dir / filename
#         if not font_path.exists():
#             continue

#         # add font เข้า matplotlib
#         font_manager.fontManager.addfont(str(font_path))
#         matplotlib.rcParams["font.family"] = font_name
#         matplotlib.rcParams["axes.unicode_minus"] = False

#         # ป้องกัน sns.set_style() reset font ทีหลัง
#         # โดย patch sns ให้ restore font หลัง set_style ทุกครั้ง
#         try:
#             import seaborn as sns
#             _original_set_style = sns.set_style

#             def _patched_set_style(*args, **kwargs):
#                 _original_set_style(*args, **kwargs)
#                 matplotlib.rcParams["font.family"] = font_name
#                 matplotlib.rcParams["axes.unicode_minus"] = False

#             sns.set_style = _patched_set_style
#         except ImportError:
#             pass  # ถ้าไม่มี seaborn ก็ข้ามไป

#         return font_name

#     print("[warn] ไม่พบฟอนต์ไทย")
#     return None

def setup_thai_font() -> str | None:
    """ตั้งค่าฟอนต์ Leelawadee UI ให้ matplotlib พร้อมดาวน์โหลดให้อัตโนมัติหากไม่พบ."""
    import urllib.request
    from pathlib import Path
    import matplotlib
    from matplotlib import font_manager

    font_name = "Leelawadee UI"

    # 1. เช็คว่าระบบมีฟอนต์ชื่อนี้ติดตั้งอยู่แล้วหรือไม่
    installed_fonts = {f.name for f in font_manager.fontManager.ttflist}
    if font_name in installed_fonts:
        _apply_font(font_name)
        return font_name

    # 2. รายการ Path ไฟล์ในเครื่อง (โปรเจกต์, Windows, WSL)
    local_target = Path("data/fonts/LeelawUI.ttf")
    candidates = [
        local_target,
        Path("C:/Windows/Fonts/LeelawUI.ttf"),
        Path("/mnt/c/Windows/Fonts/LeelawUI.ttf"),
    ]

    for p in candidates:
        if p.exists():
            font_manager.fontManager.addfont(str(p))
            _apply_font(font_name)
            return font_name

    # 3. หากไม่พบไฟล์ในเครื่อง ให้ดาวน์โหลดลง data/fonts/ อัตโนมัติ
    font_url = "https://raw.githubusercontent.com/matthras/fonts/master/Leelawadee%20UI/LeelawUI.ttf"
    try:
        local_target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[info] ไม่พบ {font_name} ในระบบ กำลังดาวน์โหลดฟอนต์อัตโนมัติ...")
        
        req = urllib.request.Request(font_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp, open(local_target, "wb") as f:
            f.write(resp.read())

        font_manager.fontManager.addfont(str(local_target))
        _apply_font(font_name)
        print(f"[info] ติดตั้งฟอนต์ {font_name} เรียบร้อยแล้วที่ {local_target}")
        return font_name
    except Exception as e:
        print(f"[warn] ไม่สามารถดาวน์โหลด {font_name} อัตโนมัติได้: {e}")
        print("       ตัวอักษรไทยในกราฟอาจแสดงเป็นสี่เหลี่ยม")
        return None


def _apply_font(font_name: str) -> None:
    """ฟังก์ชันผู้ช่วยสำหรับผูกฟอนต์เข้ากับ Matplotlib และ Seaborn."""
    import matplotlib
    matplotlib.rcParams["font.family"] = font_name
    matplotlib.rcParams["axes.unicode_minus"] = False

    try:
        import seaborn as sns
        _original_set_style = sns.set_style

        def _patched_set_style(*args, **kwargs):
            _original_set_style(*args, **kwargs)
            matplotlib.rcParams["font.family"] = font_name
            matplotlib.rcParams["axes.unicode_minus"] = False

        sns.set_style = _patched_set_style
    except ImportError:
        pass


def count_parameters(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_number(n: int | float) -> str:
    return f"{n:,}"