"""โหลด YAML config พร้อมระบบ inherit และ override จาก CLI."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from .utils import seed_everything


class Config(dict):
    """dict ที่เข้าถึงด้วย dot notation ได้ (cfg.train.stage1.epochs)."""

    def __getattr__(self, key: str) -> Any:
        try:
            val = self[key]
        except KeyError:
            raise AttributeError(f"ไม่พบคีย์ '{key}' ใน config") from None
        return Config(val) if isinstance(val, dict) else val

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        """cfg.get_path('train.stage1.lr', 1e-4)"""
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def set_path(self, dotted: str, value: Any) -> None:
        parts = dotted.split(".")
        node: dict = self
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value

    def to_dict(self) -> dict:
        return copy.deepcopy(dict(self))


def _deep_merge(base: dict, override: dict) -> dict:
    """merge แบบ recursive — override ชนะเสมอ."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _cast(value: str) -> Any:
    """แปลง string จาก CLI เป็น type ที่เหมาะสม."""
    low = value.lower()
    if low in {"true", "yes"}:
        return True
    if low in {"false", "no"}:
        return False
    if low in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def load_config(path: str | Path, overrides: list[str] | None = None) -> Config:
    """
    โหลด config พร้อมแก้ inherit แบบ recursive
    overrides: ["train.stage1.epochs=50", "model.backbone=convnext_tiny"]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ config: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    parent_name = raw.pop("inherit", None)
    if parent_name:
        parent_path = (path.parent / parent_name).resolve()
        parent = load_config(parent_path)
        raw = _deep_merge(parent.to_dict(), raw)

    cfg = Config(raw)
    cfg["_config_file"] = str(path)

    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override ต้องเป็นรูปแบบ key=value ได้รับ: {item}")
        key, val = item.split("=", 1)
        cfg.set_path(key.strip(), _cast(val.strip()))

    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    """ตรวจค่าที่พลาดแล้วพังทั้งโปรเจกต์."""
    assert cfg.get_path("project.num_classes") == 72, \
        "num_classes ต้องเป็น 72"

    head = cfg.get_path("model.head", "linear")
    assert head in {"linear", "arcface"}, f"model.head ไม่รองรับ: {head}"

    loss_t = cfg.get_path("loss.type", "ce")
    assert loss_t in {"ce", "focal", "logit_adjusted"}, f"loss.type ไม่รองรับ: {loss_t}"

    for st in ("stage1", "stage2"):
        s = cfg.get_path(f"train.{st}.sampler", "instance")
        assert s in {"instance", "sqrt", "balanced", "effective"}, \
            f"train.{st}.sampler ไม่รองรับ: {s}"

    # กันคนเผลอเปิด flip — ตัวอักษรไทยพลิกแล้วเปลี่ยนความหมาย
    forb = cfg.get_path("augmentation.forbidden", {}) or {}
    for k, v in forb.items():
        assert v is False, (
            f"augmentation.forbidden.{k} ต้องเป็น false เสมอ — "
            "การพลิก/หมุน 90° ทำให้ตัวอักษรไทยเปลี่ยนความหมาย"
        )

    rot = cfg.get_path("augmentation.online.rotate_limit", 10)
    assert rot <= 15, "rotate_limit ห้ามเกิน 15° (ๆ กับ ฯ จะชนกัน)"


def apply_runtime_settings(cfg: Config) -> None:
    """ตั้ง seed + cudnn ตาม config (เรียกครั้งเดียวตอนเริ่มโปรแกรม)."""
    import torch

    seed_everything(cfg.get_path("project.seed", 42),
                    cfg.get_path("project.deterministic", True))
    if cfg.get_path("hardware.cudnn_benchmark", True) and \
       not cfg.get_path("project.deterministic", True):
        torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def print_config(cfg: Config, logger=None) -> None:
    text = yaml.safe_dump(cfg.to_dict(), allow_unicode=True,
                          sort_keys=False, default_flow_style=False)
    msg = "\n" + "=" * 70 + "\nCONFIG\n" + "=" * 70 + f"\n{text}" + "=" * 70
    (logger.info if logger else print)(msg)