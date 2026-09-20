"""
Single source of truth ของการ map: folder_id ↔ อักษรไทย ↔ label index

⚠️ ห้ามสร้าง mapping เองที่อื่นเด็ดขาด — index ต้องตรงกันระหว่าง train กับ inference

Folder ID ของ dataset นี้คือรหัส TIS-620 ของอักษรไทย (161=0xA1=ก ... 249=0xF9=๙)
จึง decode กลับเป็นตัวอักษรได้โดยตรง ไม่ต้องพึ่งตารางฮาร์ดโค้ด
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .utils import ensure_dir, list_images, load_json, save_json

# ----------------------------------------------------------------------------
# Folder ID ทั้ง 72 คลาสของ dataset นี้ (ยืนยันจาก Dataset.csv)
# ----------------------------------------------------------------------------
EXPECTED_FOLDERS: list[int] = [
    161, 162, 163, 164, 167, 168, 169, 170, 171, 173, 175, 176, 177, 178,
    179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191, 192,
    193, 194, 195, 196, 197, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    209, 210, 212, 213, 214, 215, 216, 217, 224, 225, 226, 227, 228, 229,
    230, 231, 232, 233, 234, 236, 240, 241, 242, 243, 244, 245, 246, 247,
    248, 249,
]

# fallback เผื่อ codec tis-620 ไม่มีในระบบ
_FALLBACK: dict[int, str] = {
    161: "ก", 162: "ข", 163: "ฃ", 164: "ค", 167: "ง", 168: "จ", 169: "ฉ",
    170: "ช", 171: "ซ", 173: "ญ", 175: "ฏ", 176: "ฐ", 177: "ฑ", 178: "ฒ",
    179: "ณ", 180: "ด", 181: "ต", 182: "ถ", 183: "ท", 184: "ธ", 185: "น",
    186: "บ", 187: "ป", 188: "ผ", 189: "ฝ", 190: "พ", 191: "ฟ", 192: "ภ",
    193: "ม", 194: "ย", 195: "ร", 196: "ฤ", 197: "ล", 199: "ว", 200: "ศ",
    201: "ษ", 202: "ส", 203: "ห", 204: "ฬ", 205: "อ", 206: "ฮ", 207: "ฯ",
    209: "ั", 210: "า", 212: "ิ", 213: "ี", 214: "ึ", 215: "ื", 216: "ุ",
    217: "ู", 224: "เ", 225: "แ", 226: "โ", 227: "ใ", 228: "ไ", 229: "ๅ",
    230: "ๆ", 231: "็", 232: "่", 233: "้", 234: "๊", 236: "์", 240: "๐",
    241: "๑", 242: "๒", 243: "๓", 244: "๔", 245: "๕", 246: "๖", 247: "๗",
    248: "๘", 249: "๙",
}

# ชื่ออ่านออกเสียงได้ สำหรับใส่กราฟ/CSV (กันปัญหาสระลอยไม่มีฐาน)
_READABLE: dict[str, str] = {
    "ั": "-ั (ไม้หันอากาศ)", "า": "-า (สระอา)", "ิ": "-ิ (สระอิ)",
    "ี": "-ี (สระอี)", "ึ": "-ึ (สระอึ)", "ื": "-ื (สระอือ)",
    "ุ": "-ุ (สระอุ)", "ู": "-ู (สระอู)", "็": "-็ (ไม้ไต่คู้)",
    "่": "-่ (ไม้เอก)", "้": "-้ (ไม้โท)", "๊": "-๊ (ไม้ตรี)",
    "์": "-์ (ทัณฑฆาต)", "ๅ": "ๅ (ลากข้าง)", "ๆ": "ๆ (ไม้ยมก)",
    "ฯ": "ฯ (ไปยาลน้อย)",
}

# กลุ่มที่หน้าตาคล้ายกัน — ใช้ใน metrics/evaluate
SIMILAR_GROUPS: list[list[str]] = [
    ["ข", "ฃ"],
    ["ท", "ฑ"],
    ["ช", "ซ"],
    ["ก", "ภ", "ถ", "ฤ"],
    ["พ", "ผ", "ฝ", "ฟ", "ฬ"],
    ["ด", "ต", "ค"],
    ["บ", "ป", "ษ"],
    ["น", "ม"],
    ["ิ", "ี", "ึ", "ื"],
    ["่", "้", "๊"],
    ["็", "๘"],
    ["เ", "แ", "โ", "ใ", "ไ"],
    ["า", "ๅ", "ว"],
    ["๓", "๗", "๊"],
    ["๔", "๕"],
    ["๐", "๑", "๒", "๓", "๔", "๕", "๖", "๗", "๘", "๙"],
]


def folder_to_char(folder_id: int) -> str:
    """แปลง folder id (TIS-620 code) → อักษรไทย."""
    try:
        ch = bytes([folder_id]).decode("tis-620")
        if ch.strip():
            return ch
    except (LookupError, UnicodeDecodeError, ValueError):
        pass
    if folder_id in _FALLBACK:
        return _FALLBACK[folder_id]
    raise ValueError(f"ไม่รู้จัก folder id: {folder_id}")


def readable_name(char: str) -> str:
    """ชื่อที่อ่านออกสำหรับกราฟ/รายงาน."""
    return _READABLE.get(char, char)


class ClassMap:
    """ตัวกลางเดียวสำหรับการแปลง folder ↔ char ↔ label."""

    def __init__(self, folders: list[int], counts: dict[int, int] | None = None):
        self.folders = sorted(int(f) for f in folders)
        self.chars = [folder_to_char(f) for f in self.folders]
        self.folder_to_label = {f: i for i, f in enumerate(self.folders)}
        self.label_to_folder = {i: f for f, i in self.folder_to_label.items()}
        self.label_to_char = {i: c for i, c in enumerate(self.chars)}
        self.char_to_label = {c: i for i, c in enumerate(self.chars)}
        self.counts = {int(k): int(v) for k, v in (counts or {}).items()}

    # ---------------- properties ----------------
    @property
    def num_classes(self) -> int:
        return len(self.folders)

    @property
    def class_names(self) -> list[str]:
        return list(self.chars)

    @property
    def readable_names(self) -> list[str]:
        return [readable_name(c) for c in self.chars]

    def counts_by_label(self) -> list[int]:
        """จำนวนภาพต่อคลาสเรียงตาม label index — ใช้ใน loss/sampler."""
        return [self.counts.get(self.label_to_folder[i], 0)
                for i in range(self.num_classes)]

    def tail_labels(self, threshold: int = 50) -> list[int]:
        return [i for i, c in enumerate(self.counts_by_label()) if 0 < c < threshold]

    def tail_chars(self, threshold: int = 50) -> list[str]:
        return [self.label_to_char[i] for i in self.tail_labels(threshold)]

    def similar_groups_as_labels(self) -> list[list[int]]:
        out = []
        for g in SIMILAR_GROUPS:
            labels = [self.char_to_label[c] for c in g if c in self.char_to_label]
            if len(labels) >= 2:
                out.append(labels)
        return out

    # ---------------- io ----------------
    def to_dict(self) -> dict:
        return {
            "num_classes": self.num_classes,
            "folders": self.folders,
            "folder_to_label": {str(f): l for f, l in self.folder_to_label.items()},
            "label_to_folder": {str(l): f for l, f in self.label_to_folder.items()},
            "label_to_char": {str(l): c for l, c in self.label_to_char.items()},
            "char_to_label": {c: l for c, l in self.char_to_label.items()},
            "class_counts": {str(f): self.counts.get(f, 0) for f in self.folders},
            "counts_by_label": self.counts_by_label(),
            "readable_names": self.readable_names,
            "tail_classes_lt50": self.tail_chars(50),
        }

    def save(self, path: str | Path) -> None:
        ensure_dir(path)
        save_json(self.to_dict(), path)

    @classmethod
    def load(cls, path: str | Path) -> "ClassMap":
        d = load_json(path)
        counts = {int(k): int(v) for k, v in d.get("class_counts", {}).items()}
        return cls([int(f) for f in d["folders"]], counts)

    @classmethod
    def from_raw_dir(cls, raw_dir: str | Path, strict: bool = True) -> "ClassMap":
        """สแกน data/raw/ หาโฟลเดอร์คลาสและนับจำนวนภาพจริง."""
        raw = Path(raw_dir)
        if not raw.is_dir():
            raise FileNotFoundError(f"ไม่พบโฟลเดอร์ข้อมูล: {raw}")

        found = sorted(int(p.name) for p in raw.iterdir()
                       if p.is_dir() and p.name.isdigit())
        if not found:
            raise RuntimeError(
                f"ไม่พบโฟลเดอร์คลาสใน {raw}\n"
                "โครงสร้างที่ถูกต้อง: data/raw/161/*.png, data/raw/162/*.png, ..."
            )

        missing = set(EXPECTED_FOLDERS) - set(found)
        extra = set(found) - set(EXPECTED_FOLDERS)
        if strict and (missing or extra):
            raise RuntimeError(
                f"โฟลเดอร์ไม่ตรงกับที่คาดไว้\n"
                f"  ขาด : {sorted(missing)}\n"
                f"  เกิน: {sorted(extra)}\n"
                "ใช้ --no-strict ถ้าต้องการข้ามการตรวจนี้"
            )

        counts = {f: len(list_images(raw / str(f))) for f in found}
        return cls(found, counts)


def build_and_save(raw_dir: str | Path, out_path: str | Path,
                   strict: bool = True) -> ClassMap:
    cm = ClassMap.from_raw_dir(raw_dir, strict=strict)
    cm.save(out_path)
    return cm


def main() -> None:
    ap = argparse.ArgumentParser(description="สร้าง class_map.json จาก data/raw/")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/splits/class_map.json")
    ap.add_argument("--no-strict", action="store_true")
    args = ap.parse_args()

    cm = build_and_save(args.raw_dir, args.out, strict=not args.no_strict)
    total = sum(cm.counts.values())
    counts = cm.counts_by_label()

    print(f"✓ บันทึก {args.out}")
    print(f"  คลาส      : {cm.num_classes}")
    print(f"  ภาพทั้งหมด : {total:,}")
    print(f"  มากสุด    : {max(counts):,} | น้อยสุด: {min(counts):,} "
          f"| ratio {max(counts)/max(min(counts),1):.0f}:1")
    print(f"  tail (<50): {len(cm.tail_labels(50))} คลาส → "
          f"{' '.join(cm.tail_chars(50))}")


if __name__ == "__main__":
    main()