"""data.py - dataset scanning, image auditing, leakage-safe splitting, Dataset class.

Image-data pitfalls handled here
  * corrupt / unreadable files            -> detected by audit(), excluded from training
  * RGBA / palette / 16-bit / EXIF rotate -> load_image() converts safely (transparent -> white)
  * exact / near-duplicate images         -> grouped by dHash so a duplicate never lands in
                                             train AND val (that would inflate val accuracy)
  * cross-class exact duplicates          -> reported (likely label noise)
  * classes with 1 image                  -> kept in train only (cannot be validated)
"""
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset, WeightedRandomSampler

IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".gif"}

# glyph groups (used for the auxiliary multi-task head and for sample-aware augmentation)
GROUP_NAMES = ["consonant", "upper_mark", "lower_mark", "digit", "other"]
_UPPER = {0x0E31, 0x0E34, 0x0E35, 0x0E36, 0x0E37, 0x0E47, 0x0E48, 0x0E49, 0x0E4A,
          0x0E4B, 0x0E4C, 0x0E4D, 0x0E4E}          # ั ิ ี ึ ื ็ ่ ้ ๊ ๋ ์ ํ ๎
_LOWER = {0x0E38, 0x0E39, 0x0E3A}                   # ุ ู ฺ


def folder_to_char(name: str) -> str:
    """Folders 161..249 are TIS-620 byte codes (161 -> 'ก'). Fallback: folder name."""
    if name.isdigit():
        n = int(name)
        if 128 <= n <= 255:
            try:
                return bytes([n]).decode("tis-620")
            except UnicodeDecodeError:
                pass
    return name


def glyph_group(ch: str) -> int:
    if len(ch) == 1:
        o = ord(ch)
        if o in _UPPER:
            return 1
        if o in _LOWER:
            return 2
        if 0x0E01 <= o <= 0x0E2E:
            return 0
        if 0x0E50 <= o <= 0x0E59 or ch.isdigit():
            return 3
    return 4


def scan_dataset(root):
    """Return (samples[(path, class_idx)], folder_names, chars)."""
    root = Path(root)
    if (root / "round2").is_dir():
        root = root / "round2"
    dirs = [d for d in root.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: (int(d.name) if d.name.isdigit() else 10 ** 9, d.name))
    folders, samples = [], []
    for d in dirs:
        files = [p for p in sorted(d.rglob("*")) if p.suffix.lower() in IMG_EXT]
        if not files:
            continue
        idx = len(folders)
        folders.append(d.name)
        samples += [(str(p), idx) for p in files]
    if not folders:
        raise RuntimeError(f"No class folders with images found under {root}")
    return samples, folders, [folder_to_char(f) for f in folders]


# ----------------------------------------------------------------------------
# robust image loading
# ----------------------------------------------------------------------------
def load_image(path) -> Image.Image:
    img = Image.open(path)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")                      # transparent pixels -> WHITE (not black)
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    if img.mode in ("I;16", "I", "F"):
        a = np.asarray(img).astype(np.float32)
        a = (a - a.min()) / (a.max() - a.min() + 1e-6) * 255
        return Image.fromarray(a.astype(np.uint8)).convert("RGB")
    return img.convert("RGB")


def thumb_stats(img: Image.Image):
    """(median colour, luminance) of a small thumbnail = robust background estimate."""
    t = img.convert("RGB")
    t.thumbnail((32, 32))
    a = np.asarray(t).reshape(-1, 3)
    med = np.median(a, 0)
    lum = 0.299 * med[0] + 0.587 * med[1] + 0.114 * med[2]
    return tuple(int(v) for v in med), float(lum)


def dhash(img: Image.Image, size=8) -> str:
    g = img.convert("L").resize((size + 1, size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.int16)
    return np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes().hex()


def audit(samples, cache=None, progress=True):
    """Read every file once: md5, dHash, size, mode, luminance. Cached to JSON."""
    info = {}
    if cache and Path(cache).exists():
        info = json.load(open(cache, encoding="utf-8"))
    todo = [p for p, _ in samples if p not in info]
    for i, p in enumerate(todo):
        try:
            raw = open(p, "rb").read()
            with Image.open(p) as im0:
                mode = im0.mode
            img = load_image(p)
            w, h = img.size
            if w < 1 or h < 1:
                raise ValueError("empty image")
            info[p] = {"ok": True, "md5": hashlib.md5(raw).hexdigest(), "dhash": dhash(img),
                       "w": w, "h": h, "mode": mode, "lum": thumb_stats(img)[1]}
        except Exception as e:  # corrupt / truncated / unsupported
            info[p] = {"ok": False, "err": str(e)[:120]}
        if progress and (i + 1) % 5000 == 0:
            print(f"  audited {i + 1}/{len(todo)} images")
    if cache and todo:
        json.dump(info, open(cache, "w", encoding="utf-8"))
    return info


# ----------------------------------------------------------------------------
# leakage-safe stratified split
# ----------------------------------------------------------------------------
def grouped_stratified_split(samples, info=None, val_ratio=0.2, seed=42):
    """80/20 per class where near-duplicate images (same dHash) stay on the same side."""
    rng = random.Random(seed)
    report = {"corrupt": [], "cross_class_duplicates": [], "duplicate_groups": 0}
    usable = []
    for s in samples:
        if info is not None and not info[s[0]]["ok"]:
            report["corrupt"].append(s[0])
        else:
            usable.append(s)
    if info is not None:
        by_md5 = defaultdict(list)
        for p, y in usable:
            by_md5[info[p]["md5"]].append((p, y))
        report["cross_class_duplicates"] = [
            v for v in by_md5.values() if len({y for _, y in v}) > 1][:200]

    by_cls = defaultdict(lambda: defaultdict(list))
    for p, y in usable:
        by_cls[y][info[p]["dhash"] if info is not None else p].append((p, y))

    train, val = [], []
    for y in sorted(by_cls):
        groups = [by_cls[y][k] for k in sorted(by_cls[y])]
        rng.shuffle(groups)
        report["duplicate_groups"] += sum(len(g) > 1 for g in groups)
        n = sum(len(g) for g in groups)
        if len(groups) < 2:                       # 1 image / 1 duplicate cluster -> train only
            train += [s for g in groups for s in g]
            continue
        target, got, moved = max(1, round(n * val_ratio)), 0, 0
        for g in groups[:-1]:                     # always keep >=1 group in train
            if got >= target:
                break
            val += g
            got += len(g)
            moved += 1
        train += [s for g in groups[moved:] for s in g]
    return train, val, report, usable


# ----------------------------------------------------------------------------
# dataset / sampler
# ----------------------------------------------------------------------------
class CharDataset(Dataset):
    """transform(img, y, strength) -> tensor. `strength` is the global augmentation
    strength set by the adaptive scheduler before every epoch."""

    def __init__(self, samples, transform):
        self.samples, self.transform, self.strength = samples, transform, 1.0

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        for _ in range(5):                        # skip unreadable files instead of crashing
            path, y = self.samples[i]
            try:
                return self.transform(load_image(path), y, self.strength), y
            except Exception:
                i = random.randrange(len(self.samples))
        raise RuntimeError("too many unreadable images")


def make_sampler(samples, n_cls, power):
    """Weight ~ count^-power. power=0 natural, 1 fully balanced, 0.5 = sqrt (default)."""
    counts = np.bincount([y for _, y in samples], minlength=n_cls).astype(float)
    w = np.array([max(counts[y], 1) ** (-power) for _, y in samples])
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), len(samples),
                                 replacement=True), counts


def pick_thai_font():
    """Return a Thai-capable matplotlib font name if one is installed, else None."""
    try:
        from matplotlib import font_manager
        names = {f.name for f in font_manager.fontManager.ttflist}
        for n in ["Leelawadee UI", "Tahoma", "Noto Sans Thai", "TH Sarabun New", "Angsana New",
                  "Cordia New", "Loma", "Garuda", "Waree", "Norasi", "Sarabun"]:
            if n in names:
                return n
    except Exception:
        pass
    return None
