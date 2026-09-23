"""augment.py - preprocessing + Data Augmentation (Chapter 8 / 10).

Techniques
  * PadResize            : pad to square with background colour (never stretch a glyph)
  * Geometric distortion : small rotate / shift / scale / shear
  * Photometric          : brightness, contrast, colour, blur, low-resolution, noise, stroke width
  * Image Occlusion      : Random Erase, Cutout, Hide-and-Seek (ink dropout)  [Chapter 8]
  * Progressive schedule : weak -> strong augmentation as training proceeds    [Chapter 8]
  * Adaptive schedule    : strength is raised / lowered from the train-val gap
  * Sample-aware         : strength depends on the sample's CLASS RARITY (rare -> stronger)
                           and GLYPH GROUP (tiny tone-marks/vowels: gentle geometry, no occlusion)
NEVER used: flips (they create letters that do not exist), CutMix/MixUp (destroy tiny marks).
"""
import math
import random

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import v2

from data import thumb_stats

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
BICUBIC = T.InterpolationMode.BICUBIC


def _range(img):
    """Dynamic range (p99-p1) of the grey image: guards against washed-out / blank samples."""
    g = np.asarray(img.convert("L").resize((32, 32)), dtype=np.float32)
    return float(np.percentile(g, 99) - np.percentile(g, 1))


class PadResize:
    """Deterministic preprocessing shared by train / val / test."""

    def __init__(self, size, margin=0.12, auto_invert=True):
        self.size, self.margin, self.auto_invert = size, margin, auto_invert

    def __call__(self, img):
        img = img.convert("RGB")
        bg, lum = thumb_stats(img)
        if self.auto_invert and lum < 110:       # light-on-dark -> dark-on-light
            img = ImageOps.invert(img)
            bg = tuple(255 - c for c in bg)
        w, h = img.size
        s = max(1, int(round(max(w, h) * (1 + self.margin))))
        canvas = Image.new("RGB", (s, s), bg)
        canvas.paste(img, ((s - w) // 2, (s - h) // 2))
        rs = Image.Resampling.LANCZOS if s > self.size else Image.Resampling.BICUBIC
        return canvas.resize((self.size, self.size), rs)


class EvalTransform:
    """Val / test: PadResize -> tensor -> Normalize. Signature matches the train transform."""

    def __init__(self, size, auto_invert=True):
        self.pad = PadResize(size, auto_invert=auto_invert)
        self.to_float = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
        self.normalize = v2.Normalize(MEAN, STD)

    def __call__(self, img, y=None, strength=1.0):
        return self.normalize(self.to_float(self.pad(img)))


class SampleAwareAugment:
    def __init__(self, size, class_counts, class_groups, auto_invert=True, occlusion=True):
        self.size, self.occlusion = size, occlusion
        self.pad = PadResize(size, auto_invert=auto_invert)
        n = np.maximum(np.asarray(class_counts, dtype=float), 1.0)
        self.rarity = np.clip(1 - np.log(n) / max(np.log(n.max()), 1e-6), 0, 1)  # 0=common .. 1=rarest
        self.groups = list(class_groups)
        self.to_float = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
        self.normalize = v2.Normalize(MEAN, STD)

    def sample_strength(self, y, s):
        """Sample-aware: rare classes get up to 1.3x, common classes 0.7x the global strength."""
        return float(np.clip(s * (0.7 + 0.6 * self.rarity[y]), 0.05, 1.5))

    # ---- Image Occlusion (Chapter 8) ------------------------------------------------
    def _occlude(self, x, s):
        _, H, W = x.shape
        bg = x[:, :3, :3].mean((1, 2), keepdim=True)
        mode = random.choices(["bg", "noise", "mean"], [0.6, 0.2, 0.2])[0]
        mean = torch.tensor(MEAN).view(3, 1, 1)

        def fill(t, l, b, r):
            if b <= t or r <= l:
                return
            if mode == "bg":
                x[:, t:b, l:r] = bg
            elif mode == "noise":
                x[:, t:b, l:r] = torch.rand(3, b - t, r - l)
            else:
                x[:, t:b, l:r] = mean

        kind = random.choice(["erase", "cutout", "hide_seek"])
        k = min(s, 1.2)
        if kind == "erase":                              # Random Erase: rectangle, random aspect
            area = random.uniform(0.02, 0.08) * k * H * W
            ratio = math.exp(random.uniform(math.log(0.3), math.log(3.3)))
            h = min(H - 1, int(math.sqrt(area * ratio)))
            w = min(W - 1, int(math.sqrt(area / ratio)))
            t, l = random.randint(0, H - h), random.randint(0, W - w)
            fill(t, l, t + h, l + w)
        elif kind == "cutout":                           # Cutout: square, centre may be near edge
            side = int(random.uniform(0.10, 0.22) * H * k)
            cy, cx = random.randint(0, H - 1), random.randint(0, W - 1)
            fill(max(cy - side // 2, 0), max(cx - side // 2, 0), min(cy + side // 2, H), min(cx + side // 2, W))
        else:                                            # Hide-and-Seek: grid, hide cells randomly
            g = 6
            ph, pw = H // g, W // g
            cells = [(i, j) for i in range(g) for j in range(g)]
            for i, j in cells:
                if random.random() < 0.08 * k:
                    fill(i * ph, j * pw, (i + 1) * ph, (j + 1) * pw)
        return x

    def __call__(self, img, y, strength=1.0):
        s = self.sample_strength(y, strength)
        is_mark = self.groups[y] in (1, 2)               # tiny vowels / tone marks
        gf = 0.5 if is_mark else 1.0                     # gentler geometry for marks
        img = self.pad(img)
        W = img.size[0]
        bg = list(img.getpixel((0, 0)))

        # 1) geometric distortion
        if random.random() < 0.9:
            img = TF.affine(
                img, angle=random.uniform(-10, 10) * s * gf,
                translate=[int(random.uniform(-0.08, 0.08) * s * gf * W),
                           int(random.uniform(-0.08, 0.08) * s * gf * W)],
                scale=1 + random.uniform(-0.10, 0.10) * s * gf,
                shear=[random.uniform(-8, 8) * s * gf, 0.0],
                interpolation=BICUBIC, fill=bg)
        # 2) stroke-width jitter (different pens / fonts): thicken or thin the ink
        if W >= 64 and random.random() < 0.25 * min(s, 1.0):
            img = img.filter(random.choice([ImageFilter.MinFilter(3), ImageFilter.MaxFilter(3)]))
        # 3) photometric (strength capped at 1.0; guarded so the glyph never becomes invisible)
        sp, before, r0 = min(s, 1.0), img, _range(img)
        if random.random() < 0.7:
            img = T.ColorJitter(0.35 * sp, 0.35 * sp, 0.3 * sp, 0.03 * sp)(img)
        if random.random() < min(0.3 * s, 0.5):
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.2) * max(sp, 0.5)))
        if random.random() < min(0.3 * s, 0.5):          # low resolution
            f = random.uniform(0.4, 0.85)
            small = img.resize((max(8, int(W * f)),) * 2, Image.Resampling.BILINEAR)
            img = small.resize((W, W), Image.Resampling.BICUBIC)
        if random.random() < 0.1:
            img = ImageOps.grayscale(img).convert("RGB")
        if _range(img) < max(50.0, 0.6 * r0):            # washed out -> fall back
            img = before

        x = self.to_float(img)
        if random.random() < min(0.3 * s, 0.5):          # sensor noise
            x = (x + torch.randn_like(x) * random.uniform(0.01, 0.05) * max(s, 0.5)).clamp_(0, 1)
        # 4) image occlusion (never on tiny marks - they would simply vanish)
        if self.occlusion and not is_mark and random.random() < min(0.3 * s, 0.4):
            x = self._occlude(x, s)
        return self.normalize(x)


class AugController:
    """Progressive + Adaptive augmentation schedule (Chapter 8: dynamic augmentation).

    progressive : s_min -> s_max over the first `ramp_frac` of training
                  (weak augmentation first, so the model learns basic shapes before hard cases)
    adaptive    : after every epoch, multiplier *= step if the generalisation gap
                  (clean-train acc - val acc) is large (over-fitting -> stronger augmentation),
                  multiplier /= step if the gap is tiny (under-fitting -> weaker augmentation)
    """

    def __init__(self, epochs, s_min=0.3, s_max=1.0, ramp_frac=0.4, adaptive=True,
                 gap_hi=0.04, gap_lo=0.01, step=1.15, mult_range=(0.6, 1.5)):
        self.epochs, self.s_min, self.s_max, self.ramp_frac = epochs, s_min, s_max, ramp_frac
        self.adaptive, self.gap_hi, self.gap_lo, self.step = adaptive, gap_hi, gap_lo, step
        self.mult_range, self.mult = mult_range, 1.0

    def strength(self, epoch):
        ramp = max(1, int(self.epochs * self.ramp_frac))
        base = self.s_min + (self.s_max - self.s_min) * min(1.0, epoch / ramp)
        return float(np.clip(base * self.mult, 0.15, 1.5))

    def update(self, gap):
        if not self.adaptive:
            return
        if gap > self.gap_hi:
            self.mult *= self.step
        elif gap < self.gap_lo:
            self.mult /= self.step
        self.mult = float(np.clip(self.mult, *self.mult_range))


# ----------------------------------------------------------------------------
# perturbations to estimate robustness on "unseen" imagery
# ----------------------------------------------------------------------------
def _bg(im):
    return list(thumb_stats(im)[0])


def _rot(deg):
    return lambda im: TF.rotate(im, deg, interpolation=BICUBIC, fill=_bg(im))


def _shift(im):
    w, h = im.size
    return TF.affine(im, 0, [int(0.1 * w), int(0.1 * h)], 1.0, [0.0, 0.0], interpolation=BICUBIC, fill=_bg(im))


def _noise(im):
    a = np.asarray(im.convert("RGB")).astype(np.float32) + np.random.randn(*np.asarray(im.convert("RGB")).shape) * 15
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def _lowres(im):
    w, h = im.size
    return im.resize((max(8, w // 3), max(8, h // 3)), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BICUBIC)


def _occ(im):
    im = im.convert("RGB").copy()
    w, h = im.size
    box = (w // 3, h // 3, w // 3 + max(1, w // 5), h // 3 + max(1, h // 5))
    im.paste(tuple(_bg(im)), box)
    return im


PERTURBATIONS = {
    "clean": lambda im: im,
    "rotate+12": _rot(12), "rotate-12": _rot(-12), "shift10%": _shift,
    "blur": lambda im: im.filter(ImageFilter.GaussianBlur(1.5)),
    "lowres": _lowres,
    "dark": lambda im: ImageEnhance.Brightness(im.convert("RGB")).enhance(0.5),
    "bright": lambda im: ImageEnhance.Brightness(im.convert("RGB")).enhance(1.6),
    "lowcontrast": lambda im: ImageEnhance.Contrast(im.convert("RGB")).enhance(0.5),
    "noise": _noise, "occlusion": _occ,
    "invert": lambda im: ImageOps.invert(im.convert("RGB")),
}


class PerturbedTransform:
    def __init__(self, perturb, size, auto_invert=True):
        self.perturb, self.base = perturb, EvalTransform(size, auto_invert)

    def __call__(self, img, y=None, strength=1.0):
        return self.base(self.perturb(img.convert("RGB")))
