"""TestingCNN.py - predict on unseen images (Chapter 10: Testing phase loads model.pt).

  # unlabeled images (a folder or a single file)
  python TestingCNN.py --ckpt runs/r50/model.pt --input test_images/ --out predictions.csv --tta

  # test folder organised like the training set (numeric class folders) -> also reports accuracy
  python TestingCNN.py --ckpt runs/r50/model.pt --input test_dir/ --labeled --tta

  # ensemble of several models (probabilities are averaged)
  python TestingCNN.py --ckpt runs/a/model.pt runs/b/model.pt --input test_images/ --tta
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from Net import build_from_config
from augment import MEAN, STD, PadResize
from data import IMG_EXT, glyph_group, load_image, scan_dataset
from metrics import adjust_logits, compute_metrics, top_confusions

BICUBIC = T.InterpolationMode.BICUBIC


class TestDataset(Dataset):
    """Returns [V,3,H,W]: the image plus optional label-preserving TTA views."""

    def __init__(self, paths, size, auto_invert, tta):
        self.paths, self.tta = paths, tta
        self.pad = PadResize(size, auto_invert=auto_invert)
        self.to_float = v2.Compose([v2.PILToTensor(), v2.ToDtype(torch.float32, scale=True)])
        self.norm = v2.Normalize(MEAN, STD)
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            base = self.pad(load_image(self.paths[i]))
            ok = True
        except Exception:
            base, ok = None, False
        if not ok:
            return torch.zeros(self.n_views(), 3, self.size, self.size), i, False
        views = [base]
        if self.tta:
            bg = list(base.getpixel((0, 0)))
            views += [TF.rotate(base, d, interpolation=BICUBIC, fill=bg) for d in (-5, 5)]
            views += [TF.affine(base, 0, [dx, dy], 1.0, [0.0, 0.0], interpolation=BICUBIC, fill=bg)
                      for dx, dy in ((-3, 0), (3, 0), (0, -3), (0, 3))]
        return torch.stack([self.norm(self.to_float(v)) for v in views]), i, True

    def n_views(self):
        return 7 if self.tta else 1


def load_ckpt(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = build_from_config(ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval(), ck


@torch.no_grad()
def predict_probs(model, ck, paths, device, tta, alpha, bs, workers):
    cfg = ck["config"]
    ds = TestDataset(paths, cfg["img"], cfg.get("auto_invert", True), tta)
    log_n = np.log(np.maximum(np.asarray(ck["class_counts"], float), 1))
    power = ck.get("sampler_power", 0.5)
    if alpha is None:
        alpha = ck.get("alpha") if ck.get("alpha") is not None else 0.5
    probs, oks = torch.zeros(len(paths), cfg["num_classes"]), torch.ones(len(paths), dtype=torch.bool)
    for x, idx, ok in DataLoader(ds, bs, num_workers=workers):
        B, V = x.shape[:2]
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            lo = model(x.flatten(0, 1).to(device))[0].float().cpu()
        p = adjust_logits(lo, log_n, alpha, power).softmax(1).view(B, V, -1).mean(1)
        probs[idx], oks[idx] = p, ok
    return probs, oks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", nargs="+", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--tta", action="store_true", help="test-time augmentation (7 views)")
    ap.add_argument("--alpha", type=float, default=None, help="prior exponent (default: value tuned in training)")
    ap.add_argument("--labeled", action="store_true")
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = [load_ckpt(c, device) for c in a.ckpt]
    folders, chars = models[0][1]["folders"], models[0][1]["chars"]
    assert all(m[1]["folders"] == folders for m in models), "all checkpoints must share the same classes"
    n_cls = len(folders)

    true = None
    if a.labeled:
        samples, f2, c2 = scan_dataset(a.input)
        f_idx = {f: i for i, f in enumerate(folders)}
        char_idx = {c: i for i, c in enumerate(chars)}  # fallback: match by character, not folder name
        unknown = {}
        keep = []
        for p, y in samples:
            fname, ch = f2[y], c2[y]
            if fname in f_idx:
                keep.append((p, f_idx[fname]))
            elif ch in char_idx:                        # folder naming differs but the char matches
                keep.append((p, char_idx[ch]))
            else:
                unknown.setdefault(f"{fname} ({ch})", 0)
                unknown[f"{fname} ({ch})"] += 1
        if unknown:
            total_unknown = sum(unknown.values())
            print(f"[WARNING] {total_unknown} image(s) skipped: folder/char not in the trained model's "
                  f"{n_cls} classes -> {dict(list(unknown.items())[:10])}"
                  + (" ..." if len(unknown) > 10 else ""))
        assert keep, "no test image matched a class the model was trained on - check folder naming"
        paths, true = [p for p, _ in keep], np.array([y for _, y in keep])
        print(f"matched {len(paths)}/{len(samples)} test images to the model's {n_cls} classes")
    else:
        p = Path(a.input)
        paths = [str(p)] if p.is_file() else sorted(str(q) for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
    print(f"{len(paths)} images | {len(models)} model(s) | TTA={a.tta}")

    probs, oks = None, None
    for model, ck in models:
        pr, ok = predict_probs(model, ck, paths, device, a.tta, a.alpha, a.bs, a.workers)
        probs = pr if probs is None else probs + pr
        oks = ok if oks is None else oks & ok
    probs /= len(models)

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["file", "pred_char", "pred_folder", "confidence", "top3", "true_char"])
        for i, path in enumerate(paths):
            if not oks[i]:
                w.writerow([path, "ERROR", "", "", "unreadable image", ""]); continue
            top = probs[i].topk(3)
            j = int(top.indices[0])
            w.writerow([path, chars[j], folders[j], round(float(top.values[0]), 4),
                        " ".join(f"{chars[int(k)]}:{float(v):.2f}" for v, k in zip(top.values, top.indices)),
                        chars[true[i]] if true is not None else ""])
    print(f"predictions -> {a.out}  (unreadable: {int((~oks).sum())})")

    if true is not None:
        m, cm, recall, support = compute_metrics(probs.argmax(1).numpy(), true, n_cls)
        print(f"ACCURACY {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} | macro-recall {m['macro_recall']:.4f}")
        print("most confused:", [(t, p_, c) for c, t, p_ in top_confusions(cm, chars, 8)])
        for g in range(5):
            idx = [i for i, c in enumerate(chars) if glyph_group(c) == g and support[i] > 0]
            if idx:
                print(f"  group {g}: mean recall {recall[idx].mean():.3f} over {len(idx)} classes")


if __name__ == "__main__":
    main()
