"""analyze_data.py - describe the dataset for the presentation.

    python analyze_data.py --data /path/to/dataset --out reports

Creates: class_counts.csv, class_distribution.png, sample_grid.png, augmentation_preview.png,
         data_report.json  (+ prints a summary of challenges: imbalance, duplicates, polarity...)
"""
import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from augment import MEAN, STD, PadResize, SampleAwareAugment
from data import GROUP_NAMES, audit, glyph_group, load_image, pick_thai_font, scan_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples, folders, chars = scan_dataset(a.data)
    n_cls = len(folders)
    info = audit(samples, out / "audit_cache.json")
    ok = [s for s in samples if info[s[0]]["ok"]]
    counts = np.bincount([y for _, y in ok], minlength=n_cls)
    groups = [glyph_group(c) for c in chars]
    font = pick_thai_font()
    if font:
        plt.rcParams["font.family"] = font
    label = lambda i: chars[i] if font else folders[i]

    # ---- table
    n = counts.astype(float)
    rarity = np.clip(1 - np.log(np.maximum(n, 1)) / np.log(n.max()), 0, 1)
    with open(out / "class_counts.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["folder", "char", "group", "count", "share_%", "rarity"])
        for i in np.argsort(-counts):
            w.writerow([folders[i], chars[i], GROUP_NAMES[groups[i]], int(counts[i]),
                        round(100 * counts[i] / counts.sum(), 2), round(float(rarity[i]), 3)])

    # ---- distribution chart
    order = np.argsort(-counts)
    colors = plt.cm.tab10(np.array(groups)[order])
    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(n_cls), counts[order], color=colors)
    ax.set_yscale("log"); ax.set_xticks(range(n_cls))
    ax.set_xticklabels([label(i) for i in order], fontsize=7, rotation=90)
    ax.set_ylabel("images (log scale)"); ax.set_title(f"Images per class ({n_cls} classes, {int(counts.sum()):,} images)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.cm.tab10(g)) for g in range(len(GROUP_NAMES))]
    ax.legend(handles, GROUP_NAMES); fig.tight_layout(); fig.savefig(out / "class_distribution.png", dpi=130); plt.close(fig)

    # ---- one sample per class
    pad = PadResize(64)
    cols = 9
    rows = int(np.ceil(n_cls / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.7))
    first = {}
    for p, y in ok:
        first.setdefault(y, p)
    for k, axx in enumerate(axes.ravel()):
        axx.axis("off")
        if k < n_cls and k in first:
            axx.imshow(pad(load_image(first[k])))
            axx.set_title(f"{label(k)} ({counts[k]})", fontsize=8)
    fig.tight_layout(); fig.savefig(out / "sample_grid.png", dpi=120); plt.close(fig)

    # ---- augmentation preview (sample-aware: rare classes are augmented more strongly)
    aug = SampleAwareAugment(96, counts, groups)
    pick = [int(order[0]), int(order[n_cls // 2]), int(order[-1])]
    for g in (1, 3):
        c = [i for i in range(n_cls) if groups[i] == g]
        if c:
            pick.append(c[0])
    random.seed(0); torch.manual_seed(0)
    fig, axes = plt.subplots(len(pick), 9, figsize=(13, 1.6 * len(pick)))
    for r, cls in enumerate(pick):
        p = next(p for p, y in ok if y == cls)
        img = load_image(p)
        axes[r, 0].imshow(PadResize(96)(img)); axes[r, 0].set_ylabel(f"{label(cls)}\nn={counts[cls]}\n{GROUP_NAMES[groups[cls]]}", fontsize=8)
        for c in range(1, 9):
            x = aug(img, cls, 1.0) * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
            axes[r, c].imshow(x.clamp(0, 1).permute(1, 2, 0).numpy())
        for c in range(9):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_title("original" if r == 0 else "", fontsize=8)
    fig.suptitle("Sample-aware augmentation at strength 1.0 (col 1 = original)"); fig.tight_layout()
    fig.savefig(out / "augmentation_preview.png", dpi=120); plt.close(fig)

    # ---- report
    good = [info[p] for p, _ in ok]
    W, H = np.array([g["w"] for g in good]), np.array([g["h"] for g in good])
    lum = np.array([g["lum"] for g in good])
    by_md5 = {}
    for p, y in ok:
        by_md5.setdefault(info[p]["md5"], set()).add(y)
    rep = {
        "classes": n_cls, "images": int(counts.sum()), "corrupt_files": len(samples) - len(ok),
        "imbalance_ratio_max_over_min": float(counts.max() / max(counts.min(), 1)),
        "median_per_class": float(np.median(counts)), "mean_per_class": float(counts.mean()),
        "top10_share_%": round(100 * np.sort(counts)[::-1][:10].sum() / counts.sum(), 1),
        "classes_lt_10": int((counts < 10).sum()), "classes_lt_50": int((counts < 50).sum()),
        "classes_lt_100": int((counts < 100).sum()),
        "group_class_counts": dict(Counter(GROUP_NAMES[g] for g in groups)),
        "width_min_med_max": [int(W.min()), float(np.median(W)), int(W.max())],
        "height_min_med_max": [int(H.min()), float(np.median(H)), int(H.max())],
        "aspect_ratio_p5_p50_p95": [round(float(v), 2) for v in np.percentile(W / H, [5, 50, 95])],
        "image_modes": dict(Counter(g["mode"] for g in good)),
        "dark_background_images_%": round(float((lum < 110).mean() * 100), 2),
        "exact_duplicate_files": int(len(ok) - len(by_md5)),
        "exact_duplicates_across_classes": int(sum(len(v) > 1 for v in by_md5.values())),
    }
    json.dump(rep, open(out / "data_report.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(rep, indent=1, ensure_ascii=False))
    print(f"\nfigures/tables saved to {out}/")


if __name__ == "__main__":
    main()
