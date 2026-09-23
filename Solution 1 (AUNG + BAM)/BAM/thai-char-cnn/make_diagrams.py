"""make_diagrams.py - draws architecture.png and pipeline.png for the slides.

    python make_diagrams.py [--out diagrams] [--arch resnet50] [--img 128]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, PURPLE, GREEN, GRAY, YELLOW, RED = "#cfe2ff", "#ffe0b3", "#e2d4f5", "#d3f0d8", "#e6e6e6", "#fff6c2", "#f8d0d0"


def box(ax, x, y, w, h, text, fc, fs=9, bold=False, ec="#333", ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.0,rounding_size=1.0", fc=fc, ec=ec, lw=1.3, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(ax, p1, p2, color="#333", ls="-", rad=0.0, lw=1.6):
    ax.annotate("", xy=p2, xytext=p1, arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=ls,
                                                     connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def architecture(out, arch="resnet50", img=128):
    fig, ax = plt.subplots(figsize=(19, 8.4))
    ax.set_xlim(0, 168); ax.set_ylim(0, 84); ax.axis("off")
    ax.text(84, 80.5, f"ThaiCharNet - CNN architecture  (backbone: {arch}, input {img}x{img})",
            ha="center", fontsize=16, fontweight="bold")

    box(ax, 1, 34, 14, 16, "Input image\nany size\nRGB / RGBA / gray\n(scan or photo)", GRAY, 9)
    box(ax, 19, 34, 18, 16, f"Pre-process\nauto polarity\nPadResize -> {img}x{img}\n(no stretching)\nImageNet normalise", YELLOW, 8.5)
    arrow(ax, (15, 42), (19, 42)); arrow(ax, (37, 42), (42.5, 42))

    # backbone stages (ResNet50 numbers for 128 px input)
    s = img // 4
    stages = [("Stem\nconv7x7 s2\n+ maxpool", f"64x{s}x{s}", 26), (f"layer1\n3 blocks", f"256x{s}x{s}", 23),
              (f"layer2\n4 blocks", f"512x{s // 2}x{s // 2}", 20), (f"layer3\n6 blocks", f"1024x{s // 4}x{s // 4}", 17),
              (f"layer4\n3 blocks", f"2048x{s // 8}x{s // 8}", 14)]
    ax.add_patch(FancyBboxPatch((41.5, 15), 55, 53, boxstyle="round,pad=0,rounding_size=1.5", fc="#f2f7ff", ec="#3b6fd4", lw=1.6, ls="--"))
    ax.text(69, 65.3, f"Backbone: {arch}  (ImageNet-pretrained)", ha="center", fontsize=10.5, fontweight="bold", color="#1f4fb0")
    x = 43.0
    for i, (name, shape, h) in enumerate(stages):
        box(ax, x, 42 - h / 2, 9.6, h, name, BLUE, 7.5)
        ax.text(x + 4.8, 42 - h / 2 - 2.2, shape, ha="center", fontsize=7.5, color="#1f4fb0")
        if i:
            arrow(ax, (x - 0.4, 42), (x, 42), lw=1.2)
        x += 10.8
    ax.text(69, 17.3, "Transfer Learning: all layers fine-tuned, backbone LR = 0.1 x head LR", ha="center", fontsize=8.5, color="#1f4fb0")

    box(ax, 99, 34, 12, 16, "SE block\n(channel\nattention)\n2048", ORANGE, 8.5)
    arrow(ax, (96.5, 42), (99, 42))
    box(ax, 114, 34, 14, 16, "Dropout2d 0.1\nGAP  +  GMP\nconcat\n4096-d", ORANGE, 8.5)
    arrow(ax, (111, 42), (114, 42))
    box(ax, 131, 30, 15, 24, "Dropout 0.3\nFC 4096 -> 512\nBatchNorm\nReLU\nDropout 0.3", ORANGE, 8.5)
    arrow(ax, (128, 42), (131, 42))

    box(ax, 149, 44, 15, 13, "Head A (main)\nFC 512 -> 72\nSoftmax over\n72 Thai classes", GREEN, 8, True)
    box(ax, 149, 25, 15, 13, "Head B (aux.)\nFC 512 -> 5\nglyph group", PURPLE, 8, True)
    arrow(ax, (146, 44), (149, 50)); arrow(ax, (146, 40), (149, 31))
    ax.text(156.5, 41, "Multi-task Learning", ha="center", va="center", fontsize=7.5, color="#5b3a99", fontweight="bold")

    box(ax, 100, 3, 67, 12, "Training loss = CrossEntropy(72 classes, label smoothing 0.1)\n+ 0.3 x CrossEntropy(5 groups: consonant / upper mark / lower mark / digit / other)", RED, 8)
    ax.plot([164, 166.6, 166.6], [50.5, 50.5, 16.5], color="#a33", lw=1.6); arrow(ax, (166.6, 19), (166.6, 15), "#a33")
    arrow(ax, (156.5, 25), (156.5, 15), "#a33")
    box(ax, 1, 3, 96, 8.5,
        "Inference:  softmax( logits + (alpha - (1 - p)) * log n_c )  ->  prior-corrected class probabilities  ->  argmax\n"
        "(alpha tuned on validation for imbalance;  optional TTA x7 views and model ensemble)", GREEN, 8.5)
    ax.text(1, 29, "Regularisation:\nDropout, Dropout2d, weight decay\n(AdamW), label smoothing, EMA,\nsample-aware augmentation\nwith image occlusion", fontsize=8.5, va="top", linespacing=1.4)
    # legend
    for k, (c, t) in enumerate([(BLUE, "pretrained (transfer)"), (ORANGE, "new layers"), (GREEN, "main task"), (PURPLE, "auxiliary task")]):
        box(ax, 1 + k * 24, 70, 3.2, 3.2, "", c); ax.text(5 + k * 24, 71.6, t, fontsize=8.5, va="center")
    fig.savefig(Path(out) / "architecture.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)


def poly(ax, pts, color="#333", ls="-", lw=1.6):
    xs, ys = zip(*pts)
    ax.plot(xs[:-1], ys[:-1], color=color, ls=ls, lw=lw)
    ax.plot(xs[-2:], ys[-2:], color=color, ls=ls, lw=lw)
    arrow(ax, (xs[-2] + (xs[-1] - xs[-2]) * 0.6, ys[-2] + (ys[-1] - ys[-2]) * 0.6), (xs[-1], ys[-1]), color, lw=lw)


def pipeline(out):
    fig, ax = plt.subplots(figsize=(19, 10.5))
    ax.set_xlim(0, 170); ax.set_ylim(0, 100); ax.axis("off")
    ax.text(85, 97.5, "Full pipeline: data -> train / validate -> test", ha="center", fontsize=16, fontweight="bold")

    # column 1: data preparation
    box(ax, 1, 76, 29, 18, "Dataset\n72 classes, 62,707 images\nfolder code (TIS-620)\n-> Thai character", GRAY, 8.5, True)
    box(ax, 1, 53, 29, 19, "Image audit (data.py)\ncorrupt / RGBA / EXIF\npolarity check\nmd5 + dHash duplicates", YELLOW, 8.5)
    box(ax, 1, 28, 29, 21, "Grouped stratified\nsplit 80 / 20 per class\nduplicates stay together\n(no leakage);\n1-image classes -> train", YELLOW, 8.5)
    arrow(ax, (15.5, 76), (15.5, 72)); arrow(ax, (15.5, 53), (15.5, 49))
    ax.text(1, 21, "AFTER TRAINING\n/ TEST", fontsize=10.5, fontweight="bold", color="#5b3a99", va="top")

    # train row
    ax.text(36, 91.5, "TRAIN  (80 %)", fontsize=11.5, fontweight="bold", color="#1f4fb0")
    box(ax, 36, 64, 23, 24, "Sqrt-inverse-\nfrequency sampler\n(class imbalance)", ORANGE, 8.8)
    box(ax, 63, 64, 42, 24, "Sample-aware augmentation\n(augment.py)\n- geometric / photometric / stroke width\n- occlusion: Random Erase, Cutout, Hide-and-Seek\n- strength = progressive x adaptive\n  x class rarity x glyph group", ORANGE, 8.2)
    box(ax, 109, 64, 26, 24, "ThaiCharNet (Net.py)\npretrained backbone\n+ SE attention\n+ dropout\n+ 2 heads", BLUE, 8.8, True)
    box(ax, 139, 64, 30, 24, "Multi-task loss\nAdamW, warm-up + cosine\nadaptive LR (plateau)\ngradient clipping, AMP\nEMA weights", RED, 8.5)
    for a_, b_ in [(59, 63), (105, 109), (135, 139)]:
        arrow(ax, (a_, 76), (b_, 76))
    poly(ax, [(30, 40), (33, 40), (33, 76), (36, 76)])

    # validate row
    ax.text(36, 53.5, "VALIDATE  (20 %)", fontsize=11.5, fontweight="bold", color="#1a7a35")
    box(ax, 36, 31, 23, 17, "Eval transform\nPadResize + Normalize\n(no augmentation)", GREEN, 8.8)
    box(ax, 63, 31, 42, 17, "Every epoch: accuracy, macro-F1 (raw & EMA)\nclean-train accuracy -> generalisation gap\nbest checkpoint = 0.5 x (acc + macro-F1)", GREEN, 8.5)
    box(ax, 109, 31, 26, 17, "Early stopping\nsave best -> model.pt", GREEN, 8.8)
    arrow(ax, (30, 40), (36, 40)); arrow(ax, (59, 40), (63, 40)); arrow(ax, (105, 40), (109, 40))
    arrow(ax, (122, 64), (122, 48), "#777")
    ax.text(123.5, 56, "weights", fontsize=8, color="#555")
    arrow(ax, (84, 48), (84, 64), "#c0392b", "--", lw=1.8)
    ax.text(85.5, 56, "gap -> adapt augmentation strength", fontsize=8.5, color="#c0392b")
    poly(ax, [(135, 36), (154, 36), (154, 64)], "#c0392b", "--", 1.8)
    ax.text(136, 38.2, "plateau -> LR x 0.5", fontsize=8, color="#c0392b")

    # after training row
    box(ax, 36, 5, 22, 15, "model.pt\n(weights + config\n+ class list)", PURPLE, 8.8, True)
    box(ax, 63, 5, 33, 15, "alpha sweep (0 ... 1)\nprior correction\nfor class imbalance", PURPLE, 8.8)
    box(ax, 101, 5, 30, 15, "Analysis: per-class recall,\nconfusion pairs, error gallery,\nrobustness x 11 perturbations", PURPLE, 8.2)
    box(ax, 136, 5, 33, 15, "TestingCNN.py on unseen data\nsame pre-processing, TTA x7,\nensemble -> predictions.csv", GREEN, 8.5)
    for a_, b_ in [(58, 63), (96, 101), (131, 136)]:
        arrow(ax, (a_, 12.5), (b_, 12.5))
    poly(ax, [(122, 31), (122, 25.5), (47, 25.5), (47, 20)], "#5b3a99")
    fig.savefig(Path(out) / "pipeline.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="diagrams")
    ap.add_argument("--arch", default="resnet50")
    ap.add_argument("--img", type=int, default=128)
    a = ap.parse_args()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    architecture(a.out, a.arch, a.img)
    pipeline(a.out)
    print(f"saved {a.out}/architecture.png and {a.out}/pipeline.png")
