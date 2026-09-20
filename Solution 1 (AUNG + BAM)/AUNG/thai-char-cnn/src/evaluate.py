"""
ประเมินผลเต็มรูปแบบ + สร้างรูปสำหรับสไลด์

output (outputs/figures/):
  eval_confusion_matrix.png      — เต็ม 72x72
  eval_confusion_worst.png       — โฟกัสเฉพาะคลาสที่แย่ที่สุด
  eval_per_class_recall.png      — recall เทียบกับจำนวนภาพ train
  eval_recall_vs_count.png       — scatter พิสูจน์ความสัมพันธ์ imbalance ↔ recall
  eval_gradcam.png               — โมเดลดูตรงไหน
  eval_report.txt / per_class_metrics.csv / confused_pairs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader

from .class_map import ClassMap
from .config import load_config
from .dataset import ThaiCharDataset
from .engine import validate, validate_tta
from .metrics import (format_report, get_confusion_matrix, group_analysis,
                      per_class_table, top_confused_pairs)
from .models import build_model
from .transforms import build_tta_transforms, build_val_transform, denormalize
from .utils import (amp_dtype_from_str, describe_device, ensure_dir, get_device,
                    load_checkpoint, setup_logger, setup_thai_font)


def plot_confusion(cm: np.ndarray, names: list[str], out: Path,
                   title: str = "Confusion Matrix (normalized)") -> None:
    fig, ax = plt.subplots(figsize=(20, 17))
    sns.heatmap(cm, xticklabels=names, yticklabels=names, cmap="viridis",
                vmin=0, vmax=1, square=True, cbar_kws={"shrink": 0.6}, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title, fontsize=15)
    plt.xticks(fontsize=7, rotation=90)
    plt.yticks(fontsize=7, rotation=0)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def plot_confusion_subset(targets, preds, names, labels, out: Path,
                          title: str) -> None:
    from sklearn.metrics import confusion_matrix
    # 1. คำนวณ matrix รวม 72 คลาสก่อน เพื่อไม่ให้ตัวอย่างที่ทายหลุดนอกกลุ่มหายไป
    full_cm = confusion_matrix(targets, preds, labels=np.arange(len(names)))
    with np.errstate(divide="ignore", invalid="ignore"):
        full_cmn = np.nan_to_num(full_cm.astype(np.float64) / full_cm.sum(1, keepdims=True))

    # 2. Slice ดึงเฉพาะกลุ่มคลาสที่ต้องการมาพล็อต
    cm = full_cm[np.ix_(labels, labels)]
    cmn = full_cmn[np.ix_(labels, labels)]
    sub = [names[i] for i in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.55),
                                    max(7, len(labels) * 0.5)))
    sns.heatmap(cmn, annot=cm, fmt="d", xticklabels=sub, yticklabels=sub,
                cmap="Reds", vmin=0, vmax=1, square=True, ax=ax,
                annot_kws={"size": 8})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)

def plot_per_class_recall(df: pd.DataFrame, out: Path) -> None:
    d = df.sort_values("train_count", ascending=False)
    fig, ax1 = plt.subplots(figsize=(19, 6))
    x = np.arange(len(d))
    colors = ["#d62728" if c < 50 else "#ff7f0e" if c < 200 else "#2ca02c"
              for c in d["train_count"]]
    ax1.bar(x, d["recall"] * 100, color=colors)
    ax1.set_ylabel("Recall (%)")
    ax1.set_ylim(0, 105)
    ax1.set_xticks(x)
    ax1.set_xticklabels(d["class"], fontsize=8)

    ax2 = ax1.twinx()
    ax2.plot(x, d["train_count"], color="black", lw=1.2, alpha=0.7,
             label="จำนวนภาพใน train")
    ax2.set_yscale("log")
    ax2.set_ylabel("จำนวนภาพ train (log)")
    ax2.legend(loc="lower left")
    ax1.set_title("Per-class Recall เทียบกับปริมาณข้อมูล  "
                  "(แดง = tail <50, ส้ม = 50-199, เขียว = ≥200)")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_recall_vs_count(df: pd.DataFrame, out: Path) -> None:
    d = df[df["val_support"] > 0]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(d["train_count"], d["recall"] * 100, s=55, alpha=0.75,
               c=np.where(d["train_count"] < 50, "#d62728", "#1f77b4"))
    for _, r in d[(d["recall"] < 0.6) | (d["train_count"] < 50)].iterrows():
        ax.annotate(r["class"], (r["train_count"], r["recall"] * 100),
                    fontsize=10, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("จำนวนภาพใน train (log)")
    ax.set_ylabel("Recall (%)")
    ax.axvline(50, ls="--", c="red", lw=1, label="tail threshold")
    ax.set_title("ความสัมพันธ์ระหว่างปริมาณข้อมูลกับ Recall")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_gradcam(model, dataset, device, class_map, out: Path,
                 n: int = 12, targets=None, preds=None) -> None:
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
    except ImportError:
        print("[warn] ไม่พบ grad-cam — ข้าม (pip install grad-cam)")
        return

    # เลือกตัวอย่างที่ทายผิด เพื่อวิเคราะห์ว่าโมเดลดูอะไรผิด
    if targets is not None and preds is not None:
        wrong = np.where(np.asarray(targets) != np.asarray(preds))[0]
        idxs = wrong[:n] if len(wrong) >= n else \
            np.concatenate([wrong, np.arange(n - len(wrong))])
    else:
        idxs = np.linspace(0, len(dataset) - 1, n).astype(int)

    layer = model.get_cam_target_layer()
    cam = GradCAM(model=model, target_layers=[layer])

    cols = 4
    rows = int(np.ceil(len(idxs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3.2))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, idxs):
        img, label = dataset[int(i)]
        tensor = img.unsqueeze(0).to(device)
        try:
            grayscale = cam(input_tensor=tensor)[0]
        except Exception:
            continue
        rgb = denormalize(img).permute(1, 2, 0).cpu().numpy()
        ax.imshow(show_cam_on_image(rgb, grayscale, use_rgb=True))
        true_c = class_map.label_to_char[int(label)]
        title = f"true: {true_c}"
        if preds is not None:
            title += f" | pred: {class_map.label_to_char[int(preds[i])]}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    for ax in axes[len(idxs):]:
        ax.axis("off")

    fig.suptitle("Grad-CAM — บริเวณที่โมเดลใช้ตัดสินใจ", fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="ประเมินผลโมเดล")
    ap.add_argument("--ckpt", default="outputs/checkpoints/best_model.pth")
    ap.add_argument("--config", default=None)
    ap.add_argument("--csv", default=None, help="ค่าเริ่มต้นคือ val.csv")
    ap.add_argument("--tta", action="store_true", help="เปิด TTA")
    ap.add_argument("--no-gradcam", action="store_true")
    args = ap.parse_args()

    ckpt = load_checkpoint(args.ckpt)
    cfg = load_config(args.config) if args.config else None
    if cfg is None:
        from .config import Config
        cfg = Config(ckpt["config"])

    fig_dir = Path(cfg.get_path("paths.figure_dir", "outputs/figures"))
    ensure_dir(fig_dir)
    logger = setup_logger("evaluate",
                          Path(cfg.get_path("paths.log_dir", "outputs/logs")) / "evaluate.log")
    setup_thai_font()

    device = get_device(cfg.get_path("hardware.device", "cuda"))
    logger.info(f"Device: {describe_device(device)}")

    class_map = ClassMap.load(cfg.get_path("paths.class_map"))
    names = class_map.class_names
    n_cls = class_map.num_classes

    train_counts = np.bincount(pd.read_csv(cfg.get_path("paths.train_csv"))["label"],
                               minlength=n_cls)

    model = build_model(cfg, class_counts=train_counts).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    logger.info(f"โหลด checkpoint: stage {ckpt.get('stage')} "
                f"epoch {ckpt.get('epoch')} | {ckpt.get('metrics', {})}")

    csv_path = args.csv or cfg.get_path("paths.val_csv")
    val_ds = ThaiCharDataset(csv_path, transform=build_val_transform(cfg))
    logger.info(f"ชุดประเมิน: {val_ds.summary()}")

    amp = cfg.get_path("hardware.amp", True)
    amp_dtype = amp_dtype_from_str(cfg.get_path("hardware.amp_dtype", "bfloat16"))
    nw = cfg.get_path("hardware.num_workers", 8)

    use_tta = args.tta or cfg.get_path("eval.tta.enabled", False)
    
    raw_counts = class_map.counts_by_label()  # สถิติจำนวนภาพจริง 72 คลาส

    if use_tta:
        result, raw = validate_tta(
            # แก้ไขบรรทัดที่ 221 โดยระบุ path ของ cfg โดยตรง
            model, val_ds, build_tta_transforms(cfg, cfg.eval.tta.n_views), device,
            batch_size=128, num_workers=nw, num_classes=n_cls,
            class_counts=raw_counts, class_names=names,  # <-- ส่ง raw_counts
            amp=amp, amp_dtype=amp_dtype)
    else:
        loader = DataLoader(val_ds, batch_size=128, shuffle=False,
                            num_workers=nw, pin_memory=True)
        result, raw = validate(model, loader, torch.nn.CrossEntropyLoss(),
                               device, num_classes=n_cls,
                               class_counts=raw_counts, class_names=names,  # <-- ส่ง raw_counts
                               amp=amp, amp_dtype=amp_dtype, return_raw=True)

    targets, preds = raw["targets"], raw["preds"]

    report = format_report(result, worst_k=20)
    logger.info("\n" + report)
    (fig_dir / "eval_report.txt").write_text(report, encoding="utf-8")

    df = per_class_table(result, sort_by="recall")
    df.to_csv(fig_dir / "per_class_metrics.csv", index=False, encoding="utf-8-sig")

    pairs = top_confused_pairs(targets, preds, names, n_cls,
                               cfg.get_path("eval.report.confusion_top_k", 25))
    pd.DataFrame(pairs).to_csv(fig_dir / "confused_pairs.csv",
                               index=False, encoding="utf-8-sig")
    logger.info("\nคู่ที่สับสนมากที่สุด:")
    for p in pairs[:12]:
        logger.info(f"  {p['true']} → {p['pred']} : {p['count']} ครั้ง "
                    f"({p['rate']*100:.1f}% ของ {p['true']})")

    groups = group_analysis(targets, preds, class_map.similar_groups_as_labels(), names)
    pd.DataFrame(groups).to_csv(fig_dir / "group_analysis.csv",
                                index=False, encoding="utf-8-sig")
    logger.info("\nผลตามกลุ่มที่คล้ายกัน (เรียงจากแย่สุด):")
    for g in groups[:8]:
        logger.info(f"  [{g['group']}] acc {g['accuracy']*100:.1f}% | "
                    f"สับสนในกลุ่ม {g['within_group_error']*100:.1f}% | "
                    f"หลุดนอกกลุ่ม {g['outside_group_error']*100:.1f}%")

    # ---- figures ----
    plot_confusion(get_confusion_matrix(targets, preds, n_cls), names,
                   fig_dir / "eval_confusion_matrix.png",
                   f"Confusion Matrix — Acc {result.accuracy*100:.2f}% / "
                   f"Macro-F1 {result.macro_f1*100:.2f}%")

    worst = df.nsmallest(20, "recall")["label"].tolist()
    plot_confusion_subset(targets, preds, names, sorted(worst),
                          fig_dir / "eval_confusion_worst.png",
                          "Confusion Matrix — 20 คลาสที่ recall ต่ำสุด")
    plot_per_class_recall(df, fig_dir / "eval_per_class_recall.png")
    plot_recall_vs_count(df, fig_dir / "eval_recall_vs_count.png")

    if not args.no_gradcam:
        plot_gradcam(model, val_ds, device, class_map,
                     fig_dir / "eval_gradcam.png",
                     n=cfg.get_path("eval.report.gradcam_samples", 12),
                     targets=targets, preds=preds)

    logger.info(f"\n✓ บันทึกผลทั้งหมดที่ {fig_dir}")


if __name__ == "__main__":
    main()