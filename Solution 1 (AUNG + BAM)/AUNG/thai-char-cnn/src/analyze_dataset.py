"""
วิเคราะห์ dataset + สร้างกราฟสำหรับสไลด์

output ที่ได้ (outputs/figures/):
  01_class_distribution.png  — bar chart เรียงจากมากไปน้อย (log scale)
  02_imbalance_curve.png     — cumulative coverage curve
  03_tail_classes.png        — โฟกัส 23 คลาสที่มี < 50 ภาพ
  04_sample_grid.png         — ตัวอย่างภาพจากกลุ่มที่คล้ายกัน
  dataset_stats.csv          — ตารางใส่สไลด์ได้เลย
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .class_map import SIMILAR_GROUPS, ClassMap, readable_name
from .config import load_config
from .dataset import read_image
from .utils import ensure_dir, list_images, setup_logger, setup_thai_font


def build_stats(cm: ClassMap) -> pd.DataFrame:
    counts = cm.counts_by_label()
    total = sum(counts)
    df = pd.DataFrame({
        "label": range(cm.num_classes),
        "folder_id": [cm.label_to_folder[i] for i in range(cm.num_classes)],
        "class": cm.class_names,
        "readable": cm.readable_names,
        "count": counts,
    })
    df["percent"] = df["count"] / total * 100
    df["tier"] = pd.cut(df["count"], [-1, 19, 49, 199, 10**9],
                        labels=["<20", "20-49", "50-199", ">=200"])
    return df.sort_values("count", ascending=False).reset_index(drop=True)


def plot_distribution(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(18, 6))
    colors = {"<20": "#d62728", "20-49": "#ff7f0e",
              "50-199": "#1f77b4", ">=200": "#2ca02c"}
    ax.bar(range(len(df)), df["count"],
           color=[colors[str(t)] for t in df["tier"]])
    ax.set_yscale("log")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["class"], fontsize=9)
    ax.axhline(50, ls="--", c="red", lw=1, label="tail threshold = 50")
    ax.axhline(200, ls="--", c="green", lw=1, label="head threshold = 200")
    ax.set_ylabel("จำนวนภาพ (log scale)")
    ax.set_title(f"การกระจายตัวของข้อมูล 72 คลาส "
                 f"(รวม {df['count'].sum():,} ภาพ | ratio "
                 f"{df['count'].max()}:{df['count'].min()})")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_imbalance_curve(df: pd.DataFrame, out: Path) -> None:
    cum = df["count"].cumsum() / df["count"].sum() * 100
    x = np.arange(1, len(df) + 1) / len(df) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, cum, lw=2.5, color="#1f77b4")
    ax.plot([0, 100], [0, 100], "--", c="gray", lw=1, label="กรณีสมดุลสมบูรณ์")
    ax.fill_between(x, cum, x, alpha=0.2, color="#d62728")

    idx = int(len(df) * 0.25)
    ax.annotate(f"25% ของคลาส\nครอบคลุม {cum.iloc[idx]:.1f}% ของข้อมูล",
                xy=(25, cum.iloc[idx]), xytext=(40, 45),
                arrowprops=dict(arrowstyle="->"), fontsize=11)
    ax.set_xlabel("% ของคลาส (เรียงจากมากไปน้อย)")
    ax.set_ylabel("% ของข้อมูลสะสม")
    ax.set_title("Imbalance Curve — ยิ่งโค้งห่างเส้นทแยงยิ่งไม่สมดุล")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_tail(df: pd.DataFrame, out: Path, threshold: int = 50) -> None:
    tail = df[df["count"] < threshold].sort_values("count")
    fig, ax = plt.subplots(figsize=(10, max(6, len(tail) * 0.32)))
    bars = ax.barh(range(len(tail)), tail["count"], color="#d62728")
    ax.set_yticks(range(len(tail)))
    ax.set_yticklabels(tail["readable"], fontsize=10)
    for b, c in zip(bars, tail["count"]):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2,
                f"{c}", va="center", fontsize=9)
    ax.set_xlabel("จำนวนภาพ")
    ax.set_title(f"{len(tail)} คลาสวิกฤต (< {threshold} ภาพ) — "
                 f"รวมกันเพียง {tail['count'].sum():,} ภาพ "
                 f"({tail['count'].sum()/df['count'].sum()*100:.1f}% ของทั้งหมด)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_similar_samples(cm: ClassMap, raw_dir: Path, out: Path,
                         n_groups: int = 6, n_per: int = 4) -> None:
    groups = [g for g in SIMILAR_GROUPS
              if all(c in cm.char_to_label for c in g)][:n_groups]
    max_cols = max(len(g) for g in groups) * n_per

    fig, axes = plt.subplots(len(groups), max_cols,
                             figsize=(max_cols * 1.15, len(groups) * 1.35))
    axes = np.atleast_2d(axes)
    for a in axes.ravel():
        a.axis("off")

    for r, group in enumerate(groups):
        col = 0
        for ch in group:
            label = cm.char_to_label[ch]
            folder = cm.label_to_folder[label]
            files = list_images(raw_dir / str(folder))[:n_per]
            for k in range(n_per):
                if col >= max_cols:
                    break
                ax = axes[r, col]
                if k < len(files):
                    try:
                        ax.imshow(read_image(files[k]))
                    except Exception:
                        pass
                if k == 0:
                    ax.set_title(f"{ch} ({cm.counts.get(folder,0):,})", fontsize=9)
                col += 1
    fig.suptitle("กลุ่มตัวอักษรที่หน้าตาคล้ายกัน — ความท้าทายหลักของงานนี้",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def text_report(df: pd.DataFrame) -> str:
    total = df["count"].sum()
    tiers = df.groupby("tier", observed=True).agg(
        n_classes=("count", "size"), n_images=("count", "sum"))
    lines = [
        "=" * 70, "DATASET ANALYSIS", "=" * 70,
        f"  จำนวนคลาส     : {len(df)}",
        f"  จำนวนภาพรวม   : {total:,}",
        f"  เฉลี่ยต่อคลาส : {total/len(df):,.0f}",
        f"  มัธยฐาน       : {df['count'].median():,.0f}",
        f"  มากสุด        : {df.iloc[0]['class']} = {df.iloc[0]['count']:,}",
        f"  น้อยสุด       : {df.iloc[-1]['class']} = {df.iloc[-1]['count']:,}",
        f"  Imbalance     : {df['count'].max()/max(df['count'].min(),1):,.0f} : 1",
        "-" * 70, "  การกระจายตามช่วง:",
        f"  {'ช่วง':<12}{'คลาส':>8}{'ภาพ':>12}{'% ข้อมูล':>12}",
    ]
    for tier, row in tiers.iterrows():
        lines.append(f"  {str(tier):<12}{row['n_classes']:>8}"
                    f"{row['n_images']:>12,}{row['n_images']/total*100:>11.1f}%")

    # สร้างข้อความ Top 5 และ Bottom 5 แยกออกมาก่อน เพื่อป้องกันปัญหา f-string ซ้อนกัน
    top5_str = ' '.join(f"{r['class']}({r['count']:,})" for _, r in df.head(5).iterrows())
    bottom5_str = ' '.join(f"{r['class']}({r['count']})" for _, r in df.tail(5).iterrows())

    lines += [
        "-" * 70,
        f"  Top 5   : {top5_str}",
        f"  Bottom 5: {bottom5_str}",
        "=" * 70,
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="วิเคราะห์ dataset + สร้างกราฟ")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--raw-dir", default=None)
    ap.add_argument("--no-samples", action="store_true",
                    help="ข้ามการสร้าง sample grid (เร็วขึ้น)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    raw_dir = Path(args.raw_dir or cfg.get_path("paths.raw_dir", "data/raw"))
    fig_dir = Path(cfg.get_path("paths.figure_dir", "outputs/figures"))
    ensure_dir(fig_dir)

    logger = setup_logger("analyze",
                          Path(cfg.get_path("paths.log_dir", "outputs/logs")) / "analyze.log")
    font = setup_thai_font()
    logger.info(f"ฟอนต์ที่ใช้: {font or 'default (อาจแสดงไทยไม่ได้)'}")

    cm = ClassMap.from_raw_dir(raw_dir, strict=False)
    cm.save(cfg.get_path("paths.class_map", "data/splits/class_map.json"))

    df = build_stats(cm)
    df.to_csv(fig_dir / "dataset_stats.csv", index=False, encoding="utf-8-sig")
    logger.info("\n" + text_report(df))

    plot_distribution(df, fig_dir / "01_class_distribution.png")
    plot_imbalance_curve(df, fig_dir / "02_imbalance_curve.png")
    plot_tail(df, fig_dir / "03_tail_classes.png",
              cfg.get_path("augmentation.offline.tail_threshold", 50))
    if not args.no_samples:
        plot_similar_samples(cm, raw_dir, fig_dir / "04_sample_grid.png")

    logger.info(f"✓ บันทึกกราฟทั้งหมดที่ {fig_dir}")
    logger.info("ขั้นถัดไป: python -m src.make_splits --ratio 0.8")


if __name__ == "__main__":
    main()