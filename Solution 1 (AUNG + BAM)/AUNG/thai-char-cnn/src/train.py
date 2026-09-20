"""
Orchestrator — Two-Stage Decoupled Training

Stage 1: instance sampling, backbone trainable
         → เรียน representation ของลายเส้นอักษรไทยจากข้อมูลตามธรรมชาติ
Stage 2: sqrt-balanced sampling, backbone frozen
         → ปรับเฉพาะ decision boundary ให้ tail class

อ้างอิง: Kang et al., "Decoupling Representation and Classifier for
Long-Tailed Recognition", ICLR 2020
"""
from __future__ import annotations

import math
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .class_map import ClassMap
from .config import apply_runtime_settings, load_config, print_config
from .dataset import build_dataloaders
from .engine import (EarlyStopping, build_optimizer, build_scheduler,
                     train_one_epoch, validate)
from .losses import build_loss, build_mixup
from .metrics import format_report
from .models import build_model
from .sampler import build_sampler, describe_sampler
from .utils import (Timer, amp_dtype_from_str, check_blackwell_support,
                    count_parameters, describe_device, ensure_dir, get_device,
                    save_checkpoint, setup_logger)


def run_stage(stage_num: int, cfg, model, class_map, device, logger,
              history: list, best_state: dict, writer=None) -> dict:
    stage = f"stage{stage_num}"
    if not cfg.get_path(f"train.{stage}.enabled", True):
        logger.info(f"ข้าม {stage} (enabled=false)")
        return best_state

    logger.info("\n" + "=" * 74)
    logger.info(f"  STAGE {stage_num}  |  "
                f"sampler={cfg.get_path(f'train.{stage}.sampler')}  |  "
                f"freeze_backbone={cfg.get_path(f'train.{stage}.freeze_backbone')}")
    logger.info("=" * 74)

    freeze = cfg.get_path(f"train.{stage}.freeze_backbone", False)
    model.freeze_backbone(freeze)
    total, trainable = count_parameters(model)
    logger.info(f"พารามิเตอร์: {total:,} ทั้งหมด | {trainable:,} trainable "
                f"({trainable/total*100:.1f}%)")

    # ---- data ----
    train_csv = pd.read_csv(cfg.get_path("paths.train_csv"))
    labels = train_csv["label"].to_numpy()
    n_cls = class_map.num_classes
    mode = cfg.get_path(f"train.{stage}.sampler", "instance")

    g = torch.Generator().manual_seed(cfg.get_path("project.seed", 42))
    sampler = build_sampler(labels, n_cls, mode=mode, generator=g)
    logger.info("\n" + describe_sampler(labels, n_cls, mode,
                                        class_map.class_names))

    train_loader, val_loader, train_ds, val_ds = build_dataloaders(
        cfg, class_map, stage=stage, sampler=sampler)
    logger.info(f"train: {train_ds.summary()}")
    logger.info(f"val  : {val_ds.summary()}")

    train_counts = train_ds.class_counts(n_cls)

    # ---- loss / optim ----
    criterion = build_loss(cfg, train_counts).to(device)
    mixup = build_mixup(cfg, n_cls, train_counts) if stage_num == 1 else None
    optimizer = build_optimizer(model, cfg, stage)
    epochs = cfg.get_path(f"train.{stage}.epochs", 30)

    # คำนวณจำนวนก้าวที่แท้จริงต่อ epoch เมื่อมี gradient accumulation
    grad_accum = cfg.get_path("hardware.grad_accum_steps", 1)
    actual_steps = max(1, math.ceil(len(train_loader) / grad_accum))
    scheduler = build_scheduler(optimizer, cfg, stage, actual_steps)

    amp = cfg.get_path("hardware.amp", True)
    amp_dtype = amp_dtype_from_str(cfg.get_path("hardware.amp_dtype", "bfloat16"))
    scaler = torch.amp.GradScaler("cuda") if (amp and amp_dtype == torch.float16) else None
    channels_last = cfg.get_path("hardware.channels_last", True)

    es = EarlyStopping(
        patience=cfg.get_path("eval.early_stopping.patience", 12),
        min_delta=cfg.get_path("eval.early_stopping.min_delta", 1e-3),
        mode=cfg.get_path("eval.monitor_mode", "max"))
    enable_es = cfg.get_path("eval.early_stopping.enabled", True)
    metric_key = cfg.get_path("eval.primary_metric", "macro_f1")
    ckpt_dir = Path(cfg.get_path("paths.ckpt_dir", "outputs/checkpoints"))
    ensure_dir(ckpt_dir)

    for epoch in range(epochs):
        tr = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch=epoch, total_epochs=epochs, scheduler=scheduler,
            scaler=scaler, amp=amp, amp_dtype=amp_dtype,
            channels_last=channels_last, mixup=mixup,
            max_grad_norm=cfg.get_path("hardware.max_grad_norm", 5.0),
            grad_accum=cfg.get_path("hardware.grad_accum_steps", 1),
            label_smoothing=cfg.get_path("loss.label_smoothing", 0.1),
            log_interval=cfg.get_path("logging.log_interval", 20),
            logger=logger, stage=stage_num)


        result, raw = validate(
            model, val_loader, criterion, device, num_classes=n_cls,
            # แก้บรรทัดนี้: เปลี่ยน train_counts เป็น class_map.counts_by_label()
            class_counts=class_map.counts_by_label(), class_names=class_map.class_names,
            amp=amp, amp_dtype=amp_dtype, channels_last=channels_last)

        score = getattr(result, metric_key)
        improved = es.step(score)

        logger.info(
            f"[S{stage_num}] ep {epoch+1:>3}/{epochs} | "
            f"loss {tr['loss']:.4f} | train_acc {tr['acc']*100:5.2f}% | "
            f"{result.summary()} | lr {tr['lr']:.2e} | {tr['time']:.0f}s"
            + ("  ★ best" if improved else ""))

        history.append({"stage": stage_num, "epoch": epoch + 1,
                        "train_loss": tr["loss"], "train_acc": tr["acc"],
                        "val_loss": raw["loss"], "lr": tr["lr"],
                        **result.to_dict()})

        if writer:
            writer.add_scalar(f"stage{stage_num}/train_loss", tr["loss"], epoch)
            writer.add_scalar(f"stage{stage_num}/val_macro_f1", result.macro_f1, epoch)
            writer.add_scalar(f"stage{stage_num}/val_accuracy", result.accuracy, epoch)
            writer.add_scalar(f"stage{stage_num}/tail_recall", result.tail_recall, epoch)

        if improved and score > best_state.get("score", -1):
            best_state = {"score": score, "stage": stage_num,
                          "epoch": epoch + 1, "metrics": result.to_dict()}
            save_checkpoint(ckpt_dir / "best_model.pth", model, optimizer,
                            scheduler, epoch + 1, stage_num, result.to_dict(),
                            cfg.to_dict(), class_map.to_dict())
            (ckpt_dir / "best_report.txt").write_text(
                format_report(result), encoding="utf-8")

        if enable_es and es.should_stop:
            logger.info(f"Early stopping ที่ epoch {epoch+1} "
                        f"(ไม่ดีขึ้น {es.patience} epochs)")
            break

    if cfg.get_path("logging.save_last", True):
        save_checkpoint(ckpt_dir / f"last_stage{stage_num}.pth", model,
                        optimizer, scheduler, epochs, stage_num,
                        {}, cfg.to_dict(), class_map.to_dict())
    return best_state


def main() -> None:
    ap = argparse.ArgumentParser(description="เทรนโมเดลแบบ two-stage")
    ap.add_argument("--config", default="configs/effnet_arcface.yaml")
    ap.add_argument("--override", nargs="*", default=[],
                    help="เช่น train.stage1.epochs=50 model.backbone=convnext_tiny")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, args.override)
    apply_runtime_settings(cfg)

    log_dir = Path(cfg.get_path("paths.log_dir", "outputs/logs"))
    exp = cfg.get_path("experiment.name", "run")
    logger = setup_logger("train", log_dir / f"{exp}.log",
                          cfg.get_path("logging.level", "INFO"))

    device = get_device(cfg.get_path("hardware.device", "cuda"))
    check_blackwell_support(device)
    logger.info(f"Device: {describe_device(device)}")
    print_config(cfg, logger)

    class_map = ClassMap.load(cfg.get_path("paths.class_map"))
    logger.info(f"โหลด class_map: {class_map.num_classes} คลาส")

    train_counts = np.bincount(
        pd.read_csv(cfg.get_path("paths.train_csv"))["label"],
        minlength=class_map.num_classes)

    model = build_model(cfg, class_counts=train_counts).to(device)
    if cfg.get_path("hardware.channels_last", True):
        model = model.to(memory_format=torch.channels_last)
    if cfg.get_path("hardware.compile", False):
        logger.info("torch.compile(): กำลัง compile (ครั้งแรกใช้เวลา 1-2 นาที)")
        model = torch.compile(model)

    total, _ = count_parameters(model)
    logger.info(f"Model: {cfg.get_path('model.backbone')} + "
                f"{cfg.get_path('model.head')} head | {total:,} params")

    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        logger.info(f"โหลด weight จาก {args.resume} (epoch {ck.get('epoch')})")

    writer = None
    if cfg.get_path("logging.tensorboard", True):
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(log_dir / "tb" / exp)
        except ImportError:
            logger.warning("ไม่พบ tensorboard — ข้าม")

    history, best = [], {"score": -1.0}
    with Timer() as t:
        best = run_stage(1, cfg, model, class_map, device, logger, history, best, writer)
        best = run_stage(2, cfg, model, class_map, device, logger, history, best, writer)

    if cfg.get_path("logging.save_history_csv", True):
        pd.DataFrame(history).to_csv(log_dir / f"{exp}_history.csv",
                                     index=False, encoding="utf-8-sig")
    (log_dir / f"{exp}_best.json").write_text(
        json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    if writer:
        writer.close()

    logger.info("\n" + "=" * 74)
    logger.info(f"  เทรนเสร็จใน {t.pretty}")
    logger.info(f"  Best {cfg.get_path('eval.primary_metric')} = "
                f"{best['score']*100:.2f}%  (stage {best['stage']}, "
                f"epoch {best['epoch']})")
    logger.info(f"  Checkpoint: {cfg.get_path('paths.ckpt_dir')}/best_model.pth")
    logger.info("=" * 74)
    logger.info("ขั้นถัดไป: python -m src.evaluate "
                "--ckpt outputs/checkpoints/best_model.pth")


if __name__ == "__main__":
    main()