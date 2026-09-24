"""
Training and Fine-Tuning Entry Point for Thai Character Recognition.
Executes the full training pipeline or Fine-tuning from Terminal or VS Code:
1. Data scanning & stratified splitting (80:20) with single-sample safety
2. Train-only Data Augmentation (minimum 80 samples/class, Validation untouched)
3. DataLoader creation with Augmentations
4. Stage 1: Frozen Backbone training (can be skipped with --finetune-only)
5. Stage 2: Fine-Tuning top backbone blocks (35 Epochs)
6. Best checkpoint saving (.pth)
7. Training curve visualization & evaluation reports

Usage:
    # 1. รัน Full 2-Stage Training
    python train.py --data-dir ./data

    # 2. รัน Fine-tuning เท่านั้น โดยต่อยอดจาก best_thai_character_model_v2.pth
    python train.py --finetune-only --data-dir "D:/ThaiCharacter Dataset v2"
"""

import argparse
import os
from pathlib import Path
import sys

import pandas as pd
import torch

from src.config import (
    BATCH_SIZE,
    CLASSIFICATION_REPORT_PATH,
    CONFUSION_MATRIX_PATH,
    CURVE_IMAGE_PATH,
    DATA_DIR,
    DEFAULT_CHECKPOINT_PATH,
    DEVICE,
    DROPOUT_RATE,
    HISTORY_CSV_PATH,
    MIN_TRAIN_SAMPLES_PER_CLASS,
    NUM_WORKERS,
    OUTPUT_DIR,
    PIN_MEMORY,
    SEED,
    STAGE1_EPOCHS,
    STAGE2_EPOCHS,
    VAL_RATIO,
    resolve_checkpoint_path,
    set_seed,
)
from src.dataset import (
    augment_train_set_only,
    create_dataloaders,
    scan_dataset,
    stratified_split_per_group,
)
from src.losses import compute_class_weights, get_loss_function
from src.metrics import (
    compute_classification_report,
    compute_confusion_matrix,
    get_confused_pairs,
)
from src.models import create_model, load_model_from_checkpoint
from src.trainer import (
    evaluate_by_source,
    plot_and_save_confusion_matrix,
    plot_and_save_training_curves,
    run_epoch,
    setup_stage1,
    setup_stage2,
    train_stage,
)
from src.transforms import get_train_transforms, get_val_transforms


def parse_args():
    parser = argparse.ArgumentParser(description="เทรนและ Fine-tune โมเดลจำแนกตัวอักษรไทย (EfficientNet-B0)")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help=f"โฟลเดอร์เก็บข้อมูล Dataset (ค่าเริ่มต้น: {DATA_DIR})"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT_PATH),
        help=f"Path สำหรับบันทึกไฟล์โมเดล Checkpoint (ค่าเริ่มต้น: {DEFAULT_CHECKPOINT_PATH})"
    )
    parser.add_argument(
        "--base-checkpoint",
        type=str,
        default="best_thai_character_model_v2.pth",
        help="Path ของโมเดล Checkpoint เริ่มต้นสำหรับโหมด --finetune-only"
    )
    parser.add_argument(
        "--finetune-only",
        action="store_true",
        help="ข้าม Stage 1 และโหลดโมเดล base checkpoint มาทำ Fine-tuning สู่ best_thai_character_finetuned.pth ทันที"
    )
    parser.add_argument(
        "--min-train-samples",
        type=int,
        default=MIN_TRAIN_SAMPLES_PER_CLASS,
        help=f"จำนวนภาพขั้นต่ำต่อคลาสใน Train Set ด้วย Data Augmentation (ค่าเริ่มต้น: {MIN_TRAIN_SAMPLES_PER_CLASS})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help=f"ขนาด Batch Size (ค่าเริ่มต้น: {BATCH_SIZE})"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=NUM_WORKERS,
        help=f"จำนวน DataLoader Workers (ค่าเริ่มต้น: {NUM_WORKERS})"
    )
    parser.add_argument(
        "--stage1-epochs",
        type=int,
        default=STAGE1_EPOCHS,
        help=f"จำนวน Epoch สำหรับ Stage 1: Freeze Backbone (ค่าเริ่มต้น: {STAGE1_EPOCHS})"
    )
    parser.add_argument(
        "--stage2-epochs",
        type=int,
        default=STAGE2_EPOCHS,
        help=f"จำนวน Epoch สำหรับ Stage 2: Fine-Tuning (ค่าเริ่มต้น: {STAGE2_EPOCHS})"
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=VAL_RATIO,
        help=f"สัดส่วนชุดข้อมูล Validation (ค่าเริ่มต้น: {VAL_RATIO})"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help=f"Random Seed (ค่าเริ่มต้น: {SEED})"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    print("=" * 65)
    print("  ระบบฝึกสอนและ Fine-tuning ตัวอักษรไทย (EfficientNet-B0)")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Data Directory: {args.data_dir}")
    print(f"Checkpoint Target: {args.checkpoint}")
    print(f"Mode: {'Fine-tuning Only' if args.finetune_only else 'Full 2-Stage Training'}")
    print(f"Batch Size: {args.batch_size} | Workers: {args.num_workers}")
    if not args.finetune_only:
        print(f"Epochs: Stage 1 = {args.stage1_epochs}, Stage 2 = {args.stage2_epochs}")
    else:
        print(f"Epochs: Fine-tuning = {args.stage2_epochs} (ข้าม Stage 1)")
    print("-" * 65)

    data_path = Path(args.data_dir)
    # ตรวจสอบว่าโฟลเดอร์มีอยู่จริงหรือไม่ หากไม่พบ ให้ค้นหาโฟลเดอร์ทางเลือกอัตโนมัติ
    if not data_path.exists():
        fallback_candidates = [
            Path("D:/ThaiCharacter Dataset v2"),
            Path("./ThaiCharacter Dataset v2"),
            Path("../ThaiCharacter Dataset v2"),
        ]
        found = False
        for candidate in fallback_candidates:
            if candidate.exists():
                print(f"[*] ไม่พบโฟลเดอร์ '{data_path}' สลับไปใช้โฟลเดอร์ที่พบ: {candidate.resolve()}")
                data_path = candidate
                found = True
                break
        if not found:
            print(f"[ข้อผิดพลาด] ไม่พบโฟลเดอร์ Dataset: {data_path.resolve()}", file=sys.stderr)
            print("กรุณาวาง Dataset ไว้ที่โฟลเดอร์ ./data หรือระบุพารามิเตอร์ --data-dir เช่น:", file=sys.stderr)
            print('  python train.py --data-dir "D:/ThaiCharacter Dataset v2"', file=sys.stderr)
            sys.exit(1)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. สแกน Dataset
    print("\n[1/7] สแกนและจัดโครงสร้างข้อมูล Dataset...")
    data_df, class_to_idx, idx_to_class, num_classes = scan_dataset(data_path)
    print(f"  -> พบภาพทั้งหมด: {len(data_df)} ภาพ | จำนวนคลาส: {num_classes} คลาส")
    print(f"  -> การกระจายสไตล์:\n{data_df['source'].value_counts().to_string()}")

    # 2. แบ่ง Train / Validation
    print("\n[2/7] แบ่ง Train / Validation Set (สัดส่วน 80:20 พร้อมป้องกัน Error คลาส 1 ภาพ)...")
    train_df, val_df = stratified_split_per_group(
        dataframe=data_df,
        group_cols=["label", "source"],
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    print(f"  -> ต้นฉบับ: Train: {len(train_df)} ภาพ | Val: {len(val_df)} ภาพ")

    # ทำ Train-only Augmentation (ฝั่ง Val คงเป็นภาพต้นฉบับ 100% ป้องกัน Data Leakage)
    if args.min_train_samples > 0:
        print(f"\n[+] ตรวจสอบและทำ Data Augmentation เฉพาะ Train Set (เป้าหมายขั้นต่ำ {args.min_train_samples} ภาพ/คลาส)...")
        train_df = augment_train_set_only(
            train_df=train_df,
            target_count=args.min_train_samples,
            seed=args.seed
        )
        print(f"  -> จำนวน Train หลัง Augment: {len(train_df)} ภาพ | Val ยังคงเดิม: {len(val_df)} ภาพ")

    # 3. เตรียม DataLoader
    print("\n[3/7] สร้าง DataLoader...")
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        train_transform=train_transform,
        val_transform=val_transform,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=PIN_MEMORY
    )
    print(f"  -> Train Batches: {len(train_loader)} | Val Batches: {len(val_loader)}")

    # 4. คำนวณ Class Weight และ Loss Function
    print("\n[4/7] คำนวณ Class Weights และเตรียม Loss Function...")
    class_weights = compute_class_weights(train_df, num_classes=num_classes, device=DEVICE)
    criterion = get_loss_function(class_weights=class_weights)

    history = []
    best_score = -1.0

    # 5. เตรียมโมเดล
    if args.finetune_only:
        print(f"\n[5/7] โหมด Fine-tuning Only: กำลังค้นหา Base Checkpoint '{args.base_checkpoint}'...")
        base_ckpt_path = resolve_checkpoint_path(args.base_checkpoint)
        print(f"  -> โหลด Base Checkpoint จาก: {base_ckpt_path.resolve()}")
        model, class_to_idx, idx_to_class, prev_best = load_model_from_checkpoint(
            base_ckpt_path, device=DEVICE, dropout=DROPOUT_RATE
        )
        if prev_best is not None:
            best_score = prev_best
            print(f"  -> คะแนน Best Score เดิม: {best_score:.4f}")
    else:
        print("\n[5/7] สร้างโมเดล EfficientNet-B0 เริ่มต้นจาก ImageNet Pretrained...")
        model = create_model(num_classes=num_classes, pretrained=True, dropout=DROPOUT_RATE)
        model = model.to(DEVICE)

    # 6. Stage 1: Freeze Backbone (ข้ามหากเป็น --finetune-only)
    if not args.finetune_only and args.stage1_epochs > 0:
        print(f"\n[6/7 - Stage 1] Freeze Backbone เทรนเฉพาะ Classifier Head ({args.stage1_epochs} Epochs)...")
        optimizer1, scheduler1 = setup_stage1(model)
        model, best_score, history = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer1,
            scheduler=scheduler1,
            criterion=criterion,
            epochs=args.stage1_epochs,
            stage_name="Frozen backbone",
            checkpoint_path=checkpoint_path,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            best_score=best_score,
            history=history,
            device=DEVICE
        )

    # 7. Stage 2: Fine-Tuning (35 Epochs)
    if args.stage2_epochs > 0:
        stage2_name = "Fine-tuning"
        print(f"\n[6/7 - Stage 2] Fine-Tuning Backbone บล็อกท้ายสุด ({args.stage2_epochs} Epochs)...")
        optimizer2, scheduler2 = setup_stage2(
            model=model,
            checkpoint_path=checkpoint_path if (not args.finetune_only and checkpoint_path.exists()) else base_ckpt_path,
            device=DEVICE
        )
        model, best_score, history = train_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer2,
            scheduler=scheduler2,
            criterion=criterion,
            epochs=args.stage2_epochs,
            stage_name=stage2_name,
            checkpoint_path=checkpoint_path,
            class_to_idx=class_to_idx,
            idx_to_class=idx_to_class,
            best_score=best_score,
            history=history,
            device=DEVICE
        )

    # 8. สรุปผลและประเมินผลอย่างละเอียด
    print("\n[7/7] ประเมินผลโมเดลที่ดีที่สุดและบันทึกรายงาน...")
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_result = run_epoch(model, val_loader, criterion, device=DEVICE)
    print(f"\n== สรุปผลรวม Validation Set ==")
    print(f"  Loss: {val_result['loss']:.4f}")
    print(f"  Accuracy: {val_result['accuracy']*100:.2f}%")
    print(f"  Macro-F1: {val_result['macro_f1']:.4f}")

    print("\n== ประเมินแยกตามสไตล์ (Handwritten vs Printed) ==")
    evaluate_by_source(
        model=model,
        val_df=val_df,
        val_transform=val_transform,
        criterion=criterion,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=DEVICE
    )

    # บันทึกประวัติการเทรน
    if history:
        history_df = pd.DataFrame(history)
        history_df.to_csv(HISTORY_CSV_PATH, index=False, encoding="utf-8")
        print(f"[*] บันทึกประวัติการเทรนที่: {HISTORY_CSV_PATH.resolve()}")
        plot_and_save_training_curves(history, CURVE_IMAGE_PATH)

    # บันทึก Classification Report
    present_labels = sorted(val_df["label"].unique())
    present_class_names = [idx_to_class[label] for label in present_labels]
    report_df = compute_classification_report(
        val_result["targets"],
        val_result["predictions"],
        labels=present_labels,
        target_names=present_class_names
    )
    report_df.to_csv(CLASSIFICATION_REPORT_PATH, encoding="utf-8-sig")
    print(f"[*] บันทึก Classification Report ที่: {CLASSIFICATION_REPORT_PATH.resolve()}")

    # Confusion Matrix
    confusion = compute_confusion_matrix(
        val_result["targets"],
        val_result["predictions"],
        labels=present_labels
    )
    plot_and_save_confusion_matrix(confusion, present_class_names, CONFUSION_MATRIX_PATH)

    # คู่อักษรที่สับสน
    confused_pairs = get_confused_pairs(confusion, present_class_names, top_n=10)
    if not confused_pairs.empty:
        print("\n== 10 คู่ตัวอักษรที่โมเดลสับสนมากที่สุด ==")
        print(confused_pairs.to_string(index=False))

    print("\n" + "=" * 65)
    print(f"[สำเร็จ] เสร็จสิ้นกระบวนการ! โมเดลถูกบันทึกไว้ที่: {checkpoint_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
