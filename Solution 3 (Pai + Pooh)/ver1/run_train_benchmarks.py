"""
Benchmark Training Runner for Thai Character Classification Models (ver1).

Trains and saves checkpoints for:
1. CustomGlyphCNN (4-stage Conv-BN-Mish-SE)
2. AdaptedResNet18 (Stem-adapted ImageNet Transfer Learning)
3. AdaptedMobileNetV3 (Inverted Residuals with SE)

Outputs:
- checkpoints/<model_name>/best_model.pt
- benchmarks/benchmark_results.json
- benchmarks/benchmark_comparison.csv
"""

import json
import os
from pathlib import Path
import sys
import time

# Ensure UTF-8 output encoding on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import pandas as pd
import torch

from src.dataset import create_dataloaders, get_class_weights
from src.evaluate import evaluate_model_full
from src.losses import build_loss_fn
from src.models import build_model
from src.trainer import ThaiCharacterTrainer, build_optimizer, build_scheduler
from src.transforms import get_train_transform, get_val_transform


def run_benchmarks(
    dataset_dir: str = r"D:\vscode\kmitl\3-1\dlmed\ThaiCharacter Dataset\round2",
    epochs: int = 5,
    batch_size: int = 64,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing Benchmark Suite on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    benchmark_dir = Path("benchmarks")
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading Zero-Leakage DataLoaders...")
    train_loader, test_loader, train_df, test_df, class_to_idx = create_dataloaders(
        dataset_dir=dataset_dir,
        batch_size=batch_size,
        target_size=(32, 32),
        train_transform=get_train_transform(max_angle=8.0, p_morphology=0.3),
        test_transform=get_val_transform(),
        use_balanced_sampler=True,
        num_workers=2,
    )
    num_classes = len(class_to_idx)
    class_weights = get_class_weights(train_df, num_classes=num_classes, beta=0.999)

    models_config = [
        {
            "name": "custom_cnn",
            "display_name": "Custom GlyphCNN (4-Stage Mish-SE)",
            "differential_lr": False,
            "base_lr": 1e-3,
            "weight_decay": 1e-2,
            "save_dir": "checkpoints/custom_cnn",
        },
        {
            "name": "resnet18",
            "display_name": "Adapted ResNet-18 (ImageNet Weights)",
            "differential_lr": True,
            "base_lr": 1e-3,
            "backbone_lr": 1e-4,
            "weight_decay": 1e-2,
            "save_dir": "checkpoints/resnet18",
        },
        {
            "name": "mobilenet_v3",
            "display_name": "Adapted MobileNetV3-Small (Inverted Residuals)",
            "differential_lr": True,
            "base_lr": 1e-3,
            "backbone_lr": 1e-4,
            "weight_decay": 1e-2,
            "save_dir": "checkpoints/mobilenet_v3",
        },
    ]

    results = []

    for cfg in models_config:
        m_name = cfg["name"]
        print(f"\n{'='*70}")
        print(f"[+] Training Model: {cfg['display_name']} ({m_name})")
        print(f"{'='*70}")

        model = build_model(m_name, num_classes=num_classes, pretrained=True, adapt_stem=True)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Trainable Parameters: {num_params:,}")

        criterion = build_loss_fn(
            "class_balanced_focal",
            class_weights=class_weights,
            gamma=2.0,
            label_smoothing=0.08,
        )

        optimizer = build_optimizer(
            model,
            optimizer_name="adamw",
            base_lr=cfg["base_lr"],
            differential_lr=cfg["differential_lr"],
            backbone_lr=cfg.get("backbone_lr", 1e-4),
            weight_decay=cfg["weight_decay"],
        )

        scheduler = build_scheduler(
            optimizer,
            scheduler_type="cosine_warmup",
            total_epochs=epochs,
            warmup_epochs=2,
        )

        trainer = ThaiCharacterTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            save_dir=cfg["save_dir"],
        )

        t_start = time.time()
        history = trainer.fit(num_epochs=epochs)
        train_duration = time.time() - t_start

        # Full Evaluation on Best Checkpoint
        best_ckpt_path = Path(cfg["save_dir"]) / "best_model.pt"
        if best_ckpt_path.exists():
            ckpt = torch.load(best_ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])

        eval_stats = evaluate_model_full(model, test_loader, class_to_idx, device=device)

        summary_item = {
            "model_key": m_name,
            "model_name": cfg["display_name"],
            "parameters": num_params,
            "train_duration_sec": round(train_duration, 2),
            "throughput_img_per_sec": round((len(train_df) * epochs) / max(1, train_duration), 1),
            "top1_accuracy": round(float(eval_stats["top1_accuracy"]) * 100, 2),
            "top5_accuracy": round(float(eval_stats["top5_accuracy"]) * 100, 2),
            "macro_f1": round(float(eval_stats["macro_f1"]) * 100, 2),
            "weighted_f1": round(float(eval_stats["weighted_f1"]) * 100, 2),
        }
        results.append(summary_item)
        print(f"[RESULT] {cfg['display_name']}: Top-1 = {summary_item['top1_accuracy']}%, Top-5 = {summary_item['top5_accuracy']}%, Macro-F1 = {summary_item['macro_f1']}%")

    # Save summary files
    with open(benchmark_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    df_results = pd.DataFrame(results)
    df_results.to_csv(benchmark_dir / "benchmark_comparison.csv", index=False)

    print("\n" + "="*80)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("="*80)
    print(df_results[["model_name", "parameters", "top1_accuracy", "top5_accuracy", "macro_f1", "throughput_img_per_sec"]].to_string(index=False))
    print("="*80)


if __name__ == "__main__":
    run_benchmarks(epochs=5, batch_size=64)
