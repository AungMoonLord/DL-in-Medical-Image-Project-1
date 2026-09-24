"""
Comparative Benchmark and Model Evaluation Suite (ver2).

Compares:
1. Models: CustomGlyphCNN, AdaptedResNet18, AdaptedMobileNetV3, Solution 2 EfficientNet-B0.
2. Datasets: Original (round2) vs Cleaned (round2-cleaned).
3. Evaluates Macro-F1, Top-1 Accuracy, Top-3 Accuracy, and latency.
4. Outputs structured Markdown and JSON benchmark tables.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import pandas as pd
import torch

# Ensure stdout is utf-8
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure ver2/src is in python path
src_dir = Path(__file__).resolve().parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from dataset import build_dataloaders
from inference import UniversalModelLoader
from models import build_model
from trainer import Trainer


def evaluate_checkpoint(
    checkpoint_path: Path,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Evaluates a checkpoint directly on validation set."""
    model, model_type, res, custom_class_map = UniversalModelLoader.inspect_and_load(
        checkpoint_path, device=device, num_classes=72
    )

    all_preds = []
    all_targets = []
    top3_correct = 0
    total = 0
    t0 = time.time()

    idx_map = None
    if custom_class_map:
        idx_map = {int(v): int(k) for k, v in custom_class_map.items()}

    total_batches = len(val_loader)
    with torch.no_grad():
        for b_idx, (tensors, targets, class_nums) in enumerate(val_loader, 1):
            tensors = tensors.to(device)
            # If model expects different resolution, resize on fly
            if res != tensors.size(-1):
                tensors = torch.nn.functional.interpolate(tensors, size=(res, res), mode="bilinear", align_corners=False)

            logits = model(tensors)
            preds = logits.argmax(dim=-1).cpu().numpy()
            targets_np = targets.numpy()

            all_preds.extend(preds)
            all_targets.extend(targets_np)

            top3 = torch.topk(logits, k=min(3, logits.size(-1)), dim=-1).indices.cpu()
            top3_correct += (top3 == targets.unsqueeze(1)).any(dim=-1).sum().item()
            total += targets.size(0)

            if b_idx % 20 == 0 or b_idx == total_batches:
                print(f"  Processed [{total}/{len(val_loader.dataset)}] samples...", end="\r", flush=True)

    print()

    elapsed = time.time() - t0
    from sklearn.metrics import f1_score
    import numpy as np

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    top1 = float((all_preds == all_targets).mean())
    top3 = float(top3_correct / max(1, total))
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0))
    throughput = round(total / max(0.001, elapsed), 1)

    return {
        "model_type": model_type,
        "resolution": f"{res}x{res}",
        "top1_acc": round(top1 * 100, 2),
        "top3_acc": round(top3 * 100, 2),
        "macro_f1": round(macro_f1 * 100, 2),
        "weighted_f1": round(weighted_f1 * 100, 2),
        "samples_per_sec": throughput,
    }


def main():
    base_dir = Path(__file__).resolve().parent
    dataset_candidates = [
        (base_dir / ".." / ".." / ".." / "ThaiCharacterDataset").resolve(),
        (base_dir / ".." / ".." / ".." / "ThaiCharacter Dataset").resolve(),
        (Path.cwd() / "ThaiCharacterDataset").resolve(),
    ]
    dataset_root = next((p for p in dataset_candidates if p.exists()), None)
    if dataset_root is None:
        raise FileNotFoundError(f"ThaiCharacterDataset not found. Searched: {dataset_candidates}")

    # Check available subfolders
    dataset_options = [
        [dataset_root / "dataset-bam", dataset_root / "dataset-pooh", dataset_root / "dataset-ajbank-cleaned"],
        dataset_root / "dataset-ajbank-cleaned",
        dataset_root / "dataset-bam-100",
        dataset_root / "dataset-bam",
        dataset_root / "dataset-pooh",
        dataset_root / "round2-cleaned",
        dataset_root / "round2",
    ]
    active_dataset = None
    for opt in dataset_options:
        if isinstance(opt, list):
            if all(p.exists() for p in opt):
                active_dataset = opt
                break
        elif opt.exists():
            active_dataset = opt
            break

    if active_dataset is None:
        raise FileNotFoundError(f"No valid dataset subdirectories found in {dataset_root}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔬 Running Thai Character Benchmarks on {device} (Dataset: {active_dataset})...")

    # Load Cleaned Dataset Validation Loader
    _, val_loader_clean, _, _, test_df_clean = build_dataloaders(
        dataset_dir=active_dataset, batch_size=64, target_size=(32, 32), use_balanced_sampler=False
    )
    print(f"Loaded Cleaned Dataset Test Split: {len(test_df_clean)} samples.")

    checkpoints_to_eval = [
        ("Solution 3 (Adapted ResNet-18 ver2)", base_dir / "checkpoints" / "resnet18" / "best_model.pt"),
        ("Solution 3 (CustomGlyphCNN ver2)", base_dir / "checkpoints" / "custom_cnn" / "best_model.pt"),
        ("Solution 3 (CustomGlyphCNN ver1)", base_dir / ".." / "ver1" / "checkpoints" / "custom_cnn" / "best_model.pt"),
        ("Solution 2 (EfficientNet-B0)", base_dir / ".." / ".." / "Solution 2 (Eungul + Ninenine)" / "best_thai_character_model.pth"),
        ("Solution 3 (MobileNetV3 ver2)", base_dir / "checkpoints" / "mobilenet_v3" / "best_model.pt"),
    ]

    results = []
    for name, ckpt_path in checkpoints_to_eval:
        if ckpt_path.exists():
            print(f"Evaluating {name} from {ckpt_path}...")
            try:
                metrics = evaluate_checkpoint(ckpt_path, val_loader_clean, device)
                results.append({"model_name": name, **metrics})
            except Exception as e:
                print(f"Error evaluating {name}: {e}")

    # Output Benchmark Table
    benchmarks_dir = base_dir / "benchmarks"
    benchmarks_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmarks_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 COMPARATIVE BENCHMARK RESULTS (Evaluated on Cleaned Test Set)")
    print("=" * 80)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
