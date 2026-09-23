"""TrainingCNN.py - train + validate the Thai character/number CNN (72 classes).

    python TrainingCNN.py --data /path/to/dataset --out runs/r50 --arch resnet50 --img 128 --epochs 40

Outputs in --out:  model.pt  classes.json  split.json  data_audit.json  history.csv  curves.png
                   alpha_sweep.json  final_metrics.json  per_class.csv  top_confusions.txt
                   confusion_matrix.png  errors/ (mis-classified val images)
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from Net import ARCHS, EMA, ThaiCharNet, build_from_config
from augment import (PERTURBATIONS, AugController, EvalTransform, PerturbedTransform, SampleAwareAugment)
from data import (GROUP_NAMES, CharDataset, audit, glyph_group, grouped_stratified_split, load_image,
                  make_sampler, pick_thai_font, scan_dataset)
from metrics import adjust_logits, compute_metrics, top_confusions


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="folder with class folders (or containing round2/)")
    p.add_argument("--out", default="runs/exp")
    p.add_argument("--arch", default="resnet50", choices=ARCHS)
    p.add_argument("--img", type=int, default=128)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--bs", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3, help="LR of new layers; backbone = lr*backbone_mult")
    p.add_argument("--backbone_mult", type=float, default=0.1)
    p.add_argument("--wd", type=float, default=0.05)
    p.add_argument("--ls", type=float, default=0.1, help="label smoothing")
    p.add_argument("--drop", type=float, default=0.3, help="dropout in the neck")
    p.add_argument("--sampler_power", type=float, default=0.5, help="0 natural, 1 balanced, 0.5 sqrt")
    p.add_argument("--group_w", type=float, default=0.3, help="weight of auxiliary glyph-group loss")
    p.add_argument("--ema", type=float, default=0.999)
    p.add_argument("--warmup", type=float, default=2.0, help="LR warm-up epochs")
    p.add_argument("--plateau_patience", type=int, default=3, help="adaptive LR: halve LR after N epochs w/o gain")
    p.add_argument("--patience", type=int, default=10, help="early-stopping patience (epochs)")
    p.add_argument("--gap_subset", type=int, default=4000, help="#train images used to measure the gap")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.5, help="prior exponent used only with --full")
    p.add_argument("--no_pretrained", action="store_true")
    p.add_argument("--no_dedupe", action="store_true", help="skip image audit / duplicate-aware split")
    p.add_argument("--no_invert", action="store_true", help="disable auto polarity normalisation")
    p.add_argument("--no_occlusion", action="store_true")
    p.add_argument("--no_adaptive_aug", action="store_true")
    p.add_argument("--no_attention", action="store_true")
    p.add_argument("--fine_detail", action="store_true", help="keep more spatial resolution (slower)")
    p.add_argument("--full", action="store_true", help="train on 100%% data (final model, after tuning)")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def get_logits(model, loader, device):
    model.eval()
    lo, ys = [], []
    for x, y in loader:
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))[0]
        lo.append(out.float().cpu()); ys.append(y)
    return torch.cat(lo), torch.cat(ys)


def score(m):
    return 0.5 * (m["acc"] + m["macro_f1"])


def save_curves(hist, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    ax[0, 0].plot(ep, [h["train_loss"] for h in hist]); ax[0, 0].set_title("train loss")
    ax[0, 1].plot(ep, [h["val_acc"] for h in hist], label="val acc (raw)")
    ax[0, 1].plot(ep, [h["ema_acc"] for h in hist], label="val acc (EMA)")
    ax[0, 1].plot(ep, [h["ema_f1"] for h in hist], "--", label="val macro-F1 (EMA)")
    ax[0, 1].legend(); ax[0, 1].set_title("validation")
    ax[1, 0].plot(ep, [h["aug_strength"] for h in hist], label="augmentation strength")
    ax[1, 0].plot(ep, [h["lr_scale"] for h in hist], label="LR plateau scale"); ax[1, 0].legend()
    ax[1, 0].set_title("adaptive schedules")
    ax[1, 1].plot(ep, [h["gap"] for h in hist]); ax[1, 1].axhline(0, c="gray", lw=.5)
    ax[1, 1].set_title("generalisation gap (clean-train acc - val acc)")
    for a_ in ax.ravel():
        a_.set_xlabel("epoch")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def save_confusion(cm, chars, folders, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    font = pick_thai_font()
    if font:
        plt.rcParams["font.family"] = font
    labels = chars if font else folders
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=90); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("predicted"); ax.set_ylabel("true"); ax.set_title("Row-normalised confusion matrix (validation)")
    fig.colorbar(im, fraction=0.03); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    a = get_args()
    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    json.dump(vars(a), open(out / "config.json", "w"), indent=1)

    # ------------------------------------------------------------ data
    samples, folders, chars = scan_dataset(a.data)
    n_cls = len(folders)
    info = None if a.no_dedupe else audit(samples, out / "audit_cache.json")
    train_s, val_s, rep, usable = grouped_stratified_split(samples, info, 0.2, a.seed)
    if a.full:
        train_s = usable
    print(f"classes={n_cls} images={len(samples)} usable={len(usable)} corrupt={len(rep['corrupt'])} "
          f"dup-groups={rep['duplicate_groups']} cross-class-dups={len(rep['cross_class_duplicates'])}")
    print(f"train={len(train_s)} val={len(val_s)} device={device}")
    json.dump(rep, open(out / "data_audit.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    json.dump({"folders": folders, "chars": chars}, open(out / "classes.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    json.dump({"train": [p for p, _ in train_s], "val": [p for p, _ in val_s]},
              open(out / "split.json", "w", encoding="utf-8"), ensure_ascii=False)

    class_groups = [glyph_group(c) for c in chars]
    gmap = torch.tensor(class_groups, device=device)
    sampler, counts = make_sampler(train_s, n_cls, a.sampler_power)
    log_n = np.log(np.maximum(counts, 1))
    auto_invert = not a.no_invert

    train_ds = CharDataset(train_s, SampleAwareAugment(a.img, counts, class_groups, auto_invert, not a.no_occlusion))
    eval_tf = EvalTransform(a.img, auto_invert)
    rng = random.Random(a.seed)
    gap_ds = CharDataset(rng.sample(train_s, min(a.gap_subset, len(train_s))), eval_tf)
    val_ds = CharDataset(val_s, eval_tf)
    pin = device.type == "cuda"
    tr_loader = DataLoader(train_ds, a.bs, sampler=sampler, num_workers=a.workers, pin_memory=pin, drop_last=True)
    ev = lambda ds: DataLoader(ds, a.bs * 2, shuffle=False, num_workers=a.workers, pin_memory=pin)
    va_loader, gap_loader = ev(val_ds), ev(gap_ds)

    # ------------------------------------------------------------ model / optimiser
    cfg = dict(arch=a.arch, num_classes=n_cls, num_groups=len(GROUP_NAMES), drop=a.drop,
               fine_detail=a.fine_detail, attention=not a.no_attention, img=a.img, auto_invert=auto_invert)
    model = ThaiCharNet(a.arch, n_cls, len(GROUP_NAMES), not a.no_pretrained, a.drop, a.fine_detail,
                        not a.no_attention).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")
    ema = EMA(model, a.ema)
    opt = torch.optim.AdamW([{"params": model.backbone_params(), "lr": a.lr * a.backbone_mult},
                             {"params": model.new_params(), "lr": a.lr}], weight_decay=a.wd)
    spe = len(tr_loader)
    total, warm = a.epochs * spe, max(1, int(a.warmup * spe))
    lr_state = {"scale": 1.0}                                   # adaptive LR (plateau) multiplier

    def lr_lambda(step):
        base = (step + 1) / warm if step < warm else \
            0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm)))
        return base * lr_state["scale"]

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit_c, crit_g = nn.CrossEntropyLoss(label_smoothing=a.ls), nn.CrossEntropyLoss()
    ctrl = AugController(a.epochs, adaptive=not a.no_adaptive_aug)

    # ------------------------------------------------------------ train / validate
    hist, best, bad, plateau = [], -1.0, 0, 0
    fields = ["epoch", "train_loss", "val_acc", "val_f1", "ema_acc", "ema_f1", "gap", "aug_strength", "lr_scale", "time_s"]
    logf = open(out / "history.csv", "w", newline="")
    log = csv.DictWriter(logf, fieldnames=fields); log.writeheader()

    for ep in range(a.epochs):
        s = ctrl.strength(ep)
        train_ds.strength = s                                   # workers are re-spawned each epoch
        model.train(); t0 = time.time(); run = 0.0
        for x, y in tr_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=use_amp):
                lc, lg = model(x)
                loss = crit_c(lc, y) + a.group_w * crit_g(lg, gmap[y])   # multi-task loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update(); sched.step(); ema.update(model)
            run += loss.item()

        lo_r, yv = get_logits(model, va_loader, device)
        m_raw = compute_metrics(lo_r.argmax(1), yv, n_cls)[0]
        lo_e, _ = get_logits(ema.module, va_loader, device)
        m_ema = compute_metrics(lo_e.argmax(1), yv, n_cls)[0]
        lo_g, yg = get_logits(model, gap_loader, device)
        gap = float((lo_g.argmax(1) == yg).float().mean()) - m_raw["acc"]
        ctrl.update(gap)                                        # adaptive augmentation feedback

        sc_r, sc_e = score(m_raw), score(m_ema)
        cur, src = (sc_e, ema.module) if sc_e >= sc_r else (sc_r, model)
        if a.full:
            cur, src = ep, ema.module                           # no honest val in full mode
        improved = cur > best + 1e-4
        if improved:
            best, bad, plateau = cur, 0, 0
            torch.save({"state_dict": src.state_dict(), "config": cfg, "folders": folders, "chars": chars,
                        "class_counts": counts.tolist(), "sampler_power": a.sampler_power,
                        "alpha": a.alpha, "epoch": ep + 1, "group_names": GROUP_NAMES}, out / "model.pt")
        else:
            bad += 1; plateau += 1
            if plateau >= a.plateau_patience:                   # adaptive LR scheduling
                lr_state["scale"] = max(0.05, lr_state["scale"] * 0.5); plateau = 0
        row = dict(epoch=ep + 1, train_loss=run / spe, val_acc=m_raw["acc"], val_f1=m_raw["macro_f1"],
                   ema_acc=m_ema["acc"], ema_f1=m_ema["macro_f1"], gap=gap, aug_strength=s,
                   lr_scale=lr_state["scale"], time_s=int(time.time() - t0))
        hist.append(row); log.writerow(row); logf.flush()
        print(f"ep {ep + 1:02d}/{a.epochs} loss {row['train_loss']:.4f} | val acc {m_raw['acc']:.4f} "
              f"f1 {m_raw['macro_f1']:.4f} | EMA acc {m_ema['acc']:.4f} f1 {m_ema['macro_f1']:.4f} | "
              f"gap {gap:+.3f} aug {s:.2f} lr x{lr_state['scale']:.2f} | {row['time_s']}s"
              + (" *" if improved else ""))
        if bad >= a.patience and not a.full:
            print("early stopping"); break
    save_curves(hist, out / "curves.png")

    # ------------------------------------------------------------ final analysis on best model
    ck = torch.load(out / "model.pt", map_location=device, weights_only=False)
    best_model = build_from_config(ck["config"]).to(device)
    best_model.load_state_dict(ck["state_dict"])
    lo, yv = get_logits(best_model, va_loader, device)
    n_val = len(yv)

    sweep, alpha = {}, a.alpha
    for al in [0.0, 0.25, 0.5, 0.75, 1.0]:                      # imbalance: tune ONE scalar on val
        sweep[str(al)] = compute_metrics(adjust_logits(lo, log_n, al, a.sampler_power).argmax(1), yv, n_cls)[0]
        print(f"  alpha={al:.2f}: acc {sweep[str(al)]['acc']:.4f}  macro-F1 {sweep[str(al)]['macro_f1']:.4f}"
              f"  macro-recall {sweep[str(al)]['macro_recall']:.4f}")
    if not a.full and n_val:
        alpha = max((float(k) for k in sweep), key=lambda k: score(sweep[str(k)]))
    ck["alpha"] = alpha
    torch.save(ck, out / "model.pt")
    json.dump(sweep, open(out / "alpha_sweep.json", "w"), indent=1)

    adj = adjust_logits(lo, log_n, alpha, a.sampler_power)
    pred = adj.argmax(1).numpy()
    m, cm, recall, support = compute_metrics(pred, yv.numpy(), n_cls)
    print(f"\nBEST epoch {ck['epoch']} | alpha={alpha} | val acc {m['acc']:.4f} | macro-F1 {m['macro_f1']:.4f} "
          f"| macro-recall {m['macro_recall']:.4f}")

    with open(out / "per_class.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f); w.writerow(["folder", "char", "group", "train_count", "val_support", "val_recall"])
        for i in range(n_cls):
            w.writerow([folders[i], chars[i], GROUP_NAMES[class_groups[i]], int(counts[i]), int(support[i]),
                        round(float(recall[i]), 4)])
    pairs = top_confusions(cm, chars, 25)
    with open(out / "top_confusions.txt", "w", encoding="utf-8") as f:
        for c, t, p_ in pairs:
            f.write(f"true {t} -> predicted {p_}: {c}\n")
    save_confusion(cm, chars, folders, out / "confusion_matrix.png")

    err_dir = out / "errors"; err_dir.mkdir(exist_ok=True)      # inspect for label noise / look-alikes
    wrong = [i for i in range(n_val) if pred[i] != int(yv[i])]
    for k, i in enumerate(random.Random(0).sample(wrong, min(150, len(wrong)))):
        path = val_s[i][0]
        try:
            load_image(path).save(err_dir / f"true{folders[int(yv[i])]}_pred{folders[pred[i]]}_{k}.png")
        except Exception:
            pass

    rob, sub = {}, random.Random(0).sample(val_s, min(3000, len(val_s)))
    for name, fn in PERTURBATIONS.items():                      # proxy for unseen conditions
        dl = DataLoader(CharDataset(sub, PerturbedTransform(fn, a.img, auto_invert)), a.bs * 2,
                        num_workers=a.workers)
        l2, y2 = get_logits(best_model, dl, device)
        rob[name] = round(compute_metrics(adjust_logits(l2, log_n, alpha, a.sampler_power).argmax(1), y2, n_cls)[0]["acc"], 4)
        print(f"  robustness {name:12s} acc {rob[name]:.4f}")
    json.dump({"val": m, "alpha": alpha, "best_epoch": ck["epoch"], "robustness": rob,
               "train_size": len(train_s), "val_size": n_val}, open(out / "final_metrics.json", "w"), indent=1)
    print(f"\nSaved everything to {out}/")


if __name__ == "__main__":
    main()
