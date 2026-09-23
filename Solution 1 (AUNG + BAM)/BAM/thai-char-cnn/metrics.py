"""metrics.py - accuracy / macro-F1 / confusion + post-hoc prior adjustment."""
import numpy as np
import torch


def adjust_logits(logits, log_n, alpha, power):
    """Post-hoc class-prior correction for imbalance (logit adjustment).

    The model is trained with a sampler whose effective prior is  n_c^(1-power).
    Adding (alpha - (1-power)) * log(n_c) re-targets the prediction prior to  n_c^alpha:
        alpha = 1  -> natural distribution (best plain accuracy if test is distributed like train)
        alpha = 0  -> perfectly balanced   (best for rare classes / macro-F1)
    """
    bias = (alpha - (1.0 - power)) * torch.as_tensor(log_n, dtype=logits.dtype)
    return logits + bias


def compute_metrics(pred, y, n_cls):
    pred, y = np.asarray(pred), np.asarray(y)
    cm = np.zeros((n_cls, n_cls), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    support, predicted = cm.sum(1), cm.sum(0)
    recall = np.divide(tp, support, out=np.zeros(n_cls), where=support > 0)
    prec = np.divide(tp, predicted, out=np.zeros(n_cls), where=predicted > 0)
    f1 = np.divide(2 * prec * recall, prec + recall, out=np.zeros(n_cls), where=(prec + recall) > 0)
    seen = support > 0
    return (dict(acc=float((pred == y).mean()), macro_f1=float(f1[seen].mean()),
                 macro_recall=float(recall[seen].mean())), cm, recall, support)


def top_confusions(cm, chars, k=25):
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(int(off[i, j]), chars[i], chars[j]) for i in range(len(chars)) for j in range(len(chars)) if off[i, j]]
    return sorted(pairs, reverse=True)[:k]
