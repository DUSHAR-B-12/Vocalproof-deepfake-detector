"""
VoiceShield v2 — Evaluation Metrics
AUC, EER, confusion matrix, and a full classification report.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, accuracy_score,
)


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray):
    """
    Compute Equal Error Rate.
    Returns (eer, threshold_at_eer).
    """
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        print("  ⚠ EER undefined: only one class in y_true")
        return float("nan"), float("nan")

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    abs_diff = np.abs(fpr - fnr)
    if np.all(np.isnan(abs_diff)):
        return float("nan"), float("nan")
    idx = np.nanargmin(abs_diff)
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def evaluate(y_true: np.ndarray, y_prob: np.ndarray,
             threshold: float = 0.5, print_report: bool = True):
    """
    Full evaluation: AUC, EER, confusion matrix, accuracy.

    Parameters
    ----------
    y_true : 1-D array of {0, 1} labels.
    y_prob : 1-D array of predicted probabilities.
    threshold : decision threshold for class predictions.
    print_report : whether to print results to stdout.

    Returns
    -------
    dict with keys: auc, eer, eer_threshold, accuracy, cm, report_str.
    """
    n_classes = len(np.unique(y_true))
    if n_classes < 2:
        print("  ⚠ AUC undefined: only one class in y_true")
        auc = float("nan")
    else:
        auc = roc_auc_score(y_true, y_prob)
    eer, eer_thresh = compute_eer(y_true, y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report_str = classification_report(
        y_true, y_pred, target_names=["REAL", "FAKE"], digits=4,
        zero_division=0,
    )

    results = {
        "auc": auc,
        "eer": eer,
        "eer_threshold": eer_thresh,
        "accuracy": acc,
        "cm": cm,
        "report_str": report_str,
    }

    if print_report:
        print(f"\n{'='*60}")
        print(f"  AUC      : {auc:.4f}")
        print(f"  EER      : {eer:.4f}  (threshold={eer_thresh:.4f})")
        print(f"  Accuracy : {acc:.4f}  (threshold={threshold:.2f})")
        print(f"{'='*60}")
        print("\nConfusion Matrix (rows=true, cols=pred):")
        print(f"           REAL    FAKE")
        print(f"  REAL   {cm[0,0]:>6d}  {cm[0,1]:>6d}")
        print(f"  FAKE   {cm[1,0]:>6d}  {cm[1,1]:>6d}")
        print(f"\n{report_str}")

    return results
