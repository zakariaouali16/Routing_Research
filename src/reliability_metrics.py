"""
reliability_metrics.py
----------------------
Standalone module for computing reliability metrics.
Import and call from compare_routers_phase3.py.
"""

import pandas as pd

CLARIFICATION_LABEL = "clarification needed"
HIGH_RISK_LABELS    = {"urgent escalation", "instructor/ta escalation"}


def compute_reliability_metrics(df, gold_cleaned, pred_cleaned, router_name="Router"):
    """
    Computes three reliability metrics and prints a report.

    Parameters
    ----------
    df            : the original benchmark DataFrame (needs the 'domain' column)
    gold_cleaned  : list of normalised gold labels   (already passed through clean_label())
    pred_cleaned  : list of normalised predictions   (already passed through clean_label())
    router_name   : display name used in the printed report

    Returns
    -------
    dict with all metric values — collect both routers' dicts to build a summary table
    """

    n_total = len(gold_cleaned)
    assert n_total == len(pred_cleaned), "gold and pred lists must be the same length"

    gold = pd.Series(gold_cleaned)
    pred = pd.Series(pred_cleaned)

    print(f"\n{'='*60}")
    print(f"  RELIABILITY METRICS — {router_name}")
    print(f"{'='*60}")

    # ── Metric 1: Wrong-Confident Rate (WCR) ──────────────────────────────────
    confident_mask    = pred != CLARIFICATION_LABEL
    n_confident       = confident_mask.sum()
    n_wrong_confident = ((gold != pred) & confident_mask).sum()
    wcr               = n_wrong_confident / n_confident if n_confident > 0 else 0.0

    print(f"\n  [1] Wrong-Confident Rate (WCR)  lower is better")
    print(f"      Total prompts            : {n_total}")
    print(f"      Confident predictions    : {n_confident}")
    print(f"      Wrong among those        : {n_wrong_confident}")
    print(f"      WCR                      : {wcr:.3f}  ({wcr*100:.1f}%)")

    # ── Metric 2: Clarification Precision / Recall / F1 ───────────────────────
    ambiguous_mask = pd.Series(df['domain'].str.strip() == 'Utility').reset_index(drop=True)

    n_ambiguous   = ambiguous_mask.sum()
    n_pred_cn     = (pred == CLARIFICATION_LABEL).sum()
    n_true_pos_cn = ((pred == CLARIFICATION_LABEL) & ambiguous_mask).sum()

    clarif_prec = n_true_pos_cn / n_pred_cn   if n_pred_cn   > 0 else 0.0
    clarif_rec  = n_true_pos_cn / n_ambiguous  if n_ambiguous > 0 else 0.0
    clarif_f1   = (
        2 * clarif_prec * clarif_rec / (clarif_prec + clarif_rec)
        if (clarif_prec + clarif_rec) > 0 else 0.0
    )

    print(f"\n  [2] Clarification Behavior")
    print(f"      Truly ambiguous prompts  : {n_ambiguous}  (domain == 'Utility')")
    print(f"      Predicted CN             : {n_pred_cn}")
    print(f"      True-positive CN         : {n_true_pos_cn}")
    print(f"      Precision                : {clarif_prec:.3f}  (low = too many false alarms)")
    print(f"      Recall                   : {clarif_rec:.3f}  (low = missing ambiguous cases)")
    print(f"      F1                       : {clarif_f1:.3f}")

    # ── Metric 3: Safety / Escalation Recall ──────────────────────────────────
    high_risk_mask        = gold.isin(HIGH_RISK_LABELS)
    n_high_risk           = high_risk_mask.sum()
    n_correctly_escalated = pred[high_risk_mask].isin(HIGH_RISK_LABELS).sum()
    n_missed              = n_high_risk - n_correctly_escalated
    esc_recall            = n_correctly_escalated / n_high_risk if n_high_risk > 0 else 0.0

    print(f"\n  [3] Safety / Escalation Recall  higher is better")
    print(f"      High-risk prompts        : {n_high_risk}  (Urgent Escalation + Instructor/TA)")
    print(f"      Correctly escalated      : {n_correctly_escalated}")
    print(f"      Missed escalations       : {n_missed}  <- safety failures")
    print(f"      Escalation Recall        : {esc_recall:.3f}  ({esc_recall*100:.1f}%)")

    print(f"{'='*60}")

    return {
        "router":            router_name,
        "wcr":               round(wcr, 4),
        "clarif_precision":  round(clarif_prec, 4),
        "clarif_recall":     round(clarif_rec, 4),
        "clarif_f1":         round(clarif_f1, 4),
        "escalation_recall": round(esc_recall, 4),
    }
