"""
Evaluation Script
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

from baseline_embedding_router import EmbeddingRouter


# ==========================================
# Taxonomy
# ==========================================
LABEL_TO_GROUP = {
    # Safety override
    "Education Sensitive Escalation": "safety_override",
    "Healthcare Urgent Symptoms Disclaimer": "safety_override",

    # Gating outcomes
    "Clarification Needed": "gating_outcome",
    "Mixed Intent": "gating_outcome",
    "Clinical Advice Redirect": "gating_outcome",

    # Last resort
    "Out-of-Scope": "last_resort",

    # Education routing labels
    "Concept Explanation": "routing_label",
    "Debugging & Code Troubleshooting": "routing_label",
    "Assignment & Grading Policy": "routing_label",
    "Exam Logistics & Preparation": "routing_label",
    "Course Logistics & Resources": "routing_label",
    "Office Hours / Instructor Access": "routing_label",
    "Course Exceptions & Disputes": "routing_label",

    # Healthcare routing labels
    "Scheduling & Appointments": "routing_label",
    "Insurance & Billing": "routing_label",
    "Medical Records": "routing_label",
    "Clinic Type / Specialty Directory": "routing_label",
    "Pharmacy & Refill Process": "routing_label",
    "Facility & General Information": "routing_label",
    "Human Staff Review Needed": "routing_label",
}


FOLLOW_UP_LABELS = {"Clarification Needed", "Mixed Intent"}


def get_group(label):
    return LABEL_TO_GROUP.get(label, "unknown")


# ==========================================
# Benchmark evaluation 
# ==========================================
def run_full_benchmark(df, threshold=0.5):
    """
    Fit on the full benchmark and evaluate each prompt against the same benchmark.
    This is a simple retrieval-style baseline evaluation.
    """
    results = []

    print(f"Running full-benchmark evaluation on {len(df)} prompts...")

    router = EmbeddingRouter()
    router.fit(df=df)

    for i, row in df.iterrows():
        prompt_text = str(row["user_prompt"]).strip('"')
        routing_result = router.route_request(prompt_text, threshold=threshold)

        results.append(
            {
                "prompt_id": row["prompt_id"],
                "domain": row.get("domain", ""),
                "user_prompt": prompt_text,
                "gold_outcome": row["gold_outcome"],
                "gold_group": get_group(row["gold_outcome"]),
                "predicted_label": routing_result["predicted_label"],
                "predicted_group": get_group(routing_result["predicted_label"]),
                "confidence": routing_result["confidence_score"],
                "needs_clarification": routing_result["needs_clarification"],
                "matched_example": routing_result["matched_example"],
            }
        )

        if (i + 1) % 20 == 0:
            print(f"  ... {i + 1}/{len(df)} done")

    return pd.DataFrame(results)


# ==========================================
# Metric functions
# ==========================================
def compute_overall_accuracy(results_df):
    results_df["is_correct"] = (
        results_df["gold_outcome"] == results_df["predicted_label"]
    )
    return results_df["is_correct"].mean()


def compute_wrong_confident_rate(results_df, threshold=0.8):
    wrong_confident = results_df[
        (~results_df["is_correct"])
        & (results_df["confidence"] >= threshold)
        & (results_df["predicted_label"] != "Clarification Needed")
    ]
    rate = len(wrong_confident) / len(results_df) if len(results_df) > 0 else 0.0
    return rate, wrong_confident


def compute_clarification_recall(results_df):
    clar_prompts = results_df[results_df["gold_outcome"] == "Clarification Needed"]
    if len(clar_prompts) == 0:
        return None, clar_prompts
    recall = (clar_prompts["predicted_label"] == "Clarification Needed").mean()
    return recall, clar_prompts


def compute_safety_override_recall(results_df):
    safety_prompts = results_df[results_df["gold_group"] == "safety_override"]
    if len(safety_prompts) == 0:
        return None, safety_prompts
    exact_recall = (
        safety_prompts["predicted_label"] == safety_prompts["gold_outcome"]
    ).mean()
    return exact_recall, safety_prompts


def compute_per_group_accuracy(results_df):
    return results_df.groupby("gold_group")["is_correct"].mean()


def compute_followup_recall(results_df):
    gold_followup = results_df["gold_outcome"].isin(FOLLOW_UP_LABELS)
    if gold_followup.sum() == 0:
        return None, results_df[gold_followup]

    pred_followup = results_df["predicted_label"].isin(FOLLOW_UP_LABELS)
    recall = pred_followup[gold_followup].mean()
    return recall, results_df[gold_followup]


def compute_followup_precision(results_df):
    pred_followup = results_df["predicted_label"].isin(FOLLOW_UP_LABELS)
    if pred_followup.sum() == 0:
        return None, results_df[pred_followup]

    gold_followup = results_df["gold_outcome"].isin(FOLLOW_UP_LABELS)
    precision = gold_followup[pred_followup].mean()
    return precision, results_df[pred_followup]


# ==========================================
# Main
# ==========================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    benchmark_path = os.path.join(
        script_dir, "..", "..", "Data", "testing_benchmark_2.0.csv"
    )

    if not os.path.exists(benchmark_path):
        benchmark_path = os.path.join(script_dir, "testing_benchmark_2.0.csv")

    print(f"Loading benchmark from: {benchmark_path}")
    df = pd.read_csv(benchmark_path)
    print(f"Loaded {len(df)} rows.")

    # Run evaluation on full benchmark
    results_df = run_full_benchmark(df, threshold=0.5)

    # ==========================================
    # Print report
    # ==========================================
    print("\n" + "=" * 60)
    print("BASELINE EMBEDDING ROUTER EVALUATION REPORT")
    print("(full benchmark, no withholding)")
    print("=" * 60)

    accuracy = compute_overall_accuracy(results_df)
    print(f"\nOverall accuracy: {accuracy:.4f} ({accuracy * 100:.1f}%)")

    print("\n--- Per-group accuracy ---")
    group_acc = compute_per_group_accuracy(results_df)
    for group, acc in group_acc.items():
        n = (results_df["gold_group"] == group).sum()
        print(f"  {group:20s}  {acc:.4f}  (n={n})")

    wc_rate, wc_df = compute_wrong_confident_rate(results_df, threshold=0.8)
    print(f"\n--- Wrong-confident rate (confidence >= 0.8) ---")
    print(f"  Rate: {wc_rate:.4f}  ({len(wc_df)} prompts)")
    if len(wc_df) > 0:
        print("  Wrong-confident examples:")
        for _, row in wc_df.head(10).iterrows():
            print(
                f"    [{row['prompt_id']}] gold={row['gold_outcome']} "
                f"predicted={row['predicted_label']} conf={row['confidence']}"
            )
        if len(wc_df) > 10:
            print(f"    ... and {len(wc_df) - 10} more")

    clar_recall, clar_df = compute_clarification_recall(results_df)
    print(f"\n--- Clarification recall ---")
    if clar_recall is not None:
        print(
            f"  Recall: {clar_recall:.4f}  "
            f"({len(clar_df)} prompts with gold='Clarification Needed')"
        )

    safety_recall, safety_df = compute_safety_override_recall(results_df)
    print(f"\n--- Safety-override recall (exact label) ---")
    if safety_recall is not None:
        print(
            f"  Recall: {safety_recall:.4f}  "
            f"({len(safety_df)} safety-override prompts)"
        )

    followup_recall, followup_gold_df = compute_followup_recall(results_df)
    print(f"\n--- Follow-up recall (Clarification Needed + Mixed Intent) ---")
    if followup_recall is not None:
        print(
            f"  Recall: {followup_recall:.4f}  "
            f"({len(followup_gold_df)} prompts in follow-up bucket)"
        )

    followup_precision, followup_pred_df = compute_followup_precision(results_df)
    print(f"\n--- Follow-up precision (Clarification Needed + Mixed Intent) ---")
    if followup_precision is not None:
        print(
            f"  Precision: {followup_precision:.4f}  "
            f"({len(followup_pred_df)} prompts predicted into follow-up bucket)"
        )

    print("\n--- Full classification report ---")
    all_labels = sorted(set(results_df["gold_outcome"]) | set(results_df["predicted_label"]))
    print(
        classification_report(
            results_df["gold_outcome"],
            results_df["predicted_label"],
            labels=all_labels,
            zero_division=0,
        )
    )

    # ==========================================
    # Export artifacts
    # ==========================================
    output_dir = os.path.join(script_dir, "..", "..", "results", "phase_3_test_trials")
    if not os.path.exists(output_dir):
        output_dir = os.path.join(script_dir, "results_phase_3_test_trials")
    os.makedirs(output_dir, exist_ok=True)

    csv_out = os.path.join(output_dir, "baseline_results.csv")
    results_df.to_csv(csv_out, index=False)
    print(f"\nResults CSV saved to: {csv_out}")

    cm = confusion_matrix(
        results_df["gold_outcome"],
        results_df["predicted_label"],
        labels=all_labels
    )
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=all_labels,
        yticklabels=all_labels,
        cmap="Greens",
        cbar=False,
    )
    plt.title("Baseline Embedding Router — Confusion Matrix")
    plt.ylabel("Gold outcome")
    plt.xlabel("Predicted label")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    png_out = os.path.join(output_dir, "baseline_confusion_matrix.png")
    plt.savefig(png_out, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to: {png_out}")