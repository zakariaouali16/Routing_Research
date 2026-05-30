import os
import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from baselines.baseline_embedding_router import EmbeddingRouter
from llm_router_phase5_1 import LLMRouterV2
from reliability_metrics import compute_reliability_metrics
from tee_logger import Tee


def clean_label(label):
    """Standardizes labels for fair comparison."""
    return str(label).strip().lower().replace("_", " ")


# ── Domain label groupings ─────────────────────────────────────────────────────
EDUCATION_LABELS = {
    "assignment & grading policy",
    "concept explanation",
    "course logistics & environment setup",
    "debugging & code troubleshooting",
    "exam & assessment prep",
    "instructor/ta escalation",
}

HEALTHCARE_LABELS = {
    "facility & general information",
    "human staff review needed",
    "insurance & billing",
    "pharmacy & prescription logistics",
    "scheduling & appointments",
}

GATING_LABELS   = {"clarification needed", "clinical advice refusal", "urgent escalation"}
GATING_CN       = {"clarification needed"}
GATING_CLINICAL = {"clinical advice refusal"}
GATING_URGENT   = {"urgent escalation"}


def domain_accuracy(gold_cleaned, pred_cleaned, label_set, domain_name):
    """Print and return accuracy for a subset of labels."""
    indices = [i for i, g in enumerate(gold_cleaned) if g in label_set]
    if not indices:
        print(f"  {domain_name}: no prompts found")
        return 0.0
    gold_sub = [gold_cleaned[i] for i in indices]
    pred_sub = [pred_cleaned[i] for i in indices]
    acc = accuracy_score(gold_sub, pred_sub)
    n_correct = sum(g == p for g, p in zip(gold_sub, pred_sub))
    print(f"  {domain_name:<44} {n_correct}/{len(gold_sub)} = {acc*100:.1f}%")
    return acc


def run_comparison():
    # ── 1. Setup paths ─────────────────────────────────────────────────────────
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(CURRENT_DIR)

    DATA_PATH     = os.path.join(BASE_DIR, "Data", "pilot_benchmark_phase5.csv")
    TAXONOMY_PATH = os.path.join(BASE_DIR, "Data", "taxonomy_phase5.json")

    RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase5")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    taxonomy_name = os.path.splitext(os.path.basename(TAXONOMY_PATH))[0]
    router_name   = LLMRouterV2.__name__
    output_file   = os.path.join(
        RESULTS_DIR,
        f"benchmark_{router_name}_{taxonomy_name}_{timestamp}.csv"
    )

    # ── Start logging ──────────────────────────────────────────────────────────
    log_file = output_file.replace('.csv', '_run_log.txt')
    tee = Tee(log_file)
    sys.stdout = tee
    print(f"Run log: {log_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── 2. Load benchmark ──────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)

    # ── 3. Initialise routers ──────────────────────────────────────────────────
    print("\n--- Initializing Routers ---")
    baseline_router = EmbeddingRouter()

    TRAINING_PATH = os.path.join(BASE_DIR, "Data", "training_data.csv")
    baseline_router.fit(TRAINING_PATH)

    llm_router = LLMRouterV2(taxonomy_path=TAXONOMY_PATH)

    # ── 4. Run predictions ─────────────────────────────────────────────────────
    baseline_preds = []
    llm_preds      = []
    gold_outcomes  = df['gold_outcome'].tolist()

    print(f"\n--- Running benchmark on {len(df)} rows ---")
    for _, row in tqdm(df.iterrows(), total=df.shape[0]):
        prompt = row['user_prompt']

        try:
            res   = baseline_router.route_request(prompt)
            b_val = res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res)
            baseline_preds.append(b_val)
        except Exception as e:
            baseline_preds.append(f"Err: {str(e)}")

        try:
            res   = llm_router.route_request(prompt)
            l_val = res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res)
            llm_preds.append(l_val)
        except Exception as e:
            llm_preds.append(f"Err: {str(e)}")

    # ── 5. Clean labels ────────────────────────────────────────────────────────
    gold_cleaned = [clean_label(l) for l in gold_outcomes]
    base_cleaned = [clean_label(l) for l in baseline_preds]
    llm_cleaned  = [clean_label(l) for l in llm_preds]

    # ── 6. Save full results CSV ───────────────────────────────────────────────
    df['baseline_prediction'] = baseline_preds
    df['llm_prediction']      = llm_preds
    df['baseline_match']      = [g == b for g, b in zip(gold_cleaned, base_cleaned)]
    df['llm_match']           = [g == l for g, l in zip(gold_cleaned, llm_cleaned)]
    df.to_csv(output_file, index=False)

    # ── 7. Overall routing accuracy ────────────────────────────────────────────
    print("\n" + "=" * 50)
    print(f"RESULTS SAVED TO: {output_file}")
    print("=" * 50)

    baseline_acc = accuracy_score(gold_cleaned, base_cleaned)
    llm_acc      = accuracy_score(gold_cleaned, llm_cleaned)

    print(f"\nBaseline (Embedding) Accuracy : {baseline_acc * 100:.2f}%")
    print(f"LLM (Llama3) Accuracy         : {llm_acc * 100:.2f}%")

    unique_gold_labels = sorted(list(set(gold_cleaned)))

    print("\n--- DETAILED ACCURACY RATING (Baseline Router) ---")
    print(classification_report(gold_cleaned, base_cleaned,
                                 labels=unique_gold_labels, zero_division=0))

    print("\n--- DETAILED ACCURACY RATING (LLM Router) ---")
    print(classification_report(gold_cleaned, llm_cleaned,
                                 labels=unique_gold_labels, zero_division=0))

    # ── 7b. Per-domain accuracy breakdown ─────────────────────────────────────
    print("\n--- PER-DOMAIN ACCURACY (Baseline Router) ---")
    base_edu_acc      = domain_accuracy(gold_cleaned, base_cleaned, EDUCATION_LABELS,  "Education")
    base_hc_acc       = domain_accuracy(gold_cleaned, base_cleaned, HEALTHCARE_LABELS, "Healthcare")
    base_gate_acc     = domain_accuracy(gold_cleaned, base_cleaned, GATING_LABELS,     "Gating (all)")
    base_gate_cn_acc  = domain_accuracy(gold_cleaned, base_cleaned, GATING_CN,         "  Clarification Needed")
    base_gate_cli_acc = domain_accuracy(gold_cleaned, base_cleaned, GATING_CLINICAL,   "  Clinical Advice Refusal")
    base_gate_urg_acc = domain_accuracy(gold_cleaned, base_cleaned, GATING_URGENT,     "  Urgent Escalation")

    print("\n--- PER-DOMAIN ACCURACY (LLM Router) ---")
    llm_edu_acc       = domain_accuracy(gold_cleaned, llm_cleaned, EDUCATION_LABELS,   "Education")
    llm_hc_acc        = domain_accuracy(gold_cleaned, llm_cleaned, HEALTHCARE_LABELS,  "Healthcare")
    llm_gate_acc      = domain_accuracy(gold_cleaned, llm_cleaned, GATING_LABELS,      "Gating (all)")
    llm_gate_cn_acc   = domain_accuracy(gold_cleaned, llm_cleaned, GATING_CN,          "  Clarification Needed")
    llm_gate_cli_acc  = domain_accuracy(gold_cleaned, llm_cleaned, GATING_CLINICAL,    "  Clinical Advice Refusal")
    llm_gate_urg_acc  = domain_accuracy(gold_cleaned, llm_cleaned, GATING_URGENT,      "  Urgent Escalation")

    print("\n--- DOMAIN ACCURACY SUMMARY (for charts) ---")
    print(f"{'Domain':<46} {'Baseline':>10} {'LLM':>10}")
    print("-" * 68)
    print(f"{'Overall':<46} {baseline_acc*100:>9.1f}% {llm_acc*100:>9.1f}%")
    print(f"{'Education':<46} {base_edu_acc*100:>9.1f}% {llm_edu_acc*100:>9.1f}%")
    print(f"{'Healthcare':<46} {base_hc_acc*100:>9.1f}% {llm_hc_acc*100:>9.1f}%")
    print(f"{'Gating (all)':<46} {base_gate_acc*100:>9.1f}% {llm_gate_acc*100:>9.1f}%")
    print(f"{'  Clarification Needed':<46} {base_gate_cn_acc*100:>9.1f}% {llm_gate_cn_acc*100:>9.1f}%")
    print(f"{'  Clinical Advice Refusal':<46} {base_gate_cli_acc*100:>9.1f}% {llm_gate_cli_acc*100:>9.1f}%")
    print(f"{'  Urgent Escalation':<46} {base_gate_urg_acc*100:>9.1f}% {llm_gate_urg_acc*100:>9.1f}%")

    # ── 8. Reliability metrics ─────────────────────────────────────────────────
    baseline_rel = compute_reliability_metrics(df, gold_cleaned, base_cleaned, "Baseline (Embedding)")
    llm_rel      = compute_reliability_metrics(df, gold_cleaned, llm_cleaned,  "LLM (Llama3)")

    # ── 9. Reliability summary table ──────────────────────────────────────────
    summary = pd.DataFrame([baseline_rel, llm_rel])
    print("\n--- RELIABILITY SUMMARY TABLE ---")
    print(summary.to_string(index=False))

    summary_file = output_file.replace('.csv', '_reliability_summary.csv')
    summary.to_csv(summary_file, index=False)
    print(f"\nReliability summary saved to: {summary_file}")

    # ── Close logger ───────────────────────────────────────────────────────────
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run log saved to: {log_file}")
    tee.close()


if __name__ == "__main__":
    run_comparison()