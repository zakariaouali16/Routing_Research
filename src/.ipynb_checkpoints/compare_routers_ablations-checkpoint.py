import os
import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Import your classes
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


def run_single_llm(llm_router, df, gold_cleaned, ablation, label):
    """Run LLM router with a given ablation flag and return cleaned predictions + correction count."""
    preds = []
    n_corrected = 0
    print(f"\n--- Running LLM Router [{label}] ---")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=label):
        try:
            res = llm_router.route_request(row['user_prompt'], ablation=ablation)
            preds.append(res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res))
            if isinstance(res, dict) and res.get('was_corrected', False):
                n_corrected += 1
        except Exception as e:
            preds.append(f"Err: {str(e)}")
    return [clean_label(p) for p in preds], n_corrected


def print_domain_breakdown(gold_cleaned, pred_cleaned, label):
    print(f"\n--- PER-DOMAIN ACCURACY ({label}) ---")
    domain_accuracy(gold_cleaned, pred_cleaned, EDUCATION_LABELS,  "Education")
    domain_accuracy(gold_cleaned, pred_cleaned, HEALTHCARE_LABELS, "Healthcare")
    domain_accuracy(gold_cleaned, pred_cleaned, GATING_LABELS,     "Gating (all)")
    domain_accuracy(gold_cleaned, pred_cleaned, GATING_CN,         "  Clarification Needed")
    domain_accuracy(gold_cleaned, pred_cleaned, GATING_CLINICAL,   "  Clinical Advice Refusal")
    domain_accuracy(gold_cleaned, pred_cleaned, GATING_URGENT,     "  Urgent Escalation")


def run_comparison():
    # ── 1. Setup paths ─────────────────────────────────────────────────────────
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR    = os.path.dirname(CURRENT_DIR)

    DATA_PATH     = os.path.join(BASE_DIR, "Data", "pilot_benchmark_phase5.csv")
    TAXONOMY_PATH = os.path.join(BASE_DIR, "Data", "taxonomy_phase5.json")
    TRAINING_PATH = os.path.join(BASE_DIR, "Data", "training_data.csv")

    RESULTS_DIR = os.path.join(BASE_DIR, "results", "phase5", "ablations")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    taxonomy_name = os.path.splitext(os.path.basename(TAXONOMY_PATH))[0]
    output_file   = os.path.join(RESULTS_DIR, f"benchmark_ablation_{taxonomy_name}_{timestamp}.csv")

    # ── Start logging — all print() output goes to console + log file ──────────
    log_file = output_file.replace('.csv', '_run_log.txt')
    tee = Tee(log_file)
    sys.stdout = tee
    print(f"Run log: {log_file}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # ── 2. Load benchmark ──────────────────────────────────────────────────────
    df = pd.read_csv(DATA_PATH)
    gold_outcomes  = df['gold_outcome'].tolist()
    gold_cleaned   = [clean_label(l) for l in gold_outcomes]

    # ── 3. Initialise routers ──────────────────────────────────────────────────
    print("\n--- Initializing Routers ---")
    baseline_router = EmbeddingRouter()
    baseline_router.fit(TRAINING_PATH)
    llm_router = LLMRouterV2(taxonomy_path=TAXONOMY_PATH)

    # ── 4. Run baseline ────────────────────────────────────────────────────────
    baseline_preds = []
    print(f"\n--- Running Baseline Router ---")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Baseline"):
        try:
            res = baseline_router.route_request(row['user_prompt'])
            baseline_preds.append(res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res))
        except Exception as e:
            baseline_preds.append(f"Err: {str(e)}")
    base_cleaned = [clean_label(p) for p in baseline_preds]

    # ── 5. Run LLM router — 5 variants ────────────────────────────────────────
    # Full system (no ablation)
    llm_full_cleaned,      full_corrections      = run_single_llm(llm_router, df, gold_cleaned, ablation=None,                     label="Full System")
    # Ablation 1: no safety keyword override
    llm_no_safety_cleaned, no_safety_corrections = run_single_llm(llm_router, df, gold_cleaned, ablation="no_safety",              label="No Safety Override")
    # Ablation 2: no academic keyword override
    llm_no_acad_cleaned,   no_acad_corrections   = run_single_llm(llm_router, df, gold_cleaned, ablation="no_academic",            label="No Academic Override")
    # Ablation 3: no clarification gate
    llm_no_gate_cleaned,   no_gate_corrections   = run_single_llm(llm_router, df, gold_cleaned, ablation="no_gate",                label="No Clarification Gate")
    # Ablation 4: no hallucination guard
    llm_no_halluc_cleaned, no_halluc_corrections = run_single_llm(llm_router, df, gold_cleaned, ablation="no_hallucination_guard", label="No Hallucination Guard")

    # ── 6. Save full results CSV ───────────────────────────────────────────────
    df['baseline_prediction']           = baseline_preds
    df['llm_full_prediction']           = llm_full_cleaned
    df['llm_no_safety_prediction']      = llm_no_safety_cleaned
    df['llm_no_academic_prediction']    = llm_no_acad_cleaned
    df['llm_no_gate_prediction']        = llm_no_gate_cleaned
    df['llm_no_hallucguard_prediction'] = llm_no_halluc_cleaned
    df['baseline_match']                = [g == b for g, b in zip(gold_cleaned, base_cleaned)]
    df['llm_full_match']                = [g == l for g, l in zip(gold_cleaned, llm_full_cleaned)]
    df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")

    # ── 7. Overall accuracy summary ────────────────────────────────────────────
    variants = {
        "Baseline (Embedding)":              base_cleaned,
        "LLM — Full System":                 llm_full_cleaned,
        "LLM — No Safety Override":          llm_no_safety_cleaned,
        "LLM — No Academic Override":        llm_no_acad_cleaned,
        "LLM — No Clarification Gate":       llm_no_gate_cleaned,
        "LLM — No Hallucination Guard":      llm_no_halluc_cleaned,
    }

    print("\n" + "=" * 60)
    print("OVERALL ACCURACY SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<40} {'Accuracy':>10}")
    print("-" * 52)
    for name, preds in variants.items():
        acc = accuracy_score(gold_cleaned, preds)
        print(f"{name:<40} {acc*100:>9.2f}%")

    # ── 8. Per-domain breakdown for each variant ───────────────────────────────
    for name, preds in variants.items():
        print_domain_breakdown(gold_cleaned, preds, name)

    # ── 9. Detailed classification report for full system ─────────────────────
    unique_gold_labels = sorted(list(set(gold_cleaned)))

    print("\n--- DETAILED CLASSIFICATION REPORT (Baseline) ---")
    print(classification_report(gold_cleaned, base_cleaned,
                                 labels=unique_gold_labels, zero_division=0))

    print("\n--- DETAILED CLASSIFICATION REPORT (LLM Full System) ---")
    print(classification_report(gold_cleaned, llm_full_cleaned,
                                 labels=unique_gold_labels, zero_division=0))

    # ── 10. Reliability metrics for all variants ───────────────────────────────
    correction_counts = {
        "Baseline (Embedding)":              0,  # embedding baseline has no hallucination guard
        "LLM — Full System":                 full_corrections,
        "LLM — No Safety Override":          no_safety_corrections,
        "LLM — No Academic Override":        no_acad_corrections,
        "LLM — No Clarification Gate":       no_gate_corrections,
        "LLM — No Hallucination Guard":      no_halluc_corrections,
    }

    all_reliability = []
    for name, preds in variants.items():
        rel = compute_reliability_metrics(df, gold_cleaned, preds, name)
        rel["halluc_corrections"] = correction_counts[name]
        all_reliability.append(rel)

    # ── 11. Reliability summary table ─────────────────────────────────────────
    summary = pd.DataFrame(all_reliability)

    print("\n" + "=" * 60)
    print("ABLATION RELIABILITY SUMMARY TABLE")
    print("=" * 60)
    print(summary.to_string(index=False))

    summary_file = output_file.replace('.csv', '_reliability_summary.csv')
    summary.to_csv(summary_file, index=False)
    print(f"\nReliability summary saved to: {summary_file}")

    # ── 12. Print focused ablation impact table ────────────────────────────────
    print("\n" + "=" * 60)
    print("ABLATION IMPACT — KEY METRICS")
    print("(compare each row against LLM Full System)")
    print("=" * 60)

    key_cols = ["router", "wcr", "clarif_f1", "urgent_recall", "instructor_recall", "clinical_recall", "halluc_corrections"]
    ablation_summary = summary[key_cols].copy()
    print(ablation_summary.to_string(index=False))

    ablation_summary_file = output_file.replace('.csv', '_ablation_impact.csv')
    ablation_summary.to_csv(ablation_summary_file, index=False)
    print(f"\nAblation impact table saved to: {ablation_summary_file}")

    # ── Close logger ───────────────────────────────────────────────────────────
    print(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Run log saved to: {log_file}")
    tee.close()


if __name__ == "__main__":
    run_comparison()