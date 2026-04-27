import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from baselines.baseline_embedding_router import EmbeddingRouter
from llm_router_v3 import LLMRouterV3


def clean_label(label):
    return str(label).strip().lower().replace("_", " ")


LABEL_TO_GROUP = {
    "Education Sensitive Escalation": "safety_override",
    "Healthcare Urgent Symptoms Disclaimer": "safety_override",
    "Clarification Needed": "gating_outcome",
    "Mixed Intent": "gating_outcome",
    "Clinical Advice Redirect": "gating_outcome",
    "Out-of-Scope": "last_resort",
    "Concept Explanation": "routing_label",
    "Debugging & Code Troubleshooting": "routing_label",
    "Assignment & Grading Policy": "routing_label",
    "Exam Logistics & Preparation": "routing_label",
    "Course Logistics & Resources": "routing_label",
    "Office Hours / Instructor Access": "routing_label",
    "Course Exceptions & Disputes": "routing_label",
    "Scheduling & Appointments": "routing_label",
    "Insurance & Billing": "routing_label",
    "Medical Records": "routing_label",
    "Clinic Type / Specialty Directory": "routing_label",
    "Pharmacy & Refill Process": "routing_label",
    "Facility & General Information": "routing_label",
    "Human Staff Review Needed": "routing_label",
}


def get_group(label):
    return LABEL_TO_GROUP.get(label, "unknown")


def run_comparison():
    base_dir = "/content/drive/MyDrive/Routing_Research"

    data_path = os.path.join(base_dir, "Data", "testing_benchmark_2.0.csv")
    taxonomy_path = os.path.join(base_dir, "Data", "taxonomy_v4_1.json")

    results_dir = os.path.join(base_dir, "results", "phase_3_test_trials")
    os.makedirs(results_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"compare_routers_v3_{timestamp}.csv")

    print(f"Loading benchmark from: {data_path}")
    df = pd.read_csv(data_path)

    required_cols = {"prompt_id", "user_prompt", "gold_outcome"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required benchmark columns: {missing}")

    print("\n--- Initializing routers ---")
    baseline_router = EmbeddingRouter()
    baseline_router.fit(df=df)

    llm_router = LLMRouterV3(
        model_name="llama3",
        taxonomy_path=taxonomy_path
    )

    baseline_preds = []
    baseline_conf = []
    baseline_needs_clarification = []

    llm_initial_preds = []
    llm_final_preds = []
    llm_conf_levels = []
    llm_needs_clarification = []
    llm_was_corrected = []
    llm_short_reasons = []
    llm_qa_reasons = []

    print(f"\n--- Running comparison on {len(df)} rows ---")
    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc="Comparing", unit="prompt"):
        prompt = str(row["user_prompt"]).strip('"')

        # Baseline prediction
        try:
            res = baseline_router.route_request(prompt)
            baseline_preds.append(res.get("predicted_label", "Error"))
            baseline_conf.append(res.get("confidence_score", None))
            baseline_needs_clarification.append(res.get("needs_clarification", False))
        except Exception as e:
            baseline_preds.append("Error")
            baseline_conf.append(None)
            baseline_needs_clarification.append(True)
            print(f"Baseline error on {row['prompt_id']}: {e}")

        # LLM prediction (prompt-only, no domain passed)
        try:
            res = llm_router.route_request(prompt)
            llm_initial_preds.append(res.get("initial_label", "Error"))
            llm_final_preds.append(res.get("predicted_label", "Error"))
            llm_conf_levels.append(res.get("confidence_level", "Unknown"))
            llm_needs_clarification.append(res.get("needs_clarification", False))
            llm_was_corrected.append(res.get("was_corrected", False))
            llm_short_reasons.append(res.get("short_reason", ""))
            llm_qa_reasons.append(res.get("qa_reason", ""))
        except Exception as e:
            llm_initial_preds.append("Error")
            llm_final_preds.append("Error")
            llm_conf_levels.append("Unknown")
            llm_needs_clarification.append(True)
            llm_was_corrected.append(False)
            llm_short_reasons.append(f"Error: {str(e)}")
            llm_qa_reasons.append("N/A")
            print(f"LLM error on {row['prompt_id']}: {e}")

    results_df = df.copy()

    results_df["gold_group"] = results_df["gold_outcome"].apply(get_group)

    results_df["baseline_prediction"] = baseline_preds
    results_df["baseline_confidence"] = baseline_conf
    results_df["baseline_needs_clarification"] = baseline_needs_clarification
    results_df["baseline_group"] = results_df["baseline_prediction"].apply(get_group)

    results_df["llm_initial_prediction"] = llm_initial_preds
    results_df["llm_prediction"] = llm_final_preds
    results_df["llm_confidence_level"] = llm_conf_levels
    results_df["llm_needs_clarification"] = llm_needs_clarification
    results_df["llm_was_corrected_by_qa"] = llm_was_corrected
    results_df["llm_short_reason"] = llm_short_reasons
    results_df["llm_qa_reason"] = llm_qa_reasons
    results_df["llm_group"] = results_df["llm_prediction"].apply(get_group)

    gold_clean = [clean_label(x) for x in results_df["gold_outcome"]]
    baseline_clean = [clean_label(x) for x in results_df["baseline_prediction"]]
    llm_clean = [clean_label(x) for x in results_df["llm_prediction"]]

    results_df["baseline_match"] = [g == p for g, p in zip(gold_clean, baseline_clean)]
    results_df["llm_match"] = [g == p for g, p in zip(gold_clean, llm_clean)]

    results_df.to_csv(output_file, index=False)

    print("\n" + "=" * 60)
    print(f"RESULTS SAVED TO: {output_file}")
    print("=" * 60)

    baseline_acc = accuracy_score(gold_clean, baseline_clean)
    llm_acc = accuracy_score(gold_clean, llm_clean)

    print(f"\nBaseline (Embedding) Accuracy: {baseline_acc * 100:.2f}%")
    print(f"LLM Router Accuracy:           {llm_acc * 100:.2f}%")

    if "domain" in results_df.columns:
        print("\n--- Accuracy by domain (analysis only) ---")
        for dom in results_df["domain"].fillna("missing").unique():
            dom_df = results_df[results_df["domain"].fillna("missing") == dom]
            b_acc = dom_df["baseline_match"].mean()
            l_acc = dom_df["llm_match"].mean()
            print(f"{dom:12s} baseline={b_acc:.4f}  llm={l_acc:.4f}  (n={len(dom_df)})")

    print("\n--- Accuracy by group ---")
    for group in sorted(results_df["gold_group"].unique()):
        group_df = results_df[results_df["gold_group"] == group]
        b_acc = group_df["baseline_match"].mean()
        l_acc = group_df["llm_match"].mean()
        print(f"{group:20s} baseline={b_acc:.4f}  llm={l_acc:.4f}  (n={len(group_df)})")

    print("\n--- Baseline classification report ---")
    print(
        classification_report(
            results_df["gold_outcome"],
            results_df["baseline_prediction"],
            zero_division=0
        )
    )

    print("\n--- LLM classification report ---")
    print(
        classification_report(
            results_df["gold_outcome"],
            results_df["llm_prediction"],
            zero_division=0
        )
    )

    print("\n--- Clarification recall ---")
    clar_df = results_df[results_df["gold_outcome"] == "Clarification Needed"]
    if len(clar_df) > 0:
        baseline_clar_recall = (
            clar_df["baseline_prediction"] == "Clarification Needed"
        ).mean()
        llm_clar_recall = (
            clar_df["llm_prediction"] == "Clarification Needed"
        ).mean()
        print(f"Baseline: {baseline_clar_recall:.4f}")
        print(f"LLM:      {llm_clar_recall:.4f}")

    print("\n--- Safety override recall ---")
    safety_df = results_df[results_df["gold_group"] == "safety_override"]
    if len(safety_df) > 0:
        baseline_safety_recall = (
            safety_df["baseline_prediction"] == safety_df["gold_outcome"]
        ).mean()
        llm_safety_recall = (
            safety_df["llm_prediction"] == safety_df["gold_outcome"]
        ).mean()
        print(f"Baseline: {baseline_safety_recall:.4f}")
        print(f"LLM:      {llm_safety_recall:.4f}")

    print("\n--- QA correction count ---")
    print(f"LLM QA corrections: {results_df['llm_was_corrected_by_qa'].sum()}/{len(results_df)}")


if __name__ == "__main__":
    run_comparison()