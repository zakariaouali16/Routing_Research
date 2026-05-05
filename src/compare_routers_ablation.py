import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

from baselines.baseline_embedding_router import EmbeddingRouter
from llm_router_ablation import LLMRouterV1
from reliability_metrics import compute_reliability_metrics


def clean_label(label):
    return str(label).strip().lower().replace("_", " ")


def get_predictions(router, df):
    preds = []
    for _, row in tqdm(df.iterrows(), total=df.shape[0], leave=False):
        try:
            res = router.route_request(row['user_prompt'])
            val = res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res)
            preds.append(val)
        except Exception as e:
            preds.append(f"Err: {str(e)}")
    return preds


def run_ablation():
    # ── Paths ──────────────────────────────────────────────────────────────────
    CURRENT_DIR   = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR      = os.path.dirname(CURRENT_DIR)

    DATA_PATH     = os.path.join(BASE_DIR, "Data", "pilot_benchmark_mini_100.csv")  
    TAXONOMY_PATH = os.path.join(BASE_DIR, "Data", "taxonomy_phase3_1.json")
    TRAINING_PATH = os.path.join(BASE_DIR, "Data", "training_data.csv")

    RESULTS_DIR   = os.path.join(BASE_DIR, "results", "phase3")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file   = os.path.join(RESULTS_DIR, f"ablation_gate_slots_{timestamp}.csv")

    # ── Load benchmark ─────────────────────────────────────────────────────────
    print("\nLoading benchmark...")
    df            = pd.read_csv(DATA_PATH)
    gold_labels   = df['gold_label'].tolist()
    gold_cleaned  = [clean_label(l) for l in gold_labels]
    unique_labels = sorted(set(gold_cleaned))

    # ── Fit baseline ───────────────────────────────────────────────────────────
    print("\n--- Initializing Baseline Router ---")
    baseline_router = EmbeddingRouter()
    temp_train_path = TRAINING_PATH.replace('.csv', '_temp_fit.csv')
    df_train = pd.read_csv(TRAINING_PATH)
    df_train.rename(columns={'gold_label': 'gold_outcome'}).to_csv(temp_train_path, index=False)
    baseline_router.fit(temp_train_path)
    if os.path.exists(temp_train_path):
        os.remove(temp_train_path)

    # ── Define configurations ──────────────────────────────────────────────────
    configurations = [
        ("Baseline (Embedding)",          baseline_router),
        ("LLM-v3 full (gate + slots)",    LLMRouterV1(taxonomy_path=TAXONOMY_PATH, use_gate=True,  use_slots=True)),
        ("LLM-v3 no gate (slots only)",   LLMRouterV1(taxonomy_path=TAXONOMY_PATH, use_gate=False, use_slots=True)),
        ("LLM-v3 no slots (gate only)",   LLMRouterV1(taxonomy_path=TAXONOMY_PATH, use_gate=True,  use_slots=False)),
        ("LLM-v3 no gate no slots",       LLMRouterV1(taxonomy_path=TAXONOMY_PATH, use_gate=False, use_slots=False)),
    ]

    # ── Run all configurations ─────────────────────────────────────────────────
    all_results  = []
    all_preds    = {}

    for config_name, router in configurations:
        print(f"\n{'='*55}")
        print(f"  Running: {config_name}")
        print(f"{'='*55}")

        raw_preds    = get_predictions(router, df)
        pred_cleaned = [clean_label(p) for p in raw_preds]
        all_preds[config_name] = pred_cleaned

        acc = accuracy_score(gold_cleaned, pred_cleaned)
        print(f"\n  Accuracy: {acc * 100:.2f}%")
        print(classification_report(gold_cleaned, pred_cleaned,
                                     labels=unique_labels, zero_division=0))

        rel = compute_reliability_metrics(df, gold_cleaned, pred_cleaned, config_name)
        rel['accuracy'] = round(acc, 4)
        all_results.append(rel)

    # ── Save predictions CSV ───────────────────────────────────────────────────
    for config_name, pred_cleaned in all_preds.items():
        col = config_name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
        df[f'pred_{col}'] = pred_cleaned
    df.to_csv(output_file, index=False)

    # ── Print summary table ────────────────────────────────────────────────────
    cols = ['router', 'accuracy', 'wcr', 'clarif_precision',
            'clarif_recall', 'clarif_f1', 'escalation_recall']

    summary = pd.DataFrame(all_results)[cols]

    print("\n" + "="*70)
    print("  ABLATION SUMMARY TABLE")
    print("="*70)
    print(summary.to_string(index=False))

    summary_file = output_file.replace('.csv', '_summary.csv')
    summary.to_csv(summary_file, index=False)

    print(f"\nFull predictions : {output_file}")
    print(f"Summary table    : {summary_file}")


if __name__ == "__main__":
    run_ablation()