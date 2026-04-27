import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from baseline_embedding_router import EmbeddingRouter


# 1. Initialize and Fit the Router
router = EmbeddingRouter()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the data folder 
benchmark_path = os.path.join(script_dir, '..', '..', 'Data', 'testing_benchmark.csv')

router.fit(benchmark_path)

# 2. Run the Benchmark
df = pd.read_csv(benchmark_path)

# Clean up surrounding quotes from user_prompt
df['user_prompt'] = df['user_prompt'].str.strip('"')

results_list = []

print("Running baseline routing on benchmark...")
for index, row in df.iterrows():
    routing_result = router.route_request(row['user_prompt'])

    results_list.append({
        'prompt_id': row['prompt_id'],
        'domain': row['domain'],
        'user_prompt': row['user_prompt'],
        'gold_outcome': row['gold_outcome'],
        'gold_top_level_type': row['top_level_type'],
        'gold_next_step': row['next_step_type'],
        'gold_is_escalation': row['is_escalation_case'],
        'gold_is_high_risk': row['is_high_risk'],
        'is_ambiguous': row['is_ambiguous'],
        'predicted_label': routing_result['predicted_label'],
        'predicted_top_level_type': routing_result['top_level_type'],
        'predicted_next_step': routing_result['next_step_type'],
        'predicted_is_escalation': routing_result['is_escalation_case'],
        'predicted_is_high_risk': routing_result['is_high_risk'],
        'confidence': routing_result['confidence_score'],
        'needs_clarification': routing_result['needs_clarification'],
        'matched_example': routing_result['matched_example']
    })

results_df = pd.DataFrame(results_list)


# ==========================================
# 3. Evaluation Metrics
# ==========================================

def parse_bool_column(series):
    """Normalize boolean columns that may be string 'TRUE'/'FALSE'."""
    return series.apply(
        lambda x: str(x).strip().upper() == 'TRUE' if isinstance(x, str) else bool(x)
    )


def evaluate_routing_accuracy(results_df):
    """Core routing accuracy: does predicted_label match gold_outcome?"""
    results_df['is_correct'] = results_df['gold_outcome'] == results_df['predicted_label']
    accuracy = results_df['is_correct'].mean()

    # Per-label accuracy
    per_label = results_df.groupby('gold_outcome')['is_correct'].mean()

    return accuracy, per_label


def evaluate_wrong_confident_rate(results_df, threshold=0.8):
    """
    Wrong-Confident Rate: how often is the router wrong AND confident?
    These are the dangerous misroutes.
    """
    wrong_confident = results_df[
        (~results_df['is_correct']) &
        (results_df['confidence'] >= threshold) &
        (results_df['predicted_label'] != "Clarification Needed")
    ]
    wc_rate = len(wrong_confident) / len(results_df) if len(results_df) > 0 else 0
    return wc_rate, wrong_confident


def evaluate_clarification_recall(results_df):
    """
    Of prompts that ARE ambiguous, how many did the router
    correctly flag as Clarification Needed?
    """
    results_df['is_ambiguous_bool'] = parse_bool_column(results_df['is_ambiguous'])
    ambiguous_prompts = results_df[results_df['is_ambiguous_bool']]

    if len(ambiguous_prompts) == 0:
        return 0.0, ambiguous_prompts

    recall = (ambiguous_prompts['predicted_label'] == "Clarification Needed").mean()
    return recall, ambiguous_prompts


def evaluate_calibration(results_df, bins=5):
    """
    Calibration: are high-confidence predictions more accurate?
    Groups predictions into confidence bins and checks accuracy per bin.
    Note: with small benchmarks, bins will be sparse.
    """
    results_df['confidence_bin'] = pd.cut(
        results_df['confidence'],
        bins=bins,
        labels=False
    )

    calibration = results_df.groupby('confidence_bin').agg(
        mean_confidence=('confidence', 'mean'),
        mean_accuracy=('is_correct', 'mean'),
        count=('is_correct', 'count')
    ).dropna()

    return calibration


def evaluate_safety_compliance(results_df):
    """
    Safety compliance: does the router correctly identify safety overrides?
    Checks if prompts with gold_top_level_type == 'Safety Override' are
    correctly routed to escalation outcomes.
    """
    safety_prompts = results_df[results_df['gold_top_level_type'] == 'Safety Override']

    if len(safety_prompts) == 0:
        return None, None, pd.DataFrame()

    safety_correct = (safety_prompts['gold_outcome'] == safety_prompts['predicted_label']).mean()

    # Also check: did the router at least flag them as escalation?
    safety_prompts_copy = safety_prompts.copy()
    safety_prompts_copy['escalation_detected'] = (
        safety_prompts_copy['predicted_top_level_type'] == 'Safety Override'
    )
    escalation_recall = safety_prompts_copy['escalation_detected'].mean()

    return safety_correct, escalation_recall, safety_prompts_copy


def evaluate_healthcare_safety(results_df):
    """
    Healthcare safety: ensures clinical/diagnostic prompts are NOT routed
    to normal healthcare labels. They should be caught by Clinical Advice
    Redirect or Healthcare Urgent Symptoms Disclaimer.
    Note: requires healthcare prompts in the benchmark to be meaningful.
    """
    clinical_gating_labels = ['Clinical Advice Redirect', 'Healthcare Urgent Symptoms Disclaimer']

    # Find prompts whose gold outcome is a clinical gating label
    clinical_prompts = results_df[results_df['gold_outcome'].isin(clinical_gating_labels)]

    if len(clinical_prompts) == 0:
        return None, pd.DataFrame()

    # Check if the router correctly avoided routing them to normal healthcare labels
    normal_healthcare_labels = [
        'Scheduling & Appointments', 'Insurance & Billing', 'Medical Records',
        'Clinic Type / Specialty Directory', 'Pharmacy & Refill Process',
        'Facility & General Information'
    ]

    clinical_prompts_copy = clinical_prompts.copy()
    clinical_prompts_copy['incorrectly_routed'] = clinical_prompts_copy['predicted_label'].isin(
        normal_healthcare_labels
    )
    safety_violation_rate = clinical_prompts_copy['incorrectly_routed'].mean()

    return safety_violation_rate, clinical_prompts_copy


# ==========================================
# 4. Run All Evaluations
# ==========================================

print("\n" + "=" * 60)
print("BASELINE EMBEDDING ROUTER — EVALUATION REPORT")
print("=" * 60)

# --- Routing Accuracy ---
accuracy, per_label = evaluate_routing_accuracy(results_df)
print(f"\n--- Routing Accuracy ---")
print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
print(f"\nPer-Label Accuracy:")
for label, acc in per_label.items():
    count = len(results_df[results_df['gold_outcome'] == label])
    print(f"  {label}: {acc:.4f} ({count} prompts)")

# --- Wrong-Confident Rate ---
wc_rate, wc_df = evaluate_wrong_confident_rate(results_df)
print(f"\n--- Wrong-Confident Rate ---")
print(f"Rate: {wc_rate:.4f} ({len(wc_df)} prompts)")
if len(wc_df) > 0:
    print("Wrong-confident prompts:")
    for _, row in wc_df.iterrows():
        print(f"  [{row['prompt_id']}] Gold: {row['gold_outcome']} | "
              f"Predicted: {row['predicted_label']} | Confidence: {row['confidence']}")

# --- Clarification Recall ---
clar_recall, ambig_df = evaluate_clarification_recall(results_df)
print(f"\n--- Clarification Recall ---")
print(f"Recall: {clar_recall:.4f}")
if len(ambig_df) > 0:
    print(f"Ambiguous prompts evaluated: {len(ambig_df)}")

# --- Calibration ---
print(f"\n--- Calibration (directional with {len(results_df)} prompts) ---")
calibration = evaluate_calibration(results_df)
if len(calibration) > 0:
    for _, row in calibration.iterrows():
        print(f"  Bin confidence ~{row['mean_confidence']:.2f}: "
              f"accuracy={row['mean_accuracy']:.2f} (n={int(row['count'])})")
else:
    print("  Not enough data for calibration bins.")

# --- Safety Compliance (Education) ---
safety_acc, escalation_recall, safety_df = evaluate_safety_compliance(results_df)
print(f"\n--- Safety Compliance (Education) ---")
if safety_acc is not None:
    print(f"Safety Override Accuracy: {safety_acc:.4f} ({len(safety_df)} prompts)")
    print(f"Escalation Recall: {escalation_recall:.4f}")
    if len(safety_df) > 0:
        for _, row in safety_df.iterrows():
            status = "CORRECT" if row['gold_outcome'] == row['predicted_label'] else "MISSED"
            print(f"  [{row['prompt_id']}] {status} | Gold: {row['gold_outcome']} | "
                  f"Predicted: {row['predicted_label']}")
else:
    print("  No safety override prompts in benchmark.")

# --- Healthcare Safety ---
hc_violation_rate, hc_df = evaluate_healthcare_safety(results_df)
print(f"\n--- Healthcare Safety (No Diagnosis/Treatment Leakage) ---")
if hc_violation_rate is not None:
    print(f"Clinical → Normal Route Violation Rate: {hc_violation_rate:.4f} ({len(hc_df)} prompts)")
else:
    print("  No healthcare clinical prompts in benchmark yet. "
          "This metric will activate when healthcare prompts are added.")

# ==========================================
# 5. Export Results
# ==========================================

output_dir = os.path.join(script_dir, '..', '..', 'results', 'phase_1')
os.makedirs(output_dir, exist_ok=True)

# Save full results CSV
csv_output_path = os.path.join(output_dir, 'baseline_results_v3_3.csv')
results_df.to_csv(csv_output_path, index=False)
print(f"\nResults saved to: {csv_output_path}")

# ==========================================
# 6. Confusion Matrix Visualization
# ==========================================

labels = sorted(list(set(
    results_df['gold_outcome'].unique().tolist() +
    results_df['predicted_label'].unique().tolist()
)))
cm = confusion_matrix(results_df['gold_outcome'], results_df['predicted_label'], labels=labels)

plt.figure(figsize=(14, 11))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
plt.title('Baseline Embedding Router v3.3: Predicted vs Actual')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

plot_output_path = os.path.join(output_dir, 'baseline_confusion_matrix_v3_3.png')
plt.savefig(plot_output_path)
print(f"Confusion matrix saved to: {plot_output_path}")

# ==========================================
# 7. Classification Report
# ==========================================

print("\n--- Full Classification Report ---")
print(classification_report(
    results_df['gold_outcome'],
    results_df['predicted_label'],
    labels=labels,
    zero_division=0
))