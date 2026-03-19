import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from baseline_embedding_router import EmbeddingRouter  # Assumes your file is in the same folder

# 1. Initialize and Fit the Router
router = EmbeddingRouter()
script_dir = os.path.dirname(os.path.abspath(__file__))

# Build the path to the Data folder relative to this script
# (Up one level to 'src', up another to root, then into 'Data')
benchmark_path = os.path.join(script_dir, '..', '..', 'Data', 'v0_pilot_benchmark.csv')

router.fit(benchmark_path)

# 2. Run the Benchmark
df = pd.read_csv(benchmark_path)
results_list = []

print("Running baseline routing on benchmark...")
for index, row in df.iterrows():
    routing_result = router.route_request(row['prompt'])
    
    results_list.append({
        'prompt_id': row['prompt_id'],
        'domain': row['domain'],
        'prompt': row['prompt'],
        'label': row['label'],
        'is_ambiguous': row['is_ambiguous'],
        'predicted_label': routing_result['predicted_label'],
        'confidence': routing_result['confidence_score'],
        'needs_clarification': routing_result['needs_clarification']
    })

results_df = pd.DataFrame(results_list)

# 3. Define Evaluation Function
def evaluate_reliability(results_df, model_name="Baseline_Embedding"):
    # Normalize label names for comparison
    results_df['label'] = results_df['label'].replace('Needs Clarification', 'Clarification Needed')
    results_df['predicted_label'] = results_df['predicted_label'].replace('Needs Clarification', 'Clarification Needed')
    
    # Calculate Correctness
    results_df['is_correct'] = results_df['label'] == results_df['predicted_label']
    accuracy = results_df['is_correct'].mean()
    
    # Wrong-Confident Rate (Confidence >= 0.8 and wrong, not asking for clarification)
    threshold = 0.8 
    wrong_confident = results_df[
        (~results_df['is_correct']) & 
        (results_df['confidence'] >= threshold) & 
        (results_df['predicted_label'] != "Clarification Needed")
    ]
    wc_rate = len(wrong_confident) / len(results_df)
    
    # Clarification Recall (Reliability on Ambiguous Inputs)
    ambiguous_prompts = results_df[results_df['is_ambiguous'] == True]
    clarity_recall = (ambiguous_prompts['predicted_label'] == "Clarification Needed").mean() if len(ambiguous_prompts) > 0 else 0

    return {
        "Accuracy": accuracy,
        "Wrong-Confident Rate": wc_rate,
        "Clarification Recall": clarity_recall
    }

# 4. Calculate and Print Metrics
metrics = evaluate_reliability(results_df)
print("\n--- Final Baseline Metrics ---")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}")

# Create an output directory if it doesn't exist
output_dir = os.path.join(script_dir, '..', '..', 'results', 'phase_1')
os.makedirs(output_dir, exist_ok=True)

# 5. Export results using the explicit path
csv_output_path = os.path.join(output_dir, 'baseline_results_final.csv')
results_df.to_csv(csv_output_path, index=False)
print(f"Results saved to: {csv_output_path}")

# 6. Generate Confusion Matrix Visualization
labels = sorted(list(set(results_df['label'].unique())))
cm = confusion_matrix(results_df['label'], results_df['predicted_label'], labels=labels)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels, cmap='Greens')
plt.title('Baseline Embedding Router: Predicted vs Actual')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()

# Save the plot using the explicit path
plot_output_path = os.path.join(output_dir, 'baseline_confusion_matrix.png')
plt.savefig(plot_output_path)
print(f"Plot saved to: {plot_output_path}")