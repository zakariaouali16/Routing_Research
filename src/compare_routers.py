import pandas as pd
import os

def compare_models(baseline_csv_path, llm_csv_path, output_csv_path="router_comparison_results.csv"):
    print(f"Loading Baseline results from: {baseline_csv_path}")
    print(f"Loading LLM results from: {llm_csv_path}\n")

    # 1. Load the results
    try:
        baseline_df = pd.read_csv(baseline_csv_path)
        llm_df = pd.read_csv(llm_csv_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find one of the result files. {e}")
        return

    # 2. Standardize column names for the merge
    # Assuming baseline outputs 'predicted_label' and 'confidence'
    baseline_df = baseline_df.rename(columns={
        'predicted_label': 'baseline_prediction',
        'confidence': 'baseline_confidence'
    })

    # Assuming LLM outputs 'predicted_label' and potentially 'confidence' (if requested in JSON)
    llm_df = llm_df.rename(columns={
        'predicted_label': 'llm_prediction',
        'confidence': 'llm_confidence'
    })

    # 3. Merge the dataframes on 'prompt_id'
    # We keep 'prompt' and the gold 'label' from the baseline df to avoid duplicates
    comparison_df = pd.merge(
        baseline_df[['prompt_id', 'domain', 'prompt', 'label', 'is_ambiguous', 'baseline_prediction', 'baseline_confidence']],
        llm_df[['prompt_id', 'llm_prediction', 'llm_confidence']] if 'llm_confidence' in llm_df.columns else llm_df[['prompt_id', 'llm_prediction']],
        on='prompt_id',
        how='inner'
    )

    # 4. Calculate Correctness (Accuracy Boolean)
    comparison_df['baseline_is_correct'] = comparison_df['baseline_prediction'] == comparison_df['label']
    comparison_df['llm_is_correct'] = comparison_df['llm_prediction'] == comparison_df['label']

    # 5. Calculate High-Level Metrics
    total_prompts = len(comparison_df)
    
    baseline_accuracy = comparison_df['baseline_is_correct'].mean()
    llm_accuracy = comparison_df['llm_is_correct'].mean()

    baseline_error_rate = 1 - baseline_accuracy
    llm_error_rate = 1 - llm_accuracy

    baseline_avg_conf = comparison_df['baseline_confidence'].mean()
    
    # Calculate LLM confidence if available
    if 'llm_confidence' in comparison_df.columns:
        # Convert to float just in case the LLM outputted strings
        comparison_df['llm_confidence'] = pd.to_numeric(comparison_df['llm_confidence'], errors='coerce')
        llm_avg_conf = comparison_df['llm_confidence'].mean()
    else:
        llm_avg_conf = "N/A (Not tracked by LLM)"

    # 6. Print the Results
    print("-" * 40)
    print(f"COMPARISON RESULTS (Total Prompts: {total_prompts})")
    print("-" * 40)
    print(f"{'Metric':<25} | {'Baseline':<15} | {'LLM'}")
    print("-" * 40)
    print(f"{'Accuracy Rate':<25} | {baseline_accuracy:.2%}         | {llm_accuracy:.2%}")
    print(f"{'Error Rate':<25} | {baseline_error_rate:.2%}         | {llm_error_rate:.2%}")
    
    if isinstance(llm_avg_conf, float):
        print(f"{'Avg Confidence Level':<25} | {baseline_avg_conf:.4f}          | {llm_avg_conf:.4f}")
    else:
        print(f"{'Avg Confidence Level':<25} | {baseline_avg_conf:.4f}          | {llm_avg_conf}")
    print("-" * 40)

    # 7. Identify where LLM fixed Baseline errors (and vice versa)
    llm_fixed = comparison_df[(~comparison_df['baseline_is_correct']) & (comparison_df['llm_is_correct'])]
    baseline_fixed = comparison_df[(comparison_df['baseline_is_correct']) & (~comparison_df['llm_is_correct'])]
    
    print(f"\nInsights:")
    print(f"- The LLM correctly routed {len(llm_fixed)} prompts that the Baseline got wrong.")
    print(f"- The Baseline correctly routed {len(baseline_fixed)} prompts that the LLM got wrong.")

    # 8. Export the side-by-side comparison
    comparison_df.to_csv(output_csv_path, index=False)
    print(f"\nFull side-by-side analysis saved to: {output_csv_path}")

if __name__ == "__main__":
    # Update these paths to point to where your generated results live
    BASELINE_FILE = "../../results/phase_1/baseline_results_final.csv" 
    LLM_FILE = "../../Data/v1_llm_results.csv"
    OUTPUT_FILE = "../../results/router_head_to_head_comparison.csv"
    
    # Create directories if they don't exist
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    
    compare_models(BASELINE_FILE, LLM_FILE, OUTPUT_FILE)