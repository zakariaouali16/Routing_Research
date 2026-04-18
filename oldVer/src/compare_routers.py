import os
import pandas as pd
from baselines.baseline_embedding_router import EmbeddingRouter

from sklearn.metrics import accuracy_score

# Import your existing classes

from llm_router_v1 import LLMRouterV1


# Define relative paths based on your folder structure
# 1. Get the directory where THIS script is (Routing_Research/src)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the Root (Routing_Research)
BASE_DIR = os.path.dirname(CURRENT_DIR)
DATA_PATH = os.path.join(BASE_DIR, "Data", "v0_pilot_benchmark.csv")
TAXONOMY_PATH = os.path.join(BASE_DIR, "Data", "taxonomy_v1.json")
OUTPUT_PATH = os.path.join(BASE_DIR, "Results", "comparison_report.csv")

def run_comparison(benchmark_csv, taxonomy_json, output_csv):
    print("--- Initializing Routers ---")
    
    # 1. Initialize Baseline
    baseline = EmbeddingRouter()
    baseline.fit(benchmark_csv)
    
    # 2. Initialize LLM Router (Assumes Ollama is running)
    # We pass the local path to taxonomy
    llm_router = LLMRouterV1(model_name='llama3', taxonomy_path=taxonomy_json)
    
    # Load Benchmark Data
    df = pd.read_csv(benchmark_csv)
    results = []

    print(f"\n--- Running Evaluation on {len(df)} prompts ---")
    
    for idx, row in df.iterrows():
        prompt = row['prompt']
        domain = row['domain']
        gold_label = row['label']
        
        print(f"[{idx+1}/{len(df)}] Processing: {prompt[:50]}...")
        
        # Get Baseline Prediction
        # We use a standard threshold; adjust based on your needs
        b_res = baseline.route_request(prompt, threshold=0.5)
        
        # Get LLM Prediction
        l_res = llm_router.route_request(prompt, domain)
        
        # Log results
        results.append({
            "prompt_id": row.get('prompt_id', idx),
            "domain": domain,
            "prompt": prompt,
            "gold_label": gold_label,
            "baseline_pred": b_res['predicted_label'],
            "llm_pred": l_res.get('predicted_label', 'Error'),
            "baseline_score": b_res['confidence_score'],
            "llm_conf": l_res.get('confidence_level', 'Low'),
            "llm_reason": l_res.get('short_reason', ''),
            "baseline_match": b_res['matched_example']
        })

    # Create Comparison DataFrame
    comp_df = pd.DataFrame(results)
    
    # Add Boolean correctness columns
    comp_df['baseline_correct'] = comp_df['baseline_pred'] == comp_df['gold_label']
    comp_df['llm_correct'] = comp_df['llm_pred'] == comp_df['gold_label']
    comp_df['agreement'] = comp_df['baseline_pred'] == comp_df['llm_pred']

    # Save to CSV
    comp_df.to_csv(output_csv, index=False)
    print(f"\nDetailed results saved to: {output_csv}")

    # --- Summary Statistics ---
    print("\n" + "="*30)
    print("      PERFORMANCE SUMMARY")
    print("="*30)
    
    b_acc = accuracy_score(comp_df['gold_label'], comp_df['baseline_pred'])
    l_acc = accuracy_score(comp_df['gold_label'], comp_df['llm_pred'])
    agreement = comp_df['agreement'].mean()

    print(f"Baseline Accuracy: {b_acc:.2%}")
    print(f"LLM Router Accuracy: {l_acc:.2%}")
    print(f"Inter-Router Agreement: {agreement:.2%}")
    
    # Discordance Analysis: Where LLM wins vs where Baseline wins
    llm_only_correct = comp_df[(comp_df['llm_correct']) & (~comp_df['baseline_correct'])]
    baseline_only_correct = comp_df[(~comp_df['llm_correct']) & (comp_df['baseline_correct'])]
    
    print(f"\nLLM was correct when Baseline failed: {len(llm_only_correct)} times")
    print(f"Baseline was correct when LLM failed: {len(baseline_only_correct)} times")
    
    if not llm_only_correct.empty:
        print("\nTop LLM Wins (Semantic Reasoning):")
        print(llm_only_correct[['prompt', 'gold_label', 'baseline_pred', 'llm_reason']].head(3))

def main():
    # 1. Load the Baseline
    # Note: EmbeddingRouter needs to fit the data to create its vector database
    print("Initializing Baseline...")
    baseline = EmbeddingRouter()
    baseline.fit(DATA_PATH)

    # 2. Load the LLM Router
    print("Initializing LLM Router...")
    llm = LLMRouterV1(model_name='llama3', taxonomy_path=TAXONOMY_PATH)

    # 3. Load Benchmark
    df = pd.read_csv(DATA_PATH)
    comparison_results = []

    print(f"Comparing {len(df)} samples...")
    for _, row in df.iterrows():
        prompt = row['prompt']
        domain = row['domain']
        
        # Run both
        b_out = baseline.route_request(prompt)
        l_out = llm.route_request(prompt, domain)
        
        comparison_results.append({
            "prompt": prompt,
            "actual": row['label'],
            "baseline_pred": b_out['predicted_label'],
            "llm_pred": l_out.get('predicted_label'),
            "baseline_conf": b_out['confidence_score'],
            "llm_reason": l_out.get('short_reason')
        })

    # 4. Export to Results folder
    results_df = pd.DataFrame(comparison_results)
    results_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Done! Report saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()