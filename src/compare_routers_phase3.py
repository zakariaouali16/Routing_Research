import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report

# Import your classes
from baselines.baseline_embedding_router import EmbeddingRouter
from llm_router_phase3 import LLMRouterV1

def clean_label(label):
    """Standardizes labels for fair comparison."""
    return str(label).strip().lower().replace("_", " ")

def run_comparison():
    # 1. Setup paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    
    DATA_PATH = os.path.join(BASE_DIR, "Data", "pilot_benchmark_phase3.csv")
    TAXONOMY_PATH = os.path.join(BASE_DIR, "Data", "taxonomy_phase3.json")
    
    RESULTS_DIR = os.path.join(BASE_DIR, "results","phase3")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. Extract the taxonomy name
    taxonomy_name = os.path.splitext(os.path.basename(TAXONOMY_PATH))[0]
    
    # 2. Get the LLM router name 
    router_name = LLMRouterV1.__name__
    
    # 3. Format the new output file name specifically for the LLM router
    file_name = f"benchmark_{router_name}_{taxonomy_name}_{timestamp}.csv"
    output_file = os.path.join(RESULTS_DIR, file_name)

    # 2. Load data
    df = pd.read_csv(DATA_PATH)
    
    # --- FIX 1: Create a temporary DataFrame/CSV for the Baseline Router ---
    temp_data_path = DATA_PATH.replace('.csv', '_temp_fit.csv')
    df_temp = df.rename(columns={'gold_label': 'gold_outcome'})
    df_temp.to_csv(temp_data_path, index=False)
    
    # 3. Initialize Routers
    print("\n--- Initializing Routers ---")
    baseline_router = EmbeddingRouter()
    
    # --- FIXED: Use the new training dataset for fitting ---
    TRAINING_PATH = os.path.join(BASE_DIR, "Data", "training_data.csv")
    temp_train_path = TRAINING_PATH.replace('.csv', '_temp_fit.csv')
    
    # Load training data and format it for the baseline router
    df_train = pd.read_csv(TRAINING_PATH)
    df_train = df_train.rename(columns={'gold_label': 'gold_outcome'})
    df_train.to_csv(temp_train_path, index=False)
    
    # Fit the baseline on the separate training data
    baseline_router.fit(temp_train_path) 
    
    # Clean up the temporary training file
    if os.path.exists(temp_train_path):
        os.remove(temp_train_path)
        
    llm_router = LLMRouterV1(taxonomy_path=TAXONOMY_PATH)

    # 4. Run Predictions
    baseline_preds = []
    llm_first_pass_preds = [] # NEW
    llm_final_preds = []      # NEW
    
    # --- FIX 2: Map to the correct 'gold_label' column ---
    gold_labels = df['gold_label'].tolist()

    print(f"\n--- Running benchmark on {len(df)} rows ---")
    for index, row in tqdm(df.iterrows(), total=len(df)):
        
        # --- FIX 3: Map to the correct 'user_prompt' column ---
        prompt = row['user_prompt']
        domain = row['domain']
        
        # --- Baseline Prediction ---
        try:
            res = baseline_router.route_request(prompt)
            b_val = res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res)
            baseline_preds.append(b_val)
        except Exception as e:
            baseline_preds.append(f"Err: {str(e)}")

        # --- LLM Prediction ---
        try:
            # Unpack the dictionary returned by the updated router
            llm_result = llm_router.route_request(user_prompt)
            llm_first_pass_preds.append(llm_result.get("first_pass", "Error"))
            llm_final_preds.append(llm_result.get("final_pass", "Error"))
        except Exception as e:
            llm_first_pass_preds.append(f"Err: {str(e)}")
            llm_final_preds.append(f"Err: {str(e)}")
            
    # 5. Export results to CSV
    df['baseline_prediction'] = baseline_preds
    df['llm_1st_pass_prediction'] = llm_first_pass_preds # NEW
    df['llm_final_prediction'] = llm_final_preds         # NEW
    
    gold_cleaned = [clean_label(l) for l in gold_labels]
    base_cleaned = [clean_label(l) for l in baseline_preds]
    llm_first_cleaned  = [clean_label(l) for l in llm_first_pass_preds] # NEW
    llm_final_cleaned  = [clean_label(l) for l in llm_final_preds]      # NEW

    df['baseline_match'] = [g == b for g, b in zip(gold_cleaned, base_cleaned)]
    df['llm_1st_pass_match'] = [g == l for g, l in zip(gold_cleaned, llm_first_cleaned)] # NEW
    df['llm_final_match'] = [g == l for g, l in zip(gold_cleaned, llm_final_cleaned)]    # NEW

    df.to_csv(output_file, index=False)

    # 6. Final Report
    print("\n" + "="*50)
    print(f"RESULTS SAVED TO: {output_file}")
    print("="*50)

    baseline_acc = accuracy_score(gold_cleaned, base_cleaned)
    llm_1st_acc = accuracy_score(gold_cleaned, llm_first_cleaned) # NEW
    llm_final_acc = accuracy_score(gold_cleaned, llm_final_cleaned) # NEW

    print(f"Baseline (Embedding) Accuracy: {baseline_acc * 100:.2f}%")
    print(f"LLM (Llama3) 1st Pass Accuracy: {llm_1st_acc * 100:.2f}%")
    print(f"LLM (Llama3) Final Pass Accuracy: {llm_final_acc * 100:.2f}%")
    # Calculate the delta to see if the 2nd pass helped or hurt
    delta = (llm_final_acc - llm_1st_acc) * 100
    if delta > 0:
        print(f"-> The 2nd pass IMPROVED accuracy by {delta:.2f}%")
    elif delta < 0:
        print(f"-> The 2nd pass DEGRADED accuracy by {abs(delta):.2f}% (Counter-productive)")
    else:
        print("-> The 2nd pass had NO IMPACT on overall accuracy.")
    # --- FIX: Extract only the unique, actual labels from the gold standard ---
    unique_gold_labels = sorted(list(set(gold_cleaned)))

    # --- ADDED: Detailed classification reports per label ---
    print("\n--- DETAILED ACCURACY RATING (Baseline Router) ---")
    print(classification_report(gold_cleaned, base_cleaned, labels=unique_gold_labels, zero_division=0))
    
    print("\n--- DETAILED ACCURACY RATING (LLM Router) ---")
    print(classification_report(gold_cleaned, llm_cleaned, labels=unique_gold_labels, zero_division=0))
if __name__ == "__main__":
    run_comparison()