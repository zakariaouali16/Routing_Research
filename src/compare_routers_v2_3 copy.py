import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import re
from sklearn.metrics import accuracy_score, classification_report

# Import your classes
from baselines.baseline_embedding_router import EmbeddingRouter
from llm_router_v2_3 import LLMRouterV1

def clean_label(label):
    """Standardizes labels and strips hallucinated domain prefixes for fair comparison."""
    cleaned = str(label).strip().lower().replace("_", " ")
    
    # Strip out any text ending with 'domain' followed by punctuation/spaces
    # e.g., "education domain - concept explanation" -> "concept explanation"
    cleaned = re.sub(r'^.*?domain[\s\-\/\>\.]*', '', cleaned)
    
    # Handle edge cases where it just says "education / " without the word "domain"
    cleaned = re.sub(r'^(education|healthcare|utility)[\s\-\/\>\.]*', '', cleaned)
    
    return cleaned.strip()

def run_comparison():
    # 1. Setup paths
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BASE_DIR = os.path.dirname(CURRENT_DIR)
    
    DATA_PATH = os.path.join(BASE_DIR, "Data", "v1_1_pilot_benchmark.csv")
    TAXONOMY_PATH = os.path.join(BASE_DIR, "Data", "taxonomy_v2.json")
    
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
    llm_preds = []
    
    # --- FIX 2: Map to the correct 'gold_label' column ---
    gold_labels = df['gold_label'].tolist()

    print(f"\n--- Running benchmark on {len(df)} rows ---")
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        
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
            res = llm_router.route_request(prompt) # <--- REMOVE DOMAIN HERE
            l_val = res.get('predicted_label', 'Error') if isinstance(res, dict) else str(res)
            llm_preds.append(l_val)
        except Exception as e:
            llm_preds.append(f"Err: {str(e)}")
            
    # 5. Export results to CSV
    df['baseline_prediction'] = baseline_preds
    df['llm_prediction'] = llm_preds
    
    gold_cleaned = [clean_label(l) for l in gold_labels]
    base_cleaned = [clean_label(l) for l in baseline_preds]
    llm_cleaned  = [clean_label(l) for l in llm_preds]

    df['baseline_match'] = [g == b for g, b in zip(gold_cleaned, base_cleaned)]
    df['llm_match'] = [g == l for g, l in zip(gold_cleaned, llm_cleaned)]

    df.to_csv(output_file, index=False)

    # 6. Final Report
    print("\n" + "="*50)
    print(f"RESULTS SAVED TO: {output_file}")
    print("="*50)

    baseline_acc = accuracy_score(gold_cleaned, base_cleaned)
    llm_acc = accuracy_score(gold_cleaned, llm_cleaned)

    print(f"Baseline (Embedding) Accuracy: {baseline_acc * 100:.2f}%")
    print(f"LLM (Llama3) Accuracy:         {llm_acc * 100:.2f}%")
    # --- ADDED: Detailed classification reports per label ---
    print("\n--- DETAILED ACCURACY RATING (Baseline Router) ---")
    print(classification_report(gold_cleaned, base_cleaned, zero_division=0))
    
    print("\n--- DETAILED ACCURACY RATING (LLM Router) ---")
    print(classification_report(gold_cleaned, llm_cleaned, zero_division=0))
if __name__ == "__main__":
    run_comparison()