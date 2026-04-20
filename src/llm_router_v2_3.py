import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class LLMRouterV1:
    def __init__(self, model_name='llama3', taxonomy_path='../../Data/taxonomy_v2.json'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
        # Load the taxonomy
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

    def build_system_prompt(self):
        all_labels_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            all_labels_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for l in domain_data['labels']:
                slots = l.get('required_slots', [])
                all_labels_text += f"- {l['name']}: {l['definition']}\n"
                if slots:
                    all_labels_text += f"  REQUIRED INFO: {', '.join(slots)}\n"
    
        system_prompt = f"""You are an expert routing and information extraction agent.
Your task:
1. Classify the request into EXACTLY ONE category:
{all_labels_text}

2. Identify "Slots":
- Check the REQUIRED INFO for your chosen category.
- Extract any values present in the user request.
- List any missing values.

Respond ONLY in JSON format:
{{
    "predicted_label": "Category Name",
    "confidence_level": "High/Medium/Low",
    "extracted_slots": {{ "slot_name": "extracted_value" }},
    "missing_slots": ["slot_name"],
    "needs_clarification": true/false,
    "short_reason": "Reasoning"
}}"""
        return system_prompt

    def verify_prediction(self, user_prompt, domain, proposed_label):
        """A secondary lightweight verification step to act as a QA auditor."""
        domain_key = domain.lower() 
        labels = self.taxonomy['domains'][domain_key]['labels']
        labels_text = "\n".join([f"- {l['name']}: {l['definition']}" for l in labels])
        
        verification_prompt = f"""You are a strict QA auditor for a {domain_key} support system.
A previous routing agent classified a user's request, and your job is to verify if it is accurate based on the taxonomy.

VALID CATEGORIES:
{labels_text}

USER REQUEST: "{user_prompt}"
PROPOSED LABEL: "{proposed_label}"

Critically analyze if the PROPOSED LABEL is the absolute best fit for the USER REQUEST.
Output your response ONLY as a valid JSON object with these exact keys:
{{
    "is_correct": true or false,
    "verified_label": "If is_correct is true, output the PROPOSED LABEL. If false, output the corrected valid category name from the VALID CATEGORIES list.",
    "qa_reason": "One short sentence explaining why you confirmed or corrected the label."
}}

Do not include any markdown formatting, conversational text, or explanations outside of the JSON object.
"""
        payload = {
            "model": self.model_name,
            "prompt": verification_prompt,
            "stream": False,
            "format": "json", 
            "options": {            
                "temperature": 0.0, # Keep at 0 for strict validation
                "seed": 42
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            tqdm.write(f"Verification Error -> {e}")
            return {"is_correct": True, "verified_label": proposed_label, "qa_reason": "Verification failed, defaulting to original."}

    def route_request(self, user_prompt, domain):
        """Sends the prompt to Ollama, gets a prediction, and verifies it."""
        # --- PASS 1: Initial Generation ---
        # [Remove the 'domain' parameter from this method's signature]
        system_prompt = self.build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n\"{user_prompt}\""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json", 
            "options": {            
                "temperature": 0.0, 
                "seed": 42
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            initial_output = json.loads(response.json().get("response", "{}"))
        except Exception as e:
            tqdm.write(f"Error calling LLM for prompt: '{user_prompt[:30]}...' -> {e}")
            return {
                "initial_label": "Error",
                "predicted_label": "Error",
                "confidence_level": "Low",
                "short_reason": f"API Error: {str(e)}",
                "needs_clarification": True,
                "was_corrected": False,
                "qa_reason": "N/A"
            }

        initial_label = initial_output.get("predicted_label", "")

        # --- PASS 2: Lightweight Verification ---
        qa_output = self.verify_prediction(user_prompt, domain, initial_label)
        
        # Merge the outputs
        final_label = qa_output.get("verified_label", initial_label)
        
        # Build final aggregated response
        return {
            "initial_label": initial_label,
            "predicted_label": final_label,
            "confidence_level": initial_output.get("confidence_level", "Unknown"),
            "needs_clarification": initial_output.get("needs_clarification", False),
            "short_reason": initial_output.get("short_reason", ""),
            "was_corrected": not qa_output.get("is_correct", True),
            "qa_reason": qa_output.get("qa_reason", "")
        }

    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the LLM over the entire pilot benchmark and saves the results."""
        print(f"Loading benchmark data from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []
        
        print(f"Routing {len(df)} requests. This will take longer due to the 2-pass verification system...\n")
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Requests", unit="prompt"):
            prompt_text = row['prompt']
            domain = row['domain']
            
            # Ask the LLM to route and verify it
            llm_output = self.route_request(prompt_text, domain)
            
            result_row = {
                "prompt_id": row.get('prompt_id', f"ID-{index}"),
                "domain": domain,
                "user_prompt": prompt_text,
                "gold_label": row.get('label', ''),
                "is_ambiguous_gold": row.get('is_ambiguous', ''),
                "initial_predicted_label": llm_output.get("initial_label", ""),
                "final_predicted_label": llm_output.get("predicted_label", ""),
                "confidence_level": llm_output.get("confidence_level", ""),
                "needs_clarification_pred": llm_output.get("needs_clarification", False),
                "short_reason": llm_output.get("short_reason", ""),
                "was_corrected_by_qa": llm_output.get("was_corrected", False),
                "qa_reason": llm_output.get("qa_reason", "")
            }
            results.append(result_row)
            
        # Save to the new CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDone! Results saved to {output_csv}\n")
        
        # ==========================================
        # ADVANCED ACCURACY RATINGS
        # ==========================================
        
        # 1. Overall Accuracy (using final verified label)
        correct = (results_df['gold_label'] == results_df['final_predicted_label']).sum()
        total = len(results_df)
        print("="*50)
        print(f"OVERALL POST-VERIFICATION ACCURACY: {correct}/{total} ({(correct/total)*100:.2f}%)")
        print("="*50)

        # Optional: Print how many times the QA agent intervened
        corrections = results_df['was_corrected_by_qa'].sum()
        print(f"QA Interventions: {corrections} out of {total} prompts.")

        # 2. Accuracy by Domain
        if 'domain' in results_df.columns:
            print("\n--- ACCURACY BY DOMAIN ---")
            for dom in results_df['domain'].unique():
                domain_df = results_df[results_df['domain'] == dom]
                d_correct = (domain_df['gold_label'] == domain_df['final_predicted_label']).sum()
                d_total = len(domain_df)
                if d_total > 0:
                    print(f"{dom}: {d_correct}/{d_total} ({(d_correct/d_total)*100:.2f}%)")
            print("-" * 26)

        # 3. Detailed Classification Report Rating
        try:
            from sklearn.metrics import classification_report
            print("\n--- DETAILED ACCURACY RATING (Per Label) ---")
            report = classification_report(
                results_df['gold_label'], 
                results_df['final_predicted_label'], 
                zero_division=0
            )
            print(report)
        except ImportError:
            print("\n[!] Tip: Install scikit-learn (`pip install scikit-learn`) to see a detailed Accuracy Rating per category.")

# ==========================================
# Run the evaluation
# ==========================================
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_v1.json"))
    benchmark_path = os.path.abspath(os.path.join(script_dir, "../Data/v1_pilot_benchmark.csv"))
    output_path = os.path.abspath(os.path.join(script_dir, "../Data/v1_llm_results.csv"))
    
    router = LLMRouterV1(model_name='llama3', taxonomy_path=taxonomy_path)
    router.evaluate_benchmark(benchmark_path, output_path)