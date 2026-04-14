import os
import json
import pandas as pd
import requests
from tqdm import tqdm

class LLMRouterV1:
    # Updated default to taxonomy_v3.json
    def __init__(self, small_model='llama3:8b', large_model='llama3:70b', taxonomy_path='../../Data/taxonomy_v3.json', confidence_threshold=0.85):
        self.small_model = small_model
        self.large_model = large_model
        self.confidence_threshold = confidence_threshold
        self.api_url = "http://localhost:11434/api/generate"
        
        # Load the taxonomy
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

    def _call_api(self, model_name, prompt):
        """Helper function to handle Ollama API calls safely."""
        payload = {
            "model": model_name,
            "prompt": prompt,
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
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            tqdm.write(f"API Error with {model_name} -> {e}")
            return None

    def _get_all_categories_text(self, domain_key):
        """Helper to extract domain labels, safety overrides, and gating outcomes."""
        # 1. Domain Labels
        labels = self.taxonomy['domains'][domain_key]['labels']
        labels_text = "\n".join([f"- {l['name']}: {l['definition']}" for l in labels])
        
        # 2. Safety Overrides
        safety = self.taxonomy.get('safety_overrides', {}).get('categories', [])
        safety_text = "\n".join([f"- {s['name']}: {s['definition']} (Triggers: {', '.join(s['triggers'])})" for s in safety])
        
        # 3. Gating Outcomes
        gating = self.taxonomy.get('gating_outcomes', {}).get('outcomes', [])
        gating_text = "\n".join([f"- {g['name']}: {g['definition']}" for g in gating])
        
        return labels_text, safety_text, gating_text

    def build_system_prompt(self, domain):
        """Constructs the prompt using all aspects of taxonomy_v3."""
        domain_key = domain.lower() 
        if domain_key not in self.taxonomy['domains']:
            raise ValueError(f"Domain '{domain}' not found in taxonomy.")
            
        labels_text, safety_text, gating_text = self._get_all_categories_text(domain_key)
        
        system_prompt = f"""You are an expert routing agent for a {domain_key} support system.
Your task is to classify the user's request into EXACTLY ONE of the following routing categories. 

SAFETY OVERRIDES (HIGHEST PRIORITY):
{safety_text}

GATING OUTCOMES:
{gating_text}

DOMAIN LABELS:
{labels_text}

CRITICAL INSTRUCTION FOR AMBIGUITY (CONFIDENCE GATE):
1. If the user's request is one sentence, lacks a clear verb/noun, or is highly ambiguous (e.g., "I need help", "Is it done?"), you MUST classify it as 'Clarification Needed'. Do not attempt to guess the department.
2. If you cannot find at least two specific keywords relating to a specific category, default to 'Clarification Needed'.
3. Do not assume 'help' means 'emergency' unless words like 'pain', 'bleeding', or 'urgent' are mentioned.

FEW-SHOT EXAMPLES:
User: "Is it done yet?"
{{
    "needs_clarification": true, 
    "short_reason": "Prompt is too short and lacks specific keywords regarding what 'it' is.", 
    "predicted_label": "Clarification Needed", 
    "confidence_level": 0.95
}}

User: "I need to talk to someone about yesterday."
{{
    "needs_clarification": true, 
    "short_reason": "Vague timeframe reference without specific intent or department mentioned.", 
    "predicted_label": "Clarification Needed", 
    "confidence_level": 0.88
}}

Analyze the user's prompt carefully. You must output your response ONLY as a valid JSON object with the following exact keys:
{{
    "needs_clarification": true or false,
    "short_reason": "One short sentence explaining the core issue in the prompt",
    "predicted_label": "The exact name of the label from the list above",
    "confidence_level": A float between 0.0 and 1.0 representing your confidence in this prediction
}}

Do not include any markdown formatting, conversational text, or explanations outside of the JSON object.
"""
        return system_prompt

    def verify_prediction(self, user_prompt, domain, proposed_label, active_model):
        """A secondary lightweight verification step to act as a QA auditor."""
        domain_key = domain.lower() 
        labels_text, safety_text, gating_text = self._get_all_categories_text(domain_key)
        
        verification_prompt = f"""You are a strict QA auditor for a {domain_key} support system.
A previous routing agent classified a user's request, and your job is to verify if it is accurate based on the taxonomy.

VALID CATEGORIES:
{safety_text}
{gating_text}
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
        result = self._call_api(active_model, verification_prompt)
        if not result:
            return {"is_correct": True, "verified_label": proposed_label, "qa_reason": "Verification failed, defaulting to original."}
        return result

    def route_request(self, user_prompt, domain):
        """Sends prompt to small LLM, checks confidence, escalates to large LLM if needed, then verifies."""
        system_prompt = self.build_system_prompt(domain)
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n\"{user_prompt}\""
        
        # --- PASS 1: Initial Generation (Small Model) ---
        active_model = self.small_model
        initial_output = self._call_api(active_model, full_prompt)
        
        if not initial_output:
            return self._format_error("API Error during initial routing")

        # Safely parse confidence level as a float
        try:
            confidence = float(initial_output.get("confidence_level", 0.0))
        except ValueError:
            confidence = 0.0

        # --- ESCALATION: Check Confidence Gate ---
        if confidence < self.confidence_threshold:
            tqdm.write(f"Low confidence ({confidence:.2f}) from {active_model}. Escalating to {self.large_model}...")
            active_model = self.large_model
            escalated_output = self._call_api(active_model, full_prompt)
            
            if escalated_output:
                initial_output = escalated_output
                try:
                    confidence = float(initial_output.get("confidence_level", 0.0))
                except ValueError:
                    confidence = 0.0

        initial_label = initial_output.get("predicted_label", "")

        # --- PASS 2: Lightweight Verification ---
        qa_output = self.verify_prediction(user_prompt, domain, initial_label, active_model)
        
        final_label = qa_output.get("verified_label", initial_label)
        
        return {
            "initial_label": initial_label,
            "predicted_label": final_label,
            "confidence_level": confidence,
            "model_used": active_model, 
            "needs_clarification": initial_output.get("needs_clarification", False),
            "short_reason": initial_output.get("short_reason", ""),
            "was_corrected": not qa_output.get("is_correct", True),
            "qa_reason": qa_output.get("qa_reason", "")
        }

    def _format_error(self, message):
        return {
            "initial_label": "Error",
            "predicted_label": "Error",
            "confidence_level": 0.0,
            "model_used": "None",
            "short_reason": message,
            "needs_clarification": True,
            "was_corrected": False,
            "qa_reason": "N/A"
        }

    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the LLM over the entire pilot benchmark and saves the results."""
        print(f"Loading benchmark data from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []
        
        print(f"Routing {len(df)} requests. Escalation threshold is {self.confidence_threshold}...\n")
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Requests", unit="prompt"):
            prompt_text = row['prompt']
            domain = row['domain']
            
            llm_output = self.route_request(prompt_text, domain)
            
            result_row = {
                "prompt_id": row.get('prompt_id', f"ID-{index}"),
                "domain": domain,
                "user_prompt": prompt_text,
                "gold_label": row.get('label', ''),
                "is_ambiguous_gold": row.get('is_ambiguous', ''),
                "initial_predicted_label": llm_output.get("initial_label", ""),
                "final_predicted_label": llm_output.get("predicted_label", ""),
                "confidence_level": llm_output.get("confidence_level", 0.0),
                "model_used": llm_output.get("model_used", ""),
                "needs_clarification_pred": llm_output.get("needs_clarification", False),
                "short_reason": llm_output.get("short_reason", ""),
                "was_corrected_by_qa": llm_output.get("was_corrected", False),
                "qa_reason": llm_output.get("qa_reason", "")
            }
            results.append(result_row)
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDone! Results saved to {output_csv}\n")
        
        # Ratings block
        correct = (results_df['gold_label'] == results_df['final_predicted_label']).sum()
        total = len(results_df)
        print("="*50)
        print(f"OVERALL POST-VERIFICATION ACCURACY: {correct}/{total} ({(correct/total)*100:.2f}%)")
        print("="*50)

        small_uses = (results_df['model_used'] == self.small_model).sum()
        large_uses = (results_df['model_used'] == self.large_model).sum()
        print(f"Model Workload - {self.small_model}: {small_uses} | {self.large_model}: {large_uses}")

        corrections = results_df['was_corrected_by_qa'].sum()
        print(f"QA Interventions: {corrections} out of {total} prompts.")

        if 'domain' in results_df.columns:
            print("\n--- ACCURACY BY DOMAIN ---")
            for dom in results_df['domain'].unique():
                domain_df = results_df[results_df['domain'] == dom]
                d_correct = (domain_df['gold_label'] == domain_df['final_predicted_label']).sum()
                d_total = len(domain_df)
                if d_total > 0:
                    print(f"{dom}: {d_correct}/{d_total} ({(d_correct/d_total)*100:.2f}%)")
            print("-" * 26)

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


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Updated paths to reflect v3/v2 requirements
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_v3.json"))
    benchmark_path = os.path.abspath(os.path.join(script_dir, "../Data/v0_pilot_benchmark.csv")) # Updated to match your CSV name from the compare script
    output_path = os.path.abspath(os.path.join(script_dir, "../Data/v3_llm_results.csv"))
    
    router = LLMRouterV1(small_model='llama3:8b', large_model='llama3:70b', taxonomy_path=taxonomy_path, confidence_threshold=0.85)
    router.evaluate_benchmark(benchmark_path, output_path)