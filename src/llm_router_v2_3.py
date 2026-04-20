import os
import json
import pandas as pd
import requests
from tqdm import tqdm

class LLMRouterV1:
    def __init__(self, model_name='llama3', taxonomy_path='../../Data/taxonomy_v2.json'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
        # Load the taxonomy
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

    def build_system_prompt(self):
        """Constructs the prompt using all domains, labels, and required slots."""
        
        all_labels_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            all_labels_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for l in domain_data['labels']:
                # Dynamically fetch the slots for the prompt
                slots = ", ".join(l.get('required_slots', [])) if l.get('required_slots') else "None"
                all_labels_text += f"- {l['name']}: {l['definition']} (Required Slots to Extract: {slots})\n"
        
        system_prompt = f"""You are an expert, autonomous routing agent.
Your task is to classify the user's request into EXACTLY ONE of the following routing categories across all domains AND extract the required slots based on the user's text.

{all_labels_text}

CRITICAL INSTRUCTION FOR AMBIGUITY (CONFIDENCE GATE):
If the user's request is too vague, lacks context, or does not clearly fit any of the specific categories above, you MUST route it to 'Clarification Needed'.

OUTPUT FORMAT:
Output your response ONLY as a valid JSON object using this exact schema:
{{
    "predicted_label": "The exact name of the category ONLY. Do NOT include the domain name, slashes, or arrows.",
    "confidence_level": "High, Medium, or Low",
    "short_reason": "One sentence explaining why",
    "needs_clarification": true or false,
    "extracted_slots": {{
        "slot_name": "The extracted value from the prompt, or null if the user did not provide it"
    }}
}}"""
        return system_prompt

    def verify_prediction(self, user_prompt, proposed_label, proposed_slots):
        """Secondary lightweight verification step. Completely blind to the gold domain."""
        
        # Rebuild the full taxonomy text so the QA agent can cross-check everything
        all_labels_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            all_labels_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for l in domain_data['labels']:
                slots = ", ".join(l.get('required_slots', [])) if l.get('required_slots') else "None"
                all_labels_text += f"- {l['name']}: {l['definition']} (Required Slots: {slots})\n"
        
        verification_prompt = f"""You are a strict QA auditor for a multi-domain support system.
A previous routing agent classified a user's request and extracted data. Your job is to verify if it is accurate based on the taxonomy.

VALID CATEGORIES & SLOTS ACROSS ALL DOMAINS:
{all_labels_text}

USER REQUEST: "{user_prompt}"
PROPOSED LABEL: "{proposed_label}"
PROPOSED SLOTS: {json.dumps(proposed_slots)}

Critically analyze if the PROPOSED LABEL is the absolute best fit across ALL domains. If you correct the label, you MUST extract the correct slots for your new label.

Output your response ONLY as a valid JSON object with these exact keys:
{{
    "is_correct": true or false,
    "verified_label": "If is_correct is true, output the PROPOSED LABEL. If false, output the corrected valid category name from ANY domain.",
    "qa_reason": "One short sentence explaining why you confirmed or corrected the label.",
    "verified_slots": {{
        "slot_name": "The extracted value for the verified_label, or null if missing"
    }}
}}
"""
        payload = {
            "model": self.model_name,
            "prompt": verification_prompt,
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
            tqdm.write(f"Verification Error -> {e}")
            return {
                "is_correct": True, 
                "verified_label": proposed_label, 
                "qa_reason": "Verification failed, defaulting to original.",
                "verified_slots": proposed_slots
            }

    def route_request(self, user_prompt):
        """Sends the prompt to Ollama, gets a prediction + slots, and verifies it."""
        # --- PASS 1: Initial Generation ---
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
                "qa_reason": "N/A",
                "final_slots": {}
            }

        initial_label = initial_output.get("predicted_label", "")
        initial_slots = initial_output.get("extracted_slots", {})

        # --- PASS 2: Lightweight Verification ---
        qa_output = self.verify_prediction(user_prompt, initial_label, initial_slots)
        
        final_label = qa_output.get("verified_label", initial_label)
        final_slots = qa_output.get("verified_slots", initial_slots)
        
        return {
            "initial_label": initial_label,
            "predicted_label": final_label,
            "confidence_level": initial_output.get("confidence_level", "Unknown"),
            "needs_clarification": initial_output.get("needs_clarification", False),
            "short_reason": initial_output.get("short_reason", ""),
            "was_corrected": not qa_output.get("is_correct", True),
            "qa_reason": qa_output.get("qa_reason", ""),
            "initial_slots": initial_slots,
            "final_slots": final_slots
        }

    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the LLM over the entire pilot benchmark and saves the results."""
        print(f"Loading benchmark data from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []
        
        print(f"Routing and Extracting Slots for {len(df)} requests...\n")
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Requests", unit="prompt"):
            prompt_text = row['prompt']
            domain = row['domain']
            
            llm_output = self.route_request(prompt_text)
            
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
                "qa_reason": llm_output.get("qa_reason", ""),
                # Convert the slots dictionary to a JSON string so it saves cleanly in the CSV
                "extracted_slots": json.dumps(llm_output.get("final_slots", {}))
            }
            results.append(result_row)
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDone! Results saved to {output_csv}\n")
        
        # --- Advanced Accuracy Ratings (Same as original) ---
        correct = (results_df['gold_label'] == results_df['final_predicted_label']).sum()
        total = len(results_df)
        print("="*50)
        print(f"OVERALL POST-VERIFICATION ACCURACY: {correct}/{total} ({(correct/total)*100:.2f}%)")
        print("="*50)

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Updated paths to reflect v2 mapping
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_v2.json"))
    benchmark_path = os.path.abspath(os.path.join(script_dir, "../Data/v2_pilot_benchmark.csv"))
    output_path = os.path.abspath(os.path.join(script_dir, "../Data/v2_llm_results.csv"))
    
    # Initialize the updated router
    router = LLMRouterV2(model_name='llama3', taxonomy_path=taxonomy_path)
    router.evaluate_benchmark(benchmark_path, output_path)