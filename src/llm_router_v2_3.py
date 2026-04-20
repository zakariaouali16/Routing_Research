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
        """Constructs a prompt that instructs the LLM to classify and extract required slots."""
        all_labels_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            all_labels_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for l in domain_data['labels']:
                all_labels_text += f"- {l['name']}: {l['definition']}\n"
                slots = l.get('required_slots', [])
                if slots:
                    all_labels_text += f"  REQUIRED INFO TO EXTRACT: {', '.join(slots)}\n"
        
        system_prompt = f"""You are an expert, autonomous routing and information extraction agent.
    Your task is to classify the user's request and identify missing information.

    CATEGORIES AND DATA REQUIREMENTS:
    {all_labels_text}

    CRITICAL INSTRUCTION:
    1. Classify the request into EXACTLY ONE category.
    2. If the category has "REQUIRED INFO", extract those values from the user's text.
    3. If a required value is missing, list it in 'missing_slots'.
    4. If the request is too vague for any category, use 'Clarification Needed'.

    Respond ONLY as a valid JSON object with these keys:
    {{
        "predicted_label": "The category name",
        "extracted_slots": {{ "slot_name": "value_found" }},
        "missing_slots": ["list_of_missing_required_slots"],
        "confidence_level": "High/Medium/Low",
        "needs_clarification": true/false,
        "short_reason": "One sentence justification"
    }}"""
        return system_prompt

    def verify_prediction(self, user_prompt, proposed_label, extracted_slots):
        """
        A secondary QA auditor that verifies both the routing label 
        and the accuracy of extracted information.
        """
        # Automatically find the domain for the proposed label
        domain_key = None
        label_info = None
        all_labels_in_domain = []

        for d_name, d_data in self.taxonomy['domains'].items():
            match = next((l for l in d_data['labels'] if l['name'] == proposed_label), None)
            if match:
                domain_key = d_name
                label_info = match
                all_labels_in_domain = d_data['labels']
                break

        # If no domain match (e.g., "Clarification Needed"), use general context
        if not domain_key:
            labels_text = "N/A - Classification requested clarification."
            required_slots = []
        else:
            labels_text = "\n".join([f"- {l['name']}: {l['definition']}" for l in all_labels_in_domain])
            required_slots = label_info.get('required_slots', [])

        verification_prompt = f"""You are a strict QA auditor. 
    Verify the routing accuracy and data extraction.

    CONTEXT DOMAIN: {domain_key if domain_key else "General/Unknown"}
    VALID CATEGORIES IN THIS DOMAIN:
    {labels_text}

    USER REQUEST: "{user_prompt}"
    PROPOSED LABEL: "{proposed_label}"
    EXTRACTED SLOTS: {json.dumps(extracted_slots)}
    REQUIRED SLOTS FOR THIS CATEGORY: {required_slots}

    TASK:
    1. Label Check: Is the PROPOSED LABEL the absolute best fit?
    2. Slot Check: Did the agent miss info present in the request?

    Output ONLY a valid JSON object:
    {{
        "is_correct": true or false,
        "verified_label": "Correct label name",
        "verified_slots": {{ "slot_name": "corrected_value" }},
        "qa_reason": "Brief explanation"
    }}"""

        payload = {
            "model": self.model_name,
            "prompt": verification_prompt,
            "stream": False,
            "format": "json", 
            "options": { "temperature": 0.0, "seed": 42 }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            return {"is_correct": True, "verified_label": proposed_label, "verified_slots": extracted_slots, "qa_reason": f"Error: {e}"}

    def route_request(self, user_prompt):
        """Sends the prompt to LLM, gets prediction, and performs QA verification."""
        system_prompt = self.build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n\"{user_prompt}\""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json", 
            "options": { "temperature": 0.0, "seed": 42 }
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            initial_output = json.loads(response.json().get("response", "{}"))
        except Exception as e:
            return {"initial_label": "Error", "predicted_label": "Error", "confidence_level": "Low", "short_reason": str(e), "needs_clarification": True, "was_corrected": False}

        initial_label = initial_output.get("predicted_label", "Clarification Needed")
        initial_slots = initial_output.get("extracted_slots", {})

        # --- PASS 2: Verification (Fixing the argument error here) ---
        qa_output = self.verify_prediction(user_prompt, initial_label, initial_slots)
        
        final_label = qa_output.get("verified_label", initial_label)
        
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
        df = pd.read_csv(input_csv)
        results = []
        
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
            # Removed 'domain' from the call to route_request
            llm_output = self.route_request(row['prompt'])
            
            results.append({
                "prompt_id": row.get('prompt_id', f"ID-{index}"),
                "domain": row['domain'],
                "user_prompt": row['prompt'],
                "gold_label": row.get('label', ''),
                "initial_predicted_label": llm_output.get("initial_label", ""),
                "final_predicted_label": llm_output.get("predicted_label", ""),
                "was_corrected_by_qa": llm_output.get("was_corrected", False)
            })
            
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"Accuracy: {(results_df['gold_label'] == results_df['final_predicted_label']).mean()*100:.2f}%")