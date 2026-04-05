import os
import json
import pandas as pd
import requests
from tqdm import tqdm  # Added tqdm for the progress bar
from concurrent.futures import ThreadPoolExecutor, as_completed
from stepARouter import StepARouter

class LLMRouterV1:
    def __init__(self, model_name='llama3', taxonomy_path='../../Data/taxonomy_v2.json'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
        # Load the taxonomy
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

        # Build the Step A Router tool
        self.step_a_router = StepARouter(model_name=model_name)

    def build_system_prompt(self):
        """Constructs a unified prompt using all domains and labels from the taxonomy."""
        taxonomy_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            taxonomy_text += f"\nDOMAIN: {domain_name.capitalize()}\n"
            taxonomy_text += f"Description: {domain_data['domain_description']}\n"
            for label in domain_data['labels']:
                taxonomy_text += f"  - {label['name']}: {label['definition']}\n"

        system_prompt = f"""You are an expert master routing agent for a multi-domain support system.
Your task is to first determine the correct DOMAIN for the user's request, and then classify it into EXACTLY ONE of the corresponding routing categories.

{taxonomy_text}

Analyze the user's prompt carefully. You must output your response ONLY as a valid JSON object with the following exact keys:
{{
    "domain": "The exact name of the domain (Education or Healthcare)",
    "predicted_label": "The exact name of the label from the chosen domain",
    "needs_clarification": true or false,
    "clarifying_question": "If needs_clarification is true, write a specific question to ask the user to resolve the ambiguity. If false, output null.",
    "short_reason": "One short sentence explaining why you chose this domain and route",
    "confidence_level": "High, Medium, or Low"
}}

Do not include any markdown formatting, conversational text, or explanations outside of the JSON object.
"""
        return system_prompt

    def route_request(self, user_prompt):
        """Sends the prompt to the local Ollama model and parses the JSON response."""
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
            
            result_text = response.json().get("response", "{}")
            return json.loads(result_text)
            
        except Exception as e:
            return {
                "domain": "Error",
                "predicted_label": "Error",
                "confidence_level": "Low",
                "short_reason": f"API Error: {str(e)}",
                "needs_clarification": True,
                "clarifying_question": "I encountered a system error. Could you try asking again?"
            }
    def interactive_chat(self):
        """Runs an interactive session where the LLM auto-detects the domain and asks for clarification."""
        print(f"\n=== Starting Master Dispatch Router ===")
        print("Type 'exit' or  'quit' to stop.\n")

        # Memory for follow-ups
        context = ""
        
        while True:
            prompt_label = "Follow-up Request: " if context else "User Request: "
            user_input = input(f"\n{prompt_label}")

            # Stop the loop
            if user_input.lower() in ['exit', 'quit']: 
              print("Goodbye! 👋")
              break

            # Combine new answer with previous context 
            full_request = f"{context} {user_input}".strip()
        
            # --- STEP A: CLARIFY VS ROUTE ---
            triage_result = self.step_a_router.check_request(full_request)
            
            if not triage_result.get("is_actionable"):
                # If Step A fails, ask the clarifying question immediately
                print(f"🤖 Clarification Needed: {triage_result.get('clarifying_question')}")
                context = full_request # Save the context so we can add to it next time
                continue # Loop back to get more input
                
            # --- STEP B: TAXONOMY ROUTING ---
            # Only if Step A is successful, run your existing heavy taxonomy logic
            response = self.route_request(full_request)
            print(f"✅ Routed to: {response.get('predicted_label')}")

            context = ""
    
    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the LLM over the entire pilot benchmark and saves the results."""
        print(f"Loading benchmark data from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []
        
        print(f"Routing {len(df)} requests. This will take a few minutes depending on your hardware...\n")
        
        # --- ADDED TQDM PROGRESS BAR HERE ---
        # Wrapping df.iterrows() with tqdm automatically draws the progress bar
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing Requests", unit="prompt"):
            prompt_text = row['prompt']
            domain = row['domain']
            
            # Ask the LLM to route it
            llm_output = self.route_request(prompt_text, domain)
            
            result_row = {
                "prompt_id": row.get('prompt_id', f"ID-{index}"),
                "domain": domain,
                "user_prompt": prompt_text,
                "gold_label": row.get('label', ''),
                "is_ambiguous_gold": row.get('is_ambiguous', ''),
                "predicted_label": llm_output.get("predicted_label", ""),
                "confidence_level": llm_output.get("confidence_level", ""),
                "needs_clarification_pred": llm_output.get("needs_clarification", False),
                "short_reason": llm_output.get("short_reason", "")
            }
            results.append(result_row)
            
        # Save to the new CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDone! Results saved to {output_csv}\n")
        
        # ==========================================
        # ADVANCED ACCURACY RATINGS
        # ==========================================
        
        # 1. Overall Accuracy
        correct = (results_df['gold_label'] == results_df['predicted_label']).sum()
        total = len(results_df)
        print("="*50)
        print(f"OVERALL ACCURACY RATING: {correct}/{total} ({(correct/total)*100:.2f}%)")
        print("="*50)

        # 2. Accuracy by Domain
        if 'domain' in results_df.columns:
            print("\n--- ACCURACY BY DOMAIN ---")
            for dom in results_df['domain'].unique():
                domain_df = results_df[results_df['domain'] == dom]
                d_correct = (domain_df['gold_label'] == domain_df['predicted_label']).sum()
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
                results_df['predicted_label'], 
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
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_v2.json"))
    
    # Initialize the router
    router = LLMRouterV1(model_name='llama3', taxonomy_path=taxonomy_path)
    
    # Run the interactive master dispatcher
    router.interactive_chat()