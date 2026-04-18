import os
import json
import pandas as pd
import requests

class LLMRouterV1:
    def __init__(self, model_name='llama3', taxonomy_path='../../Data/taxonomy_v1.json'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        
        # Load the taxonomy
        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

    def build_system_prompt(self, domain):
        """Constructs the prompt using the specific domain labels from the taxonomy."""
        # The taxonomy has 'education' and 'healthcare' keys
        domain_key = domain.lower() 
        if domain_key not in self.taxonomy['domains']:
            raise ValueError(f"Domain '{domain}' not found in taxonomy.")
            
        labels = self.taxonomy['domains'][domain_key]['labels']
        
        # Format labels into a readable string for the LLM
        labels_text = "\n".join([f"- {l['name']}: {l['definition']}" for l in labels])
        
        system_prompt = f"""You are an expert routing agent for a {domain_key} support system.
Your task is to classify the user's request into EXACTLY ONE of the following routing categories:

{labels_text}

Analyze the user's prompt carefully. You must output your response ONLY as a valid JSON object with the following exact keys:
{{
    "predicted_label": "The exact name of the label from the list above",
    "confidence_level": "High, Medium, or Low",
    "short_reason": "One short sentence explaining why you chose this label",
    "needs_clarification": true or false (use true if the prompt is too ambiguous, vague, or missing critical details)
}}
Do not include any markdown formatting, conversational text, or explanations outside of the JSON object.
"""
        return system_prompt

    def route_request(self, user_prompt, domain):
        """Sends the prompt to the local Ollama model and parses the JSON response."""
        system_prompt = self.build_system_prompt(domain)
        
        # Combine system instructions and the actual user request
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n\"{user_prompt}\""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json" # Forces Ollama to return valid JSON
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            
            result_text = response.json().get("response", "{}")
            
            # Parse the string back into a Python dictionary
            parsed_result = json.loads(result_text)
            return parsed_result
            
        except Exception as e:
            print(f"Error calling LLM for prompt: '{user_prompt[:30]}...' -> {e}")
            # Fallback if the LLM fails or hallucinates bad JSON
            return {
                "predicted_label": "Error",
                "confidence_level": "Low",
                "short_reason": f"API Error: {str(e)}",
                "needs_clarification": True
            }

    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the LLM over the entire pilot benchmark and saves the results."""
        print(f"Loading benchmark data from {input_csv}...")
        df = pd.read_csv(input_csv)
        
        results = []
        
        print(f"Routing {len(df)} requests. This will take a few minutes depending on your hardware...")
        
        for index, row in df.iterrows():
            prompt_text = row['prompt']
            domain = row['domain']
            
            print(f"Processing [{index+1}/{len(df)}]: {prompt_text[:40]}...")
            
            # Ask the LLM to route it
            llm_output = self.route_request(prompt_text, domain)
            
            # Combine original row data with LLM predictions
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
        print(f"\\nDone! Results saved to {output_csv}")
        
        # Calculate a quick accuracy metric
        correct = (results_df['gold_label'] == results_df['predicted_label']).sum()
        total = len(results_df)
        print(f"Rough Accuracy: {correct}/{total} ({(correct/total)*100:.1f}%)")

# ==========================================
# Run the evaluation
# ==========================================
if __name__ == "__main__":
    # 1. Calculate absolute paths FIRST to avoid FileNotFoundError
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Notice we only go up one level (..) from 'src' to get to the 'Data' folder
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_v1.json"))
    benchmark_path = os.path.abspath(os.path.join(script_dir, "../Data/v0_pilot_benchmark.csv"))
    output_path = os.path.abspath(os.path.join(script_dir, "../Data/v1_llm_results.csv"))
    
    print(f"Using taxonomy file at: {taxonomy_path}")
    
    # 2. Initialize the router WITH the correct path
    router = LLMRouterV1(model_name='llama3', taxonomy_path=taxonomy_path)
    
    # 3. Run the benchmark
    router.evaluate_benchmark(benchmark_path, output_path)