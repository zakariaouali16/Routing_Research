import os
import json
import pandas as pd
import requests
from tqdm import tqdm

class LLMRouterV1:
    def __init__(
        self,
        model_name='llama3',
        taxonomy_path='../../Data/taxonomy_phase3_1.json',
        use_gate=True,       # ← ablation flag: set False to disable the gate
        use_slots=True       # ← ablation flag: set False to disable slot checking
    ):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"
        self.use_gate = use_gate
        self.use_slots = use_slots

        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

        # Log the active configuration so results are traceable
        print(f"  Config: use_gate={self.use_gate}, use_slots={self.use_slots}")

    def build_system_prompt(self):
        """Constructs the routing prompt. Slot instructions included only if use_slots=True."""
        all_labels_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            all_labels_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for l in domain_data['labels']:
                if self.use_slots:
                    slots = ", ".join(l.get('required_slots', [])) if l.get('required_slots') else "None"
                    all_labels_text += f"- {l['name']}: {l['definition']} (Required slots: {slots})\n"
                else:
                    # No slot info — model just sees the label and definition
                    all_labels_text += f"- {l['name']}: {l['definition']}\n"

        if self.use_slots:
            slot_instruction = """CRITICAL INSTRUCTION FOR MISSING INFORMATION:
1. First, determine the best-fit category for the user's request.
2. Check the "Required slots" listed for that category.
3. If the user's request DOES NOT contain the information for ALL required slots, you MUST set the predicted_label to "Clarification Needed".
4. Output your response ONLY as a valid JSON object with these exact keys:
{
    "predicted_label": "The EXACT name of the category (must be 'Clarification Needed' if any slots are missing)",
    "confidence_level": "High, Medium, or Low",
    "missing_slots": ["List the specific required slots that were missing. Leave empty [] if all are present"],
    "short_reason": "Brief explanation of why you chose this label, mentioning missing info if applicable."
}"""
        else:
            slot_instruction = """CRITICAL INSTRUCTION:
1. Classify the user's request into EXACTLY ONE of the categories above.
2. Output your response ONLY as a valid JSON object with these exact keys:
{
    "predicted_label": "The EXACT name of the category",
    "confidence_level": "High, Medium, or Low",
    "missing_slots": [],
    "short_reason": "Brief explanation of why you chose this label."
}"""

        system_prompt = f"""You are an expert, autonomous routing agent.
Your task is to classify the user's request into EXACTLY ONE of the following routing categories:
{all_labels_text}

{slot_instruction}"""

        return system_prompt

    def check_gate(self, user_prompt):
        """
        Step A — Gate.
        Decides whether the prompt has enough information to route confidently.
        Returns True (enough info) or False (not enough, return Clarification Needed).
        Only called when use_gate=True.
        """
        definitions_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            definitions_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for label in domain_data['labels']:
                definitions_text += f"- {label['name']}: {label['definition']}\n"

        gate_prompt = f"""You are a gating agent for a routing system.
Your job is to decide if the user request contains enough information to confidently match it to exactly one of the labels below.

ROUTING LABELS:
{definitions_text}

Answer "false" (not enough information) when ANY of these conditions are met:
- The request is so vague that no subject or intent can be identified
  (e.g. "I need help", "something is wrong", "can you check this")
- The intent is completely unclear even after reading the full message
- The intent is clear but the request is missing a specific referent
  needed to route it to exactly one label
  (e.g. "Is my prescription ready?" — intent is clear but which prescription,
  which patient, which pharmacy is unknown;
  "What's on the test?" — intent is clear but which exam, which course is unknown;
  "Where are you located?" — intent is clear but which clinic or location is unknown)

Answer "true" (enough information) when:
- The request clearly matches one label based on its definition, even if minor details are missing
- The subject and intent are both clear enough to commit to exactly one label without guessing

Respond ONLY as a valid JSON object:
{{
    "has_enough_info": true or false,
    "reason": "One short sentence explaining why."
}}

USER REQUEST: "{user_prompt}"
"""

        payload = {
            "model": self.model_name,
            "prompt": gate_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42}
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = json.loads(response.json().get("response", "{}"))
            return result.get("has_enough_info", True)
        except Exception as e:
            tqdm.write(f"Gate error for prompt '{user_prompt[:40]}...' -> {e}")
            return True  # fail open

    def route_request(self, user_prompt):
        """
        Routes a user prompt.
        - If use_gate=True: runs gate first, returns CN if gate fails
        - If use_slots=True: system prompt includes slot checking instructions
        """

        # ── STEP A: Gate (skipped if use_gate=False) ──────────────────────
        if self.use_gate:
            has_enough_info = self.check_gate(user_prompt)
            if not has_enough_info:
                return {
                    "initial_label": "Clarification Needed",
                    "predicted_label": "Clarification Needed",
                    "confidence_level": "High",
                    "missing_slots": [],
                    "short_reason": "Request lacks sufficient information to pass the initial Gate.",
                    "was_corrected": False,
                    "qa_reason": "Blocked by gate."
                }

        # ── STEP B: Route ──────────────────────────────────────────────────
        system_prompt = self.build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUSER REQUEST:\n\"{user_prompt}\""

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "seed": 42}
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            initial_output = json.loads(response.json().get("response", "{}"))
        except Exception as e:
            tqdm.write(f"Router error for prompt '{user_prompt[:40]}...' -> {e}")
            return {
                "initial_label": "Error",
                "predicted_label": "Error",
                "confidence_level": "Low",
                "missing_slots": [],
                "short_reason": f"API Error: {str(e)}",
                "was_corrected": False,
                "qa_reason": "N/A"
            }

        initial_label = initial_output.get("predicted_label", "")
        missing_slots = initial_output.get("missing_slots", [])

        return {
            "initial_label": initial_label,
            "predicted_label": initial_label,
            "confidence_level": initial_output.get("confidence_level", "Unknown"),
            "missing_slots": missing_slots,
            "short_reason": initial_output.get("short_reason", ""),
            "was_corrected": False,
            "qa_reason": ""
        }

    def evaluate_benchmark(self, input_csv, output_csv):
        """Runs the router over the full benchmark and saves results."""
        print(f"Loading benchmark from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []

        print(f"Routing {len(df)} requests...\n")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="prompt"):
            prompt_text = row['user_prompt']
            domain = row['domain']

            llm_output = self.route_request(prompt_text)

            results.append({
                "prompt_id": row.get('prompt_id', f"ID-{index}"),
                "domain": domain,
                "user_prompt": prompt_text,
                "gold_label": row.get('gold_label', ''),
                "final_predicted_label": llm_output.get("predicted_label", ""),
                "missing_slots": ", ".join(llm_output.get("missing_slots", [])),
                "confidence_level": llm_output.get("confidence_level", ""),
                "short_reason": llm_output.get("short_reason", ""),
                "qa_reason": llm_output.get("qa_reason", "")
            })

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_csv, index=False)
        print(f"\nResults saved to {output_csv}")

        correct = (results_df['gold_label'] == results_df['final_predicted_label']).sum()
        total = len(results_df)
        print(f"Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")

        try:
            from sklearn.metrics import classification_report
            print(classification_report(
                results_df['gold_label'],
                results_df['final_predicted_label'],
                zero_division=0
            ))
        except ImportError:
            pass


# ── Interactive mode ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    taxonomy_path = os.path.abspath(os.path.join(script_dir, "../Data/taxonomy_phase3_1.json"))

    if not os.path.exists(taxonomy_path):
        taxonomy_path = "taxonomy_phase3_1.json"

    router = LLMRouterV1(taxonomy_path=taxonomy_path, use_gate=True, use_slots=True)

    print("\n" + "="*50)
    print("LLM Router — Gate + Slot Checking")
    print("Type 'quit' to exit.")
    print("="*50 + "\n")

    while True:
        user_input = input("Enter your request: ")
        if user_input.strip().lower() in ['quit', 'exit']:
            break
        if not user_input.strip():
            continue

        result = router.route_request(user_input)

        print("\n--- Prediction ---")
        print(f"Predicted Label: {result['predicted_label']}")
        print(f"Confidence:      {result['confidence_level']}")
        print(f"Reasoning:       {result['short_reason']}")

        if result.get('missing_slots'):
            print(f"\n⚠️ ACTION REQUIRED: Missing Information ⚠️")
            print(f"Please provide the following to proceed: {', '.join(result['missing_slots'])}")

        print("-" * 25 + "\n")
