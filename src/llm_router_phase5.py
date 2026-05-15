import os
import json
import pandas as pd
import requests
from tqdm import tqdm


class LLMRouterV2:

    # ── Safety keywords for pre-gate override ─────────────────────────────────
    SAFETY_KEYWORDS = [
        # suicidal ideation
        "suicide", "kill myself", "end my life", "want to die",
        "no point in going on", "there's no point anymore", "can't go on",
        "don't want to be here anymore", "thinking about ending it",
        "feel like ending it",
        # self-harm
        "self-harm", "self harm", "cutting myself", "hurt myself",
        "going to hurt myself", "harming myself", "thoughts of hurting",
        "telling me to hurt",
        # cardiac / stroke signals
        "chest is tight", "chest feels tight", "chest feels like",
        "chest pain", "crushing chest pain",
        "arm is numb", "having a stroke", "face is drooping",
        "severe chest pain", "shortness of breath",
        # breathing
        "can't breathe", "cannot breathe", "can barely breathe",
        "not breathing", "stopped breathing",
        # throat / allergic reaction
        "throat is swelling", "throat feels like it's closing",
        "throat feels like it's swelling shut", "allergic reaction",
        "throat is starting to swell",
        # unconscious / unresponsive
        "passed out", "fainted", "collapsed", "unresponsive",
        "won't wake up", "not waking up", "collapsed on the floor",
        "not responding", "isn't waking up",
        # bleeding
        "bleed", "won't stop bleeding", "severe bleeding"
        "coughing up blood", "bleeding won't stop",
        "soaking through the towels",
        # medication / poisoning
        "overdose", "took too many pills", "took too much",
        "took more than", "took more medication", "accidentally took",
        "swallowed a whole", "swallowed something",
        "drank a cleaning chemical", "drank half a bottle",
        "feel strange", "feel weird",
        "feel sick", "took the wrong",
        "double my dose",
        # vision / neurological
        "lost all vision", "lost vision", "vision is going dark",
        "losing consciousness", "seizure", "having a seizure",
        # choking
        "choking", "making a choking sound",
        # acute physical distress
        "skin is peeling off", "thunderclap headache",
        "shaking and can't stop", "high fever", "burning up",
        "fever won't break", "hit head", "broken bone", "deep cut",
        # obstetric emergency
        "water just broke", "water broke",
        # psychiatric emergency
        "voices telling me to hurt", "can't fight them off",
        # injury / trauma
        "cannot feel my legs", "can't feel my legs",
        "fell off the roof", "fell down",
    ]

    def __init__(self, model_name='llama3', taxonomy_path='../../Data/taxonomy_phase3_1.json'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, 'r') as f:
            self.taxonomy = json.load(f)

    # ── Pre-gate safety check ──────────────────────────────────────────────────
    def _check_safety_override(self, user_prompt):
        """
        Deterministic keyword scan run BEFORE the gate or router.
        Returns True if the prompt contains an obvious crisis/emergency signal.

        Catches clear cases so they can never be mis-routed to Clarification Needed.
        Subtle crisis language is handled downstream by the Urgent Escalation taxonomy label.
        """
        prompt_lower = user_prompt.lower()
        for keyword in self.SAFETY_KEYWORDS:
            if keyword in prompt_lower:
                return True
        return False

    def build_system_prompt(self):
        """Constructs the routing prompt using all domains and labels from the taxonomy."""
        all_labels_text = ""
        for domain_name, domain_data in self.taxonomy['domains'].items():
            all_labels_text += f"\n### {domain_name.upper()} DOMAIN ###\n"
            for l in domain_data['labels']:
                slots = ", ".join(l.get('required_slots', [])) if l.get('required_slots') else "None"
                all_labels_text += f"- {l['name']}: {l['definition']} (Required slots: {slots})\n"

        system_prompt = f"""You are an expert, autonomous routing agent.
Your task is to classify the user's request into EXACTLY ONE of the following routing categories:
{all_labels_text}

CRITICAL INSTRUCTION FOR MISSING INFORMATION:
1. First, determine the best-fit category for the user's request.
2. Check the "Required slots" listed for that category.
3. If the user's request DOES NOT contain the information for ALL required slots, you MUST set the predicted_label to "Clarification Needed".
4. Output your response ONLY as a valid JSON object with these exact keys:
{{
    "predicted_label": "The EXACT name of the category (must be 'Clarification Needed' if any slots are missing)",
    "confidence_level": "High, Medium, or Low",
    "missing_slots": ["List the specific required slots that were missing. Leave empty [] if all are present"],
    "short_reason": "Brief explanation of why you chose this label, mentioning missing info if applicable."
}}"""

        return system_prompt

    def check_gate(self, user_prompt):
        """
        Step A — Gate.
        Decides whether the prompt has enough information to route confidently.
        Returns True (enough info, proceed to routing) or False (not enough, return Clarification Needed).
        Defaults to True if the gate itself fails, so no prompt is silently dropped.
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
            "options": {
                "temperature": 0.0,
                "seed": 42
            }
        }

        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            result = json.loads(response.json().get("response", "{}"))
            return result.get("has_enough_info", True)
        except Exception as e:
            tqdm.write(f"Gate error for prompt '{user_prompt[:40]}...' -> {e}")
            return True  # fail open — let it through to the router

    def route_request(self, user_prompt):
        """
        Three-step routing:
          Safety Override — pre-gate keyword check for obvious crisis signals
          Step A — Gate: check if enough information is present
          Step B — Route: only if gate passes, pick a label, check slots.
        """

        # ── SAFETY OVERRIDE (pre-gate) ─────────────────────────────────────────
        # Runs before anything else. If an obvious crisis/emergency keyword is
        # detected, hard-route to Urgent Escalation immediately, bypassing
        # both the gate and the LLM router. This prevents self-harm or emergency
        # prompts from ever being returned as Clarification Needed.
        if self._check_safety_override(user_prompt):
            return {
                "initial_label": "Urgent Escalation",
                "predicted_label": "Urgent Escalation",
                "confidence_level": "High",
                "missing_slots": [],
                "short_reason": "Safety override: message contains an obvious crisis or emergency signal.",
                "was_corrected": True,
                "qa_reason": "Pre-gate safety keyword match — bypassed gate and router."
            }

        # ── STEP A: Gate ───────────────────────────────────────────────────────
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

        # ── STEP B: Route ──────────────────────────────────────────────────────
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

    router = LLMRouterV2(taxonomy_path=taxonomy_path)

    print("\n" + "=" * 50)
    print("LLM Router V2 — Safety Override + Gate + Route")
    print("Type 'quit' to exit.")
    print("=" * 50 + "\n")

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