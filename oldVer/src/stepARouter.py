import json
import requests

class StepARouter:
    def __init__(self, model_name='llama3'):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def build_step_a_prompt(self):
        """Focuses strictly on identifying if the request is actionable."""
        return """You are a Triage Agent. Your only job is to determine if a user's request has enough detail to be sent to a specialist.

Analyze the request for:
1. Specific Intent (What do they want?)
2. Essential Entities (Account names, dates, or specific problems).

Output ONLY a JSON object:
{
    "is_actionable": true or false,
    "intent_summary": "A 5-word summary of the request",
    "missing_details": "What is missing? (e.g., 'specific department', 'error code')",
    "clarifying_question": "A polite question to ask the user if is_actionable is false, else null"
}
"""

    def check_request(self, user_prompt):
        system_prompt = self.build_step_a_prompt()
        full_prompt = f"{system_prompt}\n\nUSER REQUEST: \"{user_prompt}\""
        
        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0} # Keep it deterministic
        }
        
        try:
            response = requests.post(self.api_url, json=payload)
            # Parse the response and return the 'is_actionable' status
            return json.loads(response.json().get("response", "{}"))
        except Exception as e:
            return {"is_actionable": False, "clarifying_question": "System error. Please retry."}