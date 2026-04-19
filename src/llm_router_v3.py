import os
import json
import pandas as pd
import requests
from tqdm import tqdm
from sklearn.metrics import classification_report


class LLMRouterV3:
    def __init__(self, model_name="llama3", taxonomy_path=None):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

        if taxonomy_path is None:
            raise ValueError("Please provide taxonomy_path explicitly when running in Colab.")

        print(f"Loading taxonomy from {taxonomy_path}...")
        with open(taxonomy_path, "r") as f:
            self.taxonomy = json.load(f)

        self.non_routing_outcomes = self.taxonomy["non_routing_outcomes"]["outcomes"]
        self.precedence_rules = self.taxonomy["precedence_rules"]
        self.annotation_rules = self.taxonomy["annotation_rules"]

    def _get_all_routing_labels(self):
        labels = []
        for domain_key in self.taxonomy["domains"]:
            labels.extend(self.taxonomy["domains"][domain_key]["labels"])
        return labels

    def _get_allowed_labels(self):
        non_routing_labels = [x["name"] for x in self.non_routing_outcomes]
        routing_labels = [x["name"] for x in self._get_all_routing_labels()]
        return non_routing_labels + routing_labels

    def _format_non_routing_outcomes(self):
        lines = []
        for outcome in self.non_routing_outcomes:
            required = "; ".join(outcome.get("required_slots", []))
            lines.append(
                f"- {outcome['name']} ({outcome['group']}): {outcome['definition']} "
                f"Required slots: {required}"
            )
        return "\n".join(lines)

    def _format_all_routing_labels(self):
        lines = []
        for domain_key, domain_data in self.taxonomy["domains"].items():
            lines.append(f"{domain_key.upper()} LABELS:")
            for label in domain_data["labels"]:
                required = "; ".join(label.get("required_slots", []))
                lines.append(
                    f"- {label['name']}: {label['definition']} "
                    f"Required slots: {required}"
                )
        return "\n".join(lines)

    def build_system_prompt(self):
        non_routing_text = self._format_non_routing_outcomes()
        routing_text = self._format_all_routing_labels()
        allowed_labels = self._get_allowed_labels()

        allowed_labels_text = "\n".join([f"- {x}" for x in allowed_labels])
        precedence_text = "\n".join([f"- {x}" for x in self.precedence_rules])

        prompt = f"""You are a strict routing agent.

Your job is to assign EXACTLY ONE final outcome for the user message.

IMPORTANT:
- You are NOT given the domain in advance.
- You must infer the correct outcome directly from the user message.
- Use the taxonomy definitions exactly.
- Follow the precedence rules strictly.
- Safety override outcomes always beat gating outcomes and routing labels.
- Clarification Needed is a gating outcome, not a normal routing label.
- Mixed Intent is a gating outcome used only when the prompt contains two or more distinct separable requests.
- Out-of-Scope is a last resort only after ruling out all other outcomes.

PRECEDENCE RULES:
{precedence_text}

NON-ROUTING OUTCOMES:
{non_routing_text}

ALL ROUTING LABELS ACROSS BOTH DOMAINS:
{routing_text}

ALLOWED FINAL LABELS:
{allowed_labels_text}

Return ONLY valid JSON with these exact keys:
{{
  "predicted_label": "exact allowed label",
  "needs_clarification": true or false,
  "short_reason": "one short sentence",
  "confidence_level": "High" or "Medium" or "Low"
}}

Consistency rules:
- If predicted_label is "Clarification Needed", then needs_clarification must be true.
- If predicted_label is not "Clarification Needed", then needs_clarification should usually be false.
- Do not invent labels.
- Do not output markdown or extra text.
"""
        return prompt

    def verify_prediction(self, user_prompt, initial_output):
        allowed_labels = self._get_allowed_labels()
        allowed_labels_text = "\n".join([f"- {x}" for x in allowed_labels])
        precedence_text = "\n".join([f"- {x}" for x in self.precedence_rules])

        verification_prompt = f"""You are a strict QA verifier for a routing decision.

USER MESSAGE:
{user_prompt}

INITIAL DECISION:
{json.dumps(initial_output, ensure_ascii=False)}

ALLOWED LABELS:
{allowed_labels_text}

PRECEDENCE RULES:
{precedence_text}

Check whether the initial decision obeys the taxonomy and precedence rules.
If needed, correct the label and clarification flag.

Return ONLY valid JSON with these exact keys:
{{
  "is_correct": true or false,
  "verified_label": "exact allowed label",
  "verified_needs_clarification": true or false,
  "qa_reason": "one short sentence"
}}

Rules:
- If verified_label is "Clarification Needed", verified_needs_clarification must be true.
- Do not invent labels.
- Do not output markdown or extra text.
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
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            parsed = json.loads(response.json().get("response", "{}"))

            verified_label = parsed.get("verified_label", initial_output["predicted_label"])
            if verified_label not in allowed_labels:
                verified_label = initial_output["predicted_label"]

            verified_needs_clarification = parsed.get(
                "verified_needs_clarification",
                initial_output["needs_clarification"]
            )

            if verified_label == "Clarification Needed":
                verified_needs_clarification = True

            return {
                "is_correct": parsed.get("is_correct", True),
                "verified_label": verified_label,
                "verified_needs_clarification": verified_needs_clarification,
                "qa_reason": parsed.get("qa_reason", "")
            }

        except Exception as e:
            tqdm.write(f"Verification error -> {e}")
            return {
                "is_correct": True,
                "verified_label": initial_output["predicted_label"],
                "verified_needs_clarification": initial_output["needs_clarification"],
                "qa_reason": "Verification failed; defaulted to initial output."
            }

    def route_request(self, user_prompt):
        system_prompt = self.build_system_prompt()
        full_prompt = f"{system_prompt}\n\nUSER MESSAGE:\n{user_prompt}"

        allowed_labels = self._get_allowed_labels()

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
            response = requests.post(self.api_url, json=payload, timeout=120)
            response.raise_for_status()
            initial_output = json.loads(response.json().get("response", "{}"))
        except Exception as e:
            tqdm.write(f"Generation error for prompt '{user_prompt[:40]}...' -> {e}")
            return {
                "initial_label": "Error",
                "predicted_label": "Error",
                "confidence_level": "Low",
                "needs_clarification": True,
                "short_reason": f"API error: {str(e)}",
                "was_corrected": False,
                "qa_reason": "N/A"
            }

        initial_label = initial_output.get("predicted_label", "Clarification Needed")
        if initial_label not in allowed_labels:
            initial_label = "Clarification Needed"

        initial_needs_clarification = initial_output.get("needs_clarification", False)
        if initial_label == "Clarification Needed":
            initial_needs_clarification = True

        normalized_initial = {
            "predicted_label": initial_label,
            "needs_clarification": initial_needs_clarification,
            "short_reason": initial_output.get("short_reason", ""),
            "confidence_level": initial_output.get("confidence_level", "Low")
        }

        qa_output = self.verify_prediction(user_prompt, normalized_initial)

        final_label = qa_output["verified_label"]
        final_needs_clarification = qa_output["verified_needs_clarification"]

        return {
            "initial_label": normalized_initial["predicted_label"],
            "predicted_label": final_label,
            "confidence_level": normalized_initial["confidence_level"],
            "needs_clarification": final_needs_clarification,
            "short_reason": normalized_initial["short_reason"],
            "was_corrected": not qa_output.get("is_correct", True),
            "qa_reason": qa_output.get("qa_reason", "")
        }

    def evaluate_benchmark(self, input_csv, output_csv):
        print(f"Loading benchmark data from {input_csv}...")
        df = pd.read_csv(input_csv)
        results = []

        print(f"Routing {len(df)} prompts...\n")

        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing", unit="prompt"):
            prompt_text = str(row["user_prompt"]).strip('"')
            llm_output = self.route_request(prompt_text)

            result_row = {
                "prompt_id": row.get("prompt_id", f"ID-{index}"),
                "domain": row.get("domain", ""),
                "user_prompt": prompt_text,
                "gold_outcome": row.get("gold_outcome", ""),
                "initial_predicted_label": llm_output.get("initial_label", ""),
                "final_predicted_label": llm_output.get("predicted_label", ""),
                "confidence_level": llm_output.get("confidence_level", ""),
                "needs_clarification_pred": llm_output.get("needs_clarification", False),
                "short_reason": llm_output.get("short_reason", ""),
                "was_corrected_by_qa": llm_output.get("was_corrected", False),
                "qa_reason": llm_output.get("qa_reason", "")
            }
            results.append(result_row)

        results_df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)
        print(f"\nDone! Results saved to {output_csv}\n")

        correct = (results_df["gold_outcome"] == results_df["final_predicted_label"]).sum()
        total = len(results_df)

        print("=" * 60)
        print(f"OVERALL ACCURACY: {correct}/{total} ({(correct / total) * 100:.2f}%)")
        print("=" * 60)

        corrections = results_df["was_corrected_by_qa"].sum()
        print(f"QA corrections: {corrections}/{total}")

        print("\n--- Accuracy by domain (analysis only) ---")
        for dom in results_df["domain"].fillna("missing").unique():
            dom_df = results_df[results_df["domain"].fillna("missing") == dom]
            d_correct = (dom_df["gold_outcome"] == dom_df["final_predicted_label"]).sum()
            d_total = len(dom_df)
            print(f"{dom}: {d_correct}/{d_total} ({(d_correct/d_total)*100:.2f}%)")

        print("\n--- Classification report ---")
        print(
            classification_report(
                results_df["gold_outcome"],
                results_df["final_predicted_label"],
                zero_division=0
            )
        )


if __name__ == "__main__":
    taxonomy_path = "/content/drive/MyDrive/Routing_Research/Data/taxonomy_v4_1.json"
    benchmark_path = "/content/drive/MyDrive/Routing_Research/Data/testing_benchmark_2.0.csv"
    output_path = "/content/drive/MyDrive/Routing_Research/results/phase_3_test_trials/llm_results_v3.csv"

    router = LLMRouterV3(model_name="llama3", taxonomy_path=taxonomy_path)
    router.evaluate_benchmark(benchmark_path, output_path)