import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRouter:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.reference_embeddings = None
        self.reference_labels = None
        self.reference_prompts = None

    def fit(self, csv_path):
        """Loads the spreadsheet and embeds the benchmark prompts."""
        print(f"Loading data from {csv_path}...")
        
        # 1. Add 'utf-8-sig' encoding to safely ignore invisible BOM characters
        df = pd.read_csv(csv_path, encoding='utf-8-sig')

        # 2. Strip any hidden whitespaces from the column names
        df.columns = df.columns.str.strip()
        
        # 3. Print the detected columns to help with debugging
        print(f"Detected columns: {df.columns.tolist()}")

        if 'user_prompt' not in df.columns or 'gold_outcome' not in df.columns:
            # Inject the detected columns into the error message for easy troubleshooting
            raise ValueError(f"CSV must contain 'user_prompt' and 'gold_outcome' columns. Found: {df.columns.tolist()}")

        # Clean up any surrounding quotes from user_prompt
        df['user_prompt'] = df['user_prompt'].str.strip('"')

        self.reference_prompts = df['user_prompt'].tolist()
        self.reference_labels = df['gold_outcome'].tolist()

        # Store additional benchmark metadata for richer output
        self.reference_df = df
        self.reference_ambiguous = df['is_ambiguous'].tolist(
        ) if 'is_ambiguous' in df.columns else [False] * len(df)

        print(
            f"Embedding {len(self.reference_prompts)} prompts. This might take a moment...")
        self.reference_embeddings = self.model.encode(self.reference_prompts)
        print("Done! The router is ready.")
        
    def route_request(self, new_prompt, threshold=0.5):
        """
        Embeds a new request, finds the most similar benchmark prompt,
        and returns the predicted label along with clarification flags.
        """
        if self.reference_embeddings is None:
            raise ValueError(
                "You must call .fit() with a CSV file before routing requests.")

        # 1. Embed the incoming request
        new_embedding = self.model.encode([new_prompt])

        # 2. Compare against all known benchmark prompts
        cosine_scores = cosine_similarity(
            new_embedding, self.reference_embeddings)[0]

        # 3. Find the single best match
        best_match_idx = int(np.argmax(cosine_scores))
        best_score = cosine_scores[best_match_idx]
        predicted_label = self.reference_labels[best_match_idx]
        matched_example = self.reference_prompts[best_match_idx]

        # 4. Check the ambiguous flag of the matched example
        # Handle string 'TRUE'/'FALSE' or actual boolean True/False
        ambiguous_val = self.reference_ambiguous[best_match_idx]
        if isinstance(ambiguous_val, str):
            matched_is_ambiguous = ambiguous_val.strip().upper() == 'TRUE'
        else:
            matched_is_ambiguous = bool(ambiguous_val)

        # 5. Retrieve additional metadata from the matched benchmark row
        matched_row = self.reference_df.iloc[best_match_idx]

        def parse_bool(val):
            """Handle string 'TRUE'/'FALSE' or actual boolean."""
            if isinstance(val, str):
                return val.strip().upper() == 'TRUE'
            return bool(val)

        matched_top_level_type = matched_row.get('top_level_type', 'Unknown')
        matched_next_step = matched_row.get('next_step_type', 'Unknown')
        matched_is_escalation = parse_bool(matched_row.get('is_escalation_case', False))
        matched_is_high_risk = parse_bool(matched_row.get('is_high_risk', False))

        # 6. Apply the smarter uncertainty rule
        # Flag if: score is too low OR the matched label is "Clarification Needed" OR the match is known to be ambiguous
        needs_clarification = (
            bool(best_score < threshold) or
            predicted_label == "Clarification Needed" or
            matched_is_ambiguous
        )

        # Override metadata if clarification is triggered
        if needs_clarification:
            final_label = "Clarification Needed"
            final_top_level = "Gating Outcome"
            final_next_step = "ask_clarification"
        else:
            final_label = predicted_label
            final_top_level = matched_top_level_type
            final_next_step = matched_next_step

        return {
            "predicted_label": final_label,
            "top_level_type": final_top_level,
            "next_step_type": final_next_step,
            "confidence_score": round(float(best_score), 4),
            "matched_example": matched_example,
            "needs_clarification": needs_clarification,
            "is_escalation_case": matched_is_escalation if not needs_clarification else False,
            "is_high_risk": matched_is_high_risk if not needs_clarification else False
        }


# ==========================================
# How to use the router
# ==========================================
if __name__ == "__main__":
    # Initialize the router
    router = EmbeddingRouter()

    # Dynamically build the path to the data folder
    # 1. Get the directory where this script is located (src/baselines/)
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Go up two levels (../../) and into the data folder
    csv_path = os.path.join(script_dir, "../../Data/testing_benchmark.csv")

    # Load your labeled benchmark using the dynamic path
    router.fit(csv_path)

    # Test it with a new, unseen request
    test_prompt = "Can you help me figure out why my python code keeps throwing an index out of bounds error even after updating it?"

    print(f"\nIncoming Request: '{test_prompt}'")

    # Route it!
    result = router.route_request(test_prompt, threshold=0.4)

    print("\nRouting Decision:")
    for key, value in result.items():
        print(f"- {key}: {value}")
