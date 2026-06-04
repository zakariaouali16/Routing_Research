"""
Baseline Embedding Router
"""

import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingRouter:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.reference_embeddings = None
        self.reference_prompts = None
        self.reference_labels = None

    def fit(self, csv_path=None, df=None):
        """Load benchmark prompts and embed the benchmark prompts."""
        if df is None:
            if csv_path is None:
                raise ValueError("Must provide either csv_path or df.")
            df = pd.read_csv(csv_path)

        if "user_prompt" not in df.columns or "gold_outcome" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'user_prompt' and 'gold_outcome' columns."
            )

        self.reference_prompts = df["user_prompt"].astype(str).str.strip('"').tolist()
        self.reference_labels = df["gold_outcome"].tolist()

        self.reference_embeddings = self.model.encode(
            self.reference_prompts,
            show_progress_bar=False
        )

    def route_request(self, new_prompt, threshold=0.5):
        """
        Route a single prompt by nearest-neighbor embedding similarity.

        Clarification policy:
        - If best similarity is below threshold -> Clarification Needed
        - If nearest label is already Clarification Needed -> Clarification Needed

        Returns:
            predicted_label
            confidence_score
            matched_example
            needs_clarification
        """
        if self.reference_embeddings is None:
            raise ValueError("Must call .fit() before .route_request().")

        new_embedding = self.model.encode([new_prompt], show_progress_bar=False)
        cosine_scores = cosine_similarity(new_embedding, self.reference_embeddings)[0]

        best_match_idx = int(np.argmax(cosine_scores))
        best_score = float(cosine_scores[best_match_idx])
        nearest_label = self.reference_labels[best_match_idx]
        matched_example = self.reference_prompts[best_match_idx]

        needs_clarification = (
            best_score < threshold or nearest_label == "Clarification Needed"
        )

        predicted_label = (
            "Clarification Needed" if needs_clarification else nearest_label
        )

        return {
            "predicted_label": predicted_label,
            "confidence_score": round(best_score, 4),
            "matched_example": matched_example,
            "needs_clarification": needs_clarification,
        }


if __name__ == "__main__":
    router = EmbeddingRouter()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "..", "..", "Data", "testing_benchmark_2.0.csv")

    if not os.path.exists(csv_path):
        csv_path = os.path.join(script_dir, "testing_benchmark_2.0.csv")

    router.fit(csv_path=csv_path)

    test_prompt = (
        "Can you help me figure out why my python code keeps throwing "
        "an index out of bounds error?"
    )

    print(f"\nIncoming Request: '{test_prompt}'")
    result = router.route_request(test_prompt, threshold=0.5)

    print("\nRouting Decision:")
    for key, value in result.items():
        print(f"  {key}: {value}")