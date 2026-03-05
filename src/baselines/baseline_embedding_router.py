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
        df = pd.read_csv(csv_path)
        
        # Ensure the columns exist
        if 'prompt' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV must contain 'prompt' and 'label' columns.")
        
        self.reference_prompts = df['prompt'].tolist()
        self.reference_labels = df['label'].tolist()
        
        print(f"Embedding {len(self.reference_prompts)} prompts. This might take a moment...")
        self.reference_embeddings = self.model.encode(self.reference_prompts)
        print("Done! The router is ready.")

    def route_request(self, new_prompt, threshold=0.5):
        """
        Embeds a new request, finds the most similar benchmark prompt,
        and returns the label and confidence score.
        """
        if self.reference_embeddings is None:
            raise RuntimeError("You must call .fit() with your data before routing.")

        # 1. Embed the incoming user request
        new_embedding = self.model.encode([new_prompt])

        # 2. Calculate cosine similarity against all benchmark embeddings
        similarities = cosine_similarity(new_embedding, self.reference_embeddings)[0]

        # 3. Find the index of the highest similarity score
        best_match_idx = np.argmax(similarities)
        best_score = similarities[best_match_idx]
        predicted_label = self.reference_labels[best_match_idx]

        # 4. Apply a basic uncertainty rule
        needs_clarification = bool(best_score < threshold)

        return {
            "predicted_label": predicted_label if not needs_clarification else "needs_clarification",
            "confidence_score": round(float(best_score), 4),
            "matched_example": self.reference_prompts[best_match_idx],
            "needs_clarification": needs_clarification
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
    csv_path = os.path.join(script_dir, "../../data/v0_pilot_benchmark.csv")
    
    # Load your REAL labeled spreadsheet using the dynamic path
    router.fit(csv_path) 

    # Test it with a new, unseen request
    test_prompt = "Can you help me figure out why my python code keeps throwing an index out of bounds error?"
    
    print(f"\nIncoming Request: '{test_prompt}'")
    
    # Route it!
    result = router.route_request(test_prompt, threshold=0.4)
    
    print("\nRouting Decision:")
    for key, value in result.items():
        print(f"- {key}: {value}")