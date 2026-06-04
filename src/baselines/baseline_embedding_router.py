import os
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class EmbeddingRouter:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.labels = []
        self.label_embeddings = None
        
        # Dynamically resolve the path to taxonomy_phase5.json
        # Assuming this script is in src/baselines/ and taxonomy is in Data/
        script_dir = os.path.dirname(os.path.abspath(__file__))
        taxonomy_path = os.path.abspath(os.path.join(script_dir, "../../Data/taxonomy_phase5.json"))
        
        self._load_and_embed_taxonomy(taxonomy_path)

    def _load_and_embed_taxonomy(self, taxonomy_path):
        """Embeds the rich definitions from the taxonomy for Zero-Shot routing."""
        print(f"Loading taxonomy from {taxonomy_path}...")
        if not os.path.exists(taxonomy_path):
            raise FileNotFoundError(f"Could not find taxonomy at {taxonomy_path}. Please check your paths.")

        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
            
        descriptions = []
        
        for domain, domain_data in taxonomy['domains'].items():
            for label in domain_data['labels']:
                self.labels.append(label['name'])
                # Combine label name and definition for a richer semantic representation
                text_to_embed = f"Category: {label['name']}. Description: {label['definition']}"
                descriptions.append(text_to_embed)
                
        print(f"Embedding {len(self.labels)} taxonomy definitions...")
        self.label_embeddings = self.model.encode(descriptions)
        print("Done! The Zero-Shot Taxonomy Router is ready.")

    def fit(self, csv_path):
        """
        Kept for backwards compatibility with compare_routers_v2_3.py.
        Instead of crashing because of the tiny training CSV, it safely ignores it
        and relies on the taxonomy embeddings loaded in __init__.
        """
        print(f"Note: Ignoring sparse training data at {csv_path}. Using Zero-Shot Taxonomy instead!")
        pass

    def route_request(self, new_prompt):
        """
        Embeds a new request and finds the most semantically similar taxonomy definition.
        """
        if self.label_embeddings is None:
            raise ValueError("Taxonomy must be loaded before routing requests.")

        # 1. Embed the incoming request
        new_embedding = self.model.encode([new_prompt])

        # 2. Compare against all known taxonomy definitions
        cosine_scores = cosine_similarity(new_embedding, self.label_embeddings)[0]

        # 3. Find the single best match
        best_match_idx = int(np.argmax(cosine_scores))
        
        # Format exactly like your previous router so it plugs perfectly into your compare script
        return {
            "predicted_label": self.labels[best_match_idx],
            "confidence_score": round(float(cosine_scores[best_match_idx]), 4)
        }

# ==========================================
# Quick Local Testing
# ==========================================
if __name__ == "__main__":
    # Initialize the router
    router = EmbeddingRouter()

    # The fit method is now a dummy method, but we can still call it to simulate the pipeline
    router.fit("dummy_path.csv")

    # Test it with a new, unseen request
    test_prompt = "Can you help me figure out why my python code keeps throwing an index out of bounds error even after updating it?"
    
    print(f"\nIncoming Request: '{test_prompt}'")

    # Route
    result = router.route_request(test_prompt)
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence Score: {result['confidence_score']}")