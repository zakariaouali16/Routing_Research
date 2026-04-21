import os
import json
from transformers import pipeline

class EmbeddingRouter:
    def __init__(self, model_name="facebook/bart-large-mnli"):
        print(f"Loading Hugging Face Zero-Shot model: {model_name}...")
        
        # Initialize the Hugging Face zero-shot classification pipeline
        # You can pass device=0 if you have a GPU available
        self.classifier = pipeline("zero-shot-classification", model=model_name)
        self.labels = []
        
        # Dynamically resolve the path to taxonomy_v2.json
        script_dir = os.path.dirname(os.path.abspath(__file__))
        taxonomy_path = os.path.abspath(os.path.join(script_dir, "../../Data/taxonomy_v2.json"))
        
        self._load_taxonomy(taxonomy_path)

    def _load_taxonomy(self, taxonomy_path):
        """Loads candidate labels from the taxonomy for Zero-Shot classification."""
        print(f"Loading taxonomy from {taxonomy_path}...")
        if not os.path.exists(taxonomy_path):
            raise FileNotFoundError(f"Could not find taxonomy at {taxonomy_path}.")

        with open(taxonomy_path, 'r') as f:
            taxonomy = json.load(f)
            
        for domain, domain_data in taxonomy['domains'].items():
            for label in domain_data['labels']:
                # Note: We are only storing the label names, not the descriptions
                self.labels.append(label['name'])
                
        print(f"Loaded {len(self.labels)} candidate labels for Zero-Shot routing.")

    def fit(self, csv_path):
        """
        Dummy fit method to maintain compatibility with compare_routers_v2_3.py.
        """
        print(f"Note: HF Zero-Shot ignores training data at {csv_path}.")
        pass

    def route_request(self, new_prompt):
        """
        Routes a request by evaluating the prompt against all taxonomy labels 
        using Natural Language Inference (NLI).
        """
        if not self.labels:
            raise ValueError("Taxonomy must be loaded before routing requests.")

        # Run HF Zero-Shot Classification
        # multi_label=False assumes the prompt belongs to exactly one category
        result = self.classifier(new_prompt, self.labels, multi_label=False)

        # The pipeline returns sorted lists of 'labels' and 'scores'
        top_label = result['labels'][0]
        top_score = result['scores'][0]
        
        return {
            "predicted_label": top_label,
            "confidence_score": round(float(top_score), 4)
        }

# ==========================================
# Quick Local Testing
# ==========================================
if __name__ == "__main__":
    router = EmbeddingRouter()
    router.fit("dummy_path.csv")
    
    test_prompt = "Can you help me figure out why my python code keeps throwing an index out of bounds error even after updating it?"
    print(f"\nIncoming Request: '{test_prompt}'")

    result = router.route_request(test_prompt)
    print(f"Predicted Label: {result['predicted_label']}")
    print(f"Confidence Score: {result['confidence_score']}")