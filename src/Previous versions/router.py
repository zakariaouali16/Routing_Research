import json

def route_query(user_prompt, model_name, taxonomy_context):
    """
    Model-agnostic routing function for the triage agent.
    
    Args:
        user_prompt (str): The input query from the student or patient.
        model_name (str): Identifier for the model (e.g., 'baseline_tfidf', 'ollama/llama3.2', 'gemini_flash').
        taxonomy_context (str): The loaded JSON string containing label definitions and boundary rules.
        
    Returns:
        dict: A dictionary matching the required Output Schema.
    """
    
    # This is the master prompt that guides the agent's behavior
    system_instruction = f"""
    You are an expert routing and triage agent.
    Your task is to classify the user query into the appropriate category based ONLY on the following taxonomy:
    {taxonomy_context}
    
    You must return a valid JSON object with the following keys:
    - "reasoning" (str)
    - "predicted_label" (str or "None")
    - "confidence_level" ("High", "Medium", "Low")
    - "clarification_needed" (bool)
    - "clarifying_question" (str or null)
    """

    # ---------------------------------------------------------
    # 1. The Week 3 Baseline (Non-LLM)
    # ---------------------------------------------------------
    if model_name == "baseline_tfidf":
        # TODO: Replace with your actual sklearn TF-IDF or embedding similarity logic
        return {
            "reasoning": "Baseline exact keyword match.",
            "predicted_label": "Scheduling & Appointments", # Dummy prediction
            "confidence_level": "Medium",
            "clarification_needed": False,
            "clarifying_question": None
        }

    # ---------------------------------------------------------
    # 2. Local Open-Source LLM (e.g., Llama 3.1, Gemma 2)
    # ---------------------------------------------------------
    elif model_name.startswith("ollama/"):
        import ollama # Make sure to pip install ollama
        
        actual_model = model_name.split("/")[1]
        
        response = ollama.chat(
            model=actual_model,
            messages=[
                {'role': 'system', 'content': system_instruction},
                {'role': 'user', 'content': user_prompt}
            ],
            format='json' # This forces the local model to output valid JSON
        )
        
        return json.loads(response['message']['content'])

    # ---------------------------------------------------------
    # 3. Free API-Based LLM (e.g., Gemini Flash)
    # ---------------------------------------------------------
    elif model_name == "gemini_flash":
        from google import genai
        from google.genai import types
        
        # Initialize client (ensure YOUR_API_KEY is in your environment variables)
        client = genai.Client() 
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"{system_instruction}\n\nUser Query: {user_prompt}",
            config=types.GenerateContentConfig(
                response_mime_type="application/json", # Strict JSON mode
            )
        )
        
        return json.loads(response.text)

    # Catch unsupported models
    else:
        raise ValueError(f"Model '{model_name}' is not supported in the pipeline.")

# ==========================================
# Example usage for testing:
# ==========================================
if __name__ == "__main__":
    # Mock taxonomy for testing
    mock_taxonomy = '{"labels": ["Scheduling", "Billing"]}'
    test_prompt = "I need to cancel my appointment."
    
    # Test the baseline
    print("Baseline Output:")
    print(route_query(test_prompt, "baseline_tfidf", mock_taxonomy))
    
    # When you are ready in Week 4, you just change the string:
    # print(route_query(test_prompt, "ollama/llama3.2", mock_taxonomy))