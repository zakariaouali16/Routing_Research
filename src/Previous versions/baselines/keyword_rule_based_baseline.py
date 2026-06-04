import re

def route_prompt(prompt: str) -> dict:
    """
    Baseline Keyword-Based Router for Education and Healthcare intents.
    Returns a dictionary with the predicted domain, label, and a short reason.
    """
    prompt_clean = prompt.lower()
    word_count = len(prompt_clean.split())

    # ---------------------------------------------------------
    # STEP 1: GLOBAL RISK & URGENCY CHECK (Highest Priority)
    # ---------------------------------------------------------
    
    # Healthcare Urgent Escalation
    hc_urgent_keywords = ["chest pain", "numb", "bleeding", "stroke", "end my life", "suicide", "can't breathe", "overdose", "poison", "unresponsive"]
    if any(keyword in prompt_clean for keyword in hc_urgent_keywords):
        return {
            "domain": "Healthcare",
            "label": "Urgent Escalation (Emergency Services)",
            "reason": "Matched high-risk medical emergency keyword."
        }

    # Education Instructor/TA Escalation
    edu_escalate_keywords = ["appeal", "dispute", "accommodations", "academic integrity", "family emergency", "plagiarism", "cheated", "misconduct"]
    if any(keyword in prompt_clean for keyword in edu_escalate_keywords):
        return {
            "domain": "Education",
            "label": "Instructor/TA Escalation",
            "reason": "Matched high-risk academic or sensitive keyword."
        }

    # ---------------------------------------------------------
    # STEP 2: AMBIGUITY & LENGTH CHECK
    # ---------------------------------------------------------
    
    # Catch extremely short prompts that lack context
    if word_count < 5:
        return {
            "domain": "Unknown",
            "label": "Clarification Needed",
            "reason": f"Prompt is too short ({word_count} words) to safely route."
        }
        
    # Catch vague pronouns without context
    vague_phrases = ["this works", "this count", "do this", "it is done", "is it done"]
    if any(phrase in prompt_clean for phrase in vague_phrases):
        return {
            "domain": "Unknown",
            "label": "Clarification Needed",
            "reason": "Prompt contains vague pronouns lacking clear context."
        }

    # ---------------------------------------------------------
    # STEP 3: DOMAIN ROUTING
    # ---------------------------------------------------------
    
    edu_domain_keywords = ["python", "java", "c++", "sql", "code", "algorithm", "grade", "assignment", "homework", "syllabus", "exam", "midterm", "canvas", "autograder"]
    hc_domain_keywords = ["doctor", "appointment", "insurance", "prescription", "patient", "medical", "clinic", "pharmacy", "bill", "copay"]

    edu_score = sum(1 for w in edu_domain_keywords if w in prompt_clean)
    hc_score = sum(1 for w in hc_domain_keywords if w in prompt_clean)

    # Determine domain based on keyword frequency
    if edu_score > hc_score:
        domain = "Education"
    elif hc_score > edu_score:
        domain = "Healthcare"
    else:
        # Tie or 0 score
        return {
            "domain": "Unknown",
            "label": "Clarification Needed",
            "reason": "Could not determine domain; lacks clear domain keywords."
        }

    # ---------------------------------------------------------
    # STEP 4: INTENT LABELING (Within Domain)
    # ---------------------------------------------------------
    
    if domain == "Education":
        if any(k in prompt_clean for k in ["error", "throwing", "indexerror", "notfounderror", "fault", "compile", "bug", "output", "script isn't working"]):
            label = "Debugging & Code Troubleshooting"
        elif any(k in prompt_clean for k in ["deadline", "deduction", "rubric", "late", "policy", "points", "submit"]):
            label = "Assignment & Grading Policy"
        elif any(k in prompt_clean for k in ["midterm", "final exam", "cheat sheet", "practice", "study guide", "quiz"]):
            label = "Exam & Assessment Prep"
        elif any(k in prompt_clean for k in ["install", "zoom link", "ide", "autograder", "docker", "environment", "office hours"]):
            label = "Course Logistics & Environment Setup"
        elif any(k in prompt_clean for k in ["explain", "difference between", "theory", "concept", "what is"]):
            label = "Concept Explanation"
        else:
            label = "Clarification Needed"

    elif domain == "Healthcare":
        if any(k in prompt_clean for k in ["records", "grievance", "hipaa", "proxy", "fmla", "lawyer", "note"]):
            label = "Human Staff Review Needed"
        elif any(k in prompt_clean for k in ["refill", "pharmacy", "generic", "dose", "medication", "supply"]):
            label = "Pharmacy & Prescription Logistics"
        elif any(k in prompt_clean for k in ["blue cross", "bill", "copay", "claim", "medicare", "cost", "afford"]):
            label = "Insurance & Billing"
        elif any(k in prompt_clean for k in ["operating hours", "parking", "visitors", "wheelchairs", "wi-fi", "cafeteria", "located"]):
            label = "Facility & General Information"
        elif any(k in prompt_clean for k in ["book", "cancel", "reschedule", "visit", "appointment", "waitlist"]):
            label = "Scheduling & Appointments"
        else:
            label = "Clarification Needed"

    return {
        "domain": domain,
        "label": label,
        "reason": "Matched intent keyword in target domain."
    }

# --- TESTING THE BASELINE AGAINST YOUR BENCHMARK ---

test_prompts = [
    "My husband is complaining of crushing chest pain and his left arm is totally numb.", # Should be Urgent Escalation [cite: 111]
    "I believe the TA unfairly deducted 20 points from my project. I want to appeal this.", # Should be Instructor Escalation [cite: 42]
    "I need an appointment.", # Should trigger < 5 words rule 
    "Can you explain the difference between a list and a tuple in Python?", # Standard Concept [cite: 1]
    "Where is the patient parking garage located for the main hospital?" # Standard Facility [cite: 82]
]

for p in test_prompts:
    result = route_prompt(p)
    print(f"Prompt: '{p}'")
    print(f"-> Domain: {result['domain']}")
    print(f"-> Label:  {result['label']}")
    print(f"-> Reason: {result['reason']}\n")