items = []

def add(item_number, correctness, significance, evidence, prediction, reasoning):
    items.append({
        "item_number": item_number,
        "reasoning": reasoning,
        "correctness": correctness,
        "significance": significance,
        "evidence": evidence,
        "prediction_of_expert_judgments": prediction
    })

# Add items...
