import json

valid_classes = [
    "correct_significant_sufficient",
    "correct_significant_insufficient",
    "correct_significant_disagree_on_evidence",
    "correct_marginal_sufficient",
    "correct_marginal_insufficient",
    "correct_marginal_disagree_on_evidence",
    "correct_not_significant",
    "correct_disagree_on_significance",
    "incorrect",
    "disagree_on_correctness"
]

data = json.load(open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper12_metareview.json"))
for r in data["reviewers"]:
    for i in r["items"]:
        if i["correctness"] not in ["Correct", "Not Correct"]:
            print(f"Invalid correctness: {i['correctness']}")
        if i["prediction_of_expert_judgments"] not in valid_classes:
            print(f"Invalid prediction: {i['prediction_of_expert_judgments']}")
        
        if i["correctness"] == "Correct":
            if i["prediction_of_expert_judgments"] == "incorrect":
                print("Inconsistency: Correct but predicted incorrect")
        else:
            if i["prediction_of_expert_judgments"] not in ["incorrect", "disagree_on_correctness"]:
                print("Inconsistency: Not Correct but predicted correct_*")
print("Check done.")
