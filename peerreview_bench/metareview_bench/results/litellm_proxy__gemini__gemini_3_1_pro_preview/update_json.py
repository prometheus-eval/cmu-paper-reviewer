import json
import os

JSON_PATH = "/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper33_metareview.json"

def init_json():
    if not os.path.exists(JSON_PATH):
        with open(JSON_PATH, "w") as f:
            json.dump({"paper_id": 33, "reviewers": []}, f, indent=2)

def update_reviewer(reviewer_id, items):
    with open(JSON_PATH, "r") as f:
        data = json.load(f)
    
    # Check if reviewer exists
    exists = False
    for rev in data["reviewers"]:
        if rev["reviewer_id"] == reviewer_id:
            rev["items"] = items
            exists = True
            break
    
    if not exists:
        data["reviewers"].append({
            "reviewer_id": reviewer_id,
            "items": items
        })
        
    with open(JSON_PATH, "w") as f:
        json.dump(data, f, indent=2)

init_json()
