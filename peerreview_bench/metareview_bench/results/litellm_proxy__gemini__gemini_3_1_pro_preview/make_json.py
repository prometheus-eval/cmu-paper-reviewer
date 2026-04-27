import json

def create_base():
    return {
        "paper_id": 81,
        "reviewers": []
    }

data = create_base()
with open("litellm_proxy__gemini__gemini_3_1_pro_preview_paper81_metareview.json", "w") as f:
    json.dump(data, f, indent=2)
