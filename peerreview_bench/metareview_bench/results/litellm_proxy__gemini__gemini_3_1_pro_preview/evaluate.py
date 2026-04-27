import json

output = {"paper_id": 62, "reviewers": []}

def add_reviewer(name, items):
    output["reviewers"].append({"reviewer_id": name, "items": items})

def save():
    with open('/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper62_metareview.json', 'w') as f:
        json.dump(output, f, indent=2)

