import json

from gen_human1 import h1_items
from gen_human2 import h2_items
from gen_human3 import h3_items
from gen_claude import c_items
from gen_gemini import g_items
from gen_gpt import gpt_items

output = {
  "paper_id": 62,
  "reviewers": [
    {
      "reviewer_id": "Human_1",
      "items": h1_items
    },
    {
      "reviewer_id": "Human_2",
      "items": h2_items
    },
    {
      "reviewer_id": "Human_3",
      "items": h3_items
    },
    {
      "reviewer_id": "claude-opus-4-5",
      "items": c_items
    },
    {
      "reviewer_id": "gemini-3.0-pro-preview",
      "items": g_items
    },
    {
      "reviewer_id": "gpt-5.2",
      "items": gpt_items
    }
  ]
}

# clean up duplicate keys in items
for rev in output["reviewers"]:
    for item in rev["items"]:
        # check duplicate keys in dict.
        # Python dicts can't have duplicate keys when defined via literal, 
        # so they will already be resolved (the last one wins).
        # Let's ensure "evidence" is set to None if significance is null, etc.
        if item["correctness"] == "Not Correct":
            item["significance"] = None
            item["evidence"] = None
        elif item["significance"] == "Not Significant":
            item["evidence"] = None

with open('/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper62_metareview.json', 'w') as f:
    json.dump(output, f, indent=2)

print("Generated JSON.")
