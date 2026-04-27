import json
import re

def remove_ansi_escape_sequences(text):
    if not isinstance(text, str):
        return text
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def clean_data(data):
    if isinstance(data, dict):
        return {k: clean_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_data(item) for item in data]
    else:
        return remove_ansi_escape_sequences(data)

out_path = "/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper61_metareview.json"

with open(out_path, "r") as f:
    data = json.load(f)

data = clean_data(data)

with open(out_path, "w") as f:
    json.dump(data, f, indent=2)

print("Cleaned JSON.")
