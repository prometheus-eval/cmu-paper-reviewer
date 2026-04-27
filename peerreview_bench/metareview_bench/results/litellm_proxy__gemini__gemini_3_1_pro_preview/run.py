import json
with open("generate_json.py", "r") as f:
    text = f.read()

text = text.replace("\n\n", "\n").replace("peerreview_be nch", "peerreview_bench").replace("litel lm_proxy", "litellm_proxy")
with open("generate_json.py", "w") as f:
    f.write(text)
