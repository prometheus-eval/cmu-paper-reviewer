import json
with open('draft_metareview.py') as f:
    text = f.read()

text = text.replace("peerreview_ben ch", "peerreview_bench").replace("litell m", "litellm")
with open('draft_metareview.py', 'w') as f:
    f.write(text)
