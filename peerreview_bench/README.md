# PeerReview Bench

A benchmark for evaluating **AI reviewer systems** on real scientific papers.
Given an AI reviewer that produces review items for a paper, PeerReview Bench
measures two complementary metrics:

- **Recall** — how many of the important, human-identified issues the AI
  reviewer caught (coverage of a rubric built from expert-validated human
  review items).
- **Precision** — how good the AI reviewer's own items are, judged by an LLM
  meta-reviewer on correctness, significance, and evidence.

It accompanies the paper
[*On the limits and opportunities of AI reviewers*](https://arxiv.org/abs/2605.20668).
The papers and human review items are downloaded automatically from
HuggingFace — no private data is required.

## Layout

```
peerreview_bench/
├── evaluation/          # the evaluation harness (start here)
│   ├── README.md        # full pipeline docs, BYOJ format, cost estimates
│   ├── evaluate.py      # unified entry point (recall + precision)
│   ├── generate_reviews.py / prepare_papers.py / parse_review.py / ...
│   └── judges/          # self-contained judge prompts + LLM-call helpers
├── load_data.py         # HuggingFace dataset loaders
└── download_papers.py   # paper download helper
```

## Prerequisites

```bash
pip install -r peerreview_bench/requirements.txt

export LITELLM_API_KEY=<your-key>
export LITELLM_BASE_URL=https://cmu.litellm.ai   # or your LiteLLM endpoint
```

## Quick start

```bash
cd peerreview_bench/evaluation

# 1. Download papers + human rubric from HuggingFace (one-time)
python3 prepare_papers.py

# 2a. Run your AI reviewer agent on the papers...
python3 generate_reviews.py --model-name litellm_proxy/anthropic/claude-opus-4-6 --limit 5

# 2b. ...or bring your own review items (BYOJ): drop
#     review_items_<model>.json into each papers/paper{N}/review/ directory.

# 3. Evaluate (recall + precision, F1)
python3 evaluate.py --limit 5            # agent reviews
python3 evaluate.py --byoj --limit 5     # bring-your-own JSON
```

See [`evaluation/README.md`](evaluation/README.md) for the full pipeline,
the BYOJ JSON format, configurable parameters, and cost estimates.

## Citation

```bibtex
@article{kim2026limits,
  title={On the limits and opportunities of AI reviewers: Reviewing the reviews of Nature-family papers with 45 expert scientists},
  author={Kim, Seungone and Yoon, Dongkeun and Gashteovski, Kiril and Suk, Juyoung and Baek, Jinheon and Aggarwal, Pranjal and Wu, Ian and Zaverkin, Viktor and Petkoski, Spase and Schrider, Daniel R and others},
  journal={arXiv preprint arXiv:2605.20668},
  year={2026}
}
```
