# PeerReview Bench — Local Toolkit

This directory contains the three evaluation pipelines that sit on top of
the PeerReview Bench HuggingFace dataset
(**[`prometheus-eval/peerreview-bench`](https://huggingface.co/datasets/prometheus-eval/peerreview-bench)**).

The dataset has four configs:

| Config | What it contains | Used by |
|---|---|---|
| `expert_annotation` | Per-item meta-review annotations from human experts, one row per (paper, reviewer, item, annotator_source). Used for statistical analysis and similarity measurement. | `analysis/`, `similarity_check/` |
| `meta_reviewer` | The 27 overlap papers' items with both primary and secondary labels, plus a collapsed 10-class label that encodes both the cascade outcome and whether experts agree on each axis. | `meta_review/` |
| `reviewer` | One row per paper — paper title, preprint text, `file_refs` — for training/evaluating AI reviewers that generate reviews from scratch. | `reviewer/` (TODO if you need it) |
| `submitted_papers` | Deduplicated file storage. Each row is one unique file (keyed by SHA256). Other configs reference these via `file_refs`. | Resolved on-demand by `load_data.resolve_file_refs` |

## Directory layout

```
peerreview_bench/
├── README.md                       ← you are here
├── load_data.py                    ← shared HF loaders (all 4 configs)
├── reviewer_rankings.json          ← per-paper Best/Worst Human rankings
│                                     (the one piece of metadata not in HF)
│
├── analysis/                       ← statistical analysis pipelines
│   ├── README.md
│   ├── data_filter.py              ← validity rules shared by analysis scripts
│   ├── peerreview_analysis.py
│   ├── peerreview_analysis_per_paper.py
│   ├── peerreview_analysis_glmm.py
│   ├── inter_rater_reliability.py
│   ├── run_analysis.sh
│   └── {analysis,glmm,irr}_output/ ← generated outputs
│
├── meta_review/                    ← evaluate LLM meta-reviewers
│   ├── README.md
│   ├── prompts.py                  ← prompt templates
│   ├── predictors.py               ← Random/Majority baselines + LLM predictors
│   ├── metrics.py                  ← 10-class accuracy, per-axis, per-agreement metrics
│   └── run_meta_review.py
│
└── similarity_check/               ← per-pair similarity benchmark (238-pair eval set)
    ├── README.md
    ├── load_eval_set.py            ← loader for the similarity_check HF config
    ├── prompts.py                  ← 4-way LLM-as-judge prompt templates
    ├── embeddings.py               ← pluggable embedding backends
    ├── baselines/
    │   ├── embedding_classifier.py ← cosine-similarity baseline
    │   └── llm_classifier.py       ← 4-way LLM-as-judge (thinking mode, multimodal)
    ├── evaluate.py                 ← metrics (acc, AUROC, per-finegrained, per-pair-type)
    └── run_similarity.sh           ← full sweep: embeddings + LLM + evaluate
```

## Quick start

```bash
# 1) Install dependencies
pip install datasets pandas numpy scipy statsmodels scikit-learn sentence-transformers

# 2) Statistical analysis (uses expert_annotation)
cd analysis
./run_analysis.sh

# 3) Meta-reviewer benchmark — baselines don't need API keys
cd ../meta_review
python3 run_meta_review.py --predictor majority
python3 run_meta_review.py --predictor random

# 4) Meta-reviewer benchmark with a real LLM
export ANTHROPIC_API_KEY=...
python3 run_meta_review.py --predictor anthropic --model claude-sonnet-4-5-20250929
#   or
export OPENAI_API_KEY=...
python3 run_meta_review.py --predictor openai --model gpt-4o

# 5) Similarity benchmark (238-pair eval set; requires LITELLM_API_KEY for proxy-routed models)
cd ../similarity_check
./run_similarity.sh                                      # full sweep
LIMIT=5 ./run_similarity.sh                              # smoke test
./run_similarity.sh llm                                  # only LLM-as-judge baselines
```

## Loading from the HuggingFace dataset

Everything in this repo ultimately loads the data via `load_data.py`:

```python
from load_data import (
    load_annotations,              # expert_annotation as ReviewItem dataclasses
    load_expert_annotation_rows,   # expert_annotation as raw HF row dicts
    load_meta_reviewer,            # meta_reviewer config
    load_reviewer,                 # reviewer config
    load_submitted_papers,         # submitted_papers config (hash -> bytes)
    resolve_file_refs,             # join file_refs to bytes
)
```

The top-level `load_data.py` is imported by each subdirectory via
`sys.path.insert(0, parent_dir)` — this keeps the module flat and avoids
needing a formal Python package.

## Reviewer rankings

The per-paper Best/Worst Human assignments live in `reviewer_rankings.json`
(a tiny local sidecar). These were dropped from the HF `expert_annotation`
schema to keep the row schema clean, so `load_data._load_from_hf` reads
them from this file. Keys in that file are already the HF paper_ids
(1..85), so no translation is needed. All other data comes from HuggingFace.

## License

CC-BY-4.0 (matching the dataset license)
