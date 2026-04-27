# PeerReview Bench — Local Toolkit

This directory contains the evaluation pipelines, analysis scripts, and
benchmark infrastructure for
**[PeerReview Bench](https://huggingface.co/datasets/prometheus-eval/peerreview-bench)** —
a benchmark for AI peer reviewers of scientific papers.

## HuggingFace dataset

The dataset lives at
[`prometheus-eval/peerreview-bench`](https://huggingface.co/datasets/prometheus-eval/peerreview-bench)
and has **five configs**:

| Config | What it contains | Used by |
|---|---|---|
| `expert_annotation` | Per-item meta-review annotations from human experts (correctness, significance, evidence). One row per (paper, reviewer, item, annotator_source). 3,954 rows across 85 papers. | `analysis/`, `similarity_check/`, `metareview_bench/`, `evaluation/` |
| `meta_reviewer` | The 27 overlap papers where both primary and secondary annotators labeled each item. Includes a collapsed 10-class label encoding the cascade outcome + inter-annotator agreement. 908 rows. | `metareview_bench/expert_annotation_meta_review/` |
| `reviewer` | One row per paper (85 rows) — paper title, preprint text, `file_refs`, and a `rubric` column listing the "fully good" human review item texts for recall evaluation. | `evaluation/`, `download_papers.py` |
| `similarity_check` | 164 curated review-item pairs with expert-annotated 4-way similarity labels (near-paraphrase, convergent, topical neighbor, unrelated). | `similarity_check/expert_annotation_similarity/` |
| `submitted_papers` | Deduplicated file storage (~2 GB). Each row is one unique file (keyed by SHA256). Other configs reference these via `file_refs`. | Resolved on-demand by `load_data.resolve_file_refs` |

## Directory layout

```
peerreview_bench/
├── README.md                               ← you are here
├── load_data.py                            ← shared HF loaders (all 5 configs)
├── download_papers.py                      ← download papers + reconstruct reviews
├── reviewer_rankings.json                  ← per-paper Best/Worst Human rankings
│
├── evaluation/                             ← PeerReview Bench evaluation pipeline
│   ├── README.md                             for AI reviewer systems
│   ├── config.py                           ← configurable parameters
│   ├── prepare_papers.py                   ← download papers from HF
│   ├── generate_reviews.py                 ← run AI agent reviewer (OpenHands)
│   ├── parse_review.py                     ← extract items from review markdown
│   ├── build_rubric.py                     ← build recall rubric from human items
│   ├── evaluate_recall.py                  ← similarity-based recall metric
│   ├── evaluate_precision.py               ← meta-review-based precision metric
│   ├── evaluate.py                         ← unified entry point
│   └── run_evaluation.sh                   ← shell wrapper
│
├── analysis/                               ← statistical analysis pipelines
│   ├── data_filter.py                      ← validity rules (cascade, fully-good)
│   ├── peerreview_analysis.py              ← main per-group analysis
│   ├── peerreview_analysis_per_paper.py    ← per-paper breakdowns
│   ├── peerreview_analysis_glmm.py         ← GLMM analysis
│   ├── inter_rater_reliability.py          ← IRR metrics
│   └── main_results/                       ← paper figures and tables
│
├── metareview_bench/                       ← LLM meta-reviewer benchmark
│   ├── README.md
│   ├── litellm_client.py                   ← shared LiteLLM wrapper
│   ├── model_config.py                     ← per-model capabilities
│   ├── image_mapping.py                    ← figure detection + image loading
│   ├── metrics.py                          ← axis + 10-class scoring
│   ├── expert_annotation_meta_review/      ← curated benchmark (908 rows, 27 papers)
│   │   ├── prompts.py                      ← axis + tenclass prompt families
│   │   ├── predictors.py                   ← LLM predictor with thinking + retries
│   │   ├── run_meta_review.py              ← LLM runner (concurrent, resumable)
│   │   ├── run_meta_review_agent.py        ← OpenHands agent runner
│   │   └── evaluate.py                     ← unified scorer
│
└── similarity_check/                       ← review-item similarity benchmark
    ├── expert_annotation_similarity/       ← curated 164-pair eval set
    │   ├── prompts.py                      ← 4-way similarity taxonomy
    │   ├── load_eval_set.py                ← loader for similarity_check HF config
    │   ├── embeddings.py                   ← Azure / Gemini / Qwen3 backends
    │   ├── baselines/
    │   │   ├── embedding_classifier.py     ← cosine-similarity baseline
    │   │   └── llm_classifier.py           ← 4-way LLM judge (thinking, multimodal)
    │   ├── evaluate.py                     ← accuracy, AUROC, per-category metrics
    │   └── run_similarity.sh
    └── full_similarity/                    ← full within-paper pairs (66k pairs)
        ├── compute_full_similarity_embedding.py
        ├── compute_full_similarity_llm.py  ← resumable 4-way LLM judge
        ├── analyze_embedding.py
        ├── analyze_llm.py
        └── ANALYSIS_IDEAS.md
```

## Quick start

### (1) Practitioners: evaluate your AI reviewer on PeerReview Bench

If you have an AI reviewer system and want to measure its recall
(coverage of important issues) and precision (quality of generated items):

```bash
cd peerreview_bench/evaluation

# Step 1: Download the 85 benchmark papers (one-time, ~5 min)
python3 prepare_papers.py

# Step 2: Run your AI reviewer agent on each paper
python3 generate_reviews.py \
    --model-name litellm_proxy/anthropic/claude-opus-4-6 \
    --paper-root ../papers/ \
    --max-items 5

# Step 3: Evaluate (recall + precision)
python3 evaluate.py \
    --model-name litellm_proxy/anthropic/claude-opus-4-6 \
    --paper-root ../papers/

# Or bring your own review items JSON (BYOJ) and just evaluate
# Place review_items_mymodel.json in papers/paper{N}/review/
python3 evaluate.py --paper-root ../papers/ --byoj
```

See [`evaluation/README.md`](evaluation/README.md) for the full BYOJ
format, all configurable parameters, and cost estimates.

### (2) Researchers: reuse the analysis / meta-review / similarity scripts

If you want to understand the methodology or adapt the code to your own
review data:

**Statistical analysis** — reproduce the paper's tables and figures:
```bash
cd peerreview_bench/analysis
./run_analysis.sh          # generates tables 1-6, figures 3-5
```
Key files: `data_filter.py` (validity rules, "fully good" definition),
`peerreview_analysis.py` (per-group statistics, pairwise comparisons).
These can be pointed at any dataset with the same schema
(correctness/significance/evidence annotations).

**Meta-review experiments** — run LLM meta-reviewers on your data:
```bash
cd peerreview_bench/metareview_bench/expert_annotation_meta_review
# Run a model in axis mode (predicts correctness/significance/evidence)
python3 run_meta_review.py \
    --model litellm_proxy/gemini/gemini-3.1-pro-preview \
    --prompt-mode axis --limit 20

# Or run the agent meta-reviewer (navigates paper files)
python3 run_meta_review_agent.py \
    --model-name litellm_proxy/gemini/gemini-3.1-pro-preview \
    --paper-root ../../papers --prompt-mode axis --limit 1
```
The prompts in `prompts.py` define Correctness, Significance, and
Evidence with extensive boundary rules — you can use these directly on
your own review items by constructing rows with the expected schema.

**Similarity experiments** — measure pairwise review-item similarity:
```bash
cd peerreview_bench/similarity_check/full_similarity
# Embedding-based (fast, ~3 min per backend)
python3 compute_full_similarity_embedding.py \
    --backend litellm_proxy/azure_ai/text-embedding-3-large

# LLM-judge-based (slow but accurate, resumable)
python3 compute_full_similarity_llm.py \
    --model litellm_proxy/gemini/gemini-3.1-pro-preview \
    --concurrency 16
```
The 4-way taxonomy (near-paraphrase / convergent / topical neighbor /
unrelated) in `expert_annotation_similarity/prompts.py` can be applied
to any pair of review texts.

### (3) Researchers: benchmark your LLM meta-reviewer or similarity model

If you've built a custom LLM meta-reviewer or similarity classifier and
want to evaluate it against our expert annotations:

**Meta-reviewer benchmark** (908 items with 10-class ground truth):
```bash
cd peerreview_bench/metareview_bench/expert_annotation_meta_review

# Test your model
python3 run_meta_review.py \
    --model litellm_proxy/your-model-here \
    --prompt-mode axis --concurrency 8

# Score results
python3 evaluate.py ../../outputs/expert_annotation_meta_review/your_model/
```
Ground truth: per-axis labels from human meta-reviewers (primary +
secondary on 27 overlap papers). Reports per-axis accuracy + reduced
6-class accuracy.

**Similarity benchmark** (164 pairs with 4-way ground truth):
```bash
cd peerreview_bench/similarity_check/expert_annotation_similarity

# Embedding baseline
python3 baselines/embedding_classifier.py --backend your-embedding-model

# LLM judge baseline
python3 baselines/llm_classifier.py \
    --model litellm_proxy/your-model-here --concurrency 8

# Score
python3 evaluate.py ../../outputs/similarity_check/your_output.json
```
Ground truth: 164 expert-labeled pairs with 4-way finegrained labels.
Reports binary accuracy, 4-way accuracy, AUROC, and per-category
breakdowns.

## Loading from the HuggingFace dataset

All scripts load data via `load_data.py`:

```python
from load_data import (
    load_annotations,              # expert_annotation as ReviewItem dataclasses
    load_expert_annotation_rows,   # expert_annotation as raw HF row dicts
    load_meta_reviewer,            # meta_reviewer config (27-paper overlap set)
    load_reviewer,                 # reviewer config (85 papers + rubric)
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
