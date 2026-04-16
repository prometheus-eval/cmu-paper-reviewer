# meta_review/

Benchmark an LLM-based **meta-reviewer** — a model that reads a paper plus
review items and judges each item along three axes: *correctness*,
*significance*, and *evidence sufficiency*.

Two types of baselines:
- **LLM baseline**: the model receives the paper + one review item per call
  and returns a structured judgment via the LiteLLM proxy.
- **Agent baseline**: the model runs as an OpenHands agent with filesystem
  access to the paper's source files (preprint, images, code, supplementary).
  One conversation per paper covers ALL reviewers × ALL items.

## Directory structure

```
meta_review/
├── litellm_client.py                  # shared LiteLLM wrapper (used by similarity_check too)
├── model_config.py                    # per-model response_format, multimodal, max_output_tokens
├── image_mapping.py                   # figure-reference detection → image bytes from HF
├── metrics.py                         # axis-mode + tenclass-mode scoring
│
├── expert_annotation_meta_review/     # curated benchmark: `meta_reviewer` HF config
│   │                                  # (908 rows, 27 overlap papers, 10-class ground truth)
│   ├── prompts.py                     # 2 prompt families: axis + tenclass (extensive definitions)
│   ├── predictors.py                  # baselines + LiteLLMMetaReviewer (thinking, retries, artifacts)
│   ├── run_meta_review.py             # LLM runner: --prompt-mode {axis|tenclass} + --concurrency
│   ├── run_meta_review.sh             # 8 targets × 2 modes = 16 LLM runs
│   ├── run_meta_review_agent.py       # Agent runner: one conversation per paper
│   ├── run_meta_review_agent.sh       # 3 agents × 2 modes = 6 agent runs
│   ├── evaluate.py                    # unified scorer (handles both LLM JSONL + agent JSON)
│   └── smoke_test.py                  # quick per-model × per-mode sanity check
│
└── full_metareview/                   # production run: `expert_annotation` HF config
    │                                  # (3k+ items, all 85 papers)
    ├── run_full_metareview.py         # LLM runner on full set
    ├── run_full_metareview.sh         # 8 targets × 2 modes = 16 LLM runs
    ├── run_full_metareview_agent.py   # Agent runner on all 85 papers
    └── run_full_metareview_agent.sh   # 3 agents × 2 modes = 6 agent runs
```

## Prerequisites

### Paper download (required for agent baselines)

The agent needs local access to paper files. Run the download script once:

```bash
cd peerreview_bench
python3 download_papers.py                # creates papers/paper1/ ... papers/paper85/
python3 download_papers.py --limit 5      # smoke test: first 5 papers only
```

Each paper directory has:
```
papers/paper1/
├── preprint/           # paper source files from HF submitted_papers config
│   ├── preprint.md
│   ├── images/
│   ├── images_list.json
│   ├── supplementary/
│   └── code/
└── reviews/            # reconstructed per-reviewer review files
    ├── Claude.md
    ├── GPT.md
    ├── Gemini.md
    ├── Human_1.md
    ├── Human_2.md
    └── Human_3.md
```

Review files aggregate all items from one reviewer into a single markdown
document, with cited references collected and deduplicated at the end.

## Two prompt modes

| Mode | What the model does | Output shape | Ground truth |
|---|---|---|---|
| **axis** | Plays a single expert meta-reviewer; emits per-axis cascade labels | `{correctness, significance, evidence}` | Per-axis consensus where both annotators agreed |
| **tenclass** | Predicts what a PAIR of experts would jointly produce, including "disagree" variants | `{label}` (one of 10 collapsed classes) | 10-class label from `meta_reviewer` config |

Each model runs **4 times** in a full benchmark (2 modes × 2 subdirectories).

### The 10-class taxonomy (tenclass mode)

| ID | Label | Meaning |
|---:|:---|:---|
| 1 | `correct_significant_sufficient` | Both experts: Correct + Significant + Sufficient |
| 2 | `correct_significant_insufficient` | Both experts: Correct + Significant + Requires More |
| 3 | `correct_significant_disagree_on_evidence` | Both: Correct + Significant, disagree on evidence |
| 4 | `correct_marginal_sufficient` | Both: Correct + Marginally Sig + Sufficient |
| 5 | `correct_marginal_insufficient` | Both: Correct + Marginally Sig + Requires More |
| 6 | `correct_marginal_disagree_on_evidence` | Both: Correct + Marginally Sig, disagree on evidence |
| 7 | `correct_not_significant` | Both: Correct + Not Significant |
| 8 | `correct_disagree_on_significance` | Both: Correct, disagree on significance |
| 9 | `incorrect` | Both: Not Correct |
| 10 | `disagree_on_correctness` | Experts disagree on correctness |

## Models

All 6 LLM targets are multimodal (verified with image smoke tests):

| Model | Provider | Response format | Notes |
|---|---|---|---|
| `azure_ai/gpt-5.4` | Azure | `json_schema` | |
| `azure_ai/grok-4-1-fast-reasoning` | Azure (xAI) | `json_schema` | built-in reasoning |
| `azure_ai/Kimi-K2.5` | Azure (Moonshot) | `json_object` | built-in reasoning |
| `gemini/gemini-3.1-pro-preview` | Google | `json_schema` | |
| `anthropic/claude-opus-4-6` | Anthropic | `json_schema` | |
| `fireworks_ai/.../qwen3p6-plus` | Fireworks | `json_object` | always-on reasoning |

Plus 2 baselines (majority, random) and 3 OpenHands agent variants
(gpt-5.4, gemini-3.1-pro-preview, claude-opus-4-6).

## LLM baseline vs Agent baseline

| | LLM baseline | Agent baseline |
|---|---|---|
| **Input** | Paper content + 1 review item per API call | Filesystem access to full paper dir + all reviews |
| **Granularity** | Per-item (one call per item) | Per-paper (one conversation covers all reviewers × items) |
| **Output** | `predictions_{mode}.jsonl` — one line per item | `agent_trajectories/{mode}/paper{N}/prediction.json` — nested JSON with all reviewers |
| **Images** | Attached as base64 in the prompt (via `image_mapping.py`) | Agent opens image files directly from `preprint/images/` |
| **Code** | Text artifacts inlined in prompt (budget-capped) | Agent navigates `preprint/code/` and can read/run files |
| **Concurrency** | `--concurrency N` (ThreadPoolExecutor) | Sequential (one paper at a time) |
| **Resume** | JSONL is line-buffered; restart skips scored items | Agent checks for existing `prediction.json` |
| **Cost** | ~$0.01-0.10 per item | ~$0.50-1.00 per paper |
| **Speed** | ~10-60s per item | ~5-10 min per paper |

## Quick start

```bash
# ===== LLM baseline =====
cd meta_review/expert_annotation_meta_review

# Smoke test: one model, both modes, 3 rows
python3 smoke_test.py --models gemini --limit 3

# Full run on the 908-row benchmark subset
./run_meta_review.sh                           # all models × both modes
LIMIT=20 ./run_meta_review.sh axis             # 20-row smoke, axis only

# Single model
python3 run_meta_review.py \
    --model litellm_proxy/gemini/gemini-3.1-pro-preview \
    --prompt-mode axis --concurrency 8

# ===== Agent baseline =====
# First: download papers (one-time)
cd ../../
python3 download_papers.py

# Then run agents
cd meta_review/expert_annotation_meta_review
python3 run_meta_review_agent.py \
    --model-name litellm_proxy/gemini/gemini-3.1-pro-preview \
    --paper-root ../../papers \
    --prompt-mode axis --limit 1

# ===== Full run on all 85 papers =====
cd ../full_metareview
./run_full_metareview.sh                       # all LLM models × both modes
LIMIT=5 ./run_full_metareview.sh axis          # 5-paper smoke test

# ===== Evaluation =====
cd ../expert_annotation_meta_review
# Score a single model's outputs (auto-discovers LLM + agent files)
python3 evaluate.py ../../outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview/
# Compare multiple models
python3 evaluate.py ../../outputs/expert_annotation_meta_review/*/
```

## Output layout

### LLM baseline outputs

```
outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview/
├── predictions_axis.jsonl       # one JSON line per item (resume-safe)
├── metrics_axis.json            # per-axis accuracy, reduced 6-class accuracy
├── report_axis.txt              # human-readable report
├── predictions_tenclass.jsonl
├── metrics_tenclass.json
└── report_tenclass.txt
```

### Agent baseline outputs

```
outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview_agent/
├── metrics_axis.json
├── report_axis.txt
└── agent_trajectories/
    └── axis/
        ├── paper1/
        │   ├── prediction.json      # nested JSON: {reviewers: [{items: [...]}]}
        │   └── conversation/        # OpenHands conversation state
        ├── paper2/
        ...
```

## Unified evaluation

`evaluate.py` auto-detects both output formats:

```bash
# Score LLM JSONL
python3 evaluate.py predictions_axis.jsonl --mode axis

# Score agent directory
python3 evaluate.py agent_trajectories/axis/ --mode axis

# Auto-discover everything in a model's output dir
python3 evaluate.py ../../outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview/

# Compare all models at once
python3 evaluate.py ../../outputs/expert_annotation_meta_review/*/
```

## Shared modules (top-level)

These files live at `meta_review/` (not inside either subdirectory) because
they are imported by both subdirectories AND by `similarity_check/`:

- **`litellm_client.py`** — `call_llm()` and `call_embedding()` wrappers
- **`model_config.py`** — per-model `response_format`, `max_output_tokens`, `supports_multimodal`
- **`image_mapping.py`** — `find_figure_references()` + `select_images_for_review_item()`
- **`metrics.py`** — `evaluate_axis_predictions()`, `evaluate_tenclass_predictions()`, `format_mode_report()`
