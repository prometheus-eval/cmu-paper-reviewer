# metareview_bench/

Benchmark an LLM-based **meta-reviewer** — a model that reads a paper plus
review items and judges each item along three axes: *correctness*,
*significance*, and *evidence sufficiency*.

The agent-based meta-reviewer runs as an OpenHands agent with filesystem
access to the paper's source files (preprint, images, code, supplementary).
One conversation per paper covers ALL reviewers × ALL items.

## Directory structure

```
metareview_bench/
├── litellm_client.py                  # shared LiteLLM wrapper (used by similarity_check too)
├── model_config.py                    # per-model response_format, multimodal, max_output_tokens
├── image_mapping.py                   # figure-reference detection → image bytes from HF
├── metrics.py                         # axis-mode + tenclass-mode scoring
│
├── expert_annotation_meta_review/     # curated benchmark: `meta_reviewer` HF config
│   │                                  # (908 rows, 27 overlap papers, 10-class ground truth)
│   ├── prompts.py                     # shared definitions, label constants, answer extraction
│   ├── run_meta_review_agent.py       # Agent runner: one conversation per paper
│   ├── run_meta_review_agent.sh       # 3 agents × 2 modes = 6 agent runs
│   └── evaluate.py                    # scorer for agent JSON outputs
```

## Prerequisites

### Paper download

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

| Model | Provider | Notes |
|---|---|---|
| `azure_ai/gpt-5.4` | Azure | |
| `azure_ai/grok-4-1-fast-reasoning` | Azure (xAI) | built-in reasoning |
| `azure_ai/Kimi-K2.5` | Azure (Moonshot) | built-in reasoning |
| `gemini/gemini-3.1-pro-preview` | Google | |
| `anthropic/claude-opus-4-6` | Anthropic | |
| `fireworks_ai/.../qwen3p6-plus` | Fireworks | always-on reasoning |

## Quick start

```bash
# First: download papers (one-time)
cd peerreview_bench
python3 download_papers.py

# Run agent meta-reviewer
cd metareview_bench/expert_annotation_meta_review
python3 run_meta_review_agent.py \
    --model-name litellm_proxy/gemini/gemini-3.1-pro-preview \
    --paper-root ../../papers \
    --prompt-mode axis --limit 1

# Evaluate
python3 evaluate.py ../../outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview_agent/

# Compare multiple models
python3 evaluate.py ../../outputs/expert_annotation_meta_review/*/
```

## Output layout

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

## Evaluation

```bash
# Score an agent directory
python3 evaluate.py agent_trajectories/axis/ --mode axis

# Auto-discover everything in a model's output dir
python3 evaluate.py ../../outputs/expert_annotation_meta_review/gemini__gemini-3_1-pro-preview_agent/

# Compare all models at once
python3 evaluate.py ../../outputs/expert_annotation_meta_review/*/
```

## Shared modules (top-level)

These files live at `metareview_bench/` (not inside the subdirectory) because
they are imported by `similarity_check/` as well:

- **`litellm_client.py`** — `call_llm()` and `call_embedding()` wrappers
- **`model_config.py`** — per-model `response_format`, `max_output_tokens`, `supports_multimodal`
- **`image_mapping.py`** — `find_figure_references()` + `select_images_for_review_item()`
- **`metrics.py`** — `evaluate_axis_predictions()`, `evaluate_tenclass_predictions()`, `format_mode_report()`
