# evaluation/

**PeerReview Bench** evaluation pipeline for AI reviewer systems.

Given an AI reviewer that produces reviews of scientific papers, this
pipeline measures two complementary metrics:

- **Recall** — "How many of the important human-identified issues did the
  AI reviewer catch?" Measured by checking whether each "fully good"
  human review item has a similar match among the AI reviewer's items.

- **Precision** — "How good are the AI reviewer's items?" Measured by
  running an LLM meta-reviewer on each AI item to judge correctness,
  significance, and evidence sufficiency.

## Requirements for AI reviewer baselines

To be included as a baseline on PeerReview Bench, an AI model must meet
three hard requirements:

1. **Multimodal (vision input)** — The model must accept image content
   blocks (e.g., `image_url` with base64 data) so it can read the
   figures, tables, and diagrams in the paper. Text-only models cannot
   be used because they miss visual evidence that reviewers need to
   reference.

2. **Sufficient input context (≥128K tokens)** — The agent conversation
   includes the full paper (~15–25K tokens), tool call history, file
   contents read during navigation, and condenser summaries. For longer
   papers with code and supplementary materials, the agent context can
   exceed 100K tokens. Models with context windows below 128K risk
   truncating critical content mid-conversation.

3. **Sufficient output length (≥32K tokens)** — The agent needs room for
   thinking/reasoning tokens, tool call responses, and the final review
   output across multiple conversation turns. Models with output caps
   below 32K (e.g., Mistral-Large-3 on Azure AI at 8K) will truncate
   mid-review or fail to produce complete output.

4. **Function/tool calling support** — The review agent uses OpenHands
   with `TerminalTool` (run commands), `FileEditorTool` (read/write
   files), and `TaskTrackerTool` (track progress). Models that cannot
   produce structured tool calls cannot be used as agents. This is why
   PeerReview Bench does not accept simple single-turn LLM-call
   baselines — the context window would overflow when trying to fit the
   entire paper, code, and supplementary materials into one prompt.

Models that meet all three requirements can be used directly with
`generate_reviews.py --model-name <your-model>`. Models that don't
support the OpenHands agent framework can still participate via **BYOJ
(Bring Your Own JSON)** — run your own reviewer system externally and
provide the parsed review items for evaluation.

**Important: run one model at a time.** Do not launch multiple
`generate_reviews.py` processes concurrently for different models. The
review generation pipeline temporarily hides other review files to
prevent the agent from reading them, and concurrent runs can interfere
with each other's file hiding/restoring. Run models sequentially, or
use separate copies of the `papers/` directory for parallel runs.

## Quick start

```bash
cd peerreview_bench/evaluation

# Step 1: Download papers (one-time, ~5 min)
python3 prepare_papers.py

# Step 2a: Run your AI reviewer agent on all papers
python3 generate_reviews.py \
    --model-name litellm_proxy/anthropic/claude-opus-4-6 \
    --paper-root ../papers/ \
    --max-items 5 \
    --limit 5                   # smoke test on 5 papers

# Step 2b: OR bring your own review items JSON (BYOJ)
#   Place review_items_mymodel.json in papers/paper{N}/review/
#   See "BYOJ format" section below.

# Step 3: Evaluate
python3 evaluate.py \
    --paper-root ../papers/ \
    --model-name litellm_proxy/anthropic/claude-opus-4-6 \
    --limit 5

# Or evaluate BYOJ
python3 evaluate.py --paper-root ../papers/ --byoj --limit 5

# Run only recall or only precision
python3 evaluate.py --paper-root ../papers/ --byoj --recall-only
python3 evaluate.py --paper-root ../papers/ --byoj --precision-only
```

## Pipeline overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Step 1: prepare_papers.py                                      │
│  Download 85 papers from HuggingFace → papers/paper{N}/preprint/│
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│  Step 2: generate_reviews.py  (or BYOJ)                         │
│  Run AI agent on each paper → papers/paper{N}/review/           │
│    review_{model}.md  +  review_items_{model}.json              │
└──────────────────────────────┬──────────────────────────────────┘
                               ▼
          ┌────────────────────┴────────────────────┐
          ▼                                         ▼
┌──────────────────────────┐          ┌──────────────────────────┐
│  Step 3: evaluate_recall │          │  Step 4: evaluate_prec.  │
│                          │          │                          │
│  For each paper:         │          │  For each AI item:       │
│  rubric item × AI item   │          │  run LLM meta-reviewer   │
│  → 4-way LLM similarity  │          │  (axis mode)             │
│  → "similar" = covered   │          │  → Correct + Significant │
│                          │          │    + Sufficient = "good"  │
│  Recall = covered /      │          │                          │
│           total rubric   │          │  Precision = good /      │
│                          │          │              total AI    │
└──────────────────────────┘          └──────────────────────────┘
```

## Metrics

### Recall: coverage of human rubric items

The **rubric** is the set of "fully good" human review items — items that
human meta-reviewers rated as Correct + Significant + Sufficient Evidence.
These represent the important issues that a competent reviewer should catch.

- **80 papers** have rubric items (5 papers dropped for having zero)
- **799 rubric items** total (mean 10.0 per paper)
- Primary annotator labels only (matching the paper's Table 5)

For each rubric item, we check whether ANY of the AI reviewer's items is
"similar" to it using the 4-way LLM similarity judge (same prompt as
`similarity_check/expert_annotation_similarity/`). The two "similar"
classes are:
- "same subject, same argument, same evidence" (near-paraphrase)
- "same subject, same argument, different evidence" (convergent conclusion)

**Recall = (rubric items with at least one similar AI match) / (total rubric items)**

### Precision: quality of AI reviewer items

Each AI review item is judged by an LLM meta-reviewer (axis mode, same
prompt as `metareview_bench/expert_annotation_meta_review/`) for:
- Correctness (Correct / Not Correct)
- Significance (Significant / Marginally Significant / Not Significant)
- Evidence (Sufficient / Requires More)

An item is **"fully good"** if it is Correct + Significant + Sufficient.

**Precision = (fully good AI items) / (total AI items)**

### F1

Reported when both metrics are available:
**F1 = 2 × Recall × Precision / (Recall + Precision)**

## Files

| File | What it does |
|---|---|
| `config.py` | All configurable parameters (EvalConfig dataclass) |
| `prepare_papers.py` | Downloads papers from HF (wraps `download_papers.py`) |
| `generate_reviews.py` | Runs OpenHands agent reviewer on each paper (reuses `backend/reviewer_prompt.py`) |
| `parse_review.py` | Extracts review items from markdown → `review_items_{model}.json` |
| `build_rubric.py` | Builds the recall rubric from fully-good human items |
| `evaluate_recall.py` | Computes recall via 4-way LLM similarity judge |
| `evaluate_precision.py` | Computes precision via LLM meta-reviewer |
| `evaluate.py` | Unified entry point: runs the full pipeline or individual steps |
| `run_evaluation.sh` | Shell wrapper with common configurations |

## BYOJ: Bring Your Own JSON

If you already have a reviewer system that produces structured output,
you can skip the agent generation step entirely. Place a JSON file in
each paper's `review/` directory:

```
papers/paper1/review/review_items_mymodel.json
```

The JSON must be a list of items, each with at least a `text` field:

```json
[
  {
    "item_number": 1,
    "title": "Missing ablation study",
    "main_point": "The paper does not ablate the contribution of...",
    "text": "The paper does not ablate the contribution of the proposed attention mechanism. Without removing this component and measuring the impact, the claimed 3.2% improvement cannot be attributed to the attention module versus other changes introduced in the same experiment."
  },
  {
    "item_number": 2,
    "text": "..."
  }
]
```

Required fields:
- `text` — the full item text used for similarity comparison and meta-review

Optional fields:
- `item_number` — integer item index (auto-assigned if missing)
- `title` — short title for display
- `main_point` — the core criticism (used in meta-review prompts if available)
- `claim_full` — full claim section
- `evidence_full` — full evidence section

Then run:
```bash
python3 evaluate.py --paper-root ../papers/ --byoj
```

## Configurable parameters

| Parameter | CLI flag | Default | Description |
|---|---|---|---|
| Model name | `--model-name` | claude-opus-4-6 | AI reviewer agent model |
| Max items | `--max-items` | 5 | Max review items per paper |
| Criteria | `--criteria-preset` | nature | Evaluation criteria (nature/neurips) |
| Similarity judge | `--similarity-model` | claude-opus-4-6 | LLM for recall measurement |
| Meta-review judge | `--judge-model` | claude-opus-4-6 | LLM for precision measurement |
| Judge mode | `--judge-mode` | llm | Meta-review mode (llm/agent) |
| Concurrency | `--concurrency` | 16 | Parallel LLM calls |
| Limit | `--limit` | None | Only evaluate N papers |
| Paper root | `--paper-root` | ../papers/ | Directory with paper{N}/ subdirs |
| Output dir | `--output-dir` | ../outputs/evaluation/ | Where to write results |

## Output format

```
outputs/evaluation/
├── recall.json         # per-paper recall + pair-level details
├── precision.json      # per-item meta-review judgments
└── (per-paper details are nested inside the JSONs)
```

## Cost estimates

| Component | Per paper | 85 papers |
|---|---|---|
| Review generation (agent) | ~$1-3 | ~$85-250 |
| Recall (10 rubric × 5 AI = 50 similarity calls) | ~$0.50-2 | ~$40-170 |
| Precision (5 AI items × 1 meta-review call) | ~$0.05-0.20 | ~$4-17 |
| **Total** | ~$2-5 | ~$130-440 |

Use `--limit 5` for smoke tests (~$10-25 total).
