# similarity_check/

Per-pair benchmarking against the `similarity_check` HF eval set —
164 hand-audited review-item pairs labeled with both a binary
`similar` / `not_similar` label and a 4-category fine-grained label.
Goal: validate whether automated metrics can reproduce expert
similarity judgments.

---

## What's in the eval set

Loaded from `prometheus-eval/peerreview-bench`, config `similarity_check`,
split `eval`. **164 pairs** total, covering 32 unique papers and
drawing from 266 unique review items:

- **70 similar** (48 convergent conclusions + 22 near-paraphrases)
- **94 not-similar** (27 topical neighbors + 67 unrelated)
- 85 AI-AI / 79 AI-Human

Each pair has:
- A **binary** `binary_label` (`similar` / `not_similar`) matching the
  annotator's implicit judgment.
- A **fine-grained** `finegrained_label` string from a manual audit:

  | Long-form label                                      | Subject | Argument | Evidence | Meaning               | n  |
  |------------------------------------------------------|---------|----------|----------|-----------------------|----|
  | `same subject, same argument, same evidence`        | same    | same     | same     | Near-paraphrase       | 22 |
  | `same subject, same argument, different evidence`   | same    | same     | diff     | Convergent conclusion | 48 |
  | `same subject, different argument`                  | same    | diff     | n/a      | Topical neighbor      | 27 |
  | `different subject`                                 | diff    | diff     | n/a      | Unrelated             | 67 |

  Derivation: convergent/near-paraphrase ⇒ `similar`; topical neighbor
  / unrelated ⇒ `not_similar`.

### 4-way × pair-type cross-tab

| Category              | AI-AI | AI-Human | Total |
|-----------------------|-------|----------|-------|
| Near-paraphrase (c)   | 19    | 3        | 22    |
| Convergent (b)        | 22    | 26       | 48    |
| Topical neighbor (a)  | 18    | 9        | 27    |
| Unrelated (d)         | 26    | 41       | 67    |
| **Total**             | **85**| **79**   | **164** |

### How the pairs were constructed

The 164 pairs come from two source pools derived from the per-paper
expert annotations:

- **70 positive pairs** (`source_bucket = audited_bc`) — built from the
  annotator audit of AI-to-other-reviewer matches. A subagent matched
  non-listed AI review items to other reviewers' items, the author
  audited the matches, and a verification pass upgraded the borderline
  cases. Only pairs audited as (b) convergent-conclusion or (c)
  near-paraphrase survive in this pool, so all 70 are labeled
  `similar`, split 22 `c` + 48 `b`.

- **94 negative pairs** (`source_bucket = annotator_unique_or_items_neg`)
  — sampled from the 2,984 annotator-labeled "unique" pairs across
  papers, then pre-classified into topical-neighbor (a) versus unrelated
  (d) buckets. All 94 are labeled `not_similar`, split 27 `a` + 67 `d`.
  The rationale text distinguishes the two sub-types:
  - 27 pairs whose two items reference the same figure / section /
    method from different angles ("Both items reference the same X,
    A says P and B says Q") → (a) topical neighbor
  - 67 pairs whose two items discuss genuinely different parts of the
    paper ("A is about X, while B is about Y") → (d) unrelated

---

## Files

- `load_eval_set.py` — loads the 164-pair eval set from HF as dataclasses,
  and provides `load_paper_file_refs()` for joining against the `reviewer`
  config when multimodal LLM baselines need images.
- `prompts.py` — system + user prompt templates for the LLM-as-judge
  baselines, including the 4-way long-form-label prompt.
- `embeddings.py` — pluggable embedding backends (qwen3 local, plus any
  LiteLLM embedding model routed via the CMU proxy).
- `baselines/embedding_classifier.py` — embed each item with one of the
  backends, compute pair cosine, save raw scores.
- `baselines/llm_classifier.py` — 4-way LLM-as-judge baseline via the
  LiteLLM proxy. Supports multimodal (image) input, provider-level
  thinking mode, and the full catalog max output budget for each model.
- `evaluate.py` — read a baseline output JSON and report accuracy /
  AUROC / per-fine-grained and per-pair-type breakdowns.
- `run_similarity.sh` — run every embedding + every LLM baseline + the
  evaluate aggregation, in one go. Stages can be filtered.
- `smoke_test.py` — sanity-check that every embedding backend is reachable
  (runs on 3 toy human/AI pairs and prints the cosine matrix).

## Quick start

Run the full sweep (2 embedding models × 3 LLM models × evaluate):

```bash
cd similarity_check
./run_similarity.sh                    # full run (164 pairs per backend)
LIMIT=5 ./run_similarity.sh            # smoke test (first 5 pairs each)
./run_similarity.sh embeddings         # only embedding baselines
./run_similarity.sh llm                # only LLM-as-judge baselines
./run_similarity.sh evaluate           # re-score existing outputs in-place
```

Or run a single baseline directly:

```bash
# Embedding baseline via LiteLLM proxy
python baselines/embedding_classifier.py \
    --backend litellm_proxy/azure_ai/text-embedding-3-large

# LLM-as-judge (4-way, thinking mode, multimodal)
python baselines/llm_classifier.py \
    --model litellm_proxy/anthropic/claude-opus-4-6 \
    --limit 5

# Evaluate any output
python evaluate.py ../outputs/similarity_check/llm_judge_4way_anthropic__claude-opus-4-6.json
```

## LLM-as-judge design notes

- **Routing**: every LLM model goes through the CMU LiteLLM proxy
  (`litellm_proxy/...`), which shares the same `litellm_client.py` used
  by `meta_review/`. Adding a new model only requires an entry in
  `meta_review/model_config.py`.
- **Thinking mode**: `reasoning_effort="high"` is set for every model.
  LiteLLM maps this to `thinking_level="high"` for Gemini 3.1 Pro,
  reasoning at max effort for Azure GPT-5, and adaptive thinking for
  Claude Opus 4.6. Anthropic additionally gets an explicit
  `thinking={"type":"enabled", "budget_tokens": max_tokens - 4096}`
  block so we use the model's full 128K output budget for thinking.
- **No truncation**: the full `paper_content` is sent on every pair.
  All three supported models have ≥ 1M-token context, and the longest
  paper in the dataset is ~26K tokens, so truncation is never needed.
- **Max output**: `max_tokens` defaults to the catalog-reported
  `max_output_tokens` from `meta_review/model_config.py`.
- **Temperature**: fixed at 1.0 — Anthropic extended thinking requires
  `temperature == 1`, Gemini 3 prefers 1.0 for reasoning quality, and
  Azure GPT-5 accepts any value but defaults to 1.0.
- **Images**: if either review item in a pair contains a figure/table
  reference (regex from `meta_review/image_mapping.find_figure_references`),
  the matching images from that paper's `images_list.json` are attached
  as vision-API content blocks. Images bytes are resolved via the
  `submitted_papers` HF config.

## Output

Baseline outputs land in `../outputs/similarity_check/`:

```
outputs/similarity_check/
├── embedding_litellm__azure_ai_text-embedding-3-large.json
├── embedding_litellm__gemini_gemini-embedding-001.json
├── llm_judge_4way_anthropic__claude-opus-4-6.json
├── llm_judge_4way_azure_ai__gpt-5_4.json
└── llm_judge_4way_gemini__gemini-3_1-pro-preview.json
```

Each JSON is self-contained with a `metadata` block (model id, thinking
config, parse rate, accuracy) and a `results` array (per-pair records
with `parsed_answer`, `reasoning_content`, `image_paths`, `elapsed_seconds`).
