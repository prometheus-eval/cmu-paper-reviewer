# `full_similarity/`

**Main question:** *How similar are AI reviews to human reviews, at the
review-item level?*

Unlike `../expert_annotation_similarity/`, which runs LLM-as-judge and
embedding baselines on the 164-pair curated eval set, this pipeline takes
the full `expert_annotation` HF config — **3,046 unique review items
across 85 papers** — and scores **every within-paper pair**.

Pairs fall into three buckets:

| Pair type | What it measures                     | Approx. count |
|-----------|--------------------------------------|--------------:|
| `H-H`     | Human ↔ Human (inter-human agreement) |      ~35,000 |
| `A-A`     | AI    ↔ AI    (inter-AI agreement)    |       ~6,300 |
| `H-A`     | Human ↔ AI    (**the thing we want**) |      ~24,800 |
|           | **Total**                            |      ~66,000 |

All pairs are computed **within the same paper only** — cosines between
review items on different papers aren't meaningful because the items are
grounded in entirely different scientific content.

## Two scoring paths

There are two parallel compute + analyze pipelines. They share the same
pair set but differ in what they emit per pair.

### (A) Embedding path — fast

| File                                 | What it does |
|--------------------------------------|--------------|
| `compute_full_similarity_embedding.py` | Loads `expert_annotation` from HF, dedupes by (paper_id, reviewer_id, review_item_number), embeds every item with one backend, computes cosine for every within-paper pair, and writes `pairs_embedding_<backend>.jsonl` + `metadata_embedding_<backend>.json` + `items.json`. |
| `analyze_embedding.py`               | Aggregates a `pairs_embedding_*.jsonl` into `analysis_embedding_<backend>.json`: global H-H / A-A / H-A summary, per-AI-model human-likeness ranking, per-AI-model-pair agreement, nearest-neighbor distribution, coverage at thresholds, per-paper breakdown, AI-vs-human gap. |

Per-pair output: **continuous cosine score** ∈ [-1, 1].

Runtime per backend: **2–3 minutes** (batched embedding calls).

Supported backends: the pluggable Azure / Gemini / Qwen3 backends from
`../expert_annotation_similarity/embeddings.py` (reused, not duplicated).

### (B) LLM-judge path — slow but sharper

| File                            | What it does |
|---------------------------------|--------------|
| `compute_full_similarity_llm.py`| Same pair set, but each pair is classified by a thinking-mode frontier LLM using the 4-way taxonomy from `../expert_annotation_similarity/prompts.py`. **Resumable**: on restart, already-scored pairs are skipped (detected by an order-invariant pair id in the JSONL). Text-only — no images, no artifacts. Writes `pairs_llm_<model>.jsonl` + `metadata_llm_<model>.json`. |
| `analyze_llm.py`                | Aggregates a `pairs_llm_*.jsonl` into `analysis_llm_<model>.json`: category mix per pair type, per-AI-model similar-rate ranking, inter-AI-model agreement, per-paper breakdown, AI-vs-human gap, parse/error audit. |

Per-pair output: one of the **four categorical labels**
(near-paraphrase / convergent / topical neighbor / unrelated) plus the
derived binary label.

Runtime per model: **hours** at `--concurrency 16`. Plan to run the LLM
path as a long-running job you can check in on periodically — crashes
are non-fatal because the JSONL is flushed line-by-line and reloaded
on restart. `--save-reasoning` (off by default) dumps the full thinking
trace per pair, which multiplies output size by ~20x — only turn it on
when auditing specific pairs.

Supported models: same three as `expert_annotation_similarity/baselines/`
(GPT-5.4, Gemini 3.1 Pro Preview, Claude Opus 4.6).

## Output schemas

### Embedding path (`pairs_embedding_<backend>.jsonl`)

```jsonc
{
  "paper_id": 1,
  "pair_type": "H-A",            // "H-H" | "A-A" | "H-A"
  "same_reviewer": false,        // true if item_a.reviewer_id == item_b.reviewer_id
  "item_a": {
    "reviewer_id": "reviewer_1",
    "reviewer_type": "Human",
    "review_item_number": 3
  },
  "item_b": {
    "reviewer_id": "Claude",
    "reviewer_type": "AI",
    "review_item_number": 7
  },
  "cosine_score": 0.731
}
```

### LLM path (`pairs_llm_<model>.jsonl`)

```jsonc
{
  "paper_id": 1,
  "pair_type": "H-A",
  "same_reviewer": false,
  "item_a": {"reviewer_id": "reviewer_1", "reviewer_type": "Human", "review_item_number": 3},
  "item_b": {"reviewer_id": "Claude",     "reviewer_type": "AI",    "review_item_number": 7},
  "parsed_answer": "same subject, same argument, different evidence",
  "parsed_binary": "similar",
  "elapsed_seconds": 14.2,
  "error": null
  // "response" + "reasoning_content" also present if --save-reasoning was passed
}
```

### Shared

`items.json` — the full list of scored items with their `review_item`
text, keyed by `(paper_id, reviewer_id, review_item_number)`.

`metadata_<kind>_<slug>.json` — counts, timings, model/backend config.

`analysis_<kind>_<slug>.json` — structured report written by the
corresponding analyze script.

All outputs land in `../../outputs/full_similarity/`.

## Usage

```bash
# Full run: both paths + both analyses. WARNING: the LLM path takes hours.
./run_full_similarity.sh

# Embedding path only (fast) + analyze
./run_full_similarity.sh embedding

# LLM judge path only (slow, resumable) + analyze
./run_full_similarity.sh llm

# Re-run both analyses on whatever is already in outputs/
./run_full_similarity.sh analyze

# Smoke test: first 5 papers only, both paths
LIMIT=5 ./run_full_similarity.sh

# Bump LLM concurrency if the proxy has headroom
CONCURRENCY=32 ./run_full_similarity.sh llm
```

Or invoke scripts directly:

```bash
# Embedding
python3 compute_full_similarity_embedding.py --backend litellm_proxy/azure_ai/text-embedding-3-large
python3 analyze_embedding.py ../../outputs/full_similarity/pairs_embedding_litellm__azure_ai_text-embedding-3-large.jsonl

# LLM judge — resumable, so Ctrl-C is safe
python3 compute_full_similarity_llm.py --model litellm_proxy/anthropic/claude-opus-4-6 --concurrency 16
python3 analyze_llm.py ../../outputs/full_similarity/pairs_llm_litellm__anthropic__claude-opus-4-6.jsonl
```

## Resuming a crashed LLM run

`compute_full_similarity_llm.py` flushes every result to the JSONL on
completion (line-buffered), so a kill -9 in the middle of a run loses at
most one pair. To resume, **just rerun the same command** — the script
reads the existing `pairs_llm_<slug>.jsonl`, builds a set of already-scored
pair ids (order-invariant: `(paper_id, reviewer_a, item_a_num, reviewer_b,
item_b_num)` sorted), and only scores pairs not in that set.

If a line got truncated by the crash, the resume loader will flag it on
stderr (`dropping truncated line N`) and continue past it — the pair
will be re-scored on the rerun.
