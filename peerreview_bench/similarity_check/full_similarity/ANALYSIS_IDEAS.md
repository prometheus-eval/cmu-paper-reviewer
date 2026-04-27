# `full_similarity/` — Analysis Ideas

Planning document for the paper-side analyses that the full_similarity
pipeline outputs unlock. Organized by priority tier (Tier 1 = must-have
for the main result, Tier 3-4 = depth/robustness/extensions).

**Research question:** *How similar are AI reviews to human reviews, at
the review-item level?*

**Data:**
- `pairs_embedding_<backend>.jsonl` × {Azure, Gemini} — cosine per pair
- `pairs_llm_<model>.jsonl` × {GPT-5.4, Gemini 3.1 Pro, Claude Opus 4.6}
  — 4-way categorical label per pair
- `items.json` — every scored item's text
- `expert_annotation` HF config — per-item correctness / significance /
  evidence labels (usable for the correctness-correlated analyses)

**Unit of analysis:**
- Unit of comparison = a within-paper review-item pair
- Unit of **statistical independence** = a paper (not a pair — pairs
  within a paper share paper content, so they're not independent)
- ~66k pairs across 85 papers

---

## Main paper framing: "Can AIs replace human reviewers?"

The preliminary 2-paper signal from gpt-5.4 shows `A-A > H-H > H-A` —
AIs agree with each other more than humans agree with each other, but
AIs agree with humans LESS than humans agree with humans. If this
replicates at 85 papers, it rules out direct 1-for-1 replacement of
human reviewers with AI, for three independent reasons:

1. **Distributional mismatch.** `H-A < H-H` → AI reviews sit off the
   human review distribution. Replacing a human with an AI isn't
   "another draw from the human reviewer population" — it's a different
   kind of reviewer entirely, importing a systematic style/framing shift
   into the pipeline.

2. **Loss of perspective diversity.** `A-A > H-H` → the frontier models
   converge on a shared "house style" more than three independent humans
   do. Substituting 3 AIs for 3 humans keeps the reviewer count constant
   but reduces the number of effectively independent perspectives. Peer
   review's committee effect is weaker for an AI-only panel.

3. **Unrecoverable miss rate.** Any concern in the "things humans agree
   on but AIs don't raise" slice is lost the moment you remove the human.
   Crucially, `A-A > H-A` means adding MORE AI reviewers doesn't fix this
   — extra AIs mostly amplify the same view rather than fill in the gaps.
   There is no ensemble-averaging path out of the gap.

The same inequality is **consistent** with constructive roles for AI:

- **Augmentation / hybrid workflows.** AI catching things humans miss is
  compatible with the gap as long as those catches have low H-A
  similarity. Separating "AI missing human concerns" from "AI adding
  novel valid concerns" is what orphan-items analysis (Tier 2) is for.
- **Triage / pre-review.** Nothing here says AIs can't filter obvious
  issues before a human touches the paper — that's additive, not
  substitutional.
- **Calibration target.** `H-A = H-H` can serve as a measurable goal for
  future reviewer models. This dataset is the measuring stick.

**Essential caveat to state in the paper:** similarity measures *framing
agreement*, not *correctness*. Two reviewers can agree on framing and
both be wrong; two reviewers can disagree on framing and both be right.
Similarity is necessary but not sufficient for replaceability — you also
need quality, which requires the correctness labels (see Tier 2 §7).

---

## Tier 1 — headline results (required for the main result table)

### §1. The 3-way similarity table (THE punchline)

One table, three rows × three-ish columns:

| Pair type | Mean cosine (Azure) | P(similar \| LLM judge) | N pairs |
|-----------|---------------------|-------------------------|--------:|
| H-H       | ?                   | ?                       |  ~35,266 |
| A-A       | ?                   | ?                       |   ~6,360 |
| H-A       | ?                   | ?                       |  ~24,790 |

Within-paper bootstrap (resample papers, not pairs) to get CIs on each
cell. The independence unit is the paper because items within a paper
share the paper content.

Sanity: compare cosine ordering vs LLM-judge ordering. Agreement
strengthens both numbers; disagreement is a robustness discussion.

### §2. Per-AI-model human-likeness ranking

Split H-A by the AI side's model (Claude / GPT / Gemini). For each,
report mean H-A cosine and P(similar). Ranks the three frontier models
by how close their review style sits to humans on this dataset.

Expect the ranking to be tight (they're all frontier models with thinking
mode) but informative. If one model is clearly more human-like, that's a
paper claim.

### §3. Overlaid distributions (not just means)

Three histograms of cosine score on the same axes: H-H, A-A, H-A. Shows
the *shape* of the mismatch, not just means. Critical because with 66k
pairs a small mean gap can look big even when distributions overlap
heavily — reviewers will ask "is this a real distributional shift or a
mean shift with huge tails?"

For the LLM-judge version, show stacked bars of the 4-way category mix
per pair type.

---

## Tier 2 — supporting analyses (strong secondary results)

### §4. Coverage @ threshold

For each threshold τ ∈ {.5, .6, .7, .8}: "fraction of Human items that
have at least one AI item with cosine ≥ τ on the same paper." Reframes
the question operationally: *what fraction of human concerns does at
least one AI pick up?*

Already computed by `analyze_embedding.py`. Plot the curve (coverage as a
function of τ) — rapid dropoff at high τ = "AIs approximate, but rarely
match." Similar flat curve = "AIs routinely match."

Symmetric version: "fraction of AI items with a close human match" — low
number means AIs generate content humans don't engage with.

### §5. Orphan item analysis (qualitative appendix)

- **Orphan Human items**: 50 Human items with the LOWEST
  max-cosine to any AI item on the same paper. These are concerns no AI
  caught. Show 5-10 examples in the paper appendix — these are the
  qualitative "AI blind spots."
- **Orphan AI items**: 50 AI items with the LOWEST max-cosine to any
  Human item on the same paper. AI concerns no human raised. Show 5-10
  examples. **The question for each: hallucination, or novel valid
  concern?**

Ties to §7 below: cross-reference orphan AI items with their correctness
labels. If orphan AI items are mostly labeled "Not Correct," they're
hallucinations. If they're mostly labeled "Correct" + "Significant,"
they're evidence for augmentation.

### §6. Inter-AI pair agreement

Mean A-A cosine split by model pair: Claude-GPT, Claude-Gemini, GPT-Gemini.
Answers *is the A-A > H-H gap driven by all three models, or mostly by two
models that look alike?*

Already computed by both analyzers. Hypothesis: all three pairs are close,
but GPT-5 + Gemini may cluster tighter than either with Claude.

### §7. Correlate similarity with correctness labels

The `expert_annotation` HF config carries per-item
`correctness` / `significance` / `evidence` labels. We can stratify our
similarity findings by quality:

- **AI item correctness × max-cosine-to-Human.** Are "near-paraphrase of
  a human item" AI items more likely to be labeled Correct? Intuition:
  if the AI paraphrases a concern a human also raised, it's validating
  a real issue.
- **Does low H-A similarity predict AI errors?** Partition AI items into
  {has close human match, no close human match}. Which subset has a
  higher "Not Correct" rate? If orphan AI items skew incorrect → AIs are
  hallucinating. If they skew correct → they're augmenting.
- **Quality-filtered similarity.** Re-run §1 with only the "Correct +
  Significant + Sufficient" subset of AI items. Does `A-A > H-H > H-A`
  still hold? Possible finding: high-quality AI items are better
  distributed, and the mode collapse is concentrated in low-quality
  output.

This is the analysis that lifts the paper from "style comparison" to
"substantive replaceability claim."

---

## Tier 3 — robustness and depth

### §8. Embedding × LLM-judge agreement

Per pair, we have both a cosine and a 4-way LLM label. Cross-tabulate:

- Binary confusion matrix: (cosine ≥ τ) × (LLM label ∈ {similar}) at a
  chosen τ (say, the LLM-judge-optimal threshold from the 164-pair eval).
- Rank correlation: Spearman between cosine and a binarized LLM verdict
  (or a 4-point ordinal if you map the labels to 0/1/2/3).

A strong agreement lets you defend both measurement types simultaneously
in the paper: "we confirm this with both a geometric (embedding) and a
semantic (LLM judge) measurement."

### §9. Inter-LLM-judge agreement

Three LLMs score every pair. How often do GPT-5, Gemini 3.1, and Claude
Opus agree on the 4-way label? Pairwise agreement rates, 3-way unanimous
agreement rate, and Cohen's κ per pair.

Low disagreement → the measurement is robust. High disagreement → the
paper has a limitation to flag, and the orphan items should focus on the
3-way agreed-on ones for higher-confidence qualitative claims.

### §10. Same-reviewer vs different-reviewer H-H split

The analyzer already tracks `same_reviewer` for H-H pairs.

- Same-reviewer H-H = "one human writing multiple related items on the
  same paper" (within-reviewer consistency).
- Different-reviewer H-H = "two humans converging" (inter-reviewer
  agreement).

The **different-reviewer** rate is the real human ceiling; the
same-reviewer rate inflates H-H because a single reviewer naturally
repeats themes. Use the different-reviewer H-H number as the comparison
target in §1, not the mixed H-H number.

Same split for A-A (same model / different models) — which we already
surface because we have 3 AI models per paper.

### §11. Paper-level heterogeneity

Rank papers by mean H-A cosine. Report:

- **Easy papers** (high H-A): what are they? Short? Obvious
  methodological flaws? Common domain? Characterize what makes them easy
  for AIs to match humans.
- **Hard papers** (low H-A): case studies of AI blind spots. Show 2-3 in
  the paper.

If you can correlate difficulty with observable paper features (length,
domain, methodology type, file_refs count) you get a "what predicts AI
replaceability?" discussion section.

### §12. Perspective diversity loss

Turn the gap `A-A − H-H` into a concrete number the paper can cite:

> "Using 3 frontier AIs instead of 3 humans loses ~X% of inter-reviewer
> disagreement — i.e., the AI panel produces an effectively smaller
> number of independent perspectives."

Formalize: if we model reviewer opinions as points in cosine space, the
"perspective volume" covered by 3 humans vs 3 AIs can be approximated by
(1 − mean_pairwise_cosine). A-A volume is smaller than H-H volume by a
ratio you can report.

---

## Tier 4 — extensions (probably out of scope for this paper)

### §13. Ensemble intervention

Construct a hybrid reviewer = 1 real human + 1 AI, and measure its
similarity (union of items, dedupe near-matches) against held-out
humans. If hybrid H-A > pure-AI H-A and hybrid H-H ≈ pure-H H-H, that's
direct evidence that augmentation closes part of the gap.

### §14. Paraphrase-robustness ceiling

Take a random sample of 200 Human items, paraphrase each through an LLM,
re-embed, and measure cosine(original, paraphrase). This gives an
**upper bound** for how high H-A could realistically go if the AI
matched a human concern perfectly. It anchors the other numbers: if
paraphrase-ceiling = 0.92 and observed H-A = 0.71, the "0.21 gap" has a
natural scale.

### §15. Inter-reviewer-type quality comparison (out of this dataset's scope)

Quality-comparison is answered by the other analyses in the repo
(reviewer_rankings, metareview_bench). Flagging here so nobody tries to
re-derive a reviewer-quality ranking from similarity alone — similarity
is about framing, not correctness.

---

## What the current pipeline already gives you

| Tier / §  | Covered by analyzer | Needs extra code |
|-----------|---------------------|------------------|
| §1        | ✓ `analyze_embedding.py` + `analyze_llm.py` | just a small table-assembly script |
| §2        | ✓ (per-model rankings) | — |
| §3        | — (analyzer gives summary stats, not histograms) | ~20 lines of matplotlib |
| §4        | ✓ (coverage_at_thresholds block) | — |
| §5        | ✓ (nearest_neighbor_summary) | orphan-selection + item-text join — ~50 lines |
| §6        | ✓ (per_ai_model_pair_agreement) | — |
| §7        | — | join pairs ↔ expert_annotation rows on (paper_id, reviewer_id, item_number) |
| §8        | — | ~50 lines cross-tabbing both JSONL files |
| §9        | — | ~50 lines loading all three llm JSONLs |
| §10       | ✓ (same_vs_diff_reviewer_split) | — |
| §11       | ✓ (per_paper_top10 / bottom10) | paper-feature join for the narrative |
| §12       | partial (need the "volume" framing) | ~30 lines |

The analyzers already cover most of Tier 1–3. Tier 2 §5 and §7 are the
main pieces that need additional code beyond what's already written, and
both are small.

---

## Recommended order of work once the runs finish

1. **§1 headline table + §2 model ranking** — assemble from existing
   analyzer JSON. Get the paper's main number.
2. **§3 distributions** — plot the histograms. Check that the mean gaps
   aren't illusions from heavy-tailed distributions.
3. **§7 similarity × correctness** — the analysis that differentiates
   "mode collapse on low-quality output" from "genuine mode collapse."
   This is the one that determines how strong the replaceability claim
   can be.
4. **§5 orphan items** — pull examples for the qualitative appendix.
5. **§8 embedding × LLM-judge** — robustness check.
6. **§11 paper-level narrative** — discussion section material.

Tiers 3 §9 and §12 are nice-to-have. Tier 4 is future work / rebuttal
responses.
