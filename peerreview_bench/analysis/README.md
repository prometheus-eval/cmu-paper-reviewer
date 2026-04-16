# analysis/

Statistical analysis of the PeerReview Bench expert annotations.

Four analyses that answer different questions about the same data:

| Script | Question | Output |
|---|---|---|
| `peerreview_analysis.py` | On average, which reviewer group is better on each metric? (item-level descriptive tables) | `analysis_output/` |
| `peerreview_analysis_per_paper.py` | Same question in a per-paper paired framing, plus all pairwise tests | `analysis_output_per_paper/` |
| `peerreview_analysis_glmm.py` | Are the conclusions robust under a true logistic GLMM? | `glmm_output/` |
| `inter_rater_reliability.py` | Do primary and secondary meta-reviewers agree? (κ + Gwet's AC1) | `irr_output/` |

## Data source

All four scripts load the `expert_annotation` config from the
`prometheus-eval/peerreview-bench` HuggingFace dataset (split `eval`) via
`load_data.load_annotations`. The loader is HuggingFace-only — there is no
local JSON fallback. If the dataset hasn't been pushed, the scripts raise
a clear error telling you how to push.

By default, the loader returns **both** primary and secondary annotator
rows merged as independent data points (items from the 27 overlap papers
contribute twice, items from the other papers contribute once). Pass
`--annotator-source primary` or `secondary` to load only one side.

The per-paper Best/Worst Human rankings come from
`../reviewer_rankings.json` (a small local sidecar; the ranking column
was dropped from the HF schema to keep the row schema clean).

## Running

```bash
cd analysis
./run_analysis.sh
```

This runs all four analyses in sequence. Each script can also be run
standalone:

```bash
python3 peerreview_analysis.py --output-dir ../outputs/analysis/analysis_output
python3 peerreview_analysis.py --annotator-source primary    # primary only
python3 peerreview_analysis.py --annotator-source both       # merged (default)
```

## Data quality handling (cascade stripping)

The annotation form follows a cascade: correctness → significance →
evidence. Downstream labels are only meaningful if the upstream decision
says to fill them in. `data_filter.py` classifies each annotation into
one of these categories and handles each consistently:

- **valid_not_correct** (Not Correct, no downstream labels) — kept
- **valid_correct_not_sig** (Correct + Not Significant, no evidence) — kept
- **valid_correct_sig** (Correct + Marginally/Significant + evidence labeled) — kept
- **invalid_missing_sig** (Correct with no significance) — dropped entirely
- **invalid_missing_evi** (Correct + sig > 0 with no evidence) — dropped entirely
- **overly_annotated_sig** (Not Correct but significance was still entered) — kept
  for correctness, downstream sig/evi stripped
- **overly_annotated_evi** (Correct + Not Significant but evidence was still
  entered) — kept for correctness + significance, downstream evi stripped

Significance is always 3-class: the original 4-option "Very Significant"
radio option is merged into "Significant" upstream at HF upload time.

## Statistical methodology

### Primary analysis — `peerreview_analysis.py`
Reports a dataset overview with a cascade tree visualization, per-group
"fully good" rates (Correct + Significant + Sufficient) with Wilson 95%
CIs, and three item-level descriptive tables (Correctness / Significance /
Evidence) with correct/incorrect or sufficient/not-sufficient counts, total
items, and number of papers covered per group. It does NOT print the
pairwise paired tests — those are emitted by the per-paper analysis
(`peerreview_analysis_per_paper.py`) and are identical across both scripts,
so they're printed only once to avoid duplication. The pairwise-comparison
CSVs (`{correctness,significance,evidence}_comparisons.csv`) are still
written alongside the primary report for programmatic access.

### Per-paper paired analysis — `peerreview_analysis_per_paper.py`
Paper-level aggregated framing with **four** pairwise paired comparisons:
Correctness, Significance, Evidence, and **Fully Good**.

Per-metric test, effect size, and CI:

- **Correctness / Evidence / Fully Good** (all binary rates)
  - Item-level rate CI: Wilson score interval
  - Paired test: paired t-test on per-paper rate differences
  - Effect size: Cohen's d (paired) = mean(diff) / SD(diff)
  - Diff CI: t-interval on the paired paper-level differences
- **Significance** (ordinal 3-class: 0 / 1 / 2)
  - Item-level mean CI: non-parametric bootstrap (10,000 resamples)
  - Paired test: Wilcoxon signed-rank on per-paper mean differences
  - Effect size: rank-biserial r (Kerby 2014)
  - Diff CI: t-interval on the paired paper-level differences

Raw p-values are reported at α = 0.05. No multiple-comparison correction.

### GLMM sensitivity — `peerreview_analysis_glmm.py`
- Correctness / Evidence: true logistic GLMM via `BinomialBayesMixedGLM`
  (variational Bayes, logit link, paper random intercept). Falls back to
  GEE-Binomial if VB fails.
- Significance (ordinal): two binomial GLMMs at cut points (P(Y ≥ 1) and
  P(Y = 2)), plus a linear mixed model sensitivity check.
- Reports odds ratios vs Best Human and Worst Human reference groups,
  with 95% credible intervals. Sensitivity complement to the paired
  tests in the primary analysis.

### Inter-rater reliability — `inter_rater_reliability.py`
For the 27 overlap papers where both primary and secondary annotated the
same items, reports per-axis × per-reviewer-type agreement, plus an
"Overall (combined)" section that collapses Human + AI together:

- **Percent agreement** (raw match rate)
- **Cohen's κ** (with Landis-Koch interpretation labels)
- **Gwet's AC1** (same interpretation labels; more robust to class
  imbalance — recommended over κ when one category dominates the marginals)

Also reports the 10-class `meta_reviewer` label distribution (the joint
(primary, secondary) outcome across the cascade).

## Output files

Each analysis writes to a subdirectory under `../outputs/analysis/`:

- `*_report.txt` — human-readable report
- `*_comparisons*.csv` — pairwise comparisons in tidy format
- `dataset_overview.json` — paper/annotator/item/cascade counts
- `fully_good_stats.json` — Correct + Significant + Sufficient ratio per group, with Wilson CIs
- `irr_summary.json` — agreement table + meta_reviewer label distribution
- `irr_item_level.csv` — per-item primary-vs-secondary comparison (for auditing disagreements)

## Single-human-reviewer papers

On papers that have only one human reviewer, `reviewer_rankings.json`
encodes `best == worst == <that reviewer>`. The grouping logic puts the
single reviewer's items into *both* Best Human and Worst Human groups so
paper coverage stays balanced across the two groups. The Best vs Worst
paired comparison on such papers contributes a zero paired difference,
which Wilcoxon drops from its ranking and the t-test treats as an honest
zero — effect on the final p-value is negligible.
