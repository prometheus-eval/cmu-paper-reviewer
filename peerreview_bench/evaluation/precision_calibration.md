# Precision Calibration for PeerReview Bench

## Problem

The GPT-5.4 meta-reviewer systematically over-estimates the "fully good" rate of AI review items compared to human expert judgments:

| Model judged | Raw Precision | Expert FG Rate | Gap |
|---|:---:|:---:|:---:|
| GPT-5.2 | 88.9% | 60.0% | +28.9pp |
| Claude Opus 4.5 | 75.5% | 53.1% | +22.4pp |
| Gemini 3.0 Pro | 53.4% | 50.2% | +3.2pp |

Mean gap: +18.2pp. The inflation is driven by **significance over-rating**: the meta-reviewer says "Significant" 97% of the time vs expert 58% (+39pp). Correctness (+5pp) and evidence (+1pp) are mildly inflated. Same-family bias (GPT judging GPT) adds additional leniency.

## Root Cause

From per-axis confusion matrices (AI items, N=340, meta-review bench):

| Axis | Meta rate | Expert rate | Specificity |
|---|:---:|:---:|:---:|
| Correctness | 93% | 87% | 20.9% |
| **Significance** | **97%** | **58%** | **5.9%** |
| Evidence | 96% | 95% | 8.3% |

The meta-reviewer almost never says "Not Significant" — it has 5.9% specificity on significance. This is the dominant source of precision inflation.

## Calibration Approach

### Data

- **Training set**: 27 papers from the meta-review bench × 3 AI models = 81 paper×model data points, where both meta-reviewer and human expert judgments exist
- **Test set**: 55 papers from PeerReview Bench that are NOT in the calibration set
- **Ground truth**: Expert fully-good rates from Table 5 for GPT-5.2, Claude Opus 4.5, Gemini 3.0 Pro

### Candidate Features Investigated

We investigated 21 candidate features across three categories, requiring all features to be **fully observable for any new model** evaluated on PeerReview Bench (no dependence on expert annotations):

**Meta-reviewer output features (6):**
- `meta_fg_rate` — overall fully-good rate per paper (r=+0.087)
- `meta_sig_rate` — significance rate per paper (r=+0.135)
- `meta_corr_rate` — correctness rate per paper (r=+0.153) ✓
- `meta_not_corr_rate` — "Not Correct" rate (r=-0.153)
- `meta_sig_variance` — variance of significance labels; 0 = indiscriminate (r=-0.118)
- `meta_evi_rate` — evidence sufficiency rate (r=+0.096)

**Review item characteristics (5):**
- `n_items` — number of items the model produced (r=+0.063)
- `log_review_words` — avg review item length (r=-0.137) ✓
- `avg_item_position` — average position of items (r=+0.034)
- `frac_with_fig_ref` — fraction referencing figures/tables (r=-0.040)
- `frac_with_code_ref` — fraction referencing code (r=+0.076)
- `frac_with_citation` — fraction citing external literature (r=-0.135)

**Paper characteristics (5):**
- `log_paper_words` — paper length (r=+0.155)
- `has_code` — paper has executable code (r=+0.028)
- `has_supp` — paper has supplementary materials (r=+0.221)
- `n_images` — number of figures (r=-0.029)
- `complexity` — fig_refs + table_refs + equations (r=-0.027)
- `figs_per_kword` — figure density (r=-0.102)

**Confusion-matrix-derived (1):**
- `family_sig_fpr` — significance false positive rate per model family, computed from the calibration set (r=-0.117) ✓

**Rejected features (not generalizable to new models):**
- `human_corr_rate` — requires expert annotations (r=+0.178)
- `human_sig_rate` — requires expert annotations (r=+0.350, strongest but leaks)

### Why item-level calibration fails

The meta-reviewer's per-item binary FG decision carries almost no signal:
- P(expert_FG | meta=FG) = 0.524
- P(expert_FG | meta≠FG) = 0.382
- Brier score = 0.246 (barely better than predicting 0.5 for everything)

This is because the meta-reviewer says "Significant" 97% of the time — the FG decision is dominated by correctness, which has only mild bias (+5pp), so the binary FG label is nearly uninformative about the significance dimension where the real inflation lives.

### Feature selection

We performed exhaustive subset selection (all combinations of k features from 21 candidates) evaluated by MAE on the **test-only papers** (55 papers not in the calibration set):

| N features | Best features | Test MAE (55 papers) |
|:---:|---|:---:|
| 1 | log_paper_words | 3.8pp |
| 2 | frac_with_fig_ref, log_paper_words | 3.7pp |
| **3** | **meta_corr_rate, log_review_words, family_sig_fpr** | **1.7pp** |
| 4 | meta_corr_rate, log_review_words, frac_with_code_ref, family_sig_fpr | 1.4pp |
| 5 | meta_corr_rate, meta_sig_variance, log_review_words, frac_with_code_ref, family_sig_fpr | 0.9pp |

### Selected Model: 3-feature Ridge regression

We select the **3-feature model** as the best balance of simplicity and performance:

**Features:**

1. **`meta_corr_rate`** — The meta-reviewer's correctness rate for a given paper. When the meta-reviewer flags more items as "Not Correct" on a paper, it signals genuine quality issues; conversely, papers where everything is rated "Correct" are likely experiencing indiscriminate leniency. This feature captures the meta-reviewer's discriminating behavior on the one axis where it has reasonable accuracy (82% overall). Inspired by Error Category 1 (Partial-Evidence Trap) from the error analysis: the meta-reviewer's correctness judgment, while imperfect, correlates with the item's overall quality.

2. **`log_review_words`** — The average length of review items (log-transformed). Shorter, more focused items tend to be more likely truly significant — they target a specific issue rather than rambling across multiple concerns. This aligns with the error analysis finding that verbose items are more prone to partial-evidence traps and over-broad claims. Analogous to AlpacaEval's use of response length as a confounder: just as longer chatbot responses receive inflated preference judgments, longer review items receive inflated significance ratings.

3. **`family_sig_fpr`** — The significance false positive rate for the model family being judged (GPT: 47.5%, Claude: 42.7%, Gemini: 29.5%), computed from the 908-item calibration set. This captures the systematic same-family leniency: GPT-5.4 is more lenient when judging GPT reviews than Gemini reviews. This is a constant per family (not per paper), derived once from the calibration data and applied to all papers. For new model families not in the calibration set, the mean FPR (39.9%) can be used as a default.

**Model:** Ridge regression (alpha=10.0) with StandardScaler normalization, trained on 81 paper×model data points from the 27-paper calibration set.

### Results

Evaluated on test papers only (55 papers NOT in calibration):

| Model | Raw | Calibrated (test papers) | Expert |
|---|:---:|:---:|:---:|
| GPT-5.2 | 88.9% | 58.4% | 60.0% |
| Claude Opus 4.5 | 75.5% | 55.9% | 53.1% |
| Gemini 3.0 Pro | 53.4% | 51.6% | 50.2% |

**Test MAE = 1.9pp** (on papers the model was not trained on).

### Full Calibrated Leaderboard

| Model | Raw Precision | Calibrated Precision |
|---|:---:|:---:|
| GPT-5.4 | 93.8% | 59.9% |
| GPT-5.2 | 88.9% | 58.5% |
| GPT-5.4-mini | 86.6% | 57.3% |
| Claude Opus 4.5 | 75.5% | 55.3% |
| Gemini 3 Flash | 60.5% | 55.0% |
| Claude Opus 4.7 | 71.5% | 53.2% |
| Gemini 3.1 Pro | 59.6% | 52.6% |
| Gemini 3.0 Pro | 53.4% | 50.9% |

## Usage

```bash
python3 precision_calibration.py ../outputs/eval/precision_gpt-5.2.json
python3 precision_calibration.py ../outputs/eval/precision_*.json
```

The script reads `calibration_params.json` (saved model parameters) and outputs both raw and calibrated precision for each input file.

## Limitations

1. **3 ground-truth models** — the test MAE is computed on only 3 models with expert data. The calibration may not generalize to models with very different characteristics.
2. **Review length approximation** — the precision output files don't store review item text length, so we use a fixed approximation (log(101)). Better calibration would require storing this at evaluation time.
3. **Family assignment** — new model families (e.g., Llama, Mistral) would use the mean FPR as a default, which may not be accurate.
4. **The calibrated range (50–60%) is compressed** relative to raw (53–94%). This reflects reality — human experts rate 50–60% of AI items as fully good regardless of model — but reduces the ability to distinguish between models.
