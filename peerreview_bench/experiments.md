# Experiments Plan

## Paper Structure & Experiment Mapping

---

### Section 2: In which aspects are AI reviewers better or worse than human reviewers?

#### 2.1 Expert Annotation Study Design
- Description of 85 papers, 45 annotators, annotation protocol

#### 2.2 Experimental Results

**Main body:**
- **Table 1**: AI vs human reviewer quality (correctness, significance, evidence) — paper-level means with CIs
- **Table 5**: "Fully good" review items — descriptive stats and paired comparisons
- **Table 6**: Annotator-judged win rates (AI matching/exceeding best/worst human)
- **Figure 3**: [visualization from analysis/main_results/figure3.png]

**Appendix:**
- Table 2: Item-level descriptive statistics
- Table 3: Mixed-effects logistic regression
- Table 4: All pairwise comparisons

**Status:** ✅ All tables and figures generated in `analysis/main_results/`

---

### Section 3: To which extent do AI reviews overlap with human reviews?

#### 3.1 Experimental Setting
- 65,704 within-paper review-item pairs across 85 papers
- GPT-5.4 as the 4-way similarity judge (validated at 92.7% binary accuracy on 164-pair eval set)
- 4-way taxonomy: same issue+criticism+evidence / same issue+criticism / same issue / different issue

#### 3.2 Experimental Results

**Main body:**
- **Table 7**: Similarity judge accuracy on 164-pair eval set (7 LLM judges + 2 embedding baselines) — `similarity_check/main_results/table7.tex` ✅
- **Figure 5**: Stacked bar — overlap distribution across Human-Human, AI-AI, Human-AI pairs — `similarity_check/main_results/figure5.{png,svg}` ✅
- **Figure 6**: Coverage bar chart — 1H→1H, 1H→1AI, 3H→3AI with 3 overlap levels — `similarity_check/main_results/figure6.{png,svg}` ✅

**Appendix or supplementary:**
- **Table 8**: Full 3-way similarity rates with same/diff reviewer splits and bootstrap CIs — `similarity_check/main_results/table8.tex` ✅
- **Figure 7**: Correctness × similarity (null result: orphan items not more incorrect) — `similarity_check/main_results/figure7.pdf` ✅
- **Figure 8**: Per-reviewer coverage (1/2/3 AI reviewers, all vs fully-good items) — `similarity_check/main_results/figure8.pdf` ✅
- **Figure 9**: Novel AI contributions (matched vs orphan AI items quality) — `similarity_check/main_results/figure9.pdf` ✅
- **Table 9**: Coverage summary with bootstrap CIs — `similarity_check/main_results/table9.tex` ✅

**Key findings to highlight:**
1. AI-AI overlap (21%) >> Human-Human overlap (6.1%) ≈ Human-AI overlap (7.6%) → AI models converge on a "house style"
2. One AI covers one human's points at the same rate as another human does (26.5% vs 25.3%)
3. But 3 AIs only cover 35.2% of human criticisms — diminishing returns from AI ensemble
4. Orphan AI items (no human match) are 82% correct — AIs do surface valid novel points

**TODO:**
- [ ] Decide which figures go in main body vs appendix
- [ ] Update figure8/figure9 to output PNG+SVG instead of PDF
- [ ] Consider merging figure8 and figure9 into one composite figure

---

### Section 4: How could AI researchers develop better AI reviewers in the future?

#### 4.1 Qualitative analysis of expert feedback
- **Figure 4**: Bar chart of AI-reviewer strength/weakness categories (W1-W16, S1-S6) — `analysis/main_results/figure4.{png,svg}` ✅
- **Figure 5 (analysis)**: Case studies for top weaknesses/strengths — currently generated as image, plan to convert to promptbox LaTeX format

#### 4.2 Future directions for developing better AI reviewers
- Discussion based on qualitative patterns

#### 4.3 PEERREVIEW BENCH: A benchmark for automatically evaluating AI reviewers

**Experiments needed:**

**(A) Reviewer generation (peer review bench):**

| Model | Status | Notes |
|---|---|---|
| GPT-5.2 | ✅ Done | Pre-existing (80/80) |
| GPT-5.4 | 🔄 Running | |
| GPT-5.4-mini | ❌ Not started | |
| Gemini-3.0-Pro | ✅ Done | Pre-existing (80/80) |
| Gemini-3.1-Pro | ✅ Done | 80/80 |
| Gemini-3-Flash | 🔄 Running | |
| Claude-Opus-4.5 | ✅ Done | Pre-existing (80/80) |
| Claude-Opus-4.7 | ✅ Done | 80/80 |
| Claude-Sonnet-4.6 | ❌ Blocked | Anthropic credits exhausted |
| Grok-4.1-Fast | ❌ Blocked | FileEditorTool incompatibility |
| Kimi-K2.5 | ❌ Not started | |
| Qwen-3.6-Plus | ❌ Not started | |

**(B) Recall evaluation (similarity judge = GPT-5.4):**
- For each of the 12 models: run `evaluate_recall.py` against the 80-paper rubric
- Status: ❌ Not started (waiting for all reviews to be generated)

**(C) Precision evaluation (meta-reviewer = GPT-5.4 agent):**
- For each of the 12 models: run `evaluate_precision.py` with agent meta-reviewer
- Status: ❌ Not started (waiting for all reviews to be generated)

**(D) Recall + Precision → main results table:**
- Table showing per-model: recall, precision (per-axis), F1
- This is the headline table for Section 4.3

**Calibration benchmarks (validating the judges):**

**(E) Similarity Bench (validating recall judge):**
- Table 7: 7 LLM judges on 164-pair eval set ✅
- Key number: GPT-5.4 = 92.7% binary accuracy

**(F) Meta-Reviewer Bench (validating precision judge):**

| Model | Status | Papers |
|---|---|---|
| GPT-5.4 (v3-revert prompt) | ✅ Done | 27/27 |
| Gemini-3.1-Pro | ❌ Not started | |
| Claude-Opus-4.7 | ❌ Blocked | Anthropic credits exhausted |

Results for GPT-5.4 meta-reviewer:
- Correctness: 88.0%, Significance: 74.0%, Evidence: 92.8%, Tenclass: 30.2%

**(G) Human-Human vs Human-AI agreement (justifying the meta-reviewer):**
- Human-Human: 85.4% correctness, 58.8% significance, 87.7% evidence
- Human-AI: 82.3% correctness, 57.0% significance, 85.2% evidence
- Key finding: AI meta-reviewer agrees with humans at nearly the same rate as humans agree with each other
- Status: ✅ Computed, needs to be formatted as a table

**Figures and tables for Section 4.3:**
- [ ] Main results table: 12 models × recall/precision/F1
- [ ] Human-Human vs Human-AI agreement table
- [ ] Meta-reviewer bench results table (GPT-5.4 + other models)
- [ ] Tree visualization of 908-item annotation structure
- [ ] Per-axis accuracy breakdown figure

**Supporting documents (for appendix):**
- `metareview_bench/IMPROVING_META_REVIEWERS.md` ✅ — error analysis with concrete examples
- `similarity_check/IMPROVING_SIMILARITY_JUDGES.md` ✅ — error analysis with concrete examples
- `evaluation/model_configs.md` ✅ — hyperparameters for all 12 models

---

### Section 5: How should AI reviewers be deployed in scientific conferences and journals?

#### 5.1 Adding an AI reviewer to existing human reviewers

**Analysis needed:**
- Simulate: what if we replace 1 of 3 human reviewers with 1 AI reviewer?
- Compare panel-level metrics: correctness, significance, evidence, fully-good rate
- Use existing expert annotation data (85 papers, 6 reviewers each)
- For each paper, compute metrics for:
  - 3 humans (baseline)
  - 2 humans + best AI
  - 2 humans + worst AI
  - 2 humans + each AI model
- Status: ❌ Needs new analysis script

#### 5.2 Using an AI meta-reviewer to filter low-quality review items

**Analysis needed:**
- Simulate: apply the AI meta-reviewer to filter items rated Not Correct or Not Significant
- Measure: what fraction of human review items would be filtered? What fraction of those were actually low-quality?
- Use meta-reviewer bench data (27 papers with dual annotations)
- Precision/recall of the filter: how many actually-bad items does it catch, how many good items does it wrongly remove?
- Status: ❌ Needs new analysis script

#### 5.3 The CMU PAPER REVIEWER: An AI reviewer platform

**External baselines:**
- Stanford Reviewer: obtain reviews, evaluate via BYOJ on peer review bench
- OpenAI Reviewer: same
- CSPapers: same
- New table: external baselines vs our models on recall/precision

**Status:** ❌ Need to obtain external reviews and run evaluation

---

## Priority Order

**P0 — Blocking everything else:**
1. Finish running GPT-5.4 and Gemini-3-Flash reviews (in progress)
2. Get Anthropic credits refilled (blocks Claude-Sonnet-4.6 and Claude-Opus-4.7 meta-reviewer)

**P1 — Core results:**
3. Run remaining reviewer models (GPT-5.4-mini, Kimi-K2.5, Qwen-3.6-Plus)
4. Parse all new reviews into review_items JSON
5. Run recall evaluation for all 12 models
6. Run precision evaluation for all 12 models
7. Build the main results table (Section 4.3)

**P2 — Calibration & validation:**
8. Run Gemini-3.1-Pro on meta-reviewer bench
9. Format human-human vs human-AI agreement table
10. Build meta-reviewer bench results table

**P3 — Deployment analysis:**
11. Build Section 5.1 simulation (add AI to human panel)
12. Build Section 5.2 simulation (AI meta-reviewer filter)
13. Obtain external baseline reviews for Section 5.3

**P4 — Polish:**
14. Finalize all figures (consistent style, PNG+SVG)
15. Write IMPROVING_*.md content into appendix format
16. Update HF dataset with any remaining fixes

---

## Budget Estimate

Remaining budget: ~$1,719

| Task | Estimated cost |
|---|---|
| GPT-5.4 reviews (80 papers) | ~$200 |
| GPT-5.4-mini reviews | ~$80 |
| Gemini-3-Flash reviews | ~$40 |
| Kimi-K2.5 reviews | ~$120 |
| Qwen-3.6-Plus reviews | ~$100 |
| Recall evaluation (12 models × ~$80) | ~$960 |
| Precision evaluation (12 models × ~$80) | ~$960 |
| Gemini meta-reviewer bench | ~$50 |
| **Total estimated** | **~$2,510** |

⚠️ This exceeds the remaining budget by ~$800. Options:
- Skip some reviewer models (prioritize GPT-5.4, Gemini-3-Flash, Kimi, Qwen)
- Use a cheaper similarity judge for recall (embedding-based instead of LLM)
- Use a cheaper meta-reviewer for precision (LLM call instead of agent)
- Request additional budget
