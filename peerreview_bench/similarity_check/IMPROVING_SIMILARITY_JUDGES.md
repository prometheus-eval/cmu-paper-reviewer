# Improving LLM-Based Similarity Judges: Error Analysis and Guidelines

## Introduction

This document reports an error analysis of 7 frontier LLM judges evaluated on 164 expert-annotated review-item pairs from scientific peer review. Each pair consists of two review items (drawn from AI-generated or human reviews of the same paper) and an expert label classifying their relationship along a 4-way taxonomy:

1. **"same subject, same argument, same evidence"** (near-paraphrase) -- binary: similar
2. **"same subject, same argument, different evidence"** (convergent conclusion) -- binary: similar
3. **"same subject, different argument"** (topical neighbor) -- binary: not_similar
4. **"different subject"** (unrelated) -- binary: not_similar

Binary accuracy across most models clusters at ~92.7%, while 4-way accuracy ranges from ~80% to ~84%. This document catalogs the systematic failure modes, provides concrete examples, and proposes improvement directions.

---

## Overview of Results

### Model Accuracy Table

| Model | 4-Way Accuracy | 4-Way Errors | Binary Accuracy | Binary Errors |
|---|---|---|---|---|
| Gemini 3.1 Pro Preview | **83.5%** | 27 | 92.7% | 12 |
| Claude Opus 4.6 | 81.7% | 30 | 92.7% | 12 |
| GPT-5.4 | 81.1% | 31 | 92.7% | 12 |
| Claude Sonnet 4.6 | 80.5% | 32 | 92.7% | 12 |
| GPT-5.4 Mini | 80.5% | 32 | 89.0% | 18 |
| Claude Opus 4.7 | 79.9% | 33 | 92.7% | 12 |
| Gemini 3 Flash Preview | 79.9% | 33 | 92.1% | 13 |

Key observations:
- Six of seven models converge on exactly 92.7% binary accuracy (12 errors on 164 pairs), suggesting a shared accuracy ceiling.
- GPT-5.4 Mini is a clear outlier with 18 binary errors (89.0%), making 6 additional mistakes compared to the consensus group.
- Gemini 3.1 Pro leads on 4-way classification (83.5%) but ties on binary.
- 4-way accuracy varies more (79.9%--83.5%), indicating that fine-grained distinction is the harder and more differentiating task.

### Universal vs. Model-Specific Errors

**Pairs wrong across all 7 models (binary):**
- **eval_005** (paper 51, AI-AI): BBB resealing vs. targeted delivery -- GT: similar
- **eval_059** (paper 63, AI-Human): ONE performance claim vs. CNN baseline request -- GT: similar
- **eval_120** (paper 32, AI-AI): Statistical rigor of outperformance vs. missing LiBOB-only control -- GT: not_similar
- **eval_134** (paper 59, AI-AI): Extraction method biasing comparisons vs. underestimating ISI toxicity -- GT: not_similar

**Pairs wrong by 6 of 7 models (binary):**
- **eval_023** (paper 74, AI-AI): ML model generalizability vs. validation non-independence -- GT: similar
- **eval_078** (paper 66, AI-Human): PDM fitting parameters vs. single-layer modeling inadequacy -- GT: similar
- **eval_090** (paper 18, AI-Human): Unrealistic PIC parameters vs. unexplained simulation waves -- GT: similar
- **eval_096** (paper 28, AI-Human): Priority effects unsupported vs. term misapplication -- GT: similar
- **eval_141** (paper 28, AI-AI): gLV random parameters vs. exchangeable model universality claim -- GT: not_similar

These patterns reveal that the hardest cases are not random -- they cluster around specific types of semantic relationships that consistently confuse all frontier models.

---

## Error Category 1: Convergent Conclusions Misclassified as "Different Subject" (False Negatives -- Similar Predicted as Not Similar)

This is the dominant binary error pattern. The ground truth is "same subject, same argument, different evidence" (convergent conclusion), but the model predicts "different subject" -- the most extreme disagreement possible. These are pairs where two reviewers independently arrive at the same high-level criticism but through entirely different reasoning paths, making the surface-level text look unrelated.

### Example 1.1: eval_005 (Paper 51, All 7 Models Wrong)

**Item A (Claude):** "The paper claims that platelet-inspired nanoparticles (PIN) 'reseal the compromised BBB' and promote hemostasis, but the sole evidence for this claim is reduced IgG staining -- an indirect and non-specific marker that cannot distinguish between BBB structural repair and reduced inflammation-induced permeability. No direct markers of BBB structural integrity (e.g., tight junction proteins) were assessed..."

**Item B (Gemini):** "The paper claims that the platelet-inspired nanoparticles (PIN) achieve targeted delivery and accumulation at the implant site, but it fails to provide direct experimental evidence (such as biodistribution studies or in vivo imaging) to support this mechanism in the manuscript."

**Ground truth:** same subject, same argument, different evidence (binary: similar)
**Expert rationale:** "Both about targeted delivery lacking evidence; Claude about TJ proteins, Gemini about IVIS folder omission"

**Why every model failed:** The surface-level subjects appear different -- one discusses "BBB resealing" and the other discusses "targeted delivery." Models correctly identified these as different mechanistic claims. However, the expert annotation considers both items to be criticizing the same higher-level subject: the inadequacy of evidence for PIN's mechanism of action. The shared argument is "the paper's mechanistic claims lack direct experimental evidence."

**Lesson:** Models are too literal in subject matching. When two items criticize different facets of the same overarching claim, models fail to recognize the shared abstract subject ("evidence for PIN mechanism") because the concrete topics (BBB repair vs. targeted delivery) differ.

### Example 1.2: eval_059 (Paper 63, All 7 Models Wrong)

**Item A (Gemini):** "The paper claims that the ONE architecture shows 'comparable and better performance' than other ML models, but this claim is unsubstantiated as ONE is essentially an optical implementation of the Fourier Neural Operator (FNO) and no evidence is provided to show it outperforms the standard FNO in terms of accuracy."

**Item B (Human_2):** "I would recommend the authors perform an additional baseline comparing the performance of training only the CNN from Fig. 5 on the Darcy Flow and Navier Stokes datasets, in comparison to the current case that uses the experimental hardware in tandem with the postprocessing CNN."

**Ground truth:** same subject, same argument, different evidence (binary: similar)
**Expert rationale:** "Human_2 explicitly states ONE 'is essentially just a 2D FNO architecture' -- same underlying criticism that ONE is not a novel architecture beyond FNO, even though framed as a simulation-vs-experiment labeling request"

**Why every model failed:** Item B is phrased as a constructive suggestion ("I would recommend...") rather than a criticism, and it discusses a CNN baseline rather than FNO directly. Models interpreted Item B as a request for an ablation study (different subject) rather than recognizing it as an implicit critique of ONE's novelty. The human reviewer's underlying point -- that ONE may just be a software CNN with optical decoration -- matches Item A's explicit critique.

**Lesson:** Models struggle to match explicit criticisms against implicit ones embedded in constructive suggestions. The same argument ("ONE does not add value over its software counterpart") surfaces as a direct claim in Item A and as a baseline-request in Item B.

### Example 1.3: eval_023 (Paper 74, 6 of 7 Models Wrong)

**Item A (Claude):** "The random forest classifier was trained exclusively on samples sequenced by a single center (NYGC) using a single platform (NovaSeq 6000) at similar coverage (~30x), yet the paper claims the model generalizes well based on testing with only one sample (HG002)..."

**Item B (GPT):** "This paper's conclusion that SurVIndel2 yields a substantially more complete 1000 Genomes CNV catalogue with 'high validation rate' is insufficiently supported because the validation is reported only 'in samples with long reads' that overlap the HGSVC2 cohort used to train and benchmark SurVIndel2's filtering model, making the validation non-independent..."

**Ground truth:** same subject, same argument, different evidence (binary: similar)
**Expert rationale:** "Both about training/test non-independence; Claude cites NYGC homogeneity, GPT cites HGSVC2 overlap"

**Why models failed:** Item A is about model generalization (training on homogeneous data), while Item B is about validation circularity (validation on training-overlapping samples). Models correctly identified these as distinct analytical concerns. The expert, however, views both as instances of the same argument: "the model's evaluation is non-independent from its training data." The shared abstract argument is obscured by the different concrete manifestations (homogeneous training vs. overlapping validation).

**Lesson:** Two items can express the same abstract methodological concern (non-independent evaluation) through completely different concrete instantiations. Models need to reason at a higher level of abstraction about the nature of the criticism.

---

## Error Category 2: Topical Neighbors Misclassified as Convergent Conclusions (False Positives -- Not Similar Predicted as Similar)

The second major binary error pattern: the ground truth is "same subject, different argument" (topical neighbor, binary: not_similar), but the model predicts "same subject, same argument, different evidence" (binary: similar). These are pairs where two reviewers discuss the same part of a paper but raise genuinely distinct objections -- and the model collapses the argument distinction.

### Example 2.1: eval_120 (Paper 32, All 7 Models Wrong)

**Item A (GPT):** "The manuscript's claim that ML-identified additives 'outperformed' the baseline and the initial dataset is not statistically well supported because Table 1 reports single-point performance metrics without stating the number of replicate cells, measurement variability, or uncertainty propagation..."

**Item B (Gemini):** "The paper claims a synergistic effect for the discovered binary additive (LiBOB + SA) but fails to compare it against LiBOB alone, making it impossible to attribute the performance to the combination."

**Ground truth:** same subject, different argument (binary: not_similar)
**Expert rationale:** "Both items reference the same comparative claim in Table 1 that LiBOB+SA 'outperformed all additives in the initial dataset' -- A complains about lack of replicates, B about the missing LiBOB-only control."

**Why every model failed:** Both items critique the claim that LiBOB+SA outperformed alternatives. Models saw the shared target claim and concluded the argument was the same ("the outperformance claim is inadequately supported"). But the expert distinguishes the arguments: "lack of statistical rigor" (no replicates, no error bars) is a fundamentally different methodological objection from "missing ablation control" (no LiBOB-only baseline). The model treats "claim X is inadequately supported" as a single argument, while the expert differentiates by the nature of the inadequacy.

**Lesson:** Models over-abstract arguments. "The claim is not well supported" can encompass many distinct objections. The 4-way taxonomy requires distinguishing the specific type of inadequacy, not just the presence of a critique against the same claim.

### Example 2.2: eval_134 (Paper 59, All 7 Models Wrong)

**Item A:** The methanol/water extraction method may bias inter-sector toxicity comparisons because differential solubility of toxic species across sources distorts the comparison.

**Item B:** The methanol/water extraction method underestimates ISI (iron and steel industry) toxicity specifically because insoluble metal oxides and surface-bound reactivity are missed.

**Ground truth:** same subject, different argument (binary: not_similar)

**Why every model failed:** Both items critique the same extraction method for toxicity testing. Models collapsed this into "the extraction method is inadequate" as a shared argument. The expert distinguishes: Item A argues the method creates a comparative bias (distorting sector rankings), while Item B argues the method causes an absolute underestimate (missing specific toxic species). The direction and scope of the flaw differ even though the targeted method is identical.

**Lesson:** Even when the subject is identical and both items find fault with the same method, the specific nature of the flaw constitutes a different argument. "Biases comparisons" is not the same as "underestimates absolute toxicity."

### Example 2.3: eval_141 (Paper 28, 6 of 7 Models Wrong)

**Item A:** The gLV model uses randomly sampled interaction matrices rather than parameters fitted to experimental data, limiting its mechanistic explanatory power.

**Item B:** The gLV model's exchangeable setup (identical growth rates, carrying capacities, random interactions) makes the "universal collapse" of invasion probability onto survival fraction trivially expected rather than a meaningful finding.

**Ground truth:** same subject, different argument (binary: not_similar)

**Why models failed:** Both critique the gLV theoretical model. Models saw "gLV model is inadequate" as a shared argument. The expert distinguishes: Item A's argument is about parameter grounding (the model is not fitted to data), while Item B's argument is about mathematical triviality (the model's predictions are not informative given its assumptions). These are distinct intellectual objections.

---

## Error Category 3: Evidence Granularity Confusion (4-Way Only)

The most common 4-way error (that does not affect binary accuracy) is confusing "same evidence" with "different evidence" -- both under the "similar" binary label. This accounts for the majority of 4-way errors across all models.

### Example 3.1: eval_006 (Paper 31, 5 of 7 Models Wrong at 4-Way)

**Item A (Claude):** Argues the comparison between GECKO ecModel (eciML1515) and ME-model (iJL1678) is misleading due to different model architectures, objectives, and optimization approaches.

**Item B (Gemini):** Argues the same comparison is "superficial and misleading" because it "compares a simplified, fitted model against a mechanistic ab initio model on metrics where the former has an inherent advantage due to parameter fitting."

**Ground truth:** same subject, same argument, same evidence
**Expert rationale:** "Both cite the exact same SSE 0.27 vs 1.21 GECKO vs ME-model quote and same Figure 3B text"

**Model prediction (typical):** same subject, same argument, different evidence

**Why models failed:** Both items quote the same SSE values and Figure 3B, but they frame the evidence differently. Claude emphasizes architectural differences; Gemini emphasizes the fitted-vs-ab-initio distinction. Models interpreted the different framing as different evidence, but the expert considers the quoted paper passages (the core evidence) to be identical.

**Lesson:** Models confuse "different interpretation of the same evidence" with "different evidence." The 4-way taxonomy defines evidence by what paper passages/data are cited, not by how they are interpreted.

### Example 3.2: eval_092 (Paper 37, 4 of 7 Models Wrong at 4-Way)

**Item A (Claude):** "The paper names its model 'Enformer,' which is identical to the widely recognized transformer-based genomics model for gene expression prediction published by DeepMind in Nature Methods in 2021."

**Item B (Gemini):** "The paper names its model 'Enformer', which is identical to the name of a highly influential deep learning model for genomics developed by DeepMind and published in Nature Methods in 2021."

**Ground truth:** same subject, same argument, same evidence
**Expert rationale:** "Both cite the exact Nature Methods Enformer paper + DeepMind blog -- same name collision concern, same references"

**Model prediction (typical):** same subject, same argument, different evidence

**Why models failed:** The items are near-paraphrases, yet models still classified the evidence as "different." This appears to be a calibration issue: models are biased toward predicting "different evidence" when any textual variation exists, even when the underlying citations and data points are identical.

### Example 3.3: eval_074 (Paper 63, 5 of 7 Models Wrong at 4-Way)

**Item A (Gemini):** Cites "Reconfigurable Optical Computer" (ROC) works from 2020 as prior art contradicting the paper's novelty claim.

**Item B (Human_2):** Cites Zhao et al. works on optical PINN training as prior art contradicting the same claim.

**Ground truth:** same subject, same argument, same evidence
**Model prediction (typical):** same subject, same argument, different evidence

**Why models were arguably right:** The items cite entirely different prior works (ROC vs. Zhao et al.). The expert labeled this "same evidence," apparently interpreting "evidence" at the level of "prior art exists" rather than at the level of "which specific papers are cited." This is a genuine boundary case in the taxonomy.

**Lesson:** The definition of "same evidence" is ambiguous. Does it mean the same data/citations, or the same type of evidence? This ambiguity drives a large fraction of 4-way errors.

---

## Error Category 4: Asymmetric Formulation Pairs (AI-Human Mismatches)

A disproportionate number of binary errors involve AI-Human pairs where the human review item uses a markedly different rhetorical style from the AI item. Human reviewers tend to be terser, more implicit, or frame criticisms as questions rather than assertions.

### Example 4.1: eval_090 (Paper 18, 6 of 7 Models Wrong)

**Item A (Claude):** "The PIC simulations use a reduced mass ratio (mi/me = 100) and an unrealistically low ratio of light speed to Alfven speed (c/vA = 2), which may affect the electron dynamics central to the claimed shearing mechanism, and the paper does not quantify the impact of these choices."

**Item B (Human_2):** "As it is commonly done in PIC simulations, the authors use a nonrealistic speed of light to Alfven speed ratio (c/va). The results presented in the Extended Data section show that a larger value of c/va results in the production of waves. The author state that these plasma oscillations damp away without affecting the core relaxation mechanisms. I have a few questions on this point. Which kind of waves are produced?..."

**Ground truth:** same subject, same argument, different evidence (binary: similar)
**Expert rationale:** "Both about unrealistic c/vA ratio; Claude cites mass ratio + c/vA, Human_2 focuses on waves"

**Why models failed:** Item A is a declarative criticism about parameter realism. Item B is a series of questions about the consequences of parameter choices. Models interpreted Item A as "parameters are unrealistic" (validity concern) and Item B as "what are these waves?" (phenomenological inquiry), classifying them as different arguments. The expert considers both to express the same underlying concern: the PIC simulation parameters may not support the paper's conclusions.

**Lesson:** Human reviewers often express the same argument as a question rather than an assertion. Models need to recognize that "Does X affect Y?" is functionally equivalent to "X may affect Y" as a critique.

### Example 4.2: eval_096 (Paper 28, 6 of 7 Models Wrong)

**Item A (Gemini):** "The claim that 'emergent priority effects' explain the reduced invasibility in high-nutrient communities is speculative and not supported by direct experimental evidence of history dependence."

**Item B (Human_1):** "My second main comment concerns the notion of priority effects. Here, as I understand it, the authors use this term basically to mean that a community of strongly interacting species has established itself, which makes invasions less likely... but I am not convinced that this is the sense in which most ecologists use 'priority effects'..."

**Ground truth:** same subject, same argument, different evidence (binary: similar)

**Why models failed:** Item A says "the claim is unsupported." Item B says "the term is misused." Models classified these as different arguments (evidence gap vs. terminological misuse). The expert considers both to be saying the same thing: the paper's invocation of "priority effects" is not warranted. The human reviewer's terminological objection is, at its core, the same as the AI reviewer's evidential objection -- both say the data does not support calling this phenomenon "priority effects."

**Lesson:** Terminological critiques and evidential critiques can be the same argument when the terminological concern stems from the same evidential gap.

---

## Error Category 5: Scope Mismatch Between Items

Some errors arise when one item is narrow and specific while the other is broad, encompassing the narrow item as a sub-point. Models sometimes classify these as "different subject" because the scope differs, even though the narrow item is contained within the broad one.

### Example 5.1: eval_045 (Paper 1, 3 of 7 Models Wrong at Binary)

**Item A (Gemini):** Focuses narrowly on the graphene calculation using a 2x2 supercell, arguing this is too small and no finite-size extrapolation was done.

**Item B (Human_3):** Broadly argues that the paper's claim about investigating long-range effectiveness is weakened because calculations use "relatively small system sizes" and no evidence of convergence is presented.

**Ground truth:** same subject, same argument, different evidence (binary: similar)

**Why models failed:** Item A is specifically about graphene; Item B is about all systems in the paper. Models saw "graphene" vs. "all systems" as different subjects. The expert considers the graphene-specific critique to be an instance of the general critique about system sizes.

### Example 5.2: eval_013 (Paper 36, 4 of 7 Models Wrong at Binary)

**Item A (Claude):** Narrowly argues that multi-omics integration did not improve model performance (AUC unchanged at 0.70--0.72).

**Item B (GPT):** Broadly argues that conclusions about "OA-specific biomarkers" are not robust because (1) omics does not improve performance AND (2) feature rankings do not establish clinically actionable biomarkers.

**Ground truth:** same subject, same argument, different evidence (binary: similar)

**Why models failed:** Item B has a broader scope (biomarker conclusions) that encompasses Item A's narrower point (omics utility). Models saw "omics utility for prediction" vs. "biomarker conclusions" as different arguments. The expert considers them both to be arguing that the multi-omics integration does not deliver on its promises.

---

## Cross-Model Patterns

### Pair Type Analysis

Binary errors by pair type across the 6 models with 12 binary errors each:

| Pattern | AI-AI Pairs | AI-Human Pairs | Human-Human Pairs |
|---|---|---|---|
| False negatives (similar predicted as not_similar) | eval_005, eval_023, eval_113, eval_116 | eval_015, eval_045, eval_055, eval_059, eval_078, eval_090, eval_096 | -- |
| False positives (not_similar predicted as similar) | eval_120, eval_134, eval_140, eval_141 | eval_106 | -- |

Key findings:
- **AI-Human pairs are disproportionately represented in false negatives.** When the ground truth is "similar" but one item is from a human reviewer and the other from an AI, models often miss the similarity. This is because human reviewers use shorter, more implicit, question-based formulations.
- **AI-AI pairs dominate false positives.** When two AI-generated items discuss the same paper section, models tend to over-attribute similarity, collapsing "same subject, different argument" into "same subject, same argument."
- **No Human-Human pairs appear in binary errors**, suggesting that the evaluation set may lack Human-Human pairs in boundary regions, or that Human-Human pairs are easier to classify.

### Confusion Matrix (4-Way, Aggregated Across 6 Models with 12 Binary Errors)

The dominant 4-way confusion patterns:

| Ground Truth | Most Common Misclassification | Frequency |
|---|---|---|
| same subject, same argument, same evidence | same subject, same argument, different evidence | Very high (most common 4-way error) |
| same subject, same argument, different evidence | same subject, different argument | High (causes binary errors) |
| same subject, same argument, different evidence | different subject | Moderate (worst-case binary error) |
| same subject, different argument | same subject, same argument, different evidence | Moderate (causes binary errors) |
| same subject, different argument | different subject | Low (4-way only, binary correct) |
| different subject | same subject, different argument | Low |

The critical binary-impacting confusions are:
1. **"convergent conclusion" misclassified as "topical neighbor" or "different subject"** -- the model fails to see that different-looking critiques share the same abstract argument.
2. **"topical neighbor" misclassified as "convergent conclusion"** -- the model fails to distinguish genuinely different arguments about the same subject.

---

## Guidelines for Future Work

### 1. Prompt Engineering Improvements

**1.1. Explicit abstraction instructions.** The most common failure mode is literal subject/argument matching. Prompts should instruct the model to:
- First identify the abstract subject at the level of "what aspect of the paper is being criticized" rather than "what specific claim is discussed."
- Then abstract the argument to "what type of flaw is alleged" before comparing.

For example, add instructions like: "Before comparing, restate each item's criticism in one sentence of the form: 'The [aspect] of the paper is [flaw type] because [reason].' Compare these restated versions."

**1.2. Asymmetric formulation handling.** Add explicit guidance for AI-Human pairs: "Human reviewers may express the same criticism as a question, a suggestion, or a brief remark. Treat 'Does X affect Y?' as equivalent to 'X may affect Y' when both address the same concern."

**1.3. Scope-aware comparison.** Add instructions to handle scope mismatches: "If one item is a specific instance of the other item's general criticism, they share the same argument. A critique of 'the graphene supercell is too small' is an instance of 'the system sizes are too small.'"

**1.4. Recalibrate "different evidence" vs. "same evidence."** Provide clearer definitions: "Evidence is 'the same' when the items cite the same paper passages, data points, or external references as their primary support -- even if they interpret or frame them differently. Evidence is 'different' when the primary supporting data or citations do not overlap."

**1.5. Two-pass classification.** First classify binary (similar/not_similar), then refine to 4-way. This prevents 4-way errors from cascading into binary errors. Currently, models sometimes jump to "different subject" when a more conservative assessment would correctly land on "same subject, same argument, different evidence."

### 2. Training and Fine-Tuning Approaches

**2.1. Contrastive examples for boundary cases.** Fine-tune on triplets: (anchor pair, positive pair with same label, negative pair with adjacent label). Focus on the two critical boundaries:
- "convergent conclusion" vs. "topical neighbor" (same argument vs. different argument)
- "same evidence" vs. "different evidence"

**2.2. Data augmentation with reformulation.** Generate training pairs where the same criticism is expressed as: (a) a direct assertion, (b) a question, (c) a constructive suggestion, (d) a terse comment. Train the model to recognize equivalence across formulation styles.

**2.3. Domain-specific fine-tuning.** The current evaluation spans physics, chemistry, biology, computer science, and optics. Models may benefit from domain-specific calibration, as "same argument" boundaries differ by field (e.g., "inadequate statistical power" is more narrowly defined in clinical research than in physics).

### 3. Architecture Changes

**3.1. Multi-stage classification pipeline.**
- Stage 1: Subject classification (same/different) -- high recall for "same subject"
- Stage 2: Argument classification (same/different) conditioned on same subject
- Stage 3: Evidence classification (same/different) conditioned on same argument

This decomposition prevents the single-shot classification from short-circuiting at "different subject" when the subject match requires abstraction.

**3.2. Ensemble with diverse prompts.** Use multiple prompt formulations (literal matching, abstract matching, question-aware matching) and aggregate. The universal errors (eval_005, eval_059, eval_120, eval_134) suggest that all models share the same blind spots, so prompt diversity may help more than model diversity.

**3.3. Retrieval-augmented classification.** For the "same evidence" vs. "different evidence" distinction, extract cited paper passages from both items and compute overlap. If the quoted text overlaps substantially, classify as "same evidence" regardless of how the items frame the quotes.

### 4. Evaluation Methodology Improvements

**4.1. Adjudicate universal errors.** The 4 pairs that all 7 models get wrong at binary deserve re-examination:
- eval_005 and eval_134: Models may be defensibly correct that the subjects/arguments differ. Consider whether the expert annotation reflects a broader interpretation than the taxonomy intends.
- eval_059: The expert annotation relies on reading the human reviewer's suggestion as an implicit critique -- consider whether this inference is reasonable to expect from an automated system.
- eval_120: The distinction between "lack of replicates" and "missing ablation control" as the same vs. different argument is genuinely debatable.

**4.2. Inter-annotator agreement.** Measure expert agreement on the boundary cases. If human annotators disagree on these pairs, they should not count against model accuracy.

**4.3. Weighted scoring.** Not all errors are equal. Confusing "same evidence" with "different evidence" (both binary-similar) is far less consequential than confusing "convergent conclusion" with "different subject." Consider a weighted accuracy metric that penalizes binary-impacting errors more heavily.

**4.4. Separate AI-Human evaluation.** Given the systematic difficulty with AI-Human pairs (especially false negatives), report accuracy separately by pair type to diagnose formulation-mismatch issues independently from semantic understanding issues.

### 5. Taxonomy Refinements

**5.1. Consider a 3-way taxonomy.** The "same evidence" vs. "different evidence" distinction accounts for the largest share of 4-way errors but has no binary impact. If the practical goal is binary classification (similar/not_similar), a 3-way taxonomy (same argument, different argument, different subject) would reduce noise and focus model effort on the binary-critical boundary.

**5.2. Define "argument" more precisely.** The current taxonomy leaves ambiguous whether "the claim is unsupported" is one argument or many. Consider defining argument at the level of the specific flaw type:
- "unsupported due to lack of statistical evidence" is different from "unsupported due to missing controls"
- "method is biased" is different from "method underestimates"

**5.3. Add an "implicit match" category.** Many of the hardest cases involve one explicit and one implicit version of the same criticism. An additional label for "same argument, different formulation" could capture the AI-Human asymmetry pattern.

**5.4. Clarify scope handling.** The taxonomy should explicitly address whether a specific-instance critique (e.g., about graphene) matches a general critique (e.g., about all system sizes). The current annotation sometimes treats these as matching and sometimes does not, creating inconsistency.

---

## Summary of Key Findings

1. **Binary accuracy has hit a ceiling at ~92.7%** across 6 of 7 frontier models, with exactly the same 12 errors. The remaining errors require qualitatively different capabilities rather than incremental improvement.

2. **The hardest binary errors fall into two categories:** (a) false negatives where models fail to abstract past surface-level differences to find shared arguments, and (b) false positives where models over-abstract, collapsing genuinely different arguments into "the same critique."

3. **AI-Human pairs are systematically harder** than AI-AI pairs, due to asymmetric formulation (questions vs. assertions, terse vs. detailed).

4. **4-way accuracy is primarily limited by the evidence distinction** ("same" vs. "different" evidence), which is under-defined in the taxonomy and accounts for the majority of 4-way errors that do not affect binary classification.

5. **Universal errors suggest annotation boundary cases** rather than model deficiency. Four pairs fool all 7 models, and at least two of these represent genuinely arguable ground truth labels.

6. **The most impactful improvements** would come from prompt engineering (abstraction instructions, asymmetric formulation handling) and multi-stage classification, rather than from model scaling or training data augmentation.
