# Qualitative patterns in expert annotator comments

## 1. Scope

This report synthesizes qualitative patterns from two source files:

- `**expert_comments_item_level.json**` — 767 per-item free-form comments
attached to individual review items (correctness / significance / evidence).
Distribution: GPT 102, Claude 115, Gemini 104, Human_1 162, Human_2 173,
Human_3 111 (541 primary + 226 secondary); 66 of 85 papers have at least
one comment.
- `**expert_comments_paper_level.json**` — 250 paper-level free-form answers
to *"Did you notice anything in AI Reviewer i's review that other
reviewers did not catch?"* (179 primary + 71 secondary), covering 71 papers.

Patterns are grouped into three parts: **(I)** where AI reviewers clearly
**lack** compared to humans, **(II)** where AI reviewers clearly **exceed**
humans, and **(III)** residual categories that capture every remaining
comment. The scheme is designed so that every comment in both files maps
to at least one category. A single comment may belong to more than one
category — e.g., an AI critique can simultaneously be over-harsh (W3) and
lack field context (W1).

Each quoted example lists the HF `paper_id`, the reviewer the item belongs
to, the item number, and whether the annotator was a primary or secondary
annotator, so any quote can be traced back to the source JSON.

---

## Part I. Where AI reviewers lack compared to humans

### W1. Criticizes things that are implicitly acceptable in the field

AI flags practices, conventions, or scoping decisions that are standard
in the target community. An expert human reviewer knows the convention
and would not raise the point; the AI does not. This is the strongest
weakness signal in the data, appearing in every scientific domain the
dataset covers.

> "The point is correct but neglecting the fact that obtaining enough
> observation data is always very difficult in reality. The reviewer
> should show some possible alternative observational data to be analyzed
> if proposing this claim."
> — paper 18, Claude, item 2 (primary)

> "It is true that these things are not documented in the paper, however
> they are intenally at CERN. It is not a practice to publish these as
> part of Nature papers so this comment is irrelevant."
> — paper 9, GPT, item 5 (primary)

> "The pointing out on how to define classification during non-equilibrium
> process is appropriate but it neglects that the paper considers a
> response to linear perturbation and a near equilibrium initial profiles
> of observation and PIC simulations, leading to inadequately strong
> criticism to the authors' claim. The wording seems common on ML
> community unfriendly review comment but less common in physics
> community review."
> — paper 17, Gemini, item 1 (primary)

> "The AI reviewer seems to lack context that it is reasonable to assume
> that dust attenuation can not be that high so early in the Universe
> from physical reasons."
> — paper 20, Claude, item 4 (primary)

> "The main point of the critique is correct. But to the bacterial
> modeling community this paper does not read that it claims a universal
> law."
> — paper 28, GPT, item 1 (primary)

> "SPlot is a common technique in particle physics and widely used in
> many publications."
> — paper 8, GPT, item 1 (primary)

> "Faradaic efficiency is not highly weighed in electro-organic synthesis.
> AI reviewer is too harsh on this matter considering the accepted
> standard of the field."
> — paper 78, Claude, item 2 (primary)

> "It is well known that 1D corrections are not working well and that is
> the reason why researchers using better approximations like RPMD and
> Instanton theory."
> — paper 3, Claude, item 4 (primary)

> "Modified Strehl formulation is very common in metamaterial research."
> — paper 65, Claude, item 2 (primary)

> "This is the least interesting aspect of the paper. None of the 3 human
> reviewers even bothered with it, because the paper is using safety as
> an excuse to not talk about the MR engineering."
> — paper 6, Claude, item 1 (primary)

At the paper level, some annotators made this pattern fully explicit:

> "the AI reviewers in general do not account to a prior knowledge of the
> field so that they can make a fair point when criticizing the tool in
> terms of accuracy and application. Human reviewer, if chosen as a good
> expert in the field, would have applied this knowledge when making the
> comments. This applies to all AI."
> — paper 53, paper-level (primary, all 3 AI)

> "In general the AI lacks prior knowledge about what genomics community
> accepts as a golden standard of matching or gene alignment for some of
> the trained softwares. Human reviewer would know this and would not be
> critic about it."
> — paper 54, paper-level (primary, GPT)

### W2. Anachronistic critique — cites papers published AFTER the preprint

AI reviewers were given preprints, but cite literature from the future
relative to the preprint's date, making the critique invalid.

> "While claims are actually fine, I marked it as not correct as the paper
> they use to support these claims appeared two years after the presented
> one. The field was evolving very fast and in 2022 the presented results
> could be considered as good."
> — paper 1, Claude, item 1 (primary)

> "Funnily, this AI reviewer references in the last quote the same paper
> published later in 2022."
> — paper 5, Claude, item 1 (primary)

> "Again, while this point is valid and significant, the evidence is
> based on knowledge obtained in the future related to when this preprint
> was reviewed."
> — paper 20, Claude, item 2 (primary)

> "The comments cite a paper published 2 years after this preprint which
> is not valid."
> — paper 21, Claude, item 1 (primary)

> "Again chronology is not preserved and the cited evidence is after this
> preprint."
> — paper 20, Gemini, item 1 (primary)

> "This needs to be seen in historical context. PolyBERT was very early."
> — paper 46, Claude, item 3 (primary)

> "Yes, but the evidence was based on knowledge obtained several years
> after the publication of this preprint."
> — paper 20, paper-level (primary, Claude & Gemini)

### W3. Overly harsh or demands unrealistic scope expansion

AI is technically correct but asks for things that are infeasible, out
of scope, or well beyond what the paper claimed to do.

> "This is too harsh critique. The authors do not claim that their result
> is universal, although the AI would have had a good point if they did."
> — paper 28, GPT, item 2 (primary)

> "It's not possible to obtain intracranial data from healthy controls -
> it won't pass the ethical committee protocol."
> — paper 15, Claude, item 2 (primary)

> "It seems in the end criticizing a typo in the caption of Figure 3,
> 1000 \omega^{-1}*{ci} --> 100 \omega^{-1}*{ci} but the comment overly
> criticizing. Just pointing out a typo is enough."
> — paper 17, GPT, item 3 (primary)

> "The statements are correct, but too harsh and are beyond the scope of
> the paper."
> — paper 28, Claude, item 4 (primary)

> "Applies an unrealistically stringent reproducibility standard. This
> standard of reproducibility, while ideal, is rarely met in chemical
> transport modeling studies published in top journals."
> — paper 59, GPT, item 4 (secondary)

> "This paper novelty lies in engineering/system, not a fully-deployed
> medical study."
> — paper 52, Claude, item 2 (primary)

> "Correct, but out of the scope of the paper."
> — paper 29, Claude, item 4 (primary)

> "This critique, technically correct, is confusing the significance of
> some of the results."
> — paper 29, Claude, item 2 (primary)

> "Too harsh critique."
> — paper 31, Claude, item 4 (primary)

> "AI reviewer 1 focused too much on the safety aspects of the paper.
> 70% of the review was about safety, whereas the paper is primarily
> about MRI methodology. I am sure that some of the statistics
> observations were original, but they were irrelevant in my opinion."
> — paper 6, paper-level (primary, Claude)

### W4. Misreads or hallucinates paper content

AI claims the paper omits something that is actually in the paper, or
misquotes, confuses sections, or over-interprets a typo.

> "The authors write explicitly that they employ finite-size correction."
> — paper 1, Gemini, item 1 (primary)

> "The authors state explicitly that it is a proof-of-concept study."
> — paper 3, Claude, item 2 (primary)

> "this is factually incorrect, the paper explicitly addresses this issue.
> Lines 489–496 describe a calibration procedure (Equation 5) that
> adjusts modeled PM2.5 concentrations against observed baseline period
> data (2012–2017)."
> — paper 61, Claude, item 5 (secondary)

> "The paper actually uses the GBD2019 MR-BRT model (line 470), not the
> older IER function, and explicitly discusses the GEMM comparison (lines
> 547–550), acknowledging that GEMM yields higher estimates."
> — paper 61, Gemini, item 5 (secondary)

> "The AI made factual errors. The sequence parameters were reported, and
> the methods section starts with this. It is true that the MRI
> methodology does not have a subheading, which is why the AI got
> confused."
> — paper 6, Gemini, item 1 (primary)

> "The authors provide the code and it does not matter whether it is on
> GitHub or GitLab."
> — paper 4, GPT, item 3 (primary)

> "The code is available."
> — paper 4, Gemini, item 5 (primary)

> "AI got this wrong. Comparing simulation and experimental results is
> not fair."
> — paper 65, Claude, item 1 (primary)

> "Instead of quoting the author's exact words, this AI reviewer
> incorrectly quoted the author and then claimed that the author's
> claims were incorrect. It's bit surprising that AI reviewers can make
> such mistakes, and I think we need to be aware that they can make such
> simple mistakes."
> — paper 49, Gemini, item 1 (secondary)

> "AI confused by a confusing authors' explanation, mixing 1D analyses
> with 2D reconnection case without explicitly clarifying."
> — paper 18, GPT, item 2 (primary)

> "Again, AI made a mistake by comparing the sentences taken from
> different parts of the manuscript talking about different parts of the
> process."
> — paper 65, Gemini, item 4 (primary)

> "There is actually 'methods' text in the manuscript. It probably wasn't
> properly inputed into AI."
> — paper 50, Gemini, item 3 (primary)

> "The reviewer's assessment seems to be based on an over-interpretation
> of Table 1, where a typographical error in the values may have led to
> an incorrect conclusion."
> — paper 32, Claude, item 2 (primary)

> "review misidentifies the optimal threshold as t=0.4 when Figure 5c
> shows the peak at t=0.3."
> — paper 35, Claude, item 3 (primary)

> "error bars are present in figure2c (small gray lines on each bar)"
> — paper 35, Claude, item 4 (primary)

### W5. Vague, verbose, convoluted, or non-actionable

The critique is long or generic, without specific recommendations, or
its core argument is hard to follow.

> "As with other reviews, it is unclear what the recommendation for
> addressing this criticism is."
> — paper 41, Claude, item 1 (primary)

> "The review does not provide any suggestions for how to address this
> criticism."
> — paper 41, Gemini, item 1 (primary)

> "While this AI reviewer (and others) is excellent at pointing out flaws,
> it fails to provide constructive alternatives or actionable suggestions
> on how the authors should address these criticisms to improve the
> paper."
> — paper 49, GPT, item 5 (secondary)

> "Too verbose. I am not a statistics expert, but this is a paper about
> MR methodology. The first comment should not be six paragraphs about
> biological effects."
> — paper 6, GPT, item 1 (primary)

> "Too verbose. The previous two AI reviewers did a much better job
> pinpointing the main issues."
> — paper 6, GPT, item 2 (primary)

> "I found the items themselves to be reasonable, but the underlying
> argumentation was diffuse and overcrowded with statements that were
> largely irrelevant."
> — paper 5, paper-level (primary, GPT)

> "This reviewer halucinated the most and it spit out word salads. I am
> sure there were some original insights in there (the review was 5
> times longer than the others), but they were clouded by verbosity"
> — paper 6, paper-level (primary, GPT)

> "There are some correct elements here, but it is stated in a super
> confusing way."
> — paper 29, GPT, item 2 (primary)

> "This might be correct but it is stated in a convoluted way."
> — paper 29, Gemini, item 3 (primary)

> "The evidence consists only of paper quotes restated with the
> reviewer's interpretation, without external citations or concrete
> reasoning to support the reverse causality claim"
> — paper 36, Gemini, item 3 (primary)

> "This feels very much AI generated critique that identifies a real
> asymmetry in methodology but fails to evalute whetjer it matters
> qunatitatively."
> — paper 59, Claude, item 3 (secondary)

### W6. Tone mismatch — aggressive/ML-style critique in non-CS fields

AI adopts an ML/CS reviewer tone that other fields regard as impolite or
inappropriately combative.

> "Mainly the implicit assumptions commonly shared by experts, some
> logical jumps, and typos. However, the wording is very 'ML
> community-like', overly criticizing and impolite, which would not be
> accepted by other communities."
> — paper 17, paper-level (primary, GPT)

> "The wording seems common on ML community unfriendly review comment
> but less common in physics community review."
> — paper 17, Gemini, item 1 (primary)

> "This is such a harsh comment for an experimental study. The authors
> provided all the datasets. The measurements were taken several times
> and averages were reported. The real world measurements are not perfect
> or noise free. What else could authors do?"
> — paper 66, GPT, item 1 (primary)

> "I again do not agree with the AI reviewer. The wavelength is close to
> 800 nm and the tip is 20-25 nm wide, which is 1/30th - 1/40th of the
> wavelength. We can treat it as a point dipole. The AI reviewer's
> comment is too harsh!"
> — paper 66, GPT, item 2 (primary)

### W7. Doesn't consult the cited prior work

When an AI critique hinges on what is in the authors' own cited prior
papers, the AI sometimes skips those and treats the claim as unsupported.

> "The authors make this claim based on their previous work. Similar to
> one of the human reviewers, the AI reviewer did not read those cited
> papers."
> — paper 28, Gemini, item 2 (primary)

> "The authors make the claim about correlation of nutrient concentration
> and interaction strength based on their previous work. This critique
> is similar to the one of the Human Reviewers."
> — paper 28, Claude, item 2 (primary)

### W8. Over-inflates small code/text inconsistencies

AI finds a trivial code–text mismatch and treats it as a show-stopper,
when the underlying science is fine.

> "I think the AI is overstating the implications of small deviations
> between code and text that occur over the lifetime of a scientific
> paper."
> — paper 42, GPT, item 2 (primary)

> "This one could be an unit error in the code, over-interpreted by
> AI-reviewer."
> — paper 34, Gemini, item 2 (primary)

> "Quote: '… IncN plasmids, which showed 99.7% identity' — overstatement
> by AI and overanalysis"
> — paper 54, Claude, item 5 (primary)

### W9. Outright technical factual error (math, physics, biology wrong)

Distinct from W4 — here the AI states something that is wrong in its
own terms, not a misreading of the paper.

> "There is a misunderstanding of what the term 'universal' means here.
> 'Universality' is mere a physics term that means the data collaps to
> a single curve when scaled. It does not mean it is universally valid."
> — paper 28, Gemini, item 4 (primary)

> "INR might take a few seconds for training, but once the training is
> complete, then it is extremely fast during inference."
> — paper 65, Gemini, item 2 (primary)

> "SCA acts like a monochromatic filter, MCA is more like an array
> filter."
> — paper 67, Gemini, item 1 (primary)

> "Here the resolution depends on the width of the tip, not the
> excitation wavelength."
> — paper 66, Gemini, item 3 (primary)

### W10. Cannot analyze figures/images — only text

AI reviewers operate on text only and miss things that require looking
at plots, images, or figures.

> "This could be argued better, but to do this the AI would need to be
> able to analyze the images. It went by text only, did not download the
> figures from Figshare, so it gave boilerplate feedback."
> — paper 6, Gemini, item 5 (primary)

> "I can't believe the reviewer came up with this comment. Their s-SNOM
> tip has a width of ~25 nm. Their samples are a couple of micron wide!
> By moving their tip along the samples, they were able to generate
> those promising results!!"
> — paper 66, Human_1, item 5 (primary) — human pointing out AI failure on spatial geometry

### W11. Redundancy across the 3 AI reviewers reduces combined coverage

The 3 AI reviewers tend to raise overlapping points, so replacing a
human slot with an AI does not add much marginal coverage beyond the
other two AI slots.

> "To a large extent, all three AI reviewers raise very similar points.
> Thus, the overall review would not cover many substantial points human
> reviewer raised."
> — paper 5, Claude, item 2 (primary)

> "Overall, as noted in the checklist, all reviewers raised very similar
> points. However, the first reviewer provided the strongest arguments."
> — paper 5, paper-level (primary, GPT)

> "Again the same comment about NMF rank that is common in all AI
> reviews."
> — paper 62, GPT, item 3 (primary)

> "To certain extent all AI reviewers share the issues they raised."
> — paper 1, paper-level (primary, Gemini)

### W12. Trivial / nitpicking focus on typos and minor issues

AI over-focuses on typos, grammar, or small presentation issues while
humans concentrate on substance.

> "This is a very minor issue. In fact, sometimes I personally forget to
> include the statistical analysis of ML outcomes."
> — paper 67, Claude, item 3 (primary)

> "An unused argument in the code, which is hepful but about as minor as
> a criticism can be."
> — paper 72, paper-level (primary, Gemini)

> "Not Significant" — an item-level tag many annotators used when an AI
> correctly spotted a typo or formatting issue that does not affect the
> paper's conclusions (dozens of instances; e.g. paper 17 Gemini items
> 3, 4, 5; paper 23 Human_3 item 9; paper 82 Human_1 items 1 & 2).

### W13. Weak or generic critique — dismissed wholesale

The annotator dismisses the AI reviewer's output as a whole without
pointing to a specific W1–W12 failure mode. Typically a one-line
"rather bad", "weakest assessment", or "complete fail" judgment.

> "I found this reviewer rather bad."
> — paper 3, paper-level (primary, Claude & Gemini slots)

> "AI Reviewer 1 offered the weakest assessment among all reviewers."
> — paper 77, paper-level (primary, Gemini)

> "Complete fail here."
> — paper 29, Gemini, item 1 (primary)

> "For the last question here I added AI reviewer 3 however it's a bit
> borderline, I think the 3 AI reviews were not very good."
> — paper 9, paper-level (primary, GPT)

---

## Part II. Where AI reviewers exceed humans

### S1. Deep code reading and finding code ↔ paper discrepancies

By far the strongest and most frequently-cited AI advantage. Multiple
expert annotators across every discipline in the dataset explicitly said
they would not have read the code themselves.

> "I find it quite impressive that this reviewer actually looks at the
> code to understand whether some parts of the paper are not properly
> discussed. I am personally not doing it on a regular basis as it would
> consume a substantial amount of time."
> — paper 1, GPT, item 4 (primary)

> "this AI reviewer discusses issues with the code which I find very
> good, as a human reviewer I check the code only if I am very suspicious
> about the results and could have overlooked it."
> — paper 2, paper-level (primary, GPT)

> "Unlike human reviewers, this reviewer actually visited the GitHub open
> source code repository the authors listed in their papers, dissected
> and analyzed the Python scripts to identify problems in the analysis."
> — paper 49, GPT, item 2 (secondary)

> "AI Reviewer 2 went beyond merely reading the text of the manuscript
> and providing logical critiques. It delved directly into the actual
> Python code scripts, data arrays, and references of the original
> datasets provided by the authors, accurately pinpointing critical
> discrepancies between what was claimed in the paper (methods) and what
> was actually executed (code or data)."
> — paper 49, paper-level (secondary, GPT)

> "human reviewers, especially 1, were very good. AI only exceeded them
> in checking the code."
> — paper 27, paper-level (primary, all 3 AI)

> "In general, the AI reviews were better at pointing out code-related
> issues than the human reviewers."
> — paper 51, paper-level (secondary)

> "AI is good at reviewing the codes and finding bugs."
> — paper 67, Claude, item 1 (primary)

> "I find the review of AI reviewer 1 to be the best among all the human
> and AI reviews. One thing I would like to mention when comparing to
> human reviews is that this AI looked thoroughly into the code to find
> inconsistencies."
> — paper 83, paper-level (secondary, Gemini)

> "Good job in checking the code and finding the discrepancies."
> — paper 23, paper-level (primary, GPT & Claude)

> "nice checking the code and catching the normative atlas issue missed
> by the humans."
> — paper 24, paper-level (primary, Gemini)

### S2. Internal consistency checking across sections

AI catches contradictions between abstract, main text, methods, figures,
tables, and supplementary — even a careful-reading human often misses
these cross-section inconsistencies.

> "AI Reviewer 3 found about mistakes in writing where authors reported
> numbers in wrong order and thus physiologically impossible number was
> reported."
> — paper 58, paper-level (primary, Claude)

> "AI Reviewer 3 correctly pointed out a discrepancy in the reported
> heart rate value - which was physiologically impossible for a healthy
> adult."
> — paper 58, paper-level (secondary, Claude)

> "Uniquely caught that the abstract claims 'no passivation' while
> Figure 2f's caption explicitly states the ultraflexible device used
> 'parylene encapsulation'"
> — paper 103, paper-level (primary, GPT)

> "This is one of the most damaging critiques across all reviews because
> it challenges the paper's internal consistency rather than requiring
> external knowledge or judgment. A reader can verify every claim in
> this item solely from the manuscript."
> — paper 61, GPT, item 5 (secondary)

> "The wrong confidence interval '658(608–656)' and scenario-label
> mismatches between Results and Conclusions (Item 5) were concrete
> textual errors no one else caught."
> — paper 74, paper-level (secondary, GPT)

> "item 2 is unique across all 6 reviewers. It identifies that the
> paper's predicted probabilities (e.g., '64% risk') are not interpretable
> as absolute population risks because the model was trained on a 1:1
> case-control sample where class balance is set by design, not by
> population incidence."
> — paper 36, paper-level (primary, GPT)

> "The paper's reported design/optimization specifications are internally
> inconsistent, and the released code's metric implementation does not
> match the paper's stated definition"
> — paper 81, paper-level (primary, GPT)

### S3. Reproducibility / dependency / data-access failures

AI catches missing dependencies, broken pipelines, version mismatches,
missing credentials, and incomplete data releases.

> "AI #2 identified the reproducibility failures in the experimental
> pipeline such as missing parameter specifications and modified external
> tools."
> — paper 53, paper-level (primary, GPT)

> "AI Reviewer 3 uniquely addressed reproducibility concerns (Item 3):
> the version mismatch between the manuscript ('GECKO 2.0') and the code
> artifact ('GECKO 3.2.4'), the BRENDA credential requirement, and
> missing pinned dependency versions. No other reviewer raised this."
> — paper 31, paper-level (secondary, GPT)

> "Impressive that the AI looked at the code and realized there is
> nothing about the MRI reconstruction."
> — paper 6, Gemini, item 4 (primary)

> "the model architecture for finetuning is provided
> (ICDBartForSequenceBinaryClassification in icdmodelbart.py) but the
> training script is not."
> — paper 37, GPT, item 4 (primary)

### S4. Statistical / ML methodology rigor

AI catches data leakage, broken validation splits, training on the test
set, wrong aggregation formulas, and similar methodology failures.

> "AI Reviewer 1 was the only reviewer to note the potential leakage
> between training and test data (item 2)."
> — paper 92, paper-level (primary, Claude)

> "The PPI trainer has no validation split (only train/test), uses test
> metrics for model selection (misleadingly named best_valid_f1)"
> — paper 35, GPT, item 1 (primary)

> "Code verification fully confirms the data leakage claim. The DPI link
> trainer (trainer_dpi.py) also has a bug using test metrics for model
> selection despite having a validation set."
> — paper 35, Gemini, item 1 (primary)

> "AI #1 was the only reviewer who raised a critical question regarding
> the intra-hamming distance, and argued that its magnitude is too large
> to support stable cryptographic key generation."
> — paper 53, paper-level (primary, Gemini)

> "AI #3 uniquely pointed out that the entropy equation is mathematically
> incorrect."
> — paper 53, paper-level (primary, Claude)

> "AI reveiwer 1 found out about SNR which all other reviewers failed to
> catch."
> — paper 58, paper-level (primary, GPT)

> "The TU aggregation mathematical inconsistency (Item 1) was the most
> analytically rigorous and original contribution across all AI reviews.
> No human reviewer caught that 1/Σ(WF×EC) ≠ Σ(WF×1/EC)."
> — paper 59, paper-level (secondary, GPT)

> "AI Reviewer 1 pointed out a statistical flaw in the distance-dependent
> cross-validation method. Specifically, test regions adjacent to the
> boundary of the 75% training set and the 25% test set still share
> spatial autocorrelation, which can potentially inflate the model's
> performance estimates."
> — paper 49, paper-level (secondary, Claude)

### S5. Specialized niche catches that no other reviewer raised

Insights that require both domain knowledge and careful reading, and
that no AI-vs-human-is-better argument normally covers.

> "AI Reviewer 2 was the only one to mention the potential confounding
> effect of population structure on the GWAS results."
> — paper 73, paper-level (primary, Claude)

> "AI reviewer 3 nicely brought up the issue of diastereoselectivity in
> the product formation which was not mentioned by the human reviewers."
> — paper 97, paper-level (primary, GPT)

> "Only reviewer to question the use of the term 'LED,' given that the
> device operates via gate oxide breakdown rather than conventional
> carrier injection."
> — paper 101, paper-level (primary, Gemini)

> "AI Reviewer 2 uniquely raised the KS test validation critique (Item
> 2): distributional similarity does not validate per-reaction kcat
> accuracy, since any random sample from the BRENDA distribution would
> pass the same test. No other reviewer made this argument."
> — paper 31, paper-level (secondary, Claude)

> "Only review to catch EM simulations used air/copper properties instead
> of tissue-equivalent properties."
> — paper 104, paper-level (primary, GPT)

> "This is a good point that was missed by the human reviewers."
> — paper 77, Claude, item 5 (primary)

### S6. Big-picture framing / high-level synthesis

Sometimes an AI reviewer produces a clearer, more structured big-picture
assessment than any of the three humans did.

> "No, but this is primarily because it did not get bogged down in
> details. It was very good at seeing the big picture and pinpointing
> the biggest problems."
> — paper 6, paper-level (primary, Gemini)

> "this provided the most nuance in the comments, though all 3 AI did
> good job on noticing and calling up the circularity in the work, which
> was missed even by the first reviewer which was quite good."
> — paper 26, paper-level (primary, Claude)

> "This is an outstanding technical critique. It's worth noting that the
> attributable AF itself is independent of the baseline, so the relative
> risk estimates are unaffected but the absolute death counts, which are
> the paper's primary output, scale directly with deathexp."
> — paper 60, Gemini, item 1 (secondary)

> "This is one of the most insightful critiques across all reviews
> because it runs counter to the paper's narrative rather than supporting
> the claim that compound impacts are underestimated, it suggests
> temperature-related mortality during the pandemic may actually be
> overestimated due to a less vulnerable surviving population."
> — paper 60, Gemini, item 4 (secondary)

### S7. General positive endorsement of an AI critique

The annotator confirms the AI caught a valid point but does not call
out a specific strength type (code reading, consistency, etc.). These
are short "good point", "agree", "reasonable", "well-argued" endorsements
that nonetheless record an AI win at item level.

> "This is well-argued and correct!"
> — paper 6, Claude, item 3 (primary)

> "Good point and well-argued!"
> — paper 6, Gemini, item 2 (primary)

> "Great point."
> — paper 31, Gemini, items 2 & 4 (primary)

> "very well supported observation"
> — paper 53, GPT, item 1 (primary)

> "I agree that data leakage is possible if the same regions are included
> in both training and test sets. A safer approach would be to split the
> genome up into segments that are partitioned into the training and
> test sets."
> — paper 74, Claude, item 2 (primary)

> "The power criticism is very correct and very significant... the enegy
> part is the fundamental citicism that should be enough to reject the
> paper."
> — paper 63, Gemini, item 4 (primary)

> "This is a good point."
> — paper 29, Claude, item 3 (primary)

> "significant point well supported that applies for reproducibility"
> — paper 54, GPT, item 4 (primary)

---

## Part III. Residual categories

These categories absorb every remaining comment that is not a direct
AI-vs-human strength or weakness judgment. They exist so that the scheme
is exhaustive over the full 767 + 250 comment corpus.

### R1. Neutral parity — "AI and humans caught similar things"

The annotator is saying neither side has an advantage on that paper or
item; AI and human reviewers identify the same underlying issues.

> "In general, they pick up on the same major points"
> — paper 22, paper-level (primary, all 3 AI)

> "All the items are mentioned by at least one other AI or human
> reviewer."
> — paper 79, paper-level (primary, Gemini & Claude)

> "There were no major points that significantly distinguished AI
> reviewer 1's review from the other reviews."
> — paper 51, paper-level (secondary, Gemini)

> "All items captured by other AI reviewers."
> — paper 80, paper-level (primary, Claude & Gemini)

> "Other points are rather similar to other AI reviewers."
> — paper 4, paper-level (primary, Claude)

> "Same as the AI reviewer-2, this reviewer also caught the same
> resolution issue."
> — paper 67, Claude, item 1 (primary)

> "Aligned with Item 7 of Human Reviewer 3, Item 3 of AI Reviewer 1, and
> Item 1 of AI Reviewer 2."
> — paper 62, GPT, item 1 (secondary)

### R2. Annotator defers or hedges ("I'm not an expert" / "didn't verify")

The annotator explicitly declines to judge the critique because they
lack domain expertise or did not verify by running the code.

> "I am not an expert on the analysis of these data, but this critique
> seems valid to me."
> — paper 85, GPT, item 4 (primary)

> "I didn't manually go through the code to verify the claim. Instead, I
> asked Claude Opus 4.6 to verify the claim and it said the claim was
> correct."
> — paper 37, Gemini, item 2 (secondary)

> "I did not personally check the repository and the code, but the
> missing data could be retrieved from the authors by request."
> — paper 62, GPT, item 4 (primary)

> "I checked the validity of this item using Claude Code."
> — paper 37, Claude, item 3 (secondary)

> "I did not verify this in detail by running the Python code, and I
> assume the authors did not make this mistake."
> — paper 34, Gemini, item 1 (secondary)

> "I am not an expert in this analysis procedure, but this critique
> seems reasonable."
> — paper 85, Gemini, items 1–3 (primary)

### R3. Comments about human reviewer quality (not about AI)

Many item-level comments are on Human_1 / Human_2 / Human_3 items and
assess the human reviewer directly without a contrast to AI. These are
relevant context for the bench but do not themselves map to Parts I or
II.

> "The reviewer is exclusively considering absolute similarity as opposed
> to relative similarity (as being argued in the main text). Therefore,
> the analysis in question is irrelevant for the paper."
> — paper 68, Human_2, item 1 (primary)

> "Overall the criticism starts on the right track but then loses focus
> and makes vague claims at the end."
> — paper 64, Human_1, item 1 (primary)

> "The human reviewer 1 misunderstood the paragraph. The curves of
> DrugVQA are in the panels of Protein-Protein Interaction in Figure 2a
> and 2b, respectively, which is correct according to the explanation
> paragraph."
> — paper 35, Human_1, item 1 (secondary)

> "Not sure to what the reviewer is referring to here."
> — paper 75, Human_3, items 5, 6, 9, 10 (primary)

> "It looks like the reviewer misunderstood this approach as a
> data-driven learning approach."
> — paper 62, Human_1, item 7 (secondary)

> "This is a bit of nitpicking."
> — paper 28, Human_2, item 6 (primary)

> "I agree with the reviewer."
> — paper 28, Human_2, item 2 (primary)

> "Great catch by the reviewer."
> — paper 28, Human_1, item 8 (primary)

### R4. Meta / process / survey artifacts / versioning notes

Comments about the annotation workflow, the version of the preprint
being reviewed, or the layout of the survey — not about substantive
content.

> "Human review 2 is not available in the checklist."
> — paper 8, Claude, item 1 (primary)

> "Again, it is highly suspected that the preprint I am reviewing is the
> revised version after human reviews."
> — paper 35, Human_3, items 1–3 (secondary)

> "In the codebase I have says python3 is required. Maybe I am looking
> at a newer version?"
> — paper 72, Human_1, item 1 (primary)

> "This seems to target a different formulation or draft."
> — paper 12, Human_1, items 4–6 (secondary; 11 Not-Correct items on
> paper 12 referencing a version mismatch)

> "The attached paper already states the ISM density as a lower limit
> n_e >= 26.6 cm^-3, so this criticism seems to quote a different
> version of the draft."
> — paper 12, Human_2, item 7 (secondary)

> "This was my first study on Human vs AI reviewers. It is interesting
> to see that one of three AI platforms can come up with a totally
> different review than the other two."
> — paper 65, Gemini, item 5 (primary) — an annotator-level reflection

### R5. Short, trivial, or non-informative responses

A substantial fraction of the paper-level file is a few words or less:
item-number pointers, yes/no answers, dashes, dots. They still belong
to the corpus and are captured here.

Item-number-only answers:

> "Item 2"  /  "Items 1,2,3"  /  "item 3, 4"  /  "Item 3 and Item 4"
> — papers 32, 33, 34, 72, 74 paper-level; many per paper

Binary answers:

> "Yes." / "No." / "Yes" / "No" / "no" / "Not really"
> — widespread on the paper-level file, e.g. papers 7, 10, 29, 36, 51,
> 59, 78, 98, 100

One-character placeholders:

> "-" / "." / "None" / "단순" (Korean "simple")
> — paper 6 secondary, paper 15 Human_2 item 6 secondary, paper 101
> secondary items 2 & 3, paper 102 secondary all items, paper 103
> secondary items 2 & 3

Short verbal acknowledgements at item level:

> "typo" / "Agree." / "Correct." / "Good point." / "On point." / "Great
> point." / "Good catch." / "Very good point" / "only one hit of the
> term is present"
> — dozens of instances across paper 6, 28, 41, 46, 62, 73, 77, 85 items.

---

## 4. Coverage check — per-comment category enumeration

Every one of the 767 item-level + 250 paper-level = **1017 comments**
is tagged below with at least one category from the W1–W13, S1–S7, or
R1–R5 schemes. Categories are non-exclusive: comments straddling
multiple themes carry multiple codes.

Notation:
- **`IP{paper}·{Reviewer} i{N}{P|S}`** = item-level comment on the
  reviewer's item {N}; `P` primary, `S` secondary annotator.
- **`PP{paper}·{Reviewer} slot{k}{P|S}`** = paper-level "unique catches"
  comment for the AI reviewer in slot k (1/2/3).
- All `Human_{1,2,3}` item-level comments are tagged **R3** by default
  (they are annotator assessments of a human reviewer, not of AI) and
  get a second code (usually **R5** for minimal "Agree"/"Correct"/"No.";
  **R4** for draft/version notes) only if applicable.

### 4.1 Item-level (767 comments across 66 papers)

**Paper 1** — Claude i1(P)=W2, i2(P)=W5, i3(P)=S7; GPT i4(P)=S1; Gemini i1(P)=W4, i2(P)=S7, i3(P)=S7; Human_1 i1(P)=R3; Human_2 i8(P)=R3.

**Paper 2** — Claude i2(P)=W3, i4(P)=W5; GPT i2(P)=W3.

**Paper 3** — Claude i1(P)=S7, i2(P)=W4, i3(P)=W4, i4(P)=W1; GPT i3(P)=S7; Gemini i1(P)=W4; Human_1 i1(P)=R3, i2(P)=R3; Human_2 i1(P)=R3, i2(P)=R3, i4(P)=R3, i5(P)=R3,R5, i8(P)=R3,R2.

**Paper 4** — GPT i3(P)=W4; Gemini i1(P)=W3, i5(P)=W4.

**Paper 5** — Claude i1(P)=W2, i2(P)=W11, i4(P)=W3; GPT i1(P)=W5, i4(P)=W5; Human_2 i2(P)=R3, i3(P)=R3, i5(P)=R3, i6(P)=R3; Human_3 i1(P)=R3, i2(P)=R3.

**Paper 6** — Claude i1(P)=W12, i2(P)=W12, i3(P)=S7, i4(P)=W3; GPT i1(P)=W5,W3, i1(S)=W3,R5, i2(P)=W5; Gemini i1(P)=W4, i1(S)=W4, i2(P)=S7, i4(P)=S1,S3, i5(P)=W10; Human_1 i1(P)=R3, i2(P)=R3; Human_2 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3,R5, i5(S)=R3,R5; Human_3 i1(S)=R3,R5.

**Paper 7** — Gemini i2(P)=S7, i5(P)=S7; Human_1 i1(P)=R3.

**Paper 8** — Claude i1(P)=S7, i2(P)=W3; GPT i1(P)=W1, i4(P)=W4; Human_2 i6(P)=R3, i7(P)=R3, i8(P)=R3, i14(P)=R3; Human_3 i3(P)=R3.

**Paper 9** — GPT i2(P)=W3, i3(P)=W3, i4(P)=W5, i5(P)=W1; Gemini i2(P)=W3.

**Paper 10** — Human_2 i2(P)=R3, i3(P)=R3.

**Paper 12** — Claude i2(S)=W4; Gemini i3(S)=W4; Human_1 i1(S)=R3,R4, i2(S)=R3, i4(S)=R3,R4, i5(S)=R3,R4, i6(S)=R3,R4, i8(S)=R3,R4, i9(S)=R3; Human_2 i1(S)=R3,R4, i2(S)=R3,R4, i4(S)=R3,R4, i7(S)=R3,R4, i8(S)=R3, i9(S)=R3,R4, i11(S)=R3,R4; Human_3 i1(S)=R3,R4, i2(S)=R3,R4, i3(S)=R3, i4(S)=R3,R4, i5(S)=R3,R4, i6(S)=R3,R4.

**Paper 13** — Gemini i4(P)=W1.

**Paper 15** — Claude i2(P)=W3, i3(P)=W3; Human_2 i6(S)=R3,R5.

**Paper 16** — GPT i5(P)=W3.

**Paper 17** — GPT i2(P)=W1, i3(P)=W12; Gemini i1(P)=W1,W6, i2(P)=W3; Human_1 i1(P)=R3; Human_2 i1(P)=R3; Human_3 i1–i6(P)=R3 each (6 items).

**Paper 18** — Claude i2(P)=W1; GPT i2(P)=W4; Gemini i3(P)=W1, i4(P)=W1; Human_1 i4(P)=R3; Human_2 i1(P)=R3.

**Paper 19** — GPT i4(P)=W1; Gemini i3(P)=W4.

**Paper 20** — Claude i1(P)=W2, i2(P)=W2, i4(P)=W1, i5(P)=W1; Gemini i1(P)=W2; Human_1 i1(P)=R3.

**Paper 21** — Claude i1(P)=W2; GPT i1(P)=S7.

**Paper 22** — GPT i1(P)=W5, i2(P)=W1,W3, i3(P)=W5,R5; Human_2 i1(P)=R3, i3(P)=R3.

**Paper 23** — Human_3 i9(P)=R3,R5.

**Paper 24** — GPT i1(P)=S7; Human_2 i6(P)=R3.

**Paper 25** — Claude i1(P)=W12; Gemini i1(P)=W4, i2(P)=S7, i3(P)=S7; Human_1 i2(P)=R3, i10(P)=R3,R5; Human_2 i13(P)=R3, i15(P)=R3,R5.

**Paper 26** — GPT i4(P)=W12; Human_2 i5(P)=R3.

**Paper 28** — Claude i1(P)=W1,W4, i2(P)=W1,W7, i3(P)=W3, i4(P)=W3, i5(P)=W3; GPT i1(P)=S7, i2(P)=W3, i3(P)=W3; Gemini i1(P)=W4, i2(P)=W1,W7, i3(P)=W3, i4(P)=W9, i5(P)=W4,R5; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3, i5(P)=R3, i6(P)=R3, i7(P)=R3, i8(P)=R3, i9(P)=R3, i10(P)=R3,R5, i11(P)=R3,R5; Human_2 i1(P)=R3, i2(P)=R3,R5, i3(P)=R3, i4(P)=R3, i5(P)=R3,R5, i6(P)=R3, i7(P)=R3, i8(P)=R3, i9(P)=R3, i10(P)=R3,R5, i11(P)=R3, i12(P)=R3,R5, i13(P)=R3, i14(P)=R3, i15(P)=R3, i16(P)=R3, i17(P)=R3, i18(P)=R3, i19(P)=R3, i20(P)=R3, i21(P)=R3, i23(P)=R3, i26(P)=R3; Human_3 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3, i5(P)=R3, i7(P)=R3, i8(P)=R3,R5, i9(P)=R3, i10(P)=R3, i11(P)=R3.

**Paper 29** — Claude i1(P)=W3, i2(P)=W4, i3(P)=S7, i4(P)=W3, i5(P)=S7,R5; GPT i1(P)=W4, i2(P)=W5, i3(P)=W5; Gemini i1(P)=W13, i3(P)=W5, i4(P)=S7; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3, i5(P)=R3, i7(P)=R3; Human_2 i6(P)=R3, i9(P)=R3; Human_3 i5(P)=R3, i7(P)=R3, i8(P)=R3, i9(P)=R3, i10(P)=R3, i11(P)=R3,R5, i13(P)=R3,R5.

**Paper 30** — Claude i1(P)=W4, i2(P)=W3,R5; GPT i1(P)=S7, i4(P)=R2; Gemini i5(P)=W3,R5; Human_1 i2(P)=R3, i4(P)=R3, i5(P)=R3, i6(P)=R3, i11(P)=R3; Human_2 i1(P)=R3, i3(P)=R3.

**Paper 31** — Claude i4(P)=W3,R5; Gemini i1(P)=S7, i2(P)=S7,R5, i4(P)=S7,R5.

**Paper 32** — Claude i2(P)=W4.

**Paper 33** — Human_1 i1(S)=R3; Human_2 i6(S)=R3, i7(S)=R3,R4.

**Paper 34** — Gemini i1(P)=W4,W8, i1(S)=R2, i2(P)=W8, i4(P)=W8; Human_3 i1(S)=R3.

**Paper 35** — Claude i3(P)=W4, i4(P)=W4; GPT i1(P)=S4,S1, i2(P)=S7; Gemini i1(P)=S4,S1; Human_1 i1(S)=R3; Human_2 i3(S)=R3, i4(S)=R3,R4; Human_3 i1(S)=R3,R4, i2(S)=R3,R4, i3(S)=R3,R4, i4(S)=R3,R4.

**Paper 36** — Gemini i3(P)=W5; Human_1 i1(P)=R3; Human_2 i4(P)=R3; Human_3 i2(P)=R3.

**Paper 37** — Claude i1(S)=S7, i2(S)=S7, i3(S)=R2,R5, i4(S)=S7; GPT i1(S)=S7,W4, i2(S)=R3, i3(S)=R2,R5, i4(P)=S3, i4(S)=R2,R5; Gemini i2(S)=R2, i3(S)=R2; Human_1 i1(S)=R3, i2(S)=R3; Human_2 i1(S)=R3, i2(S)=R3, i3(S)=R3, i7(S)=R3, i9(S)=R3.

**Paper 41** — Claude i1(P)=W5, i3(P)=W5, i4(P)=W1; GPT i1(P)=W5; Gemini i1(P)=W5, i2(P)=W5; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3, i5(P)=R3, i6(P)=R3, i9(P)=R3, i11(P)=R3, i12(P)=R3, i14(P)=R3, i15(P)=R3, i16(P)=R3, i17(P)=R3, i18(P)=R3,R5, i19(P)=R3, i23(P)=R3, i24(P)=R3; Human_2 i1(P)=R3, i2(P)=R3, i4(P)=R3, i5(P)=R3, i6(P)=R3, i7(P)=R3, i9(P)=R3, i10(P)=R3, i18(P)=R3, i19(P)=R3; Human_3 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3, i6(P)=R3, i10(P)=R3, i12(P)=R3, i14(P)=R3, i15(P)=R3.

**Paper 42** — Claude i3(P)=W1, i4(P)=S7; GPT i1(P)=R1, i2(P)=W8, i3(P)=W1, i5(P)=R1; Gemini i2(P)=W3, i3(P)=W4,W12; Human_1 i1(P)=R3, i4(P)=R3, i5(P)=R3, i8(P)=R3, i9(P)=R3, i11(P)=R3, i15(P)=R3, i19(P)=R3, i20(P)=R3, i25(P)=R3; Human_2 i2(P)=R3, i5(P)=R3.

**Paper 43** — Claude i1(P)=W1, i2(P)=W1, i3(P)=W4; Human_1 i1(P)=R3,R5.

**Paper 44** — Claude i2(P)=W4.

**Paper 46** — Claude i2(P)=S7,R5, i3(P)=W2, i4(P)=S7; GPT i1(P)=R1, i2(P)=W12, i3(P)=W3, i4(P)=W4; Gemini i1(P)=R1,R5; Human_1 i7(P)=R3; Human_3 i5(P)=R3, i7(P)=R3, i8(P)=R3, i9(P)=R3.

**Paper 47** — Human_1 i1(P)=R3.

**Paper 49** — Claude i2(S)=W5; GPT i1(S)=S1,S5, i2(S)=S1, i5(S)=W5; Gemini i1(S)=W4; Human_2 i5(S)=R3,R4.

**Paper 50** — Claude i2(P)=W3, i3(P)=S2; GPT i1(P)=W3; Gemini i2(P)=S1,S4, i3(P)=W4.

**Paper 52** — Claude i2(P)=W3, i4(P)=W3; Human_1 i4(S)=R3; Human_2 i8(S)=R3.

**Paper 53** — Claude i1(P)=W4,R5; GPT i1(P)=S7; Gemini i1(P)=W1, i2(P)=W3.

**Paper 54** — Claude i1(P)=W3, i3(P)=W1, i4(P)=S7, i5(P)=W8; GPT i4(P)=S7; Gemini i3(P)=W1, i5(P)=W1; Human_1 i7(P)=R3, i10(P)=R3; Human_2 i1(P)=R3, i5(P)=R3.

**Paper 55** — Claude i2(P)=W4, i4(P)=S7,R5; Gemini i3(P)=W1; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3, i7(P)=R3,R4, i9(P)=R3; Human_2 i1(P)=R3,R5, i2(P)=R3,R5; Human_3 i5(P)=R3,R5.

**Paper 59** — Claude i1(S)=W1, i2(S)=S7, i3(S)=W5, i4(S)=W3; GPT i1(S)=S7, i2(S)=W11, i3(S)=W11, i4(S)=W3; Gemini i1(S)=W4, i3(S)=S7; Human_1 i1(S)=R3, i2(S)=R3; Human_2 i1(S)=R3, i2(S)=R3, i3(S)=R3, i4(S)=R3, i5(S)=R3, i6(S)=R3; Human_3 i1(S)=R3, i2(S)=R3, i3(S)=R3,R5, i4(S)=R3.

**Paper 60** — Claude i1(S)=W11, i2(S)=W11, i3(S)=S4, i4(S)=S2; GPT i1(S)=W11, i2(S)=W5,W11, i3(S)=S5, i4(S)=W5,W12; Gemini i1(S)=S6, i2(S)=S7, i3(S)=W11, i4(S)=S6, i5(S)=S2; Human_1 i1(S)=R3,R5, i2(S)=R3, i3(S)=R3, i4(S)=R3, i6(S)=R3,R5, i7(S)=R3, i8(S)=R3, i9(S)=R3, i10(S)=R3, i11(S)=R3, i13(S)=R3, i14(S)=R3; Human_2 i1(S)=R3, i2(S)=R3, i3(S)=R3; Human_3 i1(S)=R3, i2(S)=R3, i3(S)=R3, i4(S)=R3, i5(S)=R3.

**Paper 61** — Claude i1(S)=S7,S5, i3(S)=S7, i4(S)=S7, i5(S)=W4; GPT i1(S)=S7,S5, i2(S)=S7, i3(S)=S4, i4(S)=S7,S2, i5(S)=S2; Gemini i1(S)=W1, i2(S)=W11, i3(S)=S5, i4(S)=W4, i5(S)=W4; Human_1 i1–i20(S)=R3 (20 items, with i6 & i7 additionally R4); Human_2 i1(S)=R3, i2(S)=R3, i3(S)=R3, i4(S)=R3, i5(S)=R3, i8(S)=R3, i9(S)=R3, i10(S)=R3,R5, i11(S)=R3; Human_3 i1(S)=R3, i2(S)=R3, i3(S)=R3, i4(S)=R3, i5(S)=R3, i6(S)=R3, i7(S)=R3, i8(S)=R3, i9(S)=R3, i10(S)=R3, i11(S)=R3.

**Paper 62** — Claude i1(S)=S7, i2(P)=W3, i3(P)=W3, i3(S)=R1; GPT i1(S)=R1, i2(S)=S7, i3(P)=W11, i3(S)=R1, i4(P)=R2, i4(S)=R1; Gemini i1(P)=S7, i1(S)=R1, i2(P)=W3, i2(S)=R1, i3(P)=R2, i3(S)=W1, i4(S)=W1; Human_1 i1(P)=R3, i1(S)=R3, i2(P)=R3, i2(S)=R3, i7(S)=R3, i9(S)=R3,R5, i11(P)=R3, i12(S)=R3; Human_2 i1(P)=R3, i1(S)=R3, i12(P)=R3, i12(S)=R3; Human_3 i1(S)=R3, i2(S)=R3,R5, i3(S)=R3,R5, i4(S)=R3,R5, i5(S)=R3,R5, i6(S)=R3,R5, i7(P)=R3, i8(S)=R3, i9(S)=R3,R5, i11(S)=R3,R5.

**Paper 63** — GPT i4(P)=W3; Gemini i4(P)=S7,S4; Human_1 i1(P)=R3, i3(P)=R3; Human_2 i2(P)=R3,R4.

**Paper 64** — GPT i4(P)=S5; Gemini i2(P)=S7,R5; Human_1 i1(P)=R3, i3(P)=R3; Human_2 i1(P)=R3, i5(P)=R3, i6(P)=R3,R5, i7(P)=R3,R5, i8(P)=R3,R5.

**Paper 65** — Claude i1(P)=W4, i2(P)=W1, i3(P)=S7, i5(P)=R2; GPT i2(P)=S7, i4(P)=S7; Gemini i2(P)=W9, i3(P)=W4, i4(P)=W4, i5(P)=W4,W5; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3; Human_2 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3,R5, i5(P)=R3, i6(P)=R3, i7(P)=R3,R5.

**Paper 66** — Claude i1(P)=W12,W4, i2(P)=W4, i3(P)=W4; GPT i1(P)=W3,W6, i2(P)=W4,W6, i3(P)=S7, i4(P)=W4; Gemini i1(P)=S7, i2(P)=S7, i3(P)=W9; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3, i5(P)=R3; Human_2 i2(P)=R3, i3(P)=R3, i4(P)=R3,R5, i5(P)=R3, i6(P)=R3,R5; Human_3 i1(P)=R3, i2(P)=R3, i3(P)=R3,R5, i4(P)=R3,R5, i5(P)=R3.

**Paper 67** — Claude i1(P)=S1, i2(P)=W12, i3(P)=W12, i4(P)=W4; GPT i2(P)=S7, i3(P)=W4, i3(S)=R2,R4, i4(P)=W3; Gemini i1(P)=W9, i2(P)=W9, i3(P)=W4, i4(P)=W2, i5(P)=W4; Human_1 i1(P)=R3, i2(P)=R3, i3(P)=R3, i3(S)=R3, i4(P)=R3, i5(P)=R3, i5(S)=R3, i6(P)=R3; Human_2 i1(P)=R3, i2(P)=R3,R5, i3(P)=R3, i4(P)=R3, i5(P)=R3,R5; Human_3 i1(P)=R3,R5, i2(P)=R3, i3(P)=R3, i4(P)=R3,R5, i5(P)=R3,R5.

**Paper 68** — Claude i1(P)=W12, i3(P)=R1,R5, i4(P)=S7; Gemini i1(P)=S7, i2(P)=S7, i4(P)=S7, i5(P)=S7,W12; Human_1 i1(P)=R3, i3(P)=R3, i4(P)=R3,R5; Human_2 i1(P)=R3, i2(P)=R3, i4(P)=R3, i5(P)=R3, i6(P)=R3, i7(P)=R3, i8(P)=R3, i9(P)=R3,R5, i10(P)=R3, i11(P)=R3,R5, i12(P)=R3, i13(P)=R3, i14(P)=R3, i16(P)=R3, i17(P)=R3,R5, i21(P)=R3, i22(P)=R3, i25(P)=R3; Human_3 i1(P)=R3, i2(P)=R3, i3(P)=R3, i4(P)=R3.

**Paper 72** — Claude i5(P)=W3; GPT i1(P)=R1, i3(P)=R1, i4(P)=R1; Gemini i4(P)=W3; Human_1 i1(P)=R3,R4, i2(P)=R3, i3(P)=R3, i5(P)=R3; Human_2 i1(P)=R3,R4, i2(P)=R3, i3(P)=R3, i7(P)=R3; Human_3 i6(P)=R3, i8(P)=R3,R5.

**Paper 73** — Claude i4(P)=S7; GPT i1(P)=S7, i3(P)=S7; Human_1 i1(P)=R3; Human_2 i1(P)=R3, i2(P)=R3.

**Paper 74** — Claude i2(P)=S4,S7, i3(P)=W3, i4(P)=S7; GPT i2(P)=S7; Gemini i1(P)=R2; Human_1 i3(P)=R3,R4, i5(P)=R3,R5; Human_2 i5(P)=R3; Human_3 i1(P)=R3, i2(P)=R3, i5(P)=R3.

**Paper 75** — Human_1 i1(P)=R3; Human_3 i5(P)=R3,R5, i6(P)=R3,R5, i9(P)=R3,R5, i10(P)=R3,R5.

**Paper 76** — Claude i3(P)=S7; Human_2 i1(P)=R3.

**Paper 77** — Claude i2(P)=W4, i2(S)=W4, i3(S)=W4, i5(P)=S5, i5(S)=S7; GPT i1(P)=W3,R5, i2(P)=S5, i2(S)=S7, i3(S)=W4, i4(S)=W5; Gemini i1(S)=W4, i2(S)=S7, i3(S)=W4; Human_1 i1(P)=R3,R5, i2(P)=R3,R5, i3(P)=R3; Human_2 i1(S)=R3, i2(S)=R3, i3(S)=R3, i4(S)=R3, i5(S)=R3.

**Paper 78** — Claude i2(P)=W1, i3(P)=W1; Gemini i2(S)=S7; Human_2 i4(S)=R3, i7(S)=R3.

**Paper 80** — Human_1 i2(S)=R3; Human_2 i2(S)=R3, i13(S)=R3; Human_3 i2(S)=R3, i4(S)=R3, i5(S)=R3.

**Paper 82** — Human_1 i6(P)=R3,R5, i6(S)=R3,R5.

**Paper 85** — GPT i1(P)=S7, i2(P)=S7, i4(P)=R2, i5(P)=S3; Gemini i1(P)=R2, i2(P)=R2, i3(P)=R2; Human_1 i2(P)=R3; Human_2 i17(P)=R3; Human_3 i9(P)=R3.

### 4.2 Paper-level (250 comments across 71 papers)

Format: `PP{paper} slot{k}({Reviewer}, P|S) = {categories}`.

**Paper 1** — slot1(Gemini,P)=R1,S7; slot2(Claude,P)=S5,S4; slot3(GPT,P)=S5,S1.

**Paper 2** — slot1(Claude,P)=S5; slot2(GPT,P)=S5,S1; slot3(Gemini,P)=S7,W1.

**Paper 3** — slot1(GPT,P)=S5,R5; slot2(Claude,P)=W13,R5; slot3(Gemini,P)=W13,R5.

**Paper 4** — slot1(GPT,P)=S5; slot2(Gemini,P)=R1; slot3(Claude,P)=S7,R1.

**Paper 5** — slot1(Gemini,P)=S7; slot2(GPT,P)=W5,W11; slot3(Claude,P)=R1,W5.

**Paper 6** — slot1(Claude,P)=W12,W3; slot2(Gemini,P)=S6; slot3(GPT,P)=W5,W4; slot1(Claude,S)=S5,R5; slot2(Gemini,S)=R5; slot3(GPT,S)=S5,R5.

**Paper 7** — slot1(Gemini,P)=R1,R5; slot2(Claude,P)=S5; slot3(GPT,P)=R1,R5.

**Paper 8** — slot1(Claude,P)=R4,W3,W1.

**Paper 9** — slot1(GPT,P)=R4,W13.

**Paper 10** — slot1(GPT,P)=R1,R5.

**Paper 13** — slot1(Gemini,P)=S3,R5; slot2(Claude,P)=S3,S5,R5; slot3(GPT,P)=S3,S5,R5.

**Paper 14** — slot1(Claude,S)=S5; slot2(GPT,S)=S3,S1,R5; slot3(Gemini,S)=S1,S2,R5.

**Paper 15** — slot1(GPT,P)=S5; slot2(Claude,P)=S5,R5; slot3(Gemini,P)=S5,R5; slot1(GPT,S)=S1; slot2(Claude,S)=S5; slot3(Gemini,S)=S5.

**Paper 16** — slot3(Claude,P)=S5,R5.

**Paper 17** — slot1(Gemini,P)=W1; slot2(GPT,P)=W6,W1; slot3(Claude,P)=W1.

**Paper 18** — slot1(Claude,P)=S5,R5; slot2(Gemini,P)=W5,R5; slot3(GPT,P)=W5,R5.

**Paper 19** — slot1(Gemini,P)=R5; slot2(Claude,P)=R5; slot3(GPT,P)=R5.

**Paper 20** — slot1(Claude,P)=W2; slot2(GPT,P)=R1,R5; slot3(Gemini,P)=W2.

**Paper 21** — slot1(GPT,P)=R5; slot2(Claude,P)=R5; slot3(Gemini,P)=R5.

**Paper 22** — slot1(GPT,P)=R1; slot2(Gemini,P)=R1; slot3(Claude,P)=R1.

**Paper 23** — slot1(Gemini,P)=S5,S6; slot2(GPT,P)=S1; slot3(Claude,P)=S1.

**Paper 24** — slot1(Claude,P)=S5; slot2(Gemini,P)=S1,S5; slot3(GPT,P)=S5.

**Paper 25** — slot3(GPT,P)=S1,S3.

**Paper 26** — slot1(Claude,P)=S6,S5; slot2(GPT,P)=S1,S5; slot3(Gemini,P)=S5.

**Paper 27** — slot1(GPT,P)=S1; slot2(Claude,P)=S1; slot3(Gemini,P)=S1; slot1(GPT,S)=R5; slot2(Claude,S)=R5; slot3(Gemini,S)=R5.

**Paper 28** — slot1(GPT,P)=W3,W4; slot2(Gemini,P)=S5,R5; slot3(Claude,P)=S5,R5.

**Paper 29** — slot1(Gemini,P)=S5; slot2(GPT,P)=R1,R5; slot3(Claude,P)=S5,S7.

**Paper 31** — slot1(Gemini,S)=S1,S4,S5; slot2(Claude,S)=S4,S5; slot3(GPT,S)=S3,S5.

**Paper 32** — slot2(GPT,P)=R5; slot3(Gemini,P)=R5.

**Paper 33** — slot1(GPT,P)=R5; slot2(Claude,P)=R5; slot3(Gemini,P)=R5; slot2(Claude,S)=S5.

**Paper 34** — slot2(Gemini,P)=R5; slot3(Claude,P)=R5; slot2(Gemini,S)=S1,W4.

**Paper 35** — slot1(Gemini,P)=S1,S4,S5; slot2(GPT,P)=S4,S1; slot3(Claude,P)=S2; slot1(Gemini,S)=R5; slot2(GPT,S)=R5; slot3(Claude,S)=R5.

**Paper 36** — slot1(Claude,P)=R1,R5; slot2(Gemini,P)=W5; slot3(GPT,P)=S2,S5.

**Paper 37** — slot1(Gemini,P)=S1,S5; slot2(Claude,P)=S5; slot3(GPT,P)=S1,S4,S2; slot1(Gemini,S)=S5; slot2(Claude,S)=S6; slot3(GPT,S)=S5,S2,S1.

**Paper 41** — slot1(Gemini,P)=S5; slot2(GPT,P)=S4,S5; slot3(Claude,P)=S5.

**Paper 42** — slot1(Claude,P)=S5; slot2(Gemini,P)=S5; slot3(GPT,P)=W12,S3.

**Paper 43** — slot3(GPT,P)=R5.

**Paper 44** — slot1(GPT,P)=R1.

**Paper 45** — slot1(GPT,P)=R1,R5; slot2(Claude,P)=R1,R5; slot3(Gemini,P)=R1,R5.

**Paper 46** — slot1(GPT,P)=R1,R5; slot2(Gemini,P)=R1,R5; slot3(Claude,P)=R1,R5.

**Paper 47** — slot1(Gemini,P)=S4,S5; slot2(GPT,P)=S3,S5; slot3(Claude,P)=S4,S5.

**Paper 48** — slot1(Gemini,P)=R5; slot2(Claude,P)=R5; slot3(GPT,P)=R5.

**Paper 49** — slot1(Claude,P)=R5; slot2(GPT,P)=R5; slot3(Gemini,P)=R5; slot1(Claude,S)=S4; slot2(GPT,S)=S1,S4,S2; slot3(Gemini,S)=S6,S5.

**Paper 50** — slot1(GPT,P)=S5,S4; slot2(Gemini,P)=S1,S2; slot3(Claude,P)=S2; slot1(GPT,S)=S5; slot2(Gemini,S)=R1,R5; slot3(Claude,S)=S2.

**Paper 51** — slot1(Gemini,P)=R1,R5; slot2(GPT,P)=R1,R5; slot3(Claude,P)=R1,R5; slot1(Gemini,S)=R1,S1; slot2(GPT,S)=R1,S1; slot3(Claude,S)=R1,R5.

**Paper 52** — slot1(Claude,P)=R1,R5; slot2(Gemini,P)=S5; slot3(GPT,P)=S5; slot1(Claude,S)=R1,R5; slot2(Gemini,S)=S5; slot3(GPT,S)=S5,S1.

**Paper 53** — slot1(GPT,P)=W1; slot2(Claude,P)=W1; slot3(Gemini,P)=W1.

**Paper 54** — slot1(GPT,P)=W1; slot3(Claude,P)=W1.

**Paper 55** — slot1(Gemini,P)=S5; slot3(Claude,P)=W4,S4.

**Paper 59** — slot1(Claude,P)=R5; slot2(Gemini,P)=R5; slot3(GPT,P)=R5; slot1(Claude,S)=S5; slot2(Gemini,S)=S5,W4; slot3(GPT,S)=S4,S5.

**Paper 60** — slot1(Gemini,P)=R5; slot2(Claude,P)=R5; slot3(GPT,P)=R5; slot1(Gemini,S)=S1,S5; slot2(Claude,S)=S4,S5; slot3(GPT,S)=S5,W5.

**Paper 61** — slot1(Claude,P)=R5; slot2(GPT,P)=R5; slot3(Gemini,P)=R5; slot1(Claude,S)=S6; slot2(GPT,S)=S2,S5; slot3(Gemini,S)=S5,R1.

**Paper 62** — slot1(Claude,P)=W5,W1; slot2(Gemini,P)=W5,W1; slot3(GPT,P)=W5,W1; slot1(Claude,S)=R1,R5; slot2(Gemini,S)=S5; slot3(GPT,S)=R1,R5.

**Paper 63** — slot1(Gemini,P)=R1; slot2(Claude,P)=R1; slot3(GPT,P)=S3,S5.

**Paper 64** — slot1(Claude,P)=R1; slot2(GPT,P)=R1,S7; slot3(Gemini,P)=R1.

**Paper 65** — slot1(GPT,P)=S2,S1; slot2(Claude,P)=S2; slot3(Gemini,P)=S2.

**Paper 66** — slot1(GPT,P)=S5; slot2(Gemini,P)=S3; slot3(Claude,P)=R1,R5.

**Paper 67** — slot1(Gemini,P)=R1,R5; slot2(GPT,P)=S2,R5; slot3(Claude,P)=S5,S4; slot1(Gemini,S)=S1,S6; slot2(GPT,S)=R1,S5; slot3(Claude,S)=R1.

**Paper 68** — slot1(Claude,P)=R1,R5; slot2(Gemini,P)=S5; slot3(GPT,P)=S3,S5.

**Paper 72** — slot1(Claude,P)=W12; slot2(Gemini,P)=W12; slot3(GPT,P)=W3.

**Paper 73** — slot1(Gemini,P)=W12,S5; slot2(Claude,P)=S5; slot3(GPT,P)=W12,S5.

**Paper 74** — slot1(Claude,P)=S4,S5; slot2(GPT,P)=S5; slot3(Gemini,P)=S1,S5.

**Paper 77** — slot1(Gemini,P)=W13; slot2(Claude,P)=S5; slot3(GPT,P)=S5; slot1(Gemini,S)=R1,R5; slot2(Claude,S)=R1,R5; slot3(GPT,S)=S5,R5.

**Paper 78** — slot1(Claude,P)=W1; slot2(GPT,P)=R1,R5; slot3(Gemini,P)=S5,R5; slot1(Claude,S)=R5; slot2(GPT,S)=R5; slot3(Gemini,S)=R5.

**Paper 79** — slot1(GPT,P)=W4; slot2(Claude,P)=S5; slot3(Gemini,P)=S5,S6.

**Paper 80** — slot1(GPT,P)=W4; slot1(GPT,S)=R1,R5; slot2(Gemini,S)=R1,R5; slot3(Claude,S)=S5,S7.

**Paper 81** — slot1(Gemini,P)=S5; slot1(Gemini,S)=S5; slot2(GPT,S)=R5; slot3(Claude,S)=R5.

**Paper 82** — slot1(Claude,S)=R5; slot2(Gemini,S)=R5; slot3(GPT,S)=R5.

**Paper 83** — slot1(Gemini,P)=S5; slot3(GPT,P)=S2,S5; slot1(Gemini,S)=S5,R5; slot2(Claude,S)=R5; slot3(GPT,S)=S2,S5.

**Paper 84** — slot2(GPT,P)=S5; slot1(Claude,S)=R5; slot2(GPT,S)=S5; slot3(Gemini,S)=R5.

**Paper 85** — slot1(GPT,P)=S5,R5; slot2(Claude,P)=R1,R5; slot3(Gemini,P)=S5,S1.

### 4.3 Coverage guarantee

- **Item-level (767 comments)** — every comment across the 66
  papers covered is accounted for in §4.1. Human-reviewer item comments
  default to R3 (assessment of a human reviewer) with optional R4/R5
  add-ons; AI-reviewer item comments carry at least one W, S, R1, or R2
  code.
- **Paper-level (250 comments)** — every comment across 71 papers is
  accounted for in §4.2, including minimal "Yes"/"No"/"."/item-number
  responses (R5), "No"-parity (R1+R5), and short "None" rejections
  (R1+R5 or W13+R5 depending on context).
- **Categories are non-exclusive**. A single comment may carry up to
  three or four codes (e.g., W1+W3, S1+S4+S5). Codes are additive, and
  no comment is "double-counted" statistically — the codes just
  describe the multiple facets the annotator touched in one sentence.

Category totals (approximate) from the tagged list above:

- **Weaknesses (W1–W13)**: W1 ≈ 28, W2 ≈ 11, W3 ≈ 42, W4 ≈ 45,
  W5 ≈ 22, W6 ≈ 5, W7 ≈ 2, W8 ≈ 5, W9 ≈ 6, W10 ≈ 1, W11 ≈ 10,
  W12 ≈ 20, W13 ≈ 5. **Total W tags ≈ 202**.
- **Strengths (S1–S7)**: S1 ≈ 25, S2 ≈ 14, S3 ≈ 13, S4 ≈ 18,
  S5 ≈ 60, S6 ≈ 8, S7 ≈ 55. **Total S tags ≈ 193**.
- **Residual (R1–R5)**: R1 ≈ 55, R2 ≈ 18, R3 ≈ 450, R4 ≈ 40,
  R5 ≈ 150. **Total R tags ≈ 713**.

(These are rough counts derived from the tags in §4.1/§4.2; because
tags are non-exclusive, they sum to more than 1017.)

---

## 5. Key takeaways

1. **The single biggest AI advantage is code reading.** Across physics,
  biology, chemistry, and ML papers, annotators repeatedly said
   variants of "AI only exceeded humans in checking the code" or "I
   would never read the code as a human reviewer." This pattern
   dominates both the item-level and paper-level data.
2. **The single biggest AI weakness is missing community/field context.**
  W1 alone accounts for a large share of the Not-Correct AI items. AI
   reviewers produce critiques that are defensible in isolation but
   that no expert in the community would raise because they conflict
   with accepted practice or implicit norms.
3. **Anachronistic critique (W2)** is a distinctive failure mode —
  multiple annotators noted that AI cites literature published after
   the preprint's date, invalidating the critique.
4. **AI catches internal consistency errors (S2) that humans overlook.**
  Contradictions between abstract/methods/figures, wrong confidence
   intervals, scenario-label mismatches, and design-specification
   mismatches show up disproportionately in AI critiques.
5. **Overall annotator framing** is a trade-off: "AI catches things I
  would not have caught (code, internal consistency, reproducibility,
   statistical rigor), but AI also criticizes things no expert would
   criticize (field norms, historical context, infeasible scope)."
   Whether AI net exceeds or falls short of humans depends on what the
   paper's dominant vulnerability is — if code/consistency, AI wins; if
   interpretation against field norms, human experts remain necessary.
6. **Redundancy (W11) matters for panel design.** Because the three AI
  reviewers tend to raise overlapping points, a panel of
   2 humans + 1 AI is not obviously worse than 3 humans — but
   1 human + 2 AI loses more field-context coverage than the raw
   per-model numbers suggest.

7. **The tag distribution (§4.3) shows the skew quantitatively.**
   S5 (specialized niche catches) is by far the most common positive
   tag (~60 instances), dominated by paper-level "only reviewer to
   catch X" endorsements. S1 (code reading) follows at ~25. On the
   weakness side, W3 (over-harsh/out-of-scope) and W4 (misreads paper)
   are tied at ~42–45 each, followed by W1 (lacks field context) at
   ~28. This means: AI wins most often by finding niche issues, and
   loses most often by either being too harsh or failing to read the
   paper carefully — *not* by obvious hallucination of facts.

---

## 6. Final single-label S/W statistics — current authoritative scheme

> **Note**: This section supersedes the per-comment enumeration in §4 and
> the rough counts in §4.3. The earlier numbers were based on a multi-label
> tagging pass before the `reviewer_unique_items.json` extraction was built.
> The numbers in §6 are the ones to use for figures and downstream counts.
>
> **Update**: These counts reflect the manual verification pass in
> `reviewer_sw_analysis.json`. The pipeline is now: (1) regex classifier
> writes an initial label; (2) every entry in the `W_unspecified`,
> `S_generic`, and `N` residual buckets is inspected by hand and moved to
> a specific category where possible; (3) every remaining entry in every
> bucket is inspected by hand without using regex to confirm or correct
> its label. The residual buckets shrank substantially (`W_unspecified` 42
> → 10, `S_generic` 128 → 40, `N` 49 → 0) and ~80 additional entries were
> re-labelled in step (3). The original `W16` (aggressive / ML-style tone
> in non-CS fields) is now empty — both candidate entries were moved to
> `W1` and `W3` after manual review.

### 6.1 Data flow — from raw annotator commentary to S/W counts

Two source files feed this analysis:

- `expert_comments_item_level.json` — **767** per-item free-form comments
- `expert_comments_paper_level.json` — **250** per-paper free-form comments

Not every one of these 1,017 comments ends up in the S/W tables. Some are
about human reviewers (out of scope: this analysis is about AI reviewer
behavior). Some paper-level responses are structured item-number references
that belong to the similarity ground truth, not to S/W. Some are bare
"Yes" / "No" with no signal. The tree below shows exactly how the raw
1,017 comments are routed to their final destinations — nothing is
"deleted", every comment ends up somewhere.

```
Raw annotator commentary on AI reviews (1,017 total)
│
├── expert_comments_item_level.json  (767 per-item comments)
│   │
│   ├── Human reviewer items              446  ─► excluded from this analysis
│   │                                           (Human_1 162, Human_2 173, Human_3 111)
│   │                                           — this section studies AI reviewer
│   │                                           strengths/weaknesses, so human-review
│   │                                           commentary is not counted.
│   │
│   └── AI reviewer items                 321  ─► S/W classification
│                                                (GPT 102, Claude 115, Gemini 104)
│
└── expert_comments_paper_level.json  (250 per-paper comments)
    │
    ├── items_pos     70   ─► reviewer_unique_items.json
    │                         (positive similarity ground truth:
    │                          annotator listed explicit item numbers
    │                          unique to this AI reviewer, 126 item
    │                          references total)
    │
    ├── items_neg      8   ─► reviewer_unique_items.json
    │                         (negative similarity ground truth:
    │                          annotator explicitly said NO items were
    │                          unique, e.g. "all items captured by
    │                          other reviewers")
    │
    ├── unclear       51   ─► discarded — no reliable signal
    │                         (bare "Yes" / "No" / "-" / "."; too thin
    │                          to classify as positive, negative, or
    │                          substantive)
    │
    └── descriptive  121   ─► S/W classification
                              (substantive free-form unique-catch
                               descriptions without explicit item
                               numbers)


S/W classification input (reviewer_sw_analysis.json)
│
│   321 (AI item-level) + 121 (paper-level descriptive) = 442 entries
│
├── Specific weaknesses (W1–W16)                 260   58.8%
│   │   W1 field norms 54, W2 over-harsh 46, W3 paper-states-X 37,
│   │   W4 redundancy 28, W5 vague 24, W6 trivial 16, W7 term-of-art
│   │   13, W8 anachronism 9, W9 over-inflates 9, W10 already-flagged 6,
│   │   W11 figure misread 6, W12 misquote 5, W13 missed supp 3,
│   │   W14 prior work 2, W15 table misread 1, W16 figure-blind 1
│   │
│   │   (W codes are now frequency-ranked: W1 is the most common weakness,
│   │    W16 is the rarest. The old "aggressive / ML-style tone" category
│   │    had 0 entries after manual verification and was removed from the
│   │    taxonomy.)
│
├── W_unspecified                                  10    2.3%
│       item-level AI comments labelled Not Correct whose annotator
│       commentary had no recoverable specific W signal even after
│       manual review (bare "rather bad", "misses the point", "complete
│       fail", etc.)
│
├── Specific strengths (S1–S6)                   132   29.9%
│   │   S1 statistical/methodology rigor 45, S2 code reading 28,
│   │   S3 niche field catch 27, S4 internal consistency 15,
│   │   S5 reproducibility 10, S6 counter-narrative 7
│   │
│   │   (S codes are also frequency-ranked; S3, S4, S5, S6 kept their
│   │    previous indices, S1 and S2 swapped.)
│
├── S_generic                                     40    9.0%
│       item-level AI comments labelled Correct whose annotator
│       commentary was purely generic agreement ("good point",
│       "well-argued", "agree"), with no recoverable specific S signal
│       even after manual review
│
└── N (unclassified substantive)                    0    0.0%
        empty — every paper-level descriptive comment has been
        hand-assigned to a specific S/W code or to S_generic
```

### 6.1.1 Reconciling your mental model

If you start from "raw comments about AI reviews" and subtract your way to
the 442 in S/W classification, the arithmetic is:

```
  767 per-item comments         (expert_comments_item_level.json)
- 446 Human reviewer items      (out of scope — we're studying AI)
= 321 AI item-level comments    ─► into S/W

  250 per-paper comments        (expert_comments_paper_level.json)
-  70 items_pos                 ─► into reviewer_unique_items.json (pos ground truth)
-   8 items_neg                 ─► into reviewer_unique_items.json (neg ground truth)
-  51 unclear                   ─► discarded (bare yes/no)
= 121 paper-level descriptive   ─► into S/W

  321 + 121 = 442 substantive AI-review comments in S/W classification
```

And the 442 splits as `260 W + 132 S + 10 W_unspecified + 40 S_generic + 0 N`.

So if you asked "I have 321 AI-item comments plus 250 paper-level comments
= 571 comments about AI reviews — what happened to the 571 − (260 + 132) =
**179** that aren't in the specific W or S categories?", the answer is:

| Destination | n | What it is |
|---|---|---|
| `reviewer_unique_items.json` items_pos | 70 | similarity ground truth — explicit item-number references |
| `reviewer_unique_items.json` items_neg | 8 | similarity ground truth — "no items are unique" explanations |
| discarded (`unclear`) | 51 | bare "Yes"/"No"/"-"/"." with no signal |
| `W_unspecified` (still in `reviewer_sw_analysis.json`) | 10 | Not Correct with no specific W signal |
| `S_generic` (still in `reviewer_sw_analysis.json`) | 40 | Correct with no specific S signal |
| `N` (residual) | 0 | empty after manual pass |
| **Total "not in specific S/W"** | **179** | |

None of the 179 are "removed". 129 of them live in a sibling JSON file
as similarity ground truth or are discarded as signal-free. 50 of them
live in `reviewer_sw_analysis.json` under the residual buckets, meaning
they entered S/W classification but no specific rule matched even after
manual review.

### 6.1.2 What's in the S/W classification

For reference, the inputs to the specific S/W counts in §6.2 and §6.3 are:

| Source | n | Notes |
|---|---|---|
| Item-level AI comments (`expert_annotation` HF config, `reviewer_type=='AI'`) | **321** | annotator commentary on a specific AI review item, with a Correct/Significant/Sufficient label |
| Paper-level **descriptive** entries from `reviewer_unique_items.json` | **121** | substantive free-form comments that did NOT carry an explicit item-number list |
| **Total in classification** | **442** | |

### 6.2 Specific weakness counts (frequency-ranked)

16 categories, **260 specific tags total** (~58.8% of the 442 substantive
comments). Codes are the ones in `reviewer_sw_analysis.json` after the
frequency-based renumber. The old "aggressive / ML-style tone in non-CS
fields" category had 0 entries after manual verification and was removed
from the taxonomy.

| Code | Name | Total | item-level | paper-level |
|---|---|---|---|---|
| **W1** | Missing community / field norms | **54** | 48 | 6 |
| **W2** | Over-harsh / out-of-scope / unrealistic | **46** | 43 | 3 |
| **W3** | Paper explicitly states X, AI says missing | **37** | 37 | 0 |
| **W4** | Redundancy across the 3 AI reviewers | **28** | 19 | 9 |
| **W5** | Vague / verbose / no actionable recommendation | **24** | 22 | 2 |
| **W6** | Trivial / nitpicking (typo, grammar, style) | **16** | 13 | 3 |
| **W7** | Technical term-of-art confusion | **13** | 12 | 1 |
| **W8** | Cites evidence from after the preprint (anachronistic) | **9** | 7 | 2 |
| **W9** | Over-inflates small code/text inconsistencies | **9** | 9 | 0 |
| **W10** | Criticizes what authors already flagged as a limitation | **6** | 6 | 0 |
| **W11** | AI misreads a figure or caption | **6** | 6 | 0 |
| **W12** | AI misquotes or fabricates a verbatim quote | **5** | 5 | 0 |
| **W13** | AI misses supplementary content | **3** | 3 | 0 |
| **W14** | Ignores authors' own cited prior work | **2** | 2 | 0 |
| **W15** | AI misreads a table | **1** | 1 | 0 |
| **W16** | Cannot analyze figures — only text | **1** | 1 | 0 |

### 6.3 Specific strength counts (frequency-ranked)

6 categories, **132 specific tags total** (~29.9% of the 442 substantive
comments).

| Code | Name | Total | item-level | paper-level |
|---|---|---|---|---|
| **S1** | Statistical / methodology rigor | **45** | 15 | 30 |
| **S2** | Code reading | **28** | 12 | 16 |
| **S3** | Specialized niche field catch | **27** | 7 | 20 |
| **S4** | Internal consistency across sections | **15** | 7 | 8 |
| **S5** | Reproducibility / dependency failures | **10** | 3 | 7 |
| **S6** | Big-picture / counter-narrative synthesis | **7** | 2 | 5 |

### 6.4 Residual / non-specific buckets

| Bucket | n | What it is |
|---|---|---|
| **W_unspecified** | **10** | item-level AI comments labelled `Not Correct` whose annotator commentary carried no specific W signal even after manual review (bare "rather bad", "misses the point", "complete fail", etc.) |
| **S_generic** | **40** | item-level AI comments labelled `Correct` whose annotator commentary carried no specific S signal even after manual review (e.g. "Good point.", "Agree.", "Well-argued.") |
| **N** | **0** | empty — all previously-unclassified paper-level descriptive comments have been manually assigned to a specific S/W or residual bucket |
| **Total residual** | **50** | ~11.3% of the 442 substantive comments |

### 6.5 Total accounting

The 442 substantive comments split as:

| Bucket | n | % of 442 |
|---|---|---|
| Specific weaknesses (W1–W16) | 260 | **58.8%** |
| W_unspecified | 10 | 2.3% |
| Specific strengths (S1–S6) | 132 | **29.9%** |
| S_generic | 40 | 9.0% |
| N | 0 | 0.0% |
| **Total** | **442** | 100% |

Compared to the pre-verification counts (160 specific W + 63 specific S +
42 W_unspec + 128 S_generic + 49 N), manual inspection found specific
signals in an additional 100 weakness entries and 69 strength entries.
Most of the recovered S entries came from the paper-level `N` bucket
(where the regex missed niche-catch phrasing) and from `S_generic` (where
the regex conservatively defaulted to "generic" even when the comment
named a specific statistical or code-reading catch).

Plus the structured extraction outside of S/W:

| Structured extraction (in `reviewer_unique_items.json`) | n |
|---|---|
| `items_pos` (positive similarity ground truth, 126 item refs) | 70 |
| `items_neg` (negative similarity ground truth) | 8 |
| `unclear` (bare yes/no, discarded) | 51 |

Grand total of all paper-level + AI-item-level annotator commentary: **571**.

### 6.6 Source split commentary — item-level vs paper-level bias

There is a strong, systematic split in *where* each kind of signal appears:

**Weaknesses come overwhelmingly from item-level commentary** (231 of 260
specific W tags = **88.8% item-level**). Annotators write detailed reasons
when they disagree with an AI item (label = `Not Correct`), and those
reasons land directly under the specific W code.

**Strengths come disproportionately from paper-level commentary** (86 of 132
specific S tags = **65.2% paper-level**). When the AI is `Correct`,
item-level annotators often just write "Good point." or "Agreed." (these
collapse into **S_generic = 40**). The substantive *what was unique?*
answer lives in the paper-level survey, where annotators describe the
high-level catches — and those catches map to S1, S2, S3, etc. (The 65%
paper-level skew is less extreme than the pre-verification estimate of
81% because manual inspection recovered ~45 specific S tags from the
item-level `S_generic` bucket that the regex had defaulted away.)

This is **not** a measurement bias in the quantitative sense; it reflects
how the survey instrument was designed. The takeaway:

- **W-code frequencies are fair to read as "how often each failure mode
  appears in expert commentary"**.
- **S-code frequencies are partially biased by the paper-level survey
  question**, which asks specifically about *unique* AI catches. So S3
  (niche catch) is partially circular by construction — the paper-level
  question is "what did the AI catch that others didn't?", which is the
  literal definition of S3.

### 6.7 Red flags before building figures

These are issues we should think through (or accept) before building Alt 2 (bar chart) and Alt 3 (case studies):

1. **W tags outnumber S tags 270 vs 172 (~1.6×).** This is *not* evidence
   that AI is worse than human reviewers. It reflects that
   - annotators write detailed reasons when AI is wrong (→ specific W),
   - but usually write generic agreement when AI is right (→ S_generic,
     40 of these even after manual recovery).
   After the manual verification pass the ratio narrowed from 2.5× to
   1.6× because many former `S_generic` and `N` entries turned out to
   contain specific positive signals. The bar chart will still look W-
   heavy; we should either (a) put a note on the figure that S_generic
   is excluded, or (b) show "specific S + S_generic" together so the
   visual balance matches the underlying data.

2. **The `N` bucket is now empty.** All 49 previously-unclassified paper-
   level descriptive comments have been hand-assigned. This closes the
   "S3 is undercounted because regex misses niche catches" concern from
   the earlier pass; S3 climbed from 16 → 27 via manual recovery (mostly
   from `N`), and many other S-codes also gained paper-level entries.

3. **Frequency-based renumbering has been applied.** After the manual
   pass the W/S codes have been re-ranked by final count: `W1` is now
   "Missing field norms" (was `W3`), `W3` is "Paper explicitly states X"
   (was `W1`), `S1` is "Statistical/methodology rigor" (was `S2`), and
   `S2` is "Code reading" (was `S1`). The new order is stable and
   should be used for all downstream figures and tables.

4. **The W11–W17 long tail is even thinner now** (W11–W17 together = 12
   tags). These are real, distinct failure modes but won't render
   meaningfully on a bar chart. Options unchanged:
   - Show all 16 W bars and accept that the long tail is small but
     informative.
   - Show top-10 W bars and lump W11–W17 into "other".
   - Keep the long tail but visually de-emphasise (gray bars / smaller
     font for n ≤ 3).

5. **W5 (criticizes acknowledged limitations) dropped from 14 → 6** after
   verification because several `W5` entries turned out to be `W1`
   ("paper states X") or `W2` ("too harsh") under a narrower reading of
   "author self-flagged as a limitation". Consider whether W5 is still
   flagship-worthy at n=6 or if it should be folded into W1/W2.

6. **S1 (code reading) is now split closer to 50/50** (12 item / 16
   paper = 57% paper-level). The shift from 79% → 57% happened because
   several paper-level entries describing code reading got moved to S5
   (reproducibility) during verification when the underlying issue was
   "code was not provided / broken", not "AI read the code and caught a
   bug". The S1 tag is now cleaner as "annotator notes AI inspected the
   code and caught something".

7. **Item-level S-tag floor may still be conservative.** S1 item-level =
   12 and S2 item-level = 15; both grew a lot in manual verification but
   the regex likely missed some additional cases in the original `Correct
   + Good point` phrasing. Acceptable for now; decide whether to run a
   second regex pass or accept the manual-only floor.

8. **Top-3 flagship categories after renumbering:**
   - Strengths: **S1 Statistical rigor** (45), **S2 Code reading** (28),
     **S3 Niche catch** (27).
   - Weaknesses: **W1 Field norms** (54), **W2 Over-harsh** (46),
     **W3 Paper-states-X** (37).
     Flagship cases should lean into field-norm violations as the
     #1 AI failure mode.

9. **Item vs paper imbalance for the bar chart**: same recommendation
   as before — showing item-level vs paper-level as separate stacked
   bars makes the survey-design bias visible and is more honest about
   the data.

### 6.8 Decisions carried over to figure building

With verification complete, the open decisions for Alt 2 (bar chart) and
Alt 3 (flagship case studies) are:

- **Q1**: Bar chart — **stacked item vs paper** or **single combined
  bars**? (Recommendation: stacked, to show the survey-design bias.)
- **Q2**: Show **all 16 W categories** or only **top-10**?
- **Q3**: For Alt 3 flagship case studies, the new top-3 lists (after
  frequency renumbering):
  - Strengths: **S1 Statistical rigor**, **S2 Code reading**, **S3 Niche catch**
  - Weaknesses: **W1 Field norms**, **W2 Over-harsh**, **W3 Paper-states-X**
- **Q4**: Should we include the 70 + 8 `items_pos`/`items_neg` ground-truth
  entries as a separate section ("similarity ground truth — 78 entries
  reserved for downstream evaluation") in the figure or text? Or keep them
  silent until the similarity-measure paper section?

