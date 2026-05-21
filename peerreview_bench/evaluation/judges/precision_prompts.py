"""Dimension definitions for the precision meta-review judge.
Vendored from metareview_bench/expert_annotation_meta_review/prompts.py."""

from textwrap import dedent

_DIMENSION_DEFINITIONS = dedent("""\
    -----------------------------------------------------------------
    DIMENSION DEFINITIONS (apply these to the review item in order)
    -----------------------------------------------------------------

    (1) CORRECTNESS — "is the main point of the criticism correct and
        clearly stated?"

        A review item is CORRECT if:
          - The core factual claim the reviewer is making holds up
            when checked against the paper, AND
          - A reader can clearly understand what criticism is being
            made (it is not so vague that you have to guess).

        Imprecise WORDING is OK. The reviewer does not need to phrase
        their point perfectly. What matters is whether the SUBSTANCE
        of the claim is right. If you can understand what they mean
        and their underlying point is factually accurate, mark it
        Correct — even if they cite the wrong equation number, use
        slightly off terminology, or frame the issue awkwardly.

        A review item is NOT CORRECT when any of these apply:
          - The reviewer's core factual claim does not hold up against
            the paper (e.g., says the authors used method X when they
            actually used method Y)
          - The reviewer criticizes an approach that is actually
            justified given the paper's stated goal or scope
          - The reviewer conflates terminology in a way that makes
            the criticism inapplicable (not just imprecise — wrong
            target)
          - The reviewer's factual premise is wrong in general
          - The reviewer complains that something is missing that is
            in fact adequately present in the paper — but if the
            information is only partially present and the broader
            concern (inadequate coverage) still holds, the item may
            still be Correct
          - The reviewer's main point is so vaguely stated that you
            cannot determine what specific criticism is being made
          - The reviewer makes a reasonable-sounding claim that turns
            out to be inaccurate when you actually verify against the
            paper

        CALIBRATION RULES:
          - Be STRICT on factual accuracy. If the reviewer's core
            claim is factually wrong — even if it sounds plausible —
            mark it Not Correct.
          - Be LENIENT on expression. Imprecise wording, a wrong
            equation number, or a slightly-off supporting detail do
            NOT make the item incorrect if the underlying point is
            right. Those are evidence issues, not correctness issues.
          - For partially-correct claims: judge the MAIN point. If
            the main claim holds even though a peripheral detail is
            wrong, mark Correct. If the main claim itself is the part
            that's wrong, mark Not Correct.
          - For subjective claims (e.g., "the writing is confusing"):
            mark Correct unless the writing is objectively clear.
            Assume reviewers are honest about their reading
            experience.

    (2) SIGNIFICANCE — "is the main point of the criticism talking about
        a significant aspect of the paper that is constructive to
        enhance the paper, rather than touching a minor issue?"

        Judged CONDITIONAL on correctness — you only judge significance
        for items you marked Correct. For Not Correct items, set
        significance to null.

        Three levels:

          - "Significant": The criticism is insightful and helpful to
            improve the paper. If the authors addressed this issue, the
            paper would be meaningfully better. Includes:
              * Methodology concerns that would improve the experimental
                design
              * Missing baselines, ablations, or comparisons that would
                strengthen the evaluation
              * Unclear writing in key sections where fixing it would
                improve the reader's understanding
              * Missing discussion of related work that contextualizes
                the contribution
              * Reproducibility concerns (missing code, data, or
                specification details)
              * Suggestions that would make the argument more convincing
              * Internal inconsistencies between text and figures/tables
              * Any issue where addressing it would make the paper
                substantively stronger

          - "Marginally Significant": The criticism is NOT helpful for
            actually improving the paper, but is still worth keeping in
            the review. The item is not constructive for enhancing the
            paper's substance, but it's not so trivial that it should
            be removed. Includes:
              * Typos and grammatical errors
              * Stylistic preferences ("I would phrase this differently")
              * Suggestions to submit to a different venue
              * Formatting or presentation nits that don't affect
                understanding
              * Minor references that are nice-to-have but don't change
                the paper's story

          - "Not Significant": A very minor item that should not affect
            acceptance and is better removed from the review entirely.
            The review would be improved by deleting this item.

        CRITICAL boundary rules:
          - The key question for Significant vs Marginally Significant
            is: "would addressing this criticism actually improve the
            paper?" If yes → Significant. If no (it's just noting a
            minor issue) → Marginally Significant.
          - Do NOT set a high bar for Significant. Anything that is
            constructive and would genuinely help the paper is
            Significant. You don't need the issue to "threaten the
            paper's validity" — just to be helpful feedback.
          - Judge the underlying issue, not the reviewer's tone. A
            mildly-phrased substantive criticism is Significant. An
            urgently-phrased typo is Marginally Significant.

    (3) EVIDENCE — "does the reviewer provide enough justification for
        a meta-reviewer to independently verify the criticism without
        having to do substantial additional work?"

        Judged CONDITIONAL on correctness AND at-least-Marginal
        significance — you only judge evidence for items you marked
        Correct AND (Significant OR Marginally Significant). For items
        marked Not Correct or Not Significant, set evidence to null.

        VALID EVIDENCE TYPES (any combination of these counts):
          - Pointing at specific material in the paper: a quoted
            passage, an exact number, a figure panel, a table entry, a
            line in the code, an equation reference
          - Pointing at what is ABSENT from the paper: missing
            control, missing baseline, missing error bar, missing
            ablation, missing reference
          - Appealing to external knowledge with a specific enough
            pointer that a meta-reviewer could look it up: a named
            paper, a well-known domain standard, a specific metric
          - Building a logical derivation from stated premises in the
            paper (e.g., "if X holds then Y must be true; the paper
            asserts X; therefore Y should be shown")
          - Observing an internal inconsistency between explicitly
            quoted or referenced parts of the paper

        "Sufficient" means the reviewer's evidence chain covers ALL of
        their core claim — a meta-reviewer reading the review item
        alongside the paper could decide whether the criticism holds
        without having to guess at the reviewer's reasoning or fill in
        a missing step.

        "Requires More" means at least one core step in the reviewer's
        reasoning is not supported — the meta-reviewer would have to
        do substantive independent work to complete the chain.

        Examples of SUFFICIENT evidence:
          - "Figure 2 panel (b) shows the effect disappearing above
            strain 0.15, but the text (Section 4.2, paragraph 3)
            claims the effect persists to 0.2. These contradict each
            other."
          - "The paper reports p<0.01 with n=12 but does not report a
            multiple-comparisons correction; with 5 hypotheses tested
            in Table 3, Bonferroni would set the threshold at 0.002."
          - "The ablation in Table 4 removes component X but not
            component Y; since Y is load-bearing for the mechanism in
            Section 3, the reported gain may be from Y, not from the
            proposed addition."

        Examples of REQUIRES MORE:
          - "I think the baselines are too weak" (no specific baseline
            named, no comparison given)
          - "The results seem unlikely" (no specific number or prior
            cited)
          - "This has been done before" (no reference given)
          - "Missing error bars" that are actually present in the
            referenced figure

        CRITICAL boundary rules:
          - Short but precise evidence is Sufficient. A 1-sentence
            comment with an exact citation beats a 5-paragraph
            hand-wave every time.
          - Length is not evidence. A long review item with no
            specific citation is Requires More.
          - Appeals to domain knowledge are valid evidence if the
            appeal is specific enough to verify. "Common practice in
            structural engineering requires reporting loading rate"
            is Sufficient (a meta-reviewer can check); "this isn't
            how people do it" is Requires More.
    """)
