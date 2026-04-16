"""
Prompt templates for the meta-review task, extended to match the rigor of
similarity_check/expert_annotation_similarity/prompts.py.

Two prompt families live in this file, answering two distinct questions:

    (A) AXIS mode            — "LLM AS meta-reviewer"
        The LLM plays a single expert meta-reviewer. It reads the paper
        and one review item, decides whether the reviewer's criticism is
        Correct, how Significant it is, and whether the Evidence is
        Sufficient. It emits three cascade-aware per-axis labels.

    (B) TENCLASS mode        — "LLM PREDICTS the pair of meta-reviewers"
        The LLM predicts the collapsed 10-class label that a *pair* of
        independent expert meta-reviewers would jointly produce,
        including the "disagree" variants. This is a harder task: on top
        of the axis judgment it requires the LLM to anticipate when two
        experts would weight the same evidence differently.

Both prompts share the same definitions of Correctness, Significance, and
Evidence Sufficiency; the 10-class prompt adds a section explaining when
expert meta-reviewers typically disagree and how to predict disagreement.

Both prompts wrap the final answer in answer tags so it can be extracted
with a single regex regardless of any chain-of-thought the model emits
beforehand. For providers that support strict JSON schemas (GPT-5, Gemini,
Claude, Grok), the schema is emitted alongside; for json_object and plain
providers, the parsing is regex-based off the answer tags.
"""

from __future__ import annotations

import re
from textwrap import dedent
from typing import Any, Dict, List, Optional


# ============================================================================
# SHARED DEFINITIONS — used by both AXIS and TENCLASS system prompts
# ============================================================================
#
# These definitions are the single source of truth for what "Correct",
# "Significant", and "Sufficient evidence" mean in this benchmark. Both
# system prompts embed the same block verbatim so the axis and 10class
# runs are evaluating against identical semantics.

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
            in fact present in the paper (check the paper first)
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


_OUTPUT_FORMAT_AXIS = dedent("""\
    -----------------------------------------------------------------
    OUTPUT FORMAT (STRICT) — AXIS mode
    -----------------------------------------------------------------

    Think through the decision procedure step by step in plain text.
    After your reasoning, write the final judgment wrapped in answer
    tags EXACTLY once, using this literal shape:

    <answer>
    {
      "reasoning": "<your step-by-step analysis>",
      "correctness": "Correct" | "Not Correct",
      "significance": "Significant" | "Marginally Significant" | "Not Significant" | null,
      "evidence": "Sufficient" | "Requires More" | null
    }
    </answer>

    Rules for the JSON inside the answer tag:
      - Produce "reasoning" as the FIRST key. Your reasoning must fit
        inside the string value.
      - If correctness is "Not Correct", set significance AND evidence
        to null.
      - If correctness is "Correct" and significance is "Not
        Significant", set evidence to null.
      - Otherwise all four fields must be filled in.
      - Use exact strings for each enum label — no variants, no
        capitalization changes, no trailing punctuation.
      - Do NOT nest the answer tag inside backticks, markdown fences,
        or quotes.
      - Do NOT write anything after the closing </answer> tag.
    """)


_OUTPUT_FORMAT_TENCLASS = dedent("""\
    -----------------------------------------------------------------
    OUTPUT FORMAT (STRICT) — 10CLASS mode
    -----------------------------------------------------------------

    Think through the decision procedure step by step in plain text.
    Then decide on ONE of the ten collapsed labels below and write the
    final judgment wrapped in answer tags EXACTLY once, using this
    literal shape:

    <answer>
    {
      "reasoning": "<your step-by-step analysis>",
      "label": "<one of the ten label strings below>"
    }
    </answer>

    The ten valid labels (use EXACT strings, character-for-character):

      "correct_significant_sufficient"
      "correct_significant_insufficient"
      "correct_significant_disagree_on_evidence"
      "correct_marginal_sufficient"
      "correct_marginal_insufficient"
      "correct_marginal_disagree_on_evidence"
      "correct_not_significant"
      "correct_disagree_on_significance"
      "incorrect"
      "disagree_on_correctness"

    Rules:
      - Produce "reasoning" as the FIRST key.
      - The "label" value must be one of the ten strings above, exact
        match, no quotes stripped, no extra punctuation.
      - Use the answer tag exactly once.
      - Do NOT nest the answer tag inside backticks, markdown fences,
        or quotes.
      - Do NOT write anything after the closing </answer> tag.
    """)


# ============================================================================
# AXIS MODE — "LLM as meta-reviewer"
# ============================================================================

AXIS_SYSTEM_PROMPT = dedent("""\
    You are an expert meta-reviewer for scientific papers. Your job is
    to judge the quality of an individual review item that another
    reviewer (human or AI) wrote about a paper. You will be given the
    full paper and one review item. Your output is a structured
    per-axis judgment of the review item along three dimensions —
    correctness, significance, and evidence sufficiency — using a
    cascade-aware decision procedure.

    You are NOT being asked to write a new review or to comment on the
    paper in general. Judge only the specific review item you are
    given. Treat the paper as ground truth when deciding whether the
    review item's factual claims are correct.
    """) + "\n" + _DIMENSION_DEFINITIONS + dedent("""\

    -----------------------------------------------------------------
    DECISION PROCEDURE — follow these steps in order
    -----------------------------------------------------------------

    Step 1 (Understand). Read the review item carefully. In one short
    phrase, what is the main point the reviewer is making? Write it
    out in your reasoning.

    Step 2 (Correctness). Verify the main point against the paper.
    Does the reviewer's claim hold up?
      - If YES → correctness = "Correct". Continue to Step 3.
      - If NO  → correctness = "Not Correct". Set significance and
        evidence to null. STOP.

    Step 3 (Significance). Given that the main point is correct, how
    much does this issue affect the paper's scientific contribution?
      - Would leaving it unaddressed materially change the paper's
        validity, reproducibility, or main conclusions?
        → "Significant"
      - Worth fixing but doesn't threaten the main result?
        → "Marginally Significant"
      - Very minor; would not affect any reasonable accept/reject
        decision → "Not Significant". Set evidence to null. STOP.

    Step 4 (Evidence). Given correctness and at-least-Marginal
    significance, does the reviewer provide enough justification that
    a meta-reviewer could independently verify the criticism without
    substantial extra work?
      - Yes → evidence = "Sufficient"
      - No  → evidence = "Requires More"

    -----------------------------------------------------------------
    IMPORTANT NOTES
    -----------------------------------------------------------------

    - The paper you are given is the source of truth. If the reviewer
      claims "the paper never mentions X" and the paper does mention
      X, the reviewer is Not Correct (unless they are pointing at a
      specific figure or table that genuinely omits X).
    - Your judgment must be independent of who wrote the review. Do
      not infer reviewer identity (human/AI) from style and do not
      let that inference influence your scoring.
    - The three axes are distinct and orthogonal. A review item can
      be Correct with Requires-More evidence; an item can be Not
      Correct regardless of how strong its evidence looks on the
      surface (strong-looking evidence for a wrong claim is still
      wrong).
    - When an axis is not applicable because of the cascade, use
      `null` — not an empty string, not "N/A", not a guessed label.
    """) + "\n" + _OUTPUT_FORMAT_AXIS


AXIS_USER_PROMPT_TEMPLATE = dedent("""\
    ### Paper
    Title: {paper_title}

    {paper_content}

    ---

    ### Review item (from a reviewer whose identity is hidden from you)
    Item number: {item_number}

    Main point of criticism:
    {review_content}
    {extras}
    ---

    Follow the decision procedure in the system prompt. Verify the
    review item's claims against the paper above. Think step by step.
    Then emit your final per-axis judgment inside <answer>...</answer>
    tags using the JSON format specified in the system prompt.
    """)


# ============================================================================
# TENCLASS MODE — "LLM predicts what the pair of meta-reviewers produce"
# ============================================================================

_TENCLASS_TAXONOMY = dedent("""\
    -----------------------------------------------------------------
    THE TEN COLLAPSED CLASSES
    -----------------------------------------------------------------

    In this benchmark, each review item is labeled by two independent
    expert meta-reviewers. Their per-axis judgments are collapsed into
    exactly one of ten classes that jointly encode BOTH the cascade
    outcome AND whether the two experts agreed on each step.

    You are asked to predict the collapsed class directly. That means
    you must reason about both (a) what the correct axis labels are
    and (b) which axes two independent experts would likely disagree
    on, then pick the class that captures both.

    The ten classes:

      1. "correct_significant_sufficient"
         Both experts: Correct + Significant + Sufficient.
         A solid, well-supported criticism of an important issue.

      2. "correct_significant_insufficient"
         Both experts: Correct + Significant + Requires More.
         An important criticism but with weak supporting evidence.

      3. "correct_significant_disagree_on_evidence"
         Both experts: Correct + Significant; they disagree on
         whether the evidence is Sufficient. Happens when evidence is
         borderline — e.g., the reviewer cites a relevant passage but
         doesn't finish the logical chain.

      4. "correct_marginal_sufficient"
         Both experts: Correct + Marginally Significant + Sufficient.
         A minor-but-valid point, well supported.

      5. "correct_marginal_insufficient"
         Both experts: Correct + Marginally Significant + Requires
         More. A minor-but-valid point with weak support.

      6. "correct_marginal_disagree_on_evidence"
         Both experts: Correct + Marginally Significant; disagree on
         evidence sufficiency.

      7. "correct_not_significant"
         Both experts: Correct + Not Significant.
         A minor issue that should probably be removed from the
         review.

      8. "correct_disagree_on_significance"
         Both experts: Correct; they disagree on whether the issue is
         Significant, Marginally Significant, or Not Significant.
         Happens when the issue is real but its importance is
         debatable.

      9. "incorrect"
         Both experts: Not Correct.
         The reviewer's main point does not hold up against the paper.

     10. "disagree_on_correctness"
         The two experts disagree on correctness itself.
         Happens when the review item is genuinely ambiguous — a
         charitable reading says the reviewer is right, an
         uncharitable reading says they are wrong.
    """)


_TENCLASS_DISAGREEMENT_GUIDANCE = dedent("""\
    -----------------------------------------------------------------
    WHEN DO TWO META-REVIEWERS DISAGREE?
    -----------------------------------------------------------------

    Class 3, 6, 8, and 10 all capture disagreement between the two
    experts on a specific axis. To predict disagreement correctly,
    you need to reason about your OWN uncertainty on each axis as a
    proxy for inter-expert variance.

    Predict DISAGREEMENT on correctness (class 10) when:
      - The reviewer's claim is factually borderline — a charitable
        reader would call it correct, an uncharitable reader would
        call it wrong. E.g., the reviewer criticizes an approximation
        the paper uses; whether this is "incorrect" depends on what
        you consider the paper's scope to be.
      - The reviewer's main point is partially right and partially
        wrong and it's unclear which part is the main point.
      - The reviewer's factual premise depends on a judgment about
        domain norms that experts in the field plausibly disagree on.

    Predict DISAGREEMENT on significance (class 8) when:
      - The issue is real but whether it's "Significant" or just
        "Marginally Significant" is a judgment call — e.g., missing
        an ablation that might matter, depending on how load-bearing
        the ablated component actually is.
      - The reviewer flags something that is between "Marginally
        Significant" and "Not Significant" — e.g., a minor
        presentational issue that some experts would downgrade to
        stylistic preference.
      - The issue is important for reviewer X's specialty but not
        for reviewer Y's specialty.

    Predict DISAGREEMENT on evidence (classes 3, 6) when:
      - The reviewer cites a relevant passage but doesn't close the
        logical chain. Strict experts will call this Requires More;
        charitable experts will call it Sufficient.
      - The reviewer appeals to domain standards without naming a
        specific reference. Whether this counts as sufficient depends
        on how the meta-reviewer interprets "specific enough".
      - The reviewer's evidence is short but precise — some experts
        will find 1 sentence enough, others will want more.

    Predict AGREEMENT on a given axis when:
      - The reviewer's claim is either obviously correct or obviously
        wrong after reading the paper.
      - The significance call is unambiguous (e.g., a clear typo is
        obviously Not Significant; a factual error in the main
        result is obviously Significant).
      - The evidence is either clearly specific and thorough, or
        clearly hand-wavy with no concrete pointers.

    When in doubt about a borderline case, default to AGREEMENT —
    agreement is the more common ground-truth outcome in this
    benchmark.
    """)


TENCLASS_SYSTEM_PROMPT = dedent("""\
    You are predicting how a PAIR of independent expert meta-reviewers
    would jointly judge a specific review item. Each of the two
    experts reads the same paper and the same review item, and then
    independently produces per-axis judgments (correctness,
    significance, evidence). Their judgments are then collapsed into
    exactly one of ten classes that encodes BOTH the cascade outcome
    AND whether the two experts agreed on each axis.

    Your task is to predict that collapsed class directly. Note the
    subtlety: you are NOT being asked to give your own meta-review
    judgment — you are being asked to predict what TWO independent
    experts would jointly produce. This requires reasoning about your
    own uncertainty on each axis as a proxy for inter-expert variance.

    The definitions of Correctness, Significance, and Evidence
    Sufficiency below are the same as the axis-mode benchmark. Use
    them to reason about (a) what you believe the true per-axis
    labels are, and (b) which axes are close to the boundary and
    would likely produce expert disagreement.
    """) + "\n" + _DIMENSION_DEFINITIONS + "\n" + _TENCLASS_TAXONOMY + "\n" + _TENCLASS_DISAGREEMENT_GUIDANCE + dedent("""\

    -----------------------------------------------------------------
    DECISION PROCEDURE — follow these steps in order
    -----------------------------------------------------------------

    Step 1 (Understand). Read the review item. Summarize its main
    point in one short phrase in your reasoning.

    Step 2 (Your axis labels). Reason through the correctness,
    significance, and evidence cascade in the usual way: verify
    correctness against the paper; if Correct, decide significance;
    if at-least-Marginal, decide evidence. Call this your "best
    guess" per-axis labels.

    Step 3 (Agreement on correctness). Would two independent experts
    agree with your correctness judgment? If the call is close to
    the boundary (partially-right claims, depends on domain
    interpretation, depends on scope) → predict class 10
    (disagree_on_correctness) and stop.

    Step 4 (Agreement on significance). Only reached if Step 3 was
    "agree on Correct". Would two experts agree on your significance
    label? If the call is close to the boundary between Significant /
    Marginal / Not Significant → predict class 8
    (correct_disagree_on_significance) and stop.

    Step 5 (Not Significant branch). Only reached if experts agree
    and significance is Not Significant. Predict class 7
    (correct_not_significant) — no evidence axis to judge, no
    disagreement axis to predict.

    Step 6 (Agreement on evidence). Only reached if experts agree and
    significance is Significant or Marginal. Would two experts agree
    on your evidence label? If the call is borderline → predict
    class 3 (if Significant) or class 6 (if Marginal). Otherwise
    predict class 1, 2, 4, or 5 based on (significance, evidence)
    combination.

    Step 7 (Not Correct branch). If Step 2 gave correctness = Not
    Correct and Step 3 gave agreement = agree → predict class 9
    (incorrect).

    DISAGREEMENT CLASSES STOP THE CASCADE. Once you predict that
    experts would disagree on axis X, downstream axes don't matter
    for the label — you emit the class-X-disagree label and stop.
    """) + "\n" + _OUTPUT_FORMAT_TENCLASS


TENCLASS_USER_PROMPT_TEMPLATE = dedent("""\
    ### Paper
    Title: {paper_title}

    {paper_content}

    ---

    ### Review item (from a reviewer whose identity is hidden from you)
    Item number: {item_number}

    Main point of criticism:
    {review_content}
    {extras}
    ---

    Follow the decision procedure in the system prompt. First decide
    on your own per-axis labels, then reason about which axes two
    independent experts would likely disagree on, then pick the
    collapsed 10-class label. Think step by step. Emit your final
    answer inside <answer>...</answer> tags using the JSON format
    specified in the system prompt.
    """)


# ============================================================================
# Valid label sets (for post-hoc validation and regex matching)
# ============================================================================

AXIS_CORRECTNESS_LABELS = ("Correct", "Not Correct")
AXIS_SIGNIFICANCE_LABELS = ("Significant", "Marginally Significant", "Not Significant")
AXIS_EVIDENCE_LABELS = ("Sufficient", "Requires More")

TENCLASS_LABELS = (
    "correct_significant_sufficient",
    "correct_significant_insufficient",
    "correct_significant_disagree_on_evidence",
    "correct_marginal_sufficient",
    "correct_marginal_insufficient",
    "correct_marginal_disagree_on_evidence",
    "correct_not_significant",
    "correct_disagree_on_significance",
    "incorrect",
    "disagree_on_correctness",
)

TENCLASS_LABEL_TO_ID = {
    "correct_significant_sufficient":           1,
    "correct_significant_insufficient":         2,
    "correct_significant_disagree_on_evidence": 3,
    "correct_marginal_sufficient":              4,
    "correct_marginal_insufficient":            5,
    "correct_marginal_disagree_on_evidence":    6,
    "correct_not_significant":                  7,
    "correct_disagree_on_significance":         8,
    "incorrect":                                9,
    "disagree_on_correctness":                 10,
}


# ============================================================================
# JSON Schemas for strict-mode providers
# ============================================================================

def _enum_or_null(enum_vals):
    """Azure strict mode requires anyOf instead of mixing null into enum."""
    return {
        "anyOf": [
            {"type": "string", "enum": list(enum_vals)},
            {"type": "null"},
        ],
    }


AXIS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step analysis. Must appear first.",
        },
        "correctness": {
            "type": "string",
            "enum": list(AXIS_CORRECTNESS_LABELS),
        },
        "significance": _enum_or_null(AXIS_SIGNIFICANCE_LABELS),
        "evidence": _enum_or_null(AXIS_EVIDENCE_LABELS),
    },
    "required": ["reasoning", "correctness", "significance", "evidence"],
    "additionalProperties": False,
}


TENCLASS_JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Step-by-step analysis. Must appear first.",
        },
        "label": {
            "type": "string",
            "enum": list(TENCLASS_LABELS),
        },
    },
    "required": ["reasoning", "label"],
    "additionalProperties": False,
}


# ============================================================================
# User-text + message builders
# ============================================================================

def _format_review_extras(
    review_claim: Optional[str],
    review_evidence: Optional[str],
    review_cited_references: Optional[List[str]],
) -> str:
    """Append optional structured sections to the review text. If a section
    is empty or None, omit it silently."""
    parts: List[str] = []
    if review_claim:
        parts.append(f"\n[Full Claim section from the review]\n{review_claim}")
    if review_evidence:
        parts.append(f"\n[Full Evidence section from the review]\n{review_evidence}")
    if review_cited_references:
        refs_text = "\n".join(str(r) for r in review_cited_references)
        parts.append(f"\n[References cited in the review]\n{refs_text}")
    return "\n".join(parts)


def _format_artifacts(artifacts: List[Dict[str, Any]]) -> str:
    """Inline code/text artifacts after the main user text."""
    parts = [
        "\n\n---\n\n### Referenced files (mentioned by name in the review or pulled from the paper's supplementary bundle)\n"
    ]
    for a in artifacts:
        parts.append(f'\n#### `{a["path"]}`\n```\n{a["content"]}\n```\n')
    return "".join(parts)


def build_axis_user_text(
    *,
    paper_title: str,
    paper_content: str,
    review_content: str,
    item_number: Any,
    review_claim: Optional[str] = None,
    review_evidence: Optional[str] = None,
    review_cited_references: Optional[List[str]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user-text half of an AXIS-mode prompt."""
    extras = _format_review_extras(review_claim, review_evidence, review_cited_references)
    base = AXIS_USER_PROMPT_TEMPLATE.format(
        paper_title=paper_title or "",
        paper_content=paper_content or "",
        item_number=item_number,
        review_content=review_content or "",
        extras=extras,
    )
    if artifacts:
        base += _format_artifacts(artifacts)
    return base


def build_tenclass_user_text(
    *,
    paper_title: str,
    paper_content: str,
    review_content: str,
    item_number: Any,
    review_claim: Optional[str] = None,
    review_evidence: Optional[str] = None,
    review_cited_references: Optional[List[str]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build the user-text half of a TENCLASS-mode prompt."""
    extras = _format_review_extras(review_claim, review_evidence, review_cited_references)
    base = TENCLASS_USER_PROMPT_TEMPLATE.format(
        paper_title=paper_title or "",
        paper_content=paper_content or "",
        item_number=item_number,
        review_content=review_content or "",
        extras=extras,
    )
    if artifacts:
        base += _format_artifacts(artifacts)
    return base


def build_messages(
    *,
    mode: str,                         # "axis" | "tenclass"
    paper_title: str,
    paper_content: str,
    review_content: str,
    item_number: Any,
    review_claim: Optional[str] = None,
    review_evidence: Optional[str] = None,
    review_cited_references: Optional[List[str]] = None,
    images: Optional[List[Dict[str, Any]]] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Build an OpenAI-style messages list for either prompt mode.

    `images` is an optional list of {bytes, mime, caption, img_path}
    dicts. If provided and non-empty, the user message becomes a
    multimodal content list with image blocks interleaved after the
    text block.

    `artifacts` is an optional list of {path, content} dicts for
    code/text files referenced by the review. They are inlined into
    the user text as fenced code blocks.
    """
    if mode == "axis":
        system_prompt = AXIS_SYSTEM_PROMPT
        user_text = build_axis_user_text(
            paper_title=paper_title,
            paper_content=paper_content,
            review_content=review_content,
            item_number=item_number,
            review_claim=review_claim,
            review_evidence=review_evidence,
            review_cited_references=review_cited_references,
            artifacts=artifacts,
        )
    elif mode == "tenclass":
        system_prompt = TENCLASS_SYSTEM_PROMPT
        user_text = build_tenclass_user_text(
            paper_title=paper_title,
            paper_content=paper_content,
            review_content=review_content,
            item_number=item_number,
            review_claim=review_claim,
            review_evidence=review_evidence,
            review_cited_references=review_cited_references,
            artifacts=artifacts,
        )
    else:
        raise ValueError(f"Unknown prompt mode: {mode!r} (expected 'axis' or 'tenclass')")

    if images:
        import base64
        content: List[Dict[str, Any]] = [{"type": "text", "text": user_text}]
        for img in images:
            img_bytes = img.get("bytes")
            if not img_bytes:
                continue
            mime = img.get("mime") or "image/jpeg"
            b64 = base64.b64encode(img_bytes).decode("ascii")
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:{mime};base64,{b64}"},
            })
            caption = img.get("caption", "")
            if caption:
                content.append({
                    "type": "text",
                    "text": f"(Caption for {img.get('img_path', 'image')}): {caption}",
                })
        user_msg: Dict[str, Any] = {"role": "user", "content": content}
    else:
        user_msg = {"role": "user", "content": user_text}

    return [
        {"role": "system", "content": system_prompt},
        user_msg,
    ]


# ============================================================================
# Answer extraction — multi-strategy JSON parser
# ============================================================================

_ANSWER_TAG_RE = re.compile(
    r"<answer>\s*(.*?)\s*</answer>",
    re.IGNORECASE | re.DOTALL,
)

_JSON_BLOCK_RE = re.compile(r"\{[\s\S]*\}")
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _try_json_loads(text: str) -> Optional[Dict[str, Any]]:
    """Attempt to parse `text` as JSON. Tolerates trailing commas."""
    import json
    text = text.strip()
    if not text:
        return None
    # Strip trailing commas before } and ]
    cleaned = re.sub(r",(\s*[}\]])", r"\1", text)
    try:
        obj = json.loads(cleaned)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Multi-strategy JSON extraction from arbitrary LLM text.

    Tries, in order:
      1. Content of the last <answer>...</answer> tag (preferred)
      2. Direct load of the whole text (for strict-mode responses)
      3. Content of any ```json ... ``` fence
      4. Content of the largest { ... } block (greedy match)
    """
    if not text:
        return None

    # Strategy 1: last answer tag (most reliable)
    matches = list(_ANSWER_TAG_RE.finditer(text))
    if matches:
        for m in reversed(matches):
            inner = m.group(1).strip()
            obj = _try_json_loads(inner)
            if obj:
                return obj
            fence = _CODE_FENCE_RE.search(inner)
            if fence:
                obj = _try_json_loads(fence.group(1))
                if obj:
                    return obj
            brace = _JSON_BLOCK_RE.search(inner)
            if brace:
                obj = _try_json_loads(brace.group(0))
                if obj:
                    return obj

    # Strategy 2: whole text (for json_schema strict-mode responses)
    obj = _try_json_loads(text)
    if obj:
        return obj

    # Strategy 3: code fence anywhere
    fence = _CODE_FENCE_RE.search(text)
    if fence:
        obj = _try_json_loads(fence.group(1))
        if obj:
            return obj

    # Strategy 4: greedy brace block
    brace = _JSON_BLOCK_RE.search(text)
    if brace:
        obj = _try_json_loads(brace.group(0))
        if obj:
            return obj

    return None


def _normalize_axis_string(val: Any, allowed: tuple) -> Optional[str]:
    """Coerce a string to an allowed axis label, tolerating case/whitespace.
    Returns None if the value isn't a valid label."""
    if val is None:
        return None
    if not isinstance(val, str):
        return None
    v = val.strip().strip('"').strip("'").strip(".")
    for canonical in allowed:
        if v == canonical:
            return canonical
    v_norm = " ".join(v.lower().split())
    for canonical in allowed:
        if v_norm == canonical.lower():
            return canonical
    return None


def extract_axis_answer(text: str) -> Optional[Dict[str, Any]]:
    """Parse an axis-mode prediction out of arbitrary LLM text.

    Returns a dict {reasoning, correctness, significance, evidence}
    with canonical label strings, or None if the response can't be
    parsed. Cascade constraints are NOT enforced here — that's the
    caller's job.
    """
    obj = _extract_json_from_text(text)
    if not obj:
        return None

    corr = _normalize_axis_string(obj.get("correctness"), AXIS_CORRECTNESS_LABELS)
    if corr is None:
        return None
    sig_raw = obj.get("significance")
    evi_raw = obj.get("evidence")

    # Explicit null / None is OK; any other non-matching string is an error.
    sig = None
    if sig_raw is not None:
        sig = _normalize_axis_string(sig_raw, AXIS_SIGNIFICANCE_LABELS)
        if sig is None and str(sig_raw).lower() not in ("null", "none", ""):
            return None
    evi = None
    if evi_raw is not None:
        evi = _normalize_axis_string(evi_raw, AXIS_EVIDENCE_LABELS)
        if evi is None and str(evi_raw).lower() not in ("null", "none", ""):
            return None

    return {
        "reasoning": str(obj.get("reasoning", "") or ""),
        "correctness": corr,
        "significance": sig,
        "evidence": evi,
    }


def extract_tenclass_answer(text: str) -> Optional[Dict[str, Any]]:
    """Parse a 10-class prediction out of arbitrary LLM text.

    Returns a dict {reasoning, label, label_id} or None if the
    response can't be parsed.
    """
    obj = _extract_json_from_text(text)
    if not obj:
        # Fallback: look for any of the 10 label strings as a substring
        for lab in TENCLASS_LABELS:
            if lab in text:
                return {
                    "reasoning": "",
                    "label": lab,
                    "label_id": TENCLASS_LABEL_TO_ID[lab],
                }
        return None

    lab_raw = obj.get("label") or obj.get("class") or obj.get("class_name")
    if not isinstance(lab_raw, str):
        lab_id = obj.get("label_id")
        if isinstance(lab_id, int) and 1 <= lab_id <= 10:
            lab = TENCLASS_LABELS[lab_id - 1]
            return {
                "reasoning": str(obj.get("reasoning", "") or ""),
                "label": lab,
                "label_id": lab_id,
            }
        return None

    v = lab_raw.strip().strip('"').strip("'").strip(".")
    for canonical in TENCLASS_LABELS:
        if v == canonical:
            return {
                "reasoning": str(obj.get("reasoning", "") or ""),
                "label": canonical,
                "label_id": TENCLASS_LABEL_TO_ID[canonical],
            }
    v_norm = v.lower().replace(" ", "_")
    for canonical in TENCLASS_LABELS:
        if v_norm == canonical:
            return {
                "reasoning": str(obj.get("reasoning", "") or ""),
                "label": canonical,
                "label_id": TENCLASS_LABEL_TO_ID[canonical],
            }
    for canonical in TENCLASS_LABELS:
        if canonical in v_norm:
            return {
                "reasoning": str(obj.get("reasoning", "") or ""),
                "label": canonical,
                "label_id": TENCLASS_LABEL_TO_ID[canonical],
            }
    return None


# ============================================================================
# Utility: collapse axis prediction to 10-class label (axis → tenclass bridge)
# ============================================================================

def axis_to_tenclass_label(axis_pred: Dict[str, Any]) -> str:
    """Convert a parsed axis-mode prediction to one of the 10 class labels.

    Since axis mode does NOT predict disagreement, the result is
    ALWAYS one of the 6 "agree" classes {1, 2, 4, 5, 7, 9} — never
    3/6/8/10.
    """
    corr = axis_pred.get("correctness")
    if corr == "Not Correct":
        return "incorrect"
    sig = axis_pred.get("significance")
    if sig == "Not Significant":
        return "correct_not_significant"
    sig_is_significant = (sig == "Significant")
    evi = axis_pred.get("evidence")
    if sig_is_significant:
        if evi == "Sufficient":
            return "correct_significant_sufficient"
        if evi == "Requires More":
            return "correct_significant_insufficient"
    else:
        if evi == "Sufficient":
            return "correct_marginal_sufficient"
        if evi == "Requires More":
            return "correct_marginal_insufficient"
    # Fallback — shouldn't happen if the prediction is well-formed
    return "correct_significant_sufficient"
