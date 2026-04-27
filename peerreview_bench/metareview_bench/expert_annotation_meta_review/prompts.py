"""
Shared definitions and answer-extraction utilities for the meta-review task.

Two prompt modes are supported:
    (A) AXIS mode     — per-axis cascade labels (correctness, significance, evidence)
    (B) TENCLASS mode — one of 10 collapsed class labels

This file provides:
    - Dimension definitions (_DIMENSION_DEFINITIONS) used by agent prompts
    - 10-class taxonomy and disagreement guidance (_TENCLASS_TAXONOMY, etc.)
    - Label constants (TENCLASS_LABELS, TENCLASS_LABEL_TO_ID, axis label tuples)
    - Multi-strategy answer extraction (extract_axis_answer, extract_tenclass_answer)
    - axis_to_tenclass_label converter (used by metrics.py)

The actual agent prompts live in run_meta_review_agent.py.
"""

from __future__ import annotations

import re
from textwrap import dedent
from typing import Any, Dict, Optional


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

    Use your genuine uncertainty to guide the prediction. If you
    found the axis judgment difficult or could see it going either
    way, that is evidence that experts would disagree. Do not
    suppress disagreement predictions — they are common in this
    benchmark.
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
