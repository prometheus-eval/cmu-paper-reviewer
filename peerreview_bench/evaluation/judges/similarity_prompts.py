"""
Prompt templates for LLM-as-a-judge similarity baselines.

Two prompt families are provided:

1. **Binary**: original Yes/No prompt taken from the existing
   measure_similarity_llm.py. Predicts whether two review items are
   targeting the same underlying weakness. Maps to our
   `binary_label` (similar / not_similar) directly.

2. **4-way**: a more granular prompt that asks the LLM to classify the
   pair into one of four categories matching the dataset's
   `finegrained_label` column:

     "same subject, same argument, same evidence"     (near-paraphrase)
     "same subject, same argument, different evidence" (convergent)
     "same subject, different argument"                 (topical neighbor)
     "different subject"                                 (unrelated)

   The 4-way label can be derived back to a binary one with the rule
   {same-arg-same-evid, same-arg-diff-evid} → similar, the other two →
   not_similar.

Both prompts wrap the final answer in <answer>...</answer> tags so it
can be extracted with a single regex regardless of any chain-of-thought
the model emits beforehand.
"""

# ---------------------------------------------------------------------------
# BINARY (Yes/No) prompt
# ---------------------------------------------------------------------------

BINARY_SYSTEM_PROMPT = """You will be given a research paper and two criticism items written by different reviewers of that paper.

Your task is to determine whether the two items are targeting the same underlying weakness or issue in the paper.

Definition of "same underlying weakness":
- Both items identify the same core flaw, gap, or concern — even if they use different wording, cite different evidence, or differ in specificity.
- Both items challenge the same claim or conclusion from different angles (e.g., one cites code, the other cites text).

Definition of NOT the same underlying weakness:
- The items reference different parts of the paper entirely.
- The items mention the same paragraph or figure but raise distinct, specific concerns about it (e.g., one says "Figure 2 has no error bars" and the other says "Figure 2 mislabels the y-axis").

Mere topical overlap is NOT sufficient — the underlying problem must match.

Think step by step, then wrap your final verdict in answer tags as <answer>Yes</answer> or <answer>No</answer>."""


BINARY_USER_PROMPT_TEMPLATE = (
    "### Paper\n{paper_text}\n\n"
    "---\n\n"
    "### Criticism A (from reviewer {reviewer_a})\n{item_a}\n\n"
    "### Criticism B (from reviewer {reviewer_b})\n{item_b}\n\n"
    "---\n\n"
    "After reading the paper above, are these two criticism items targeting the "
    "same underlying weakness or issue in the paper? Think step by step, then "
    "provide your final answer at the very end as <answer>Yes</answer> or "
    "<answer>No</answer>."
)


# ---------------------------------------------------------------------------
# 4-WAY (a/b/c/d) prompt
# ---------------------------------------------------------------------------

FOURWAY_SYSTEM_PROMPT = """You are classifying the relationship between two peer-review items (Item A and Item B) written by different reviewers of the same scientific paper. You must assign exactly one of four labels. The labels are defined by three orthogonal questions — SUBJECT, ARGUMENT, and EVIDENCE — applied in that order.

-----------------------------------------------------------------
DIMENSION DEFINITIONS (apply these to each item independently first)
-----------------------------------------------------------------

(1) SUBJECT — "what part of the paper is this comment about?"
The subject is the element of the paper that is the target of the comment. Subjects can be at any granularity — a single equation, an entire figure, a whole method, or a broad experimental protocol. Valid subjects include:
  - A specific numbered figure or table (e.g. "Figure 2", "Table 3b")
  - A specific section, subsection, or paragraph (e.g. "Methods section 2.3", "the ablation paragraph in Section 4")
  - A specific claim, metric, equation, dataset, or experiment (e.g. "the p < 0.01 claim", "the binary accuracy on MNLI", "Equation 4")
  - A specific code file, function, or class (e.g. "train.py", "the BatchNorm layer in model.py")
  - A broader aspect of the paper that the comment is clearly focused on (e.g. "the hyperparameter tuning protocol", "the choice of loss function", "the overall statistical analysis")
Two items share a subject if they are BOTH pointing at the same part of the paper — even if one is more specific than the other, and even if they are raising completely different complaints about that part. For example, if Item A says "Figure 2 has no error bars" and Item B says "Figure 2 is illegible at print size", both share the subject "Figure 2" and you should treat them as having the SAME subject even though they make completely different arguments about it. Two items about the "same method" also share a subject, even if one focuses on its loss function and the other focuses on its optimizer — as long as both are directed at that method.

(2) ARGUMENT — "what type of flaw is being asserted about the subject?"
An argument is the abstract type of flaw the reviewer is asserting about the subject, stripped of the reviewer's specific reasons for asserting it. To identify an item's argument, perform this reduction:
  1. Start with the reviewer's full complaint.
  2. Strip out all specific reasons (everything after "because...", "since...", "due to...", or "as shown by...").
  3. Strip out specific citations, quoted passages, numbers, line references, and illustrative examples.
  4. What remains is a single short claim of the form "X is <FLAW_TYPE>", where X is the subject and <FLAW_TYPE> is a generic category of flaw such as: wrong, missing, inadequate, unjustified, misleading, unreliable, insufficient, unsupported, inconsistent, incomplete, overstated, non-reproducible, not-generalizable, or similar.

CRITICAL: The specific reason the reviewer gives for WHY the subject is flawed is NOT part of the argument. It is part of the EVIDENCE (the reasoning chain), which is evaluated separately in Step 3. Do NOT include the "because of Y" in the argument sentence. Two reviewers can give completely different reasons while making the same argument.

Two items make the SAME argument if both assert the same <FLAW_TYPE> about the same subject — even if the reasons they provide are entirely different. For example, all three of these items share the SAME argument ("the sleep/wake classification is inadequate"):
  - "The sleep/wake classification is inadequate because it uses patient-reported events without objective verification"
  - "The sleep/wake classification is inadequate because it is defined circularly from the LFP signal being measured"
  - "The sleep/wake classification is inadequate because it relies on clinician-chosen sensing frequencies that introduce selection bias"
The specific reasons (patient events, circular definition, clinician bias) are different reasoning chains supporting the same argument. They determine EVIDENCE, not argument.

Two items make DIFFERENT arguments only when the <FLAW_TYPE>s themselves are categorically distinct. Examples of categorically different arguments about the same figure:
  - "Figure 2 is illegible" (FLAW_TYPE = presentation / legibility)
  - "Figure 2's data contradicts the text" (FLAW_TYPE = internal inconsistency / correctness)
  - "Figure 2 is missing error bars" (FLAW_TYPE = rigor / incomplete reporting)
  - "Figure 2 uses wrong units" (FLAW_TYPE = correctness)
These are categorically distinct types of flaws, not just different reasons for the same flaw.

To distinguish "different reasons for same flaw" from "different flaw types", ask: if you strip each item down to just "X is <FLAW_TYPE>", do they collapse to the same claim? If yes → SAME argument. If the remaining claims are asserting categorically different problems (e.g. one says "figure is illegible" and the other says "figure's data is wrong") → DIFFERENT argument.

(3) EVIDENCE — "what reasoning chain does the item use to justify its argument?"
Evidence is NOT restricted to citations of concrete material from the paper. It is the entire line of reasoning the reviewer uses to support their argument. A reviewer may build evidence by any combination of:
  - Pointing at specific material in the paper (a quoted passage, an exact number, a figure panel, a table entry, a line in the code, an equation reference)
  - Pointing at what is ABSENT from the paper (missing control, missing baseline, missing error bar, missing ablation, missing reference)
  - Appealing to external knowledge (domain standards, prior work, common practice in the field)
  - Building a logical derivation from scratch (e.g., "if X holds then Y must be true; the paper asserts X; therefore Y should be shown")
  - Observing an internal inconsistency (two parts of the paper contradict each other, or the method contradicts the stated claim)
Two items use the SAME evidence if their reasoning chains CONVERGE ON THE SAME CORE SET of observations, absent features, appeals, or logical steps — even if the two items phrase them differently, emphasize different aspects, or add non-essential peripheral details. One item may be longer, better organized, or include additional illustrative elaboration, while still sharing the same core reasoning chain with the other. The test is: if you strip away each reviewer's specific wording, illustrative examples, and non-essential peripherals, do the two chains reduce to the same underlying set of observations and appeals?

Two items use DIFFERENT evidence only if they reach the same argument through substantively NON-OVERLAPPING reasoning chains. The test is: can you identify a core observation, absent-feature call, external appeal, or logical derivation that one item makes but is genuinely absent from the other? If yes → different evidence. If the core observations overlap and the differences are only in peripheral wording, non-essential illustrative examples, or different emphasis on shared observations → SAME evidence.

Crucially: the fact that one item is more elaborate than the other does NOT make the evidence different. A longer item that walks through the same core observations plus some additional peripheral details still shares the SAME evidence with a shorter item that covers only the core observations. Additional peripheral details only count as "different evidence" if they introduce a genuinely new line of reasoning (a new core observation, a new absent feature, a new domain-norm appeal, or a new logical derivation), not just more illustration of shared core reasoning.

-----------------------------------------------------------------
THE FOUR CATEGORIES
-----------------------------------------------------------------

(1) "same subject, same argument, same evidence"  —  NEAR-PARAPHRASE
    Both items reference the same subject, make the same argument (same FLAW_TYPE), AND their reasoning chains converge on the same core set of observations / absent features / domain appeals / logical derivations. One item may be more elaborate than the other, but the additional material is peripheral or illustrative, not a new line of reasoning. A reader stripping away non-essential wording from both items would say "these cover the same core observations."

(2) "same subject, same argument, different evidence"  —  CONVERGENT CONCLUSION
    Both items reference the same subject and make the same argument (same FLAW_TYPE), but they justify it through substantively NON-OVERLAPPING reasoning chains. One item's chain contains at least one core observation, absent-feature call, or domain appeal that is genuinely absent from the other. For example, one reviewer might reach the complaint through concrete data-driven validation statistics while another reaches the same complaint through an appeal to domain standards and a logical derivation from first principles — those are different reasoning chains even though the conclusion is identical.

(3) "same subject, different argument"  —  TOPICAL NEIGHBOR
    Both items reference the same subject, but their reduced argument sentences assert different core issues. They notice the same part of the paper but worry about different aspects of it. This is common when reviewers both flag "something is wrong with Figure X" for completely different reasons.

(4) "different subject"  —  UNRELATED
    The two items do not share a subject. They point at different parts of the paper.

-----------------------------------------------------------------
DECISION PROCEDURE — follow these steps in order, in your reasoning
-----------------------------------------------------------------

Step 1 (Subject). Write out Item A's subject in one phrase. Write out Item B's subject in one phrase. Ask: are both items pointing at the same part of the paper — whether the same figure, the same section, the same method, or the same broader aspect? Note: they share a subject even if one is more specific than the other, and even if their complaints about it are completely different.
  - If the two items are pointing at genuinely different parts of the paper (Item A is about the introduction, Item B is about a specific table; or Item A is about the loss function, Item B is about the dataset preprocessing): answer is "different subject". Stop.
  - Otherwise (both pointing at the same figure / section / method / concept, regardless of what they claim about it): continue to Step 2.

Step 2 (Argument). Reduce Item A to a single "X is <FLAW_TYPE>" claim by stripping out its specific reasons, citations, examples, and illustrative details. Do the same for Item B. Ask: do the two reduced claims assert the same FLAW_TYPE about the same subject?
  - If YES → continue to Step 3. (Note: the claims are "the same" if both reduce to the same generic complaint about the subject, even when the two items cite completely different specific reasons for why the flaw exists. Different reasons for the same flaw are EVIDENCE, not different arguments.)
  - If NO (the two reduced claims assert categorically different types of flaws about the same subject — e.g., one says "figure is illegible" and the other says "figure's data is wrong") → the answer is "same subject, different argument". Stop.

Step 3 (Evidence). Trace the reasoning chain each item uses to justify its argument. This chain may include pointers to concrete material in the paper, absent-feature observations, appeals to domain norms, from-scratch logical derivations, or internal-inconsistency observations. Ask: do the two items converge on the same core set of observations and appeals, even if they phrase them differently, emphasize different aspects, or add non-essential peripherals?
  - If YES (the core observations overlap, and any additional details in one item or the other are peripheral/illustrative rather than introducing a new line of reasoning) → the answer is "same subject, same argument, same evidence".
  - If NO (one item uses a core observation, absent-feature call, domain appeal, or logical derivation that is genuinely absent from the other) → the answer is "same subject, same argument, different evidence".

-----------------------------------------------------------------
IMPORTANT BOUNDARY RULES
-----------------------------------------------------------------

- Two items pointing at the same figure, table, method, or section share a SUBJECT — even if they raise completely unrelated complaints about it. Do not call two items "different subject" just because their complaints diverge; that's what "same subject, different argument" is for.
- Surface-level topical overlap IS enough for "same subject" (both about Figure 2 = same subject). Surface-level topical overlap is NOT enough for "same argument" — two reviewers can agree on which figure is interesting for opposite reasons, and that's "same subject, different argument".
- Rewording, different tone, or different reviewer length does NOT make two items have different arguments. The argument is the abstract complaint, not the writing style.
- Pointing at the same subject is a NECESSARY but NOT SUFFICIENT condition for "same argument". You must verify the reduced one-sentence criticism matches.
- The distinction between "same argument, different evidence" and "same argument, same evidence" is specifically whether the core REASONING CHAINS overlap — not whether the conclusions overlap, and not whether both items are literally word-for-word identical. Conclusions overlap by definition if both items make the same argument. Two items share evidence when their core observations overlap; they have different evidence only when one introduces a genuinely new observation, absent feature, domain appeal, or logical derivation that is absent from the other. Differences in wording, elaboration, emphasis, or illustrative examples alone do NOT make evidence different.
- When reducing an item to its argument, strip the specific reason (the "because of Y" clause) out of the argument and evaluate it as evidence instead. Two reviewers saying "the method is inadequate because X" and "the method is inadequate because Y" share the SAME argument (the method is inadequate) and differ in EVIDENCE (X vs Y). They are NOT "different arguments".
- Focus on the underlying complaint, not the reviewer's writing style.

-----------------------------------------------------------------
OUTPUT FORMAT (STRICT)
-----------------------------------------------------------------

Think through the decision procedure step by step in plain text. After your reasoning, write the final classification on its own line as EXACTLY one of these four labels, wrapped in answer tags:

<answer>same subject, same argument, same evidence</answer>
<answer>same subject, same argument, different evidence</answer>
<answer>same subject, different argument</answer>
<answer>different subject</answer>

Rules for the output:
- Use the answer tag exactly once in your entire response.
- Put the answer tag at the very end; do not write anything after it.
- The text inside the tag must match one of the four labels character-for-character (lowercase, commas and spaces as shown).
- Do not nest the answer tag inside any other formatting (no quotes, no backticks, no markdown).
"""


FOURWAY_USER_PROMPT_TEMPLATE = (
    "### Paper\n{paper_text}\n\n"
    "---\n\n"
    "### Item A (from reviewer {reviewer_a})\n{item_a}\n\n"
    "### Item B (from reviewer {reviewer_b})\n{item_b}\n\n"
    "---\n\n"
    "Using the four-category taxonomy from the system prompt, classify "
    "the relationship between Item A and Item B. Apply the decision "
    "procedure rigorously: subject first, then argument, then evidence.\n\n"
    "Provide your final answer at the very end wrapped in answer tags using "
    "exactly one of the four full label strings from the system prompt."
)


# ---------------------------------------------------------------------------
# Mapping helpers
# ---------------------------------------------------------------------------

def fourway_to_binary(label: str) -> str:
    """Map a fine-grained label to the binary label used in the eval set.

    Accepts both the descriptive long-form labels used in the HF schema
    and the short audit codes (a/b/c/d) used in the construction
    artifacts. Both map to the same binary outcome.
    """
    similar_labels = {
        'same subject, same argument, same evidence',
        'same subject, same argument, different evidence',
        'b', 'c',
    }
    not_similar_labels = {
        'same subject, different argument',
        'different subject',
        'a', 'd',
    }
    if label in similar_labels:
        return 'similar'
    if label in not_similar_labels:
        return 'not_similar'
    raise ValueError(f'Unknown fine-grained label: {label}')


def binary_to_yes_no(label: str) -> str:
    if label == 'similar':
        return 'Yes'
    if label == 'not_similar':
        return 'No'
    raise ValueError(f'Unknown binary label: {label}')
