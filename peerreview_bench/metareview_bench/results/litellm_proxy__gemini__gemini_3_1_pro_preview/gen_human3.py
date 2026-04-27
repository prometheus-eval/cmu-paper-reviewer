h3_items = [
  {
    "item_number": 1,
    "reasoning": "The reviewer suggests two paths forward: remove biological claims or do more realistic biological work. This is based on the unphysiological results in sections 2.2 and 2.3. The premise that the paper's biological claims outpace its data is Correct. Suggesting a major rewrite of claims or the addition of in-vivo experiments is Significant because it addresses the core validity of the paper's framing. The evidence is Sufficient because the reviewer points specifically to sections 2.2 and 2.3 as the problematic areas. Experts would agree this is a valid and significant high-level critique.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 2,
    "reasoning": "The reviewer points out a discrepancy in Figure 4e, which shows a mouse with an implanted miniscope, but the actual data is just beads in agarose. This observation is Correct. It is Significant because misleading figures can give readers a false impression of the achieved experimental reality. The evidence is Sufficient as the reviewer directly points out the specific figure panel and contrasts it with the actual data presented. Experts would likely agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 3,
    "reasoning": "The reviewer asks if 'a.u.' for time is a typo or implies recordings were not real-time. The observation that time is not given in standard units (seconds/milliseconds) is Correct. This is Significant because temporal resolution is a key claim of the method, and missing real-time units makes it hard to evaluate if it's suitable for fast calcium transients. The evidence is Sufficient (points at a specific missing unit). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 4,
    "reasoning": "The reviewer critiques section 2.2 (fixed slice), saying it's essentially just a projection screen like the beads and doesn't add much realism. This is factually Correct, as the slice is fixed and excited by a DMD, not exhibiting true biological dynamics. Questioning the scientific value of an entire section's experimental design is Significant. The evidence is Sufficient because the reviewer explains the logical equivalence between the slice and the beads. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 5,
    "reasoning": "Similar to Item 4, the reviewer critiques section 2.3 (miniscope) for using sparse bright beads without background, questioning what new realism it adds. This is Correct; the miniscope experiment lacks tissue background. This is Significant as background is a major hurdle for miniscope imaging. The evidence is Sufficient (logical derivation of the missing elements in the experimental design). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 6,
    "reasoning": "The reviewer points out that the traces are too noise-free, lacking shot noise and background fluorescence, and suggests experiments with realistic noise levels. This is Correct, as the DMD setup selectively illuminates targets without widefield background. It is highly Significant because real NMF performance degrades sharply with noise and overlapping background. The evidence is Sufficient, providing a detailed logical breakdown of missing real-world factors. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 7,
    "reasoning": "The reviewer suggests evaluating robustness against sample movement and MMF bending. The paper uses short MMFs but does not explicitly test bending or sample movement robustness. This is Correct. Given the target application is freely moving mice, this is a Significant limitation. The evidence is Sufficient as it identifies the absence of a critical control experiment. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 8,
    "reasoning": "The reviewer asks to check language and style, specifically noting the discussion. This is a subjective but generally Correct observation if there are awkward phrasings. Language/style corrections that don't affect technical understanding are Marginally Significant. The evidence is borderline Requires More, as the reviewer only broadly points to 'the discussion' without specific examples. A charitable expert might accept this as sufficient pointer, while a strict one would demand specific quotes, leading to disagreement on evidence.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Requires More",
    "prediction_of_expert_judgments": "correct_marginal_disagree_on_evidence"
  },
  {
    "item_number": 9,
    "reasoning": "The reviewer critiques the authors' claim that improvements are simple ('plenty of room for improvement', 'simply add...'), suggesting they should implement them if they are so easy, or rephrase. The quotes are Correct. This is mainly about the tone and framing of future work, which is typically Marginally Significant, though some might argue it's a Significant overclaim. The evidence is Sufficient (direct quotes). Experts might disagree on whether this tone issue is Significant or Marginally Significant.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  },
  {
    "item_number": 10,
    "reasoning": "The reviewer notes that the discussion is verbose on improvements but lacks a critical discussion of key limitations for the target application. This is Correct, as the discussion focuses heavily on potential and less on the hurdles (like background, as noted in previous items). Missing limitations are Significant. The evidence is arguably Sufficient because it points at the entire discussion section's structural imbalance. Experts would likely agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 11,
    "reasoning": "The reviewer suggests removing Figure 4e as it is misleading (shows an implanted mouse but no such data exists). This is Correct. Removing a misleading figure improves clarity, making it Significant, though some might argue it's just a schematic and thus Marginally Significant. The evidence is Sufficient. The borderline nature of a schematic's importance could lead to disagreement on significance.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  }
]
