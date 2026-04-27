h1_items = [
  {
    "item_number": 1,
    "reasoning": "The reviewer points out that for in-vivo experiments, the fiber length might range from 5-15mm, and asks how this length variation affects the number of 'scattering fingerprints' available for NNMF. The paper focuses on short MMFs (<10mm), so asking about the capacity limits depending on length is a valid methodological question (Correct). This is Significant because it directly impacts the in-vivo applicability claimed by the paper. The reviewer's evidence is an appeal to domain knowledge about realistic fiber lengths, which is specific enough to verify the relevance of the missing estimate (Sufficient). A skeptical expert might argue that the paper already defined its scope (<10mm), so asking for lengths up to 15mm is outside the scope, potentially leading to disagreement on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
  },
  {
    "item_number": 2,
    "reasoning": "The reviewer disputes the claim of 'realistic conditions' in lines 91-92 by listing specific missing factors (tissue scattering, labeling density, blood vessels) that would be present in vivo. The paper claims validation in a 'more realistic condition' by using a fixed brain slice rather than just beads. The reviewer is factually Correct that a fixed slice still lacks these in-vivo dynamics. This is Significant because it contextualizes the validation. The evidence is Sufficient as it points to specific missing factors. However, an expert might argue that the paper only claimed a 'more realistic condition' compared to beads, not fully in-vivo conditions, so the reviewer is being overly strict or misinterpreting the claim. This ambiguity suggests experts would disagree on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
  },
  {
    "item_number": 3,
    "reasoning": "The reviewer asks for missing methodological details (filters, objective magnification, LED power) and suggests adding them to the schematic in Figure 1. While most of these details are present in the supplementary materials, the reviewer's point that they are missing from the main text/Figure 1, and that LED power is entirely absent, is Correct. The significance is arguably Significant as these are important experimental details, but since many are in the supplement, it borders on Marginally Significant. The evidence is Sufficient because the reviewer points out exactly what is missing. A devil's advocate could argue that asking to move details from the supplement to the main figure is just a stylistic preference, leading to disagreement on significance.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  },
  {
    "item_number": 4,
    "reasoning": "The reviewer asks for additional experiments to assess the maximum number of point sources that can be resolved by increasing bead density and adding scattering to mimic tissue. This is a valid observation as the paper tests only up to 6 beads (Correct). Addressing this would clarify a key limitation of the technique, making it Significant. The evidence is Sufficient as the reviewer provides a logical derivation of why this is needed based on the current experiment's limitations. A devil's advocate might argue that this is beyond the scope of a proof-of-principle paper, but it remains a valid and significant suggestion. The experts would likely agree on this.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 5,
    "reasoning": "The reviewer suggests testing simultaneous or near-simultaneous excitation of bead groups to mimic correlated neuronal activity, as this might be difficult for NNMF to demix. The paper currently uses independently fluctuating sources, so pointing out this limitation is Correct. This is Significant because correlated activity is a major challenge in real in-vivo calcium imaging. The evidence is Sufficient as the reviewer provides a specific logical reasoning for why correlated activity poses a challenge to the method. The experts would likely agree on all axes.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 6,
    "reasoning": "The reviewer notes that Figures 2-3 do not report the excitation intensity per source. This is factually Correct, as the exact excitation intensity for the DMD setup is not explicitly stated. This is a minor reproducibility detail, so it is Marginally Significant. The evidence is Sufficient as the reviewer points out a specific absence. A devil's advocate might consider this Significant for reproducibility, but it's a borderline call, so experts might disagree on significance.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  },
  {
    "item_number": 7,
    "reasoning": "The reviewer claims the correlations in Figures 2-3 g-h are 'not informative' and suggests reporting prediction accuracy instead. While suggesting an alternative metric is fine, the claim that correlation is 'not informative' is subjective and arguably incorrect, as Pearson correlation is a standard metric for comparing time traces. A charitable reader might say the underlying point (that a predictive metric is better) is Correct. A skeptical reader would argue the paper's metric is perfectly valid and the criticism is Not Correct. This is a classic case of disagreement on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
  },
  {
    "item_number": 8,
    "reasoning": "The reviewer asks how saturation intensity (discussed in lines 188-194) would limit performance when multiple sources are distributed across the field of view. The premise that the paper discusses saturation is Correct (actually referring to the miniscope saturation in Figure S03 or text). This is a minor question for discussion, making it Marginally Significant. The evidence is Sufficient as it points to a specific text segment and poses a logical question. Experts would likely agree it's a minor but valid point.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient"
  },
  {
    "item_number": 9,
    "reasoning": "The reviewer suggests adding a recent relevant paper from the Pisanello lab to the discussion (lines 214-223). This is a valid related work suggestion (Correct). It is a nice-to-have reference that provides context but doesn't change the paper's core claims, making it Marginally Significant. The evidence is Sufficient because the full citation is provided. Experts would agree this is a valid, minor addition.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient"
  },
  {
    "item_number": 10,
    "reasoning": "The reviewer points out that cylindrical symmetry of the fiber makes it difficult to identify multiple sources placed equidistantly from the center, questioning an advantage claimed in lines 219-223. The paper actually addresses this in Supp Info 04, stating that patterns from symmetric beads 'seem to not be ambiguous' and show subtle differences. A skeptical expert would say the reviewer is Not Correct because the paper already addressed it. A charitable expert would say the reviewer is Correct because the paper itself admits it needs 'further investigation'. This ambiguity leads to a disagreement on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
  },
  {
    "item_number": 11,
    "reasoning": "The reviewer notes that the claim of 'expanding capacity' is not fully supported without in-vivo experiments, but explicitly states they 'do not suggest this as a mandatory experiment'. The observation that there are no in-vivo experiments is Correct. Suggesting in-vivo validation is Significant. The evidence is Sufficient (points at a specific claim and the absence of in-vivo data). However, since the reviewer themselves softens the blow by calling it non-mandatory, some experts might downgrade it to Marginally Significant, leading to disagreement on significance.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  },
  {
    "item_number": 12,
    "reasoning": "The reviewer suggests citing earlier works by Papadopoulos et al. and Turtaev et al. This is Correct, as these are relevant works on fiber microendoscopy. This is a minor nice-to-have addition that provides context, so it is Marginally Significant. The evidence is Sufficient because the exact references are provided. The experts would likely agree on this.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient"
  }
]
