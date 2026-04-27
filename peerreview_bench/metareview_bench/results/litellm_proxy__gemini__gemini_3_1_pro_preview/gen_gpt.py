gpt_items = [
  {
    "item_number": 1,
    "reasoning": "The reviewer critiques the extrapolation to 'freely-behaving' mice, pointing out that the paper does not test the core failure mode in such settings: time-varying fiber perturbations (bending/torsion) that would change the spatial fingerprints. This is Correct, as the experiments are entirely static. Because pattern stability under motion is the make-or-break requirement for NMF demixing in freely behaving animals, this is highly Significant. The evidence is Sufficient (cites the paper's claims and contrasts them with known physical properties of multimode fibers). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 2,
    "reasoning": "The reviewer notes that the evaluation relies on simple within-experiment average correlations, lacking independent statistical repeats, cross-talk quantification, and a systematic analysis of failure cases (like the low-SNR edge source). The paper indeed reports summary statistics from single experiments without deeper false-discovery analysis. This is Correct. A rigorous validation of a demixing algorithm requires testing against realistic noise, background, and overlap, making this Significant. The evidence is Sufficient (detailed analysis of the reported metrics and comparisons with established validation standards). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 3,
    "reasoning": "The reviewer points out that counting sources via unconstrained NMF is critically dependent on selecting the rank, which the authors currently tune using ground truth or vague heuristics. The observation that the code uses ground truth to set rank is Correct. In a real scenario where the number of sources is unknown, this is a major hurdle, making it Significant. The evidence is Sufficient (cites code excerpts, supplementary text, and NMF literature on non-convexity). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 4,
    "reasoning": "The reviewer highlights reproducibility issues: the code requires '.mat' data files that are not included, has hardcoded paths, and references a function ('norm_dimension') that is missing from the provided utility scripts. Checking the code confirms the files and function are indeed missing. This is Correct. Reproducibility of the analysis is a core requirement, making this Significant. The evidence is Sufficient (cites specific lines of code and the missing components). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  }
]
