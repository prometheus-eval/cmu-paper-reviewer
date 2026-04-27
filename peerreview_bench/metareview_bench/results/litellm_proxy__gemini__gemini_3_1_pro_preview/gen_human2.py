h2_items = [
  {
    "item_number": 1,
    "reasoning": "The reviewer points out the lack of an in-vivo experiment and argues that demixing through a short MMF is the same process as demixing through a skull (which the authors did in previous work), thus limiting novelty. While it is true there is no in-vivo experiment, the physics of short MMF transmission ('scattering fingerprints') is different from the fully developed speckles of a skull, as noted in lines 81-86. A charitable reader might say the core concern (lack of in-vivo data limits the biological claim) is Correct. A skeptical reader might say the premise (it's the exact same process as the skull) is factually Not Correct. Thus, experts would disagree on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
  },
  {
    "item_number": 2,
    "reasoning": "The reviewer asks for a more thorough explanation or simulation of the spatial signatures in terms of modes propagating through the fiber. The paper does treat the fiber mostly as a black box that generates 'fingerprints' without explicitly modeling the spatial modes. This is Correct. Providing this would strengthen the physical understanding of the method, making it Significant. The evidence is Sufficient as the reviewer points at the absence of mode explanations. A devil's advocate might argue that a purely theoretical mode analysis is not strictly necessary for an applied proof-of-principle paper, which could lead to disagreement on significance.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  },
  {
    "item_number": 3,
    "reasoning": "The reviewer suggests a more realistic widefield illumination experiment (adding neuropil background) or an in-vivo experiment. The current experiment uses targeted DMD illumination without a continuous widefield neuropil background, so pointing out this limitation is Correct. This is Significant because widefield background is a major challenge in 1P imaging that NMF would need to handle. The evidence is Sufficient, as the reviewer provides a detailed logical derivation of what happens in 1P widefield and why the current setup misses it. Experts would agree this is a strong, well-supported critique.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 4,
    "reasoning": "The reviewer points out a previous method (tapered fibers, Bianco et al.) that also uses spatial patterns and asks for a comparison of the mechanisms and advantages. The paper does not cite Bianco et al., so the observation is Correct. Comparing the proposed method with a similar existing technique is crucial for establishing novelty and advantages, making it Significant. The evidence is Sufficient because the reviewer provides a specific citation and details the similarities/differences. Experts would likely agree on all axes.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 5,
    "reasoning": "The reviewer asks for theoretical limits on the unmixing capacity, specifically the minimal radial and angular distances that generate distinct spatial signatures. The paper does not provide these theoretical limits, relying instead on empirical examples, so the observation is Correct. This is Significant because it defines the fundamental resolution limits of the technique. The evidence is Sufficient as it references specific supplementary figures and identifies the missing theoretical bounds. Experts would likely agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 6,
    "reasoning": "The reviewer notes the absence of z-axis analysis and suggests calibration at different depths. The paper uses a fixed plane (e.g., a single slice or 2D bead sample) and does not analyze z-displacement. This is Correct. In a real biological setting, sources are distributed in 3D, so understanding z-axis behavior is genuinely Significant. The evidence is Sufficient as the reviewer explicitly points out the missing dimension of analysis. Experts would agree on this.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 7,
    "reasoning": "Following up on Item 6, the reviewer suggests an experiment with beads at different z-planes to test 3D unmixing. The paper lacks such an experiment, so the premise is Correct. This is Significant for the same reason: real tissue has 3D source distributions. The evidence is Sufficient (points at a specific missing experiment). Experts would likely agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 8,
    "reasoning": "The reviewer questions the implied assumption that the field of view (FoV) is always equal to the fiber diameter, asking if increasing the z-distance extends the xy FoV due to the fiber's numerical aperture. This is a valid optical physics question, as the acceptance cone does widen with distance (Correct). Clarifying the true FoV size is a useful detail, but perhaps not critical to the main claim, making it arguably Marginally Significant or Significant. The evidence is Sufficient as it relies on basic optical principles. Experts might debate whether clarifying this detail is crucial or minor, leading to disagreement on significance.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  },
  {
    "item_number": 9,
    "reasoning": "The reviewer notes that the paper ignores tissue scattering before the fiber, which would scramble the signal of deep sources before they even enter the fiber. The paper indeed focuses on the fiber's scattering and uses clear media (beads) or a very thin slice (50 um) where pre-fiber scattering is minimal. This observation is Correct. It is highly Significant because deep brain imaging necessarily involves tissue scattering, which could blur distinct patterns into identical speckles. The evidence is Sufficient as it provides a logical derivation of the physical effect. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 10,
    "reasoning": "The reviewer points out missing y-scales in Figures 2-3 and asks how the method handles resting fluorescence (baseline), which would create a continuous background pattern. The y-scales are indeed missing, and the paper's DMD setup does not clearly simulate a non-zero baseline. This is Correct. It is Significant because NMF's ability to separate signals could be heavily impacted by a constant overlapping background pattern from resting neurons. The evidence is Sufficient, providing a clear logical chain. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 11,
    "reasoning": "The reviewer asks to see raw fluorescence videos of the experiments (e.g., of Fig 2). This is a standard request for raw data, and the videos are not embedded in the manuscript. This is Correct. As a reproducibility or nice-to-have request, it is Marginally Significant. The evidence is Sufficient (identifies missing raw data). Experts would agree on this minor point.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient"
  },
  {
    "item_number": 12,
    "reasoning": "The reviewer recommends a different venue (an optical journal) because they feel the biological conclusions are not fully demonstrated without in-vivo data. The factual basis (lack of in-vivo data) is Correct. Suggestions regarding venue transfer or overall suitability for a specific high-impact journal are typically classified as Marginally Significant, as they don't directly improve the science of the paper itself. The evidence is Sufficient. Some experts might argue venue comments should be removed (Not Significant), while others keep them, leading to potential disagreement on significance.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
  }
]
