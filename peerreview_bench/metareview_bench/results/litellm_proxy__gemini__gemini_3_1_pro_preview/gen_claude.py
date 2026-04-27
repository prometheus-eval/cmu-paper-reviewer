c_items = [
  {
    "item_number": 1,
    "reasoning": "The reviewer critiques the mismatch between the paper's broad claims of resolving single-source in-vivo activity and the fact that all experimental validations are performed ex-vivo (beads/fixed slices) with artificial DMD-generated activity. This observation is Correct. Pointing out the gap between the proof-of-principle and the target biological application is Significant. The evidence is Sufficient, providing direct quotes and referencing standard validation practices in the literature. Experts would agree on this.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 2,
    "reasoning": "The reviewer points out that NMF requires manual specification of the rank (number of components), and the paper lacks an automatic, validated way to choose this for unknown in-vivo source numbers. The paper explicitly states the ranks were manually chosen, so this is Correct. Since the number of neurons is unknown in a real experiment, this limitation is Significant for practical applicability. The evidence is Sufficient, citing the text and comparing it to automated pipelines like CaImAn. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 3,
    "reasoning": "The reviewer notes that the paper assumes 'scattering fingerprints' remain stable during in-vivo conditions, but fails to experimentally validate this under perturbations like fiber bending or tissue movement. The paper relies on theoretical rigidity arguments without experimental proof, so the observation is Correct. This is highly Significant because pattern stability is the core requirement for NMF to work. The evidence is Sufficient (logical derivation and citations). Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  },
  {
    "item_number": 4,
    "reasoning": "The reviewer notes the paper presents only single experiments per condition (one 6-bead phantom, one 4-neuron slice) without repeated trials across multiple preparations. This is factually Correct. Lack of statistical replicates is a major methodological weakness, making the critique Significant. The evidence is Sufficient, pointing specifically to the reported statistics being derived from within a single experiment rather than across replicates. Experts would agree.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
  }
]
