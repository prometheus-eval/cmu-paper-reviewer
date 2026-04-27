import json

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'r') as f:
    data = json.load(f)

h3_evals = {
  1: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer points out that using normalized (z-scored) power obscures whether diurnal changes are driven by daytime drops or nighttime spikes, and asks for absolute power comparisons. This is factually correct. Absolute power would clarify the interpretation of the results significantly. The evidence is sufficient. Experts would agree."
  },
  2: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer raises the confound of sleep spindles (9-16Hz) overlapping with the tracked low-beta band, which the paper largely ignores. This is a very strong physiological point that could explain the 'novel' finding of increased nighttime beta. Addressing this is significant. Evidence is sufficient. Experts would agree."
  },
  3: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer asks if clinical phenotype (e.g., tremor-dominant) correlates with the inverse circadian profiles. This is a valid factual question. Some experts would view this as a significant missed opportunity in the analysis, while others might view it as marginally significant due to the retrospective nature and potential lack of data."
  },
  4: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer asks for a discussion of data quality and artifact sources (ECG, motion). The paper lacks this discussion. While a standard request, experts might disagree on whether it is significant or just marginally significant for this specific journal/format."
  },
  5: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer asks if profiles are patient-specific by checking correlation between hemispheres in bilateral patients. This is a great point: they have 165 hemispheres from 119 subjects, so they can test this. This would add significant value. Evidence is sufficient. Experts would agree."
  },
  6: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for more discussion on canonical bands and sleep spindles in the introduction/discussion. The paper does discuss NREM/REM to some extent, but misses spindles. This is a valid addition, making it marginally significant as a discussion point. Evidence is sufficient. Experts would agree."
  },
  7: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer points out that smoothing across patients in Figures 1C and 2C is inappropriate and asks to split them by canonical band. This is a valid visualization concern. Disagreement on significance is likely: some might see mixing bands as highly misleading (significant), while others might note Figure 3 already breaks them down (marginal)."
  },
  8: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer notes the nighttime beta increase in Supp Fig 1 when stimulation is turned up, asking for an explanation. This is correct. Explaining this secondary effect is arguably significant, but could be seen as marginal since it's just a supplementary figure. Disagreement on significance is plausible."
  },
  9: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer highlights confusing terminology (day vs sleep) and ambiguities in the automated circadian detection algorithm for datasets with no significant differences. These are valid methodological clarity issues. Clarifying this is significant. Evidence is sufficient. Experts would agree."
  },
  10: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer notes that figure font sizes are small. This is a correct observation and a minor presentation fix. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  11: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer points out there are many typos. This is correct. It is a minor presentation issue. Marginally significant. Evidence is sufficient. Experts would agree."
  }
}

for reviewer in data["reviewers"]:
    if reviewer["reviewer_id"] == "Human_3":
        for item in reviewer["items"]:
            num = item["item_number"]
            if num in h3_evals:
                for k, v in h3_evals[num].items():
                    item[k] = v

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'w') as f:
    json.dump(data, f, indent=2)

print("Updated Human_3")
