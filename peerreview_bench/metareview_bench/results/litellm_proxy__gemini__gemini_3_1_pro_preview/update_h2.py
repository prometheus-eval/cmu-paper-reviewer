import json

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'r') as f:
    data = json.load(f)

h2_evals = {
  1: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer asks for more in-depth exploration of the novel finding that beta increases at night in GPi, citing possible confounders like spindles or artifacts. This is a valid factual concern. Addressing it would improve the paper's discussion. The reviewer cites specific external knowledge. A devil's advocate might argue the retrospective nature precludes this analysis, making it a marginal point for this specific paper."
  },
  2: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer points out that the heatmaps in Figures 1 and 2 mix different frequency bands, which is true and misleading since bands modulate differently. Splitting them would clarify the results. The reviewer cites the 46.7% theta/alpha statistic. Disagreement is plausible because Figure 3 does break down by band, so some experts might view the heatmap issue as only marginally significant."
  },
  3: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer questions the automated circadian detection algorithm, noting it could fail in cases like insomnia. This correctly identifies a major methodological weakness. Resolving or fully caveating it is significant. The reviewer explains a specific failure mode. Experts would generally agree this is an important point."
  },
  4: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer asks how ER levodopa was factored into the sleep/wake classification algorithm. It was not factored in, which is a correct observation and a significant potential confound since the medication affects the exact signals used to classify sleep. The evidence is sufficient. Experts would agree this needs addressing."
  },
  5: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer asks if sleep scales (PSQI, etc.) were collected to explain variance. They were not, as it's a retrospective study. It's a valid limitation to point out. Some experts might see it as significant to add to the discussion, while others might find it marginally significant since the data cannot be retrospectively acquired."
  },
  6: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for the rationale behind target selection given the GPi/STN imbalance. This is a valid question. It is marginally significant because it provides clinical context but doesn't invalidate the within-target findings. The evidence points to the numbers. Experts would agree."
  },
  7: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer asks for contact locations to explain the difference between beta-increase and beta-decrease patients. The paper only uses a binary dorsal/ventral factor in the GLM. Providing more precise anatomical detail could explain the variance. Disagreement is likely: some experts might see this as crucial, others as beyond the scope of the current GLM."
  },
  8: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer notes a trend in Supplementary Figure 1 and asks for an explanation. The observation is factual. Explaining it would add value. Disagreement is possible on significance, as it concerns a supplementary figure and a secondary effect."
  },
  9: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer spots the 119 vs 117 subject count discrepancy. This is a correct observation of a minor error. It is marginally significant. The evidence is sufficient. Experts would agree."
  },
  10: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer points out a figure numbering error. This is correct and a minor presentation issue. It is marginally significant. The evidence is sufficient. Experts would agree."
  }
}

for reviewer in data["reviewers"]:
    if reviewer["reviewer_id"] == "Human_2":
        for item in reviewer["items"]:
            num = item["item_number"]
            if num in h2_evals:
                for k, v in h2_evals[num].items():
                    item[k] = v

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'w') as f:
    json.dump(data, f, indent=2)

print("Updated Human_2")
