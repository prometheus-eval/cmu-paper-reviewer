import json

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'r') as f:
    data = json.load(f)

gpt_evals = {
  1: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer provides a deep statistical critique of the circadian detection algorithm, noting that defining groups based on the extrema of the same signal and then calculating a difference constitutes 'selection-on-the-outcome', guaranteeing an inflated effect. This is a mathematically correct and devastating critique of the core method. The evidence points directly to the paper's formulas. Experts would universally agree this is highly significant."
  },
  2: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer points out pseudoreplication: treating 165 hemispheres from 119 patients as independent observations without hierarchical/mixed modeling. This violates the assumption of independence and likely inflates p-values. This is a correct statistical critique. The evidence is sufficient, pointing to the subject counts and lack of hierarchical models in the methods. Experts would agree this requires re-analysis."
  },
  3: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer correctly identifies that the code depends on an internal, proprietary BRAVO server (with IP addresses) and missing Excel files, making computational reproducibility impossible. The evidence quotes the exact code lines showing these dependencies. This is a major methodological failure for a computational study. Experts would agree this is significant."
  },
  4: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer catches major internal inconsistencies: the text reports z=2.46/p=0.016 for ER levodopa while Table 2 reports z=2.825/p=0.005. Furthermore, the paper claims Mann-Whitney U and FDR correction including gamma, but the code shows one-sample t-tests and excludes gamma from FDR. These are massive reporting errors that invalidate trust in the statistical results. The evidence quotes the exact text, table, and code. Experts would agree."
  }
}

for reviewer in data["reviewers"]:
    if reviewer["reviewer_id"] == "gpt-5.2":
        for item in reviewer["items"]:
            num = item["item_number"]
            if num in gpt_evals:
                for k, v in gpt_evals[num].items():
                    item[k] = v

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'w') as f:
    json.dump(data, f, indent=2)

print("Updated gpt")
