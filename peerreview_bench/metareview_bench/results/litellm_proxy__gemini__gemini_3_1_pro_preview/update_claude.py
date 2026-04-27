import json

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'r') as f:
    data = json.load(f)

claude_evals = {
  1: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer points out the circularity of defining day/night based on LFP periods without independent validation. This accurately identifies a major methodological weakness of the study. The reviewer provides specific quotes from the paper and compares it to established standards. Experts would agree this is a significant concern that requires at least a strong caveat."
  },
  2: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer brilliantly catches a major statistical discrepancy: the paper claims to use a Mann-Whitney U test (which is inappropriate for one-sample tests against 0), but the supplementary code actually uses a parametric one-sample t-test (ttest_1samp). This is factually correct, highly significant for reproducibility and statistical rigor, and perfectly evidenced by citing the exact code lines. Experts would universally agree."
  },
  3: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer notes that the study treats nighttime as homogeneous, ignoring NREM/REM differences that strongly affect beta power. This is a correct physiological point. The evidence is sufficient. However, as it is a limitation of the retrospective data, some experts might consider it a significant confounding factor, while others might view it as a marginally significant point since it cannot be fixed without a new study."
  },
  4: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer argues that therapeutic stimulation confounds the results and that TEED is an inadequate control. The paper does acknowledge this limitation. A charitable expert might argue the authors did what they could by including TEED and adding a limitation section, making this criticism only marginally significant. A stricter expert would agree it fundamentally limits the conclusions. Prediction: disagree on significance."
  },
  5: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer highlights the severe sample size imbalance between GPi and STN, pointing out that STN sub-analyses (n=5 or n=6) lack statistical power to draw robust conclusions. This is factually true and a significant issue with the paper's claims. The evidence references specific numbers from Figure 3. Experts would agree this is a significant overstatement of results in the STN cohort."
  }
}

for reviewer in data["reviewers"]:
    if reviewer["reviewer_id"] == "claude-opus-4-5":
        for item in reviewer["items"]:
            num = item["item_number"]
            if num in claude_evals:
                for k, v in claude_evals[num].items():
                    item[k] = v

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'w') as f:
    json.dump(data, f, indent=2)

print("Updated claude")
