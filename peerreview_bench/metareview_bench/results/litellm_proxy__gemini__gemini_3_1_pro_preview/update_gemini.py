import json

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'r') as f:
    data = json.load(f)

gemini_evals = {
  1: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer uncovers a severe discrepancy between the manuscript's Methods (which describe a complex adaptive algorithm for defining day/night periods) and the supplementary code (which uses hardcoded 0-5 and 15-20 windows). This is factually correct and heavily impacts reproducibility and trust in the reported methods. The evidence directly compares the text and the code. Experts would universally agree on its significance."
  },
  2: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer points out that using fixed time windows (as revealed in the code) or signal-derived periods (as in the text) without objective sleep validation is methodologically flawed given inter-patient variability. This is a correct physiological and methodological critique. The evidence is sufficient. Experts would agree this is a significant limitation of the study design."
  },
  3: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer notes that using sparse patient-marked events in only 61% of recordings as a proxy for wakefulness is insufficient to validate circadian/sleep phases. This is factually true based on the paper's own numbers. It represents a significant methodological gap in linking LFP data to actual sleep/wake states. Evidence is sufficient. Experts would agree."
  },
  4: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer identifies that clinicians selecting the tracking frequency based on prominent peaks introduces a selection bias, skewing the cohort towards specific electrophysiological phenotypes. This is correct. Disagreement on significance is likely: some experts might see this as a critical bias affecting interpretation, while others might view it as an inherent, accepted limitation of retrospective commercial DBS data."
  },
  5: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer criticizes the causal language ('modulate') used for extended-release medications in a retrospective study, noting unmeasured confounders like disease severity. This is a correct point about causal inference. A strict expert would view causal overclaiming as a significant flaw requiring revision, while a lenient expert might see it as just a marginally significant phrasing issue given the GLM controls. Prediction: disagree on significance."
  }
}

for reviewer in data["reviewers"]:
    if reviewer["reviewer_id"] == "gemini-3.0-pro-preview":
        for item in reviewer["items"]:
            num = item["item_number"]
            if num in gemini_evals:
                for k, v in gemini_evals[num].items():
                    item[k] = v

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'w') as f:
    json.dump(data, f, indent=2)

print("Updated gemini")
