import json

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'r') as f:
    data = json.load(f)

# Update Human_1 items
h1_evals = {
  1: {
    "correctness": "Not Correct", "significance": None, "evidence": None,
    "prediction_of_expert_judgments": "disagree_on_correctness",
    "reasoning": "The reviewer claims the paper lacks a clear focus or objective. However, the objective is explicitly stated in the final paragraph of the Introduction (to characterize circadian patterns...). The reviewer's core claim is factually incorrect. A charitable expert might argue that a purely descriptive goal lacks a testable hypothesis, making it effectively an unclear objective, leading to disagreement."
  },
  2: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer notes the lack of controls, chronotype accounting, and external metrics for sleep. This correctly reflects the paper's retrospective limitations. Addressing this would improve the study. The reviewer explicitly points to the lack of sleep metrics. A devil's advocate might argue this is an inherent limitation of retrospective data that the paper already acknowledges, making the criticism only marginally significant for this specific venue, causing disagreement."
  },
  3: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness",
    "reasoning": "The reviewer highlights the circularity in using LFP power to define day/night states. The paper's Methods do state they used LFP high/low periods to determine when each period occurred. This is a significant methodological concern. An opposing expert might argue the paper mapped these periods to external clock times rather than strictly equating high LFP with daytime, meaning it isn't fully circular, leading to correctness disagreement."
  },
  4: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer asks for clarity on when extended-release medications were taken, as this impacts the LFP interpretations. This is factually true and checking this would substantively improve the conclusions drawn. The evidence is sufficient as the reviewer cites specific lines. Experts would widely agree this is a substantive and verifiable concern."
  },
  5: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer notes the paper conflates day/night with awake/asleep. The paper indeed makes this leap in places, which affects the conclusions. This is a significant issue whose correction would improve the clarity of claims. The reviewer provides sufficient evidence. No strong counterargument exists, so experts would agree."
  },
  6: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer suggests that the phrase 'altered by' implies causality. This is an accurate reading of the text. However, it is a minor phrasing issue that does not fundamentally undermine the study, making it marginally significant. The evidence is clear. Experts would agree it is a minor wording fix."
  },
  7: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for clarification on the frequency band in the abstract. It is true that the abstract phrasing could be more specific. This is a presentation nitpick, so it is marginally significant. The evidence is sufficient. Experts would agree on its marginal importance."
  },
  8: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer points out that 'direction of circadian rhythm' is ambiguous and suggests a rewrite. The abstract does use this phrasing. It is a minor stylistic fix, making it marginally significant. The evidence is sufficient. Experts would agree it's minor."
  },
  9: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer suggests including the number of hemispheres in the abstract. The abstract currently lacks this exact breakdown. Adding it would be nice but isn't critical, making it marginally significant. The evidence points to a specific absence. Experts would agree."
  },
  10: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer questions the relevance of the aDBS paragraph in the introduction. The paper does rely heavily on this framing. Some experts might see this as a marginal presentation point, while others might view it as a significant structural flaw in the paper's narrative. The evidence is sufficient. Disagreement on significance is likely."
  },
  11: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer clarifies a detail about a cited paper (van Rheede et al.) lacking external sleep validation. This is a valid correction but only affects a minor reference point, making it marginally significant. The evidence is sufficient. Experts would agree."
  },
  12: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer requests references for a specific statement in the introduction. The paper does lack references here. Adding them would be helpful but minor. The evidence points out the missing citation. Experts would agree on its marginal impact."
  },
  13: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer asks why a specific data collection window (13 +/- 35 days) was chosen. The paper doesn't justify this window, which is a significant methodological ambiguity. The evidence is sufficient by pointing out the absent justification. Experts would agree this is an important clarification."
  },
  14: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer asks if frequency bands were chosen in the on-medication or off-medication state. The paper omits this detail, which could skew results. This is a significant methodological gap. The evidence is sufficient. Experts would agree."
  },
  15: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer asks how many subjects were excluded due to incompatible sensing configurations. The paper doesn't report this attrition rate. It's a significant methodological detail. Evidence is sufficient. Experts would agree this should be reported."
  },
  16: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for fractions of each frequency band across hemispheres to be reported. It's a valid data presentation request, though somewhat minor since the main bands are mentioned. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  17: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer points out a typo ('reordered' instead of 'recorded'). This is correct. It is a trivial typo, thus marginally significant. Evidence is sufficient. Experts would agree."
  },
  18: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer questions the inclusion of the gamma group since it was targeted differently (based on dyskinesia). The paper does state this different targeting. An expert might argue it's a significant confound, while another might see it as just a marginal grouping issue given the small n. Evidence is sufficient. Prediction: disagree on significance."
  },
  19: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer requests details on the 5-day windows selected for Figures 1 and 2. The paper lacks these specifics, which are important for interpreting the representative examples. This is significant. Evidence is sufficient. Experts would agree."
  },
  20: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks to explicitly state the comparison being made in the ANOVA. It is a valid request for clarity in the Results section, though mostly presentational. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  21: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer questions the mixed use of parametric and non-parametric tests and the choice of comparisons. The paper's statistical approach is indeed arguably arbitrary here. This is a significant methodological concern. Evidence is sufficient. Experts would agree."
  },
  22: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for a missing p-value. The paper omits it. Adding it is a minor fix. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  23: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer asks for clarification on the dependent variable in the GLM. The paper does state it's the 'change in Beta band', but could be more precise. Some experts might find this significantly confusing, while others might find it only marginally so. Evidence is sufficient."
  },
  24: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer spots a discrepancy in subject numbers (119 vs 117). This is a correct observation of a minor error. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  25: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer notes missing demographic data like unilateral/bilateral fractions. The paper lacks these. Some experts would consider this a significant omission for a clinical study, while others might see it as marginally important. Evidence is sufficient."
  },
  26: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer notes the lack of description of how patient events were selected. The paper does not provide this. It is a significant missing detail for understanding the validation. Evidence is sufficient. Experts would agree."
  },
  27: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer notes the lack of description for artifact handling. The paper omits this standard preprocessing detail. This is significant for evaluating data quality. Evidence is sufficient. Experts would agree."
  },
  28: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for units for 'brain recording'. A valid but minor presentation issue. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  29: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks to describe sensing-available configurations. The paper mentions them without detail. Valid minor point. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  30: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer requests the LEDD calculation method. It's a standard metric, so the omission is minor. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  31: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer points out a contradiction regarding whether stimulation parameters were available. The paper indeed contradicts itself (Discussion vs Methods). This is a significant internal inconsistency. Evidence is sufficient. Experts would agree."
  },
  32: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient",
    "reasoning": "The reviewer identifies several missing details in Figures 1 and 2 (events, scale, generation of plot C). These are significant omissions that hinder interpretation of the figures. Evidence is sufficient. Experts would agree."
  },
  33: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer points out figure numbering and symbol definition errors in Figure 3. These are minor presentation issues. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  34: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer points out a mislabeled Figure 4. This is a minor typo. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  35: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer asks for variables in Table 2 to be defined. Minor presentation fix. Marginally significant. Evidence is sufficient. Experts would agree."
  },
  36: {
    "correctness": "Correct", "significance": "Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance",
    "reasoning": "The reviewer suggests a floor effect in the Supplementary Figure. This is a plausible alternative interpretation of the data. An expert might see this as a significant interpretive point, while another might view it as just a marginal nuance. Evidence is sufficient."
  },
  37: {
    "correctness": "Correct", "significance": "Marginally Significant", "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_marginal_sufficient",
    "reasoning": "The reviewer lists several typos. These are correct and clearly minor. Marginally significant. Evidence is sufficient. Experts would agree."
  }
}

for reviewer in data["reviewers"]:
    if reviewer["reviewer_id"] == "Human_1":
        for item in reviewer["items"]:
            num = item["item_number"]
            if num in h1_evals:
                for k, v in h1_evals[num].items():
                    item[k] = v

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper14_metareview.json", 'w') as f:
    json.dump(data, f, indent=2)

print("Updated Human_1")
