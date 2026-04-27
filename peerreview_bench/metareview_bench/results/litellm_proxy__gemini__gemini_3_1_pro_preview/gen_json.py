import json

data = {
  "paper_id": 82,
  "reviewers": []
}

def create_human_items(num_items):
    items = []
    for i in range(1, num_items + 1):
        items.append({
            "item_number": i,
            "reasoning": "The reviewer is discussing topics (OECT, cells, trypsin) that are completely unrelated to the paper, which focuses on a sub-wavelength Si LED based on gate oxide breakdown. Thus, the reviewer's core factual claims do not hold up against the paper. No domain expert could charitably interpret this as relevant, meaning both experts would agree the criticism is factually incorrect for this manuscript.",
            "correctness": "Not Correct",
            "significance": None,
            "evidence": None,
            "prediction_of_expert_judgments": "incorrect"
        })
    return items

data["reviewers"].append({
    "reviewer_id": "Human_1",
    "items": create_human_items(7)
})

data["reviewers"].append({
    "reviewer_id": "Human_2",
    "items": create_human_items(15)
})

data["reviewers"].append({
    "reviewer_id": "Human_3",
    "items": create_human_items(11)
})

claude_items = []
claude_items.append({
    "item_number": 1,
    "reasoning": "The reviewer correctly notes the paper relies on gate oxide breakdown but lacks comprehensive reliability modeling, pointing out a 25% power loss after 10^5 cycles mentioned in the text. This is factually accurate. It is a Significant issue because reliability is critical for CMOS-integrated emitters. The evidence is Sufficient, citing the specific supplementary metrics. While some experts might argue the proof-of-concept scope doesn't require full lifetime testing (possibly disagreeing on Significance), the factual basis is solid. I predict experts would agree on its significance as a major caveat.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})
claude_items.append({
    "item_number": 2,
    "reasoning": "The reviewer observes that only five devices were tested, showing a +/-20% variation, which undermines the claims of high reproducibility. This is factually correct as confirmed by the paper's supplementary text. It is Significant because yield and variance are crucial for CMOS photonic integration. The evidence is Sufficient with direct quotes. A devil's advocate could argue that 5 devices is standard for an initial physics demonstration, leading to plausible disagreement among experts on whether this issue is fully Significant or just a minor limitation for this stage of research.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
})
claude_items.append({
    "item_number": 3,
    "reasoning": "The reviewer criticizes the sub-wavelength emission area claim, noting that the simple Gaussian FWHM deconvolution formula is unstable and ignores non-Gaussian features when sizes are well below the PSF. This is factually correct from an optics perspective. It is Significant because it affects the main quantitative claim of the paper. The evidence is Sufficient. However, a devil's advocate could argue that Gaussian approximation is a standard, acceptable practice in physics papers for estimating sub-resolution features. Thus, experts are likely to disagree on the correctness of asserting this method is inadequate.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})
claude_items.append({
    "item_number": 4,
    "reasoning": "The reviewer points out that the holographic microscopy demonstration only uses 20 um beads and lacks quantitative resolution metrics, limiting practical applicability. This is factually Correct based on the paper's methodology. It is Significant as it weakens a key application claim. The evidence is Sufficient. However, a devil's advocate might argue that the holography is merely a qualitative proof of spatial coherence, not the main contribution, so demanding rigorous benchmarking like USAF targets might be seen as Marginally Significant by some experts.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_disagree_on_significance"
})
data["reviewers"].append({"reviewer_id": "claude-opus-4-5", "items": claude_items})

gemini_items = []
gemini_items.append({
    "item_number": 1,
    "reasoning": "The reviewer claims the paper asserts stability but reports a 25% drop after 10^5 cycles, contradicting standard LED reliability. This is factually Correct, as the text contains both the claim of robustness and the 25% degradation figure. It is Significant. The evidence is Sufficient. However, an expert could defend the authors, arguing that in the context of nanoscale breakdown devices, operating for 10^5 cycles *is* relatively robust, and the authors were transparent about the degradation. Therefore, experts would likely disagree on the correctness of calling this a 'contradiction'.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})
gemini_items.append({
    "item_number": 2,
    "reasoning": "The reviewer argues that comparing 'intensity' with state-of-the-art Si emitters is misleading because it ignores the extremely low EQE and total power. This is factually Correct; the intensity is high primarily due to the tiny area. It is Significant as it impacts the fair assessment of the device. The evidence is Sufficient. A devil's advocate would point out that the authors explicitly report the EQE and power in the text and only claim comparable *intensity*, making the reviewer's charge of 'misleading' subjective. This will likely cause disagreement on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})
gemini_items.append({
    "item_number": 3,
    "reasoning": "The reviewer claims using the term 'LED' without qualification is misleading because the device relies on a destructive gate oxide breakdown (antifuse) mechanism. This is factually Correct as the mechanism is breakdown-based. However, the device is technically a diode emitting light, so calling it an LED is technically accurate. The issue is purely terminological and does not affect the physical claims, making it Marginally Significant. The evidence is Sufficient. Experts are very likely to disagree on the correctness of labeling the standard term 'LED' as misleading here.",
    "correctness": "Correct",
    "significance": "Marginally Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})
gemini_items.append({
    "item_number": 4,
    "reasoning": "This item is completely empty. The reviewer's main point is nonexistent, meaning no specific criticism can be determined. A charitable domain expert could not extract a valid concern from an empty block, so there is no disagreement. The item is Not Correct.",
    "correctness": "Not Correct",
    "significance": None,
    "evidence": None,
    "prediction_of_expert_judgments": "incorrect"
})
gemini_items.append({
    "item_number": 5,
    "reasoning": "This item is completely empty. The reviewer's main point is nonexistent, meaning no specific criticism can be determined. A charitable domain expert could not extract a valid concern from an empty block, so there is no disagreement. The item is Not Correct.",
    "correctness": "Not Correct",
    "significance": None,
    "evidence": None,
    "prediction_of_expert_judgments": "incorrect"
})
data["reviewers"].append({"reviewer_id": "gemini-3.0-pro-preview", "items": gemini_items})


gpt_items = []
gpt_items.append({
    "item_number": 1,
    "reasoning": "The reviewer states the core mechanism is hard breakdown, a catastrophic reliability event, and argues the paper lacks sufficient reliability analysis for a CMOS platform claim. This is factually Correct and Significant, as integrating a fundamentally destructive mechanism into standard CMOS without extensive lifetime data is highly problematic. The evidence is Sufficient, citing both the paper's data and external literature. While some experts might excuse the lack of full reliability data for a proof-of-concept, the core concern about CMOS compatibility is too important to dismiss.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})
gpt_items.append({
    "item_number": 2,
    "reasoning": "The reviewer notes that the paper claims spatial coherence enables holography but provides no quantitative coherence metrics (e.g., fringe visibility) to compensate for the broad emission bandwidth, relying on one example with rings. This is factually Correct. It is Significant because spatial coherence is a heavily emphasized property. The evidence is Sufficient. A devil's advocate could say the holography is just a qualitative demonstration, but since it's a central abstract claim, most experts would agree the lack of quantitative coherence data is a significant weakness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})
gpt_items.append({
    "item_number": 3,
    "reasoning": "The reviewer attacks the deconvolution method, pointing out that subtracting PSF FWHM in quadrature assumes perfect Gaussian profiles and is highly sensitive to non-Gaussian tails when estimating sub-resolution features. This is factually Correct. It is Significant because it affects the primary emission area claim. The evidence is Sufficient. However, similar to other deconvolution critiques, a devil's advocate would argue that Gaussian approximations are standard, accepted practice in these device physics papers, leading experts to disagree on whether the method is actually invalid.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})
gpt_items.append({
    "item_number": 4,
    "reasoning": "The reviewer observes that the EQE calculation neglects background emission, which makes it a partial efficiency metric and potentially biases the 'no thermal droop' conclusion if the background fraction changes. This is factually Correct, as the paper explicitly states EQE only accounts for the spot. It is Significant because it challenges a major performance claim. The evidence is Sufficient. A defensive expert might argue that since the authors transparently defined their EQE for the spot, the criticism of using a non-standard definition is misplaced, causing disagreement on correctness.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})
gpt_items.append({
    "item_number": 5,
    "reasoning": "This item is completely empty. The reviewer's main point is nonexistent, meaning no specific criticism can be determined. A charitable domain expert could not extract a valid concern from an empty block, so there is no disagreement. The item is Not Correct.",
    "correctness": "Not Correct",
    "significance": None,
    "evidence": None,
    "prediction_of_expert_judgments": "incorrect"
})
data["reviewers"].append({"reviewer_id": "gpt-5.2", "items": gpt_items})

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper82_metareview.json", "w") as out:
    json.dump(data, out, indent=2)
    f.write(f'''
import json

data = {json.dumps(data, indent=2)}

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper82_metareview.json", "w") as out:
    json.dump(data, out, indent=2)

print("JSON file written.")
''')
