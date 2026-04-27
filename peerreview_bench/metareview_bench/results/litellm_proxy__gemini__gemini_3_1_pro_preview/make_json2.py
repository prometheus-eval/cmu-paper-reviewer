import json

data = {
  "paper_id": 81,
  "reviewers": [
    {
      "reviewer_id": "Human_1",
      "items": []
    },
    {
      "reviewer_id": "Human_2",
      "items": []
    },
    {
      "reviewer_id": "claude-opus-4-5",
      "items": []
    },
    {
      "reviewer_id": "gemini-3.0-pro-preview",
      "items": []
    },
    {
      "reviewer_id": "gpt-5.2",
      "items": []
    }
  ]
}

# Human_1
for i in range(1, 6):
    data["reviewers"][0]["items"].append({
        "item_number": i,
        "reasoning": "The main point of the criticism is fundamentally inapplicable to this paper. The reviewer discusses an LED device, silicon substrates, and optical output power, whereas the paper is about organic electrochemical transistors (OECTs) and single-cell impedance biosensors. The factual premise is completely wrong and refers to a different paper. Devil's advocate: Is it possible the reviewer meant something else? No, the details clearly describe an entirely different device and study. Therefore, there is no plausible defense for correctness, and experts would unanimously agree this is incorrect.",
        "correctness": "Not Correct",
        "significance": None,
        "evidence": None,
        "prediction_of_expert_judgments": "incorrect"
    })

# Human_2
for i in range(1, 6):
    data["reviewers"][1]["items"].append({
        "item_number": i,
        "reasoning": "The main point of the criticism is entirely off-topic and discusses a different paper. The reviewer asks about a micro-LED device, CMOS emitters, an 1100 nm emission spectrum, and a Si filament, none of which are present in the actual paper about PEDOT:PSS OECT impedance sensors. Devil's advocate for correctness: Could the reviewer be referring to an analogous mechanism? No, the specific quotes and references (like 'CMOS emitters' and '1100 nm') clearly demonstrate the review was uploaded for the wrong paper. Experts would agree this is completely incorrect.",
        "correctness": "Not Correct",
        "significance": None,
        "evidence": None,
        "prediction_of_expert_judgments": "incorrect"
    })

# claude-opus-4-5
data["reviewers"][2]["items"].append({
    "item_number": 1,
    "reasoning": "The reviewer correctly points out that the single-cell experiment is based on just one cell and uses temporal replicates rather than independent biological replicates to derive error bars. This is verified by the paper's description and Figure 3 caption. Devil's advocate for correctness: Could the reviewer be misinterpreting the replicates? No, the paper explicitly states 'n=4 measurements acquired at different time'. Devil's advocate for significance: Could an expert see this as Marginally Significant since the paper is primarily demonstrating a device concept rather than a biological discovery? While plausible, making quantitative claims about single-cell detection without biological replicates is a major methodological gap. The evidence is highly specific and quotes the text directly. Experts would generally agree this is a valid and significant point.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][2]["items"].append({
    "item_number": 2,
    "reasoning": "The reviewer correctly notes that the 50 um dielectric microparticle lacks the biological complexity of real cells (such as membrane capacitance), which limits the validity of the surrogate model. This is factually accurate against the paper's methodology. Devil's advocate for correctness: The paper does acknowledge this is a 'first order approximation' and uses actual cells later. However, the reviewer's concern is about using this oversimplified model to draw quantitative optimizations. The criticism holds. Devil's advocate for significance: Is this just a minor nuance? No, the lack of membrane capacitance fundamentally alters the impedance spectrum, especially at higher frequencies, making this a substantive concern. The evidence cites specific biological properties and literature. Experts would agree on correctness and significance.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][2]["items"].append({
    "item_number": 3,
    "reasoning": "The reviewer points out that the model introduces a free fitting parameter, fOECT, adjusted for each sensor, which weakens its predictive power. The paper explicitly confirms this practice in the text. Devil's advocate for correctness: Could the parameter be physically justified? The paper mentions it depends on geometry, but admits to freely fitting it rather than deriving it. The claim holds. Devil's advocate for significance: Might this be a minor modeling detail? Given the paper's core claim is a 'quantitative model' that provides 'clear guidelines', relying on a per-device fitted parameter is a significant limitation. The evidence quotes the exact methodological step. Experts would agree this is a significant and well-supported criticism.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][2]["items"].append({
    "item_number": 4,
    "reasoning": "The reviewer claims the paper focuses its analysis primarily at a single frequency (625 Hz) and misses the opportunity to extract biological parameters using multi-frequency ECIS methodology. Devil's advocate for correctness: The paper actually acquires multi-frequency spectra every 10 minutes and plots them in Figures 3c-e. A strict expert might say the factual premise is incorrect because the data is multi-frequency. A charitable expert would understand the reviewer's main point is that the *biological analysis* is limited to plotting the transient at one frequency and computing gain, without extracting ECIS parameters. Because this ambiguity makes the factual truth borderline depending on how 'single-frequency approach' is interpreted, experts would likely disagree on correctness. The underlying concern about not extracting biological parameters is significant and well-supported.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "disagree_on_correctness"
})

# gemini-3.0-pro-preview
data["reviewers"][3]["items"].append({
    "item_number": 1,
    "reasoning": "The reviewer correctly identifies that a dielectric microparticle fails to capture the frequency-dependent membrane capacitance of biological cells, limiting the model's validity at high frequencies. This accurately reflects the paper's use of a purely dielectric surrogate. Devil's advocate for correctness: The paper claims it is only a first-order approximation, but the reviewer rightly argues that ignoring membrane capacitance at high frequencies (where the paper conducts sweeps up to 100 kHz) is a substantive gap. Devil's advocate for significance: Is this a minor point? No, because the paper uses the microparticle model to establish the frequency-dependent gain curve, which would be fundamentally different for real cells. The evidence cites relevant literature. Experts would agree this is correct, significant, and sufficiently supported.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][3]["items"].append({
    "item_number": 2,
    "reasoning": "The reviewer criticizes the Bernards model used in the paper for assuming a constant transconductance derived from DC parameters, neglecting intrinsic frequency dependence due to ionic transport limitations. The paper's equations confirm the use of a static gm parameter, attributing roll-off solely to the RC circuit of the electrolyte and channel capacitance. Devil's advocate for correctness: Some experts might argue the RC circuit does capture the primary ionic charging limitation for these dimensions. However, the reviewer's claim that the model omits frequency-dependent volumetric capacitance/transconductance intrinsic to the polymer is physically accurate. Devil's advocate for significance: This is a debate over model complexity, but since the paper's main contribution is a quantitative gain model, its accuracy at high frequencies is significant. The point is well-evidenced.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][3]["items"].append({
    "item_number": 3,
    "reasoning": "The reviewer correctly points out that the single-cell experiment uses an n=1 sample size and derives uncertainty from temporal replicates, not biological ones. This matches the paper's methodology and Figure 3 caption. Devil's advocate for correctness: The factual claim is unambiguously true. Devil's advocate for significance: Some experts might argue that for a device proof-of-concept, a single successful demonstration is sufficient and biological variability is out of scope. However, most experts would agree that claiming a specific quantitative gain with an error bar from temporal averaging on one cell overstates the method's proven reliability. This is a significant methodological limitation. The evidence is specific and sufficient.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

# gpt-5.2
data["reviewers"][4]["items"].append({
    "item_number": 1,
    "reasoning": "The reviewer accurately identifies that the paper fits the fOECT parameter for each device to match the data, which compromises the model's independent predictive power. The paper states this explicitly. Devil's advocate for correctness: The paper admits to fitting this parameter, so the factual basis is indisputable. Devil's advocate for significance: Could this be considered a standard calibration step? The reviewer convincingly argues that for a paper claiming a 'quantitative model' that 'correctly predicts' performance, fitting a core partition factor per device masks potential systematic errors and weakens the claim of establishing general guidelines. The evidence quotes the text and cites a paper showing direct measurement is possible. Experts would agree this is a correct and significant point.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][4]["items"].append({
    "item_number": 2,
    "reasoning": "The reviewer correctly asserts that the dielectric microparticle surrogate fails to model the electrical properties of real cells, specifically the membrane capacitance and paracellular resistance. The paper's methodology relies heavily on this rigid bead. Devil's advocate for correctness: The paper calls it a 'first order approximation', but the reviewer's point that this approximation misses critical high-frequency impedance components is factually sound. Devil's advocate for significance: Since the device optimization is based on this surrogate, amplifying the 'wrong' impedance component could lead to suboptimal design for actual cells. This makes the criticism highly constructive and significant. The evidence is robust, citing established ECIS methodology. Experts would agree on its correctness and significance.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][4]["items"].append({
    "item_number": 3,
    "reasoning": "The reviewer highlights the lack of biological replicates, correctly noting that the paper's headline error bar (±0.9 dB) comes from n=4 temporal measurements on a single cell. Devil's advocate for correctness: The factual claim is undeniable as the paper explicitly states the error bars are from temporal averages. Devil's advocate for significance: As with similar criticisms, one could argue a device paper only needs a proof-of-concept. However, reporting a tight error bar implies statistical robustness that is not actually present. The reviewer's explanation of biological vs technical replicates makes it clear why this is a significant limitation for a single-cell sensor claim. The evidence is excellent, quoting the caption and NIH guidelines. Experts would agree this is a solid point.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

data["reviewers"][4]["items"].append({
    "item_number": 4,
    "reasoning": "The reviewer correctly observes that the paper claims the gain improves measurement conditions regarding 'noise pickup and digitization' without providing any noise spectra or SNR measurements. The paper only plots AC current amplitudes. Devil's advocate for correctness: Did the paper measure noise? No, the data is absent. Devil's advocate for significance: Is it obvious that a 20 dB gain improves SNR? The reviewer correctly argues that OECTs introduce their own noise (like 1/f noise), so gain does not automatically equal SNR improvement. Claiming a practical benefit without SNR data is a significant gap in the evaluation. The evidence is strong, citing OECT noise literature. Experts would agree this is a correct and significant criticism with sufficient evidence.",
    "correctness": "Correct",
    "significance": "Significant",
    "evidence": "Sufficient",
    "prediction_of_expert_judgments": "correct_significant_sufficient"
})

with open("litellm_proxy__gemini__gemini_3_1_pro_preview_paper81_metareview.json", "w") as f:
    json.dump(data, f, indent=2)

print("JSON file created.")
