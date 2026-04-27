import json

data = {
    "paper_id": 84,
    "reviewers": []
}

def add_reviewer(rev_id, items_data):
    items = []
    for i, d in enumerate(items_data):
        items.append({
            "item_number": i + 1,
            "reasoning": d[0],
            "correctness": d[1],
            "significance": d[2],
            "evidence": d[3],
            "prediction_of_expert_judgments": d[4]
        })
    data["reviewers"].append({
        "reviewer_id": rev_id,
        "items": items
    })

# Human 1
h1 = []
# 1
h1.append([
    "The reviewer correctly notes the absence of quantified adhesion force data for the bio-adhesive layer in wet/dry environments, which is important for long-term stability of the implantable robot. However, an expert might argue that the paper uses a commercial cyanoacrylate adhesive (3M Vetbond) whose properties are known, leading to disagreement on whether this criticism is strictly 'Correct' or required. I will predict disagreement on correctness.",
    "Correct", "Significant", "Sufficient", "disagree_on_correctness"
])
# 2
h1.append([
    "The reviewer correctly points out that the actuation hydrogel's thickness is missing in the text, which is a significant factor for actuation performance. The paper details other layer thicknesses but not the hydrogel. This is a clear, verifiable omission of an important parameter.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 3
h1.append([
    "The reviewer claims that the three arms in Figure 3N have the same composition, which is factually incorrect. The text explicitly states that each arm contains different sensors (optical, thermal, strain). Since the premise is factually wrong, the criticism is not correct.",
    "Not Correct", None, None, "incorrect"
])
# 4
h1.append([
    "The reviewer raises a valid concern about the stability of the PAAm hydrogel pressure sensor in wet environments due to swelling, which the paper does not address. This is a significant issue for implantable sensors.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 5
h1.append([
    "The reviewer notes that hydrogel adhesion is discussed but swelling and degradation in wet environments over long periods are not fully addressed. This is a valid, significant concern for hydrogel-based implants.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 6
h1.append([
    "The reviewer accurately observes that if the hydrogel wraps around tissue, the electrodes on the e-skin layer may be physically separated from the tissue by the hydrogel, hindering electrical stimulation. This points out a significant missing detail in the device's structural interface design.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 7
h1.append([
    "The reviewer points out an apparent inconsistency in Figure 5G regarding the timing of the electrical stimulation relative to bladder volume. Whether this is an error in the figure or a misunderstanding of the control logic, pointing out the inconsistency is a valid, significant critique.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 8
h1.append([
    "The reviewer questions how the electronic components at the end of the arms form a strong coupling to living tissues. The paper does mention hydrogel adhesiveness, but the reviewer's skepticism about the strength and mechanism for the electronic components is a valid concern for the design. Some experts might disagree on whether this is fully addressed or not.",
    "Correct", "Significant", "Sufficient", "disagree_on_correctness"
])
# 9
h1.append([
    "The reviewer correctly notes that electrical stimulation for voiding is used for an 'underactive' bladder, not an 'overactive' one. This is a significant terminology error that affects the medical motivation.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 10
h1.append([
    "The reviewer claims there is no motivation for using the diverse materials. This is factually incorrect, as the paper explicitly discusses the motivation (e.g., biocompatibility, sensitivity, wetting behavior) for each material used.",
    "Not Correct", None, None, "incorrect"
])
# 11
h1.append([
    "The reviewer correctly notes that while conformal contact is claimed, there is no direct comparison of mechanical modulus between the devices and tissues. This is an important parameter for matching tissue compliance.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 12
h1.append([
    "The reviewer raises a valid concern about potential signal interference in the multi-layer stacked sensors (Figure S12B), which the paper does not address. This is a significant engineering challenge in multi-modal sensors.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 13
h1.append([
    "The reviewer claims there is no description for Figures 3H and 3I. This is factually incorrect, as both the main text and the figure captions describe these figures.",
    "Not Correct", None, None, "incorrect"
])
# 14
h1.append([
    "The reviewer points out that information about SI figures is included in main figure captions, suggesting it should be removed. This is a minor stylistic preference.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 15
h1.append([
    "The reviewer complains that the order of the SI figures is a 'total mess'. This is a vague claim that lacks specific examples to verify easily. Experts would likely find the evidence insufficient, and might disagree on correctness if some figures are in order.",
    "Correct", "Marginally Significant", "Requires More", "correct_marginal_insufficient"
])
add_reviewer("Human_1", h1)

# Human 2
h2 = []
# 1
h2.append([
    "The reviewer claims the article lacks detailed information about the relationships between pH, pressure, and resistance. This is factually incorrect, as the paper explicitly presents these linear correlations in the text and figures.",
    "Not Correct", None, None, "incorrect"
])
# 2
h2.append([
    "The reviewer suggests citing specific references regarding stable tissue-interfacing performance. Missing specific references is a minor issue that does not fundamentally alter the paper's claims.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 3
h2.append([
    "The reviewer claims that motor control and sensor functions operate independently, questioning the 'responsiveness'. This is factually incorrect, as the paper describes a closed feedback loop where sensing regulates actuation.",
    "Not Correct", None, None, "incorrect"
])
# 4
h2.append([
    "The reviewer asks how the variation in curvature among different organs was taken into account to avoid squeezing too much. The paper discusses varying power to modulate bending and adapting designs to specific cases. Thus, claiming this was not considered is borderline, leading to disagreement on correctness.",
    "Correct", "Significant", "Requires More", "disagree_on_correctness"
])
# 5
h2.append([
    "The reviewer asks why the parallel strips result in twisting. The paper explicitly explains that this is due to concentrated internal stresses leading to saddle-like curvature. Thus, the premise that it's unexplained is not correct.",
    "Not Correct", None, None, "incorrect"
])
# 6
h2.append([
    "The reviewer asks if there is a particular advantage in the solution-based approach over transfer methods. The paper explicitly lists the advantages (surpassing integrative complexity of 3D printing, versatility). The reviewer missed this.",
    "Not Correct", None, None, "incorrect"
])
# 7
h2.append([
    "The reviewer claims that since PNIPAM actuates around 35-45C, it wouldn't maintain its shape at normal body temperature. The paper explicitly addresses this by incorporating AAm to raise the LCST. The reviewer missed this detail.",
    "Not Correct", None, None, "incorrect"
])
# 8
h2.append([
    "The reviewer asks a speculative question about whether the composite structure might result in lower performance compared to a 'full composite'. This is a vague, unsupported query rather than a verifiable factual criticism.",
    "Correct", "Significant", "Requires More", "disagree_on_correctness"
])
# 9
h2.append([
    "The reviewer claims that handling nanomaterials after laser patterning is difficult and asks how surrounding material was removed. The paper states that PI is spin-coated over the materials before final cutting, making handling straightforward. The premise is incorrect.",
    "Not Correct", None, None, "incorrect"
])
# 10
h2.append([
    "The reviewer asks for clarification in the caption regarding whether the SEM images are surface or cross-section, and suggests cross-sectional TEM to confirm the absence of voids. This is a valid, significant request for clarifying material characterization.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 11
h2.append([
    "The reviewer claims that the temperature distribution in Figure 3E does not seem uniform, contradicting the paper's claim. This is a subjective interpretation of an image, which experts would likely disagree on.",
    "Correct", "Significant", "Sufficient", "disagree_on_correctness"
])
# 12
h2.append([
    "The reviewer asks what the reason is for the continuous change in bending force over time in Figures 3F and 3G. Neither of these figures shows bending force over time, making the premise factually incorrect.",
    "Not Correct", None, None, "incorrect"
])
# 13
h2.append([
    "The reviewer asks if it is necessary to keep the heater on continuously to maintain bending. This is a valid question pointing to a potential practical limitation of thermal actuation, though an expert might view it as just a question.",
    "Correct", "Significant", "Requires More", "disagree_on_correctness"
])
# 14
h2.append([
    "The reviewer points out that the dual-axis graphs are difficult to comprehend intuitively. This is a valid criticism of data presentation that is marginally significant.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 15
h2.append([
    "The reviewer raises a valid question about the efficiency of heat transfer and dissipation to ensure tissue safety. However, the paper does discuss minimal heat damage. Experts might disagree on whether this criticism is entirely correct or if it demands more evidence.",
    "Correct", "Significant", "Requires More", "disagree_on_correctness"
])
# 16
h2.append([
    "The reviewer asks if the thermal sensor can accurately measure tissue temperature while the heater is applying temperature. This is a valid concern about potential interference, but experts might disagree on whether it constitutes a factual error in the paper.",
    "Correct", "Significant", "Requires More", "disagree_on_correctness"
])
# 17
h2.append([
    "The reviewer asks about the difference in the transfer method in S12b compared to the dual printing method in S16. The paper does not use these terms (it uses layer-by-layer stacking), so the reviewer seems confused. Not correct.",
    "Not Correct", None, None, "incorrect"
])
# 18
h2.append([
    "The reviewer asks how actuation was implemented based on sensing feedback. The paper explicitly describes using a temperature sensor to form a closed feedback loop for robotic control. The reviewer missed this.",
    "Not Correct", None, None, "incorrect"
])
# 19
h2.append([
    "The reviewer points out a theoretical inconsistency in Figure S31D, where output power allegedly increases with distance, contradicting the coupling coefficient trend. This is a specific, significant observation of a potential error in the data.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 20
h2.append([
    "The reviewer correctly notes that PNIPAM loses adhesive strength at higher temperatures (above LCST), contradicting the paper's reliance on adhesion during actuation. This is a significant point supported by domain knowledge.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 21
h2.append([
    "The reviewer notes that the use of Au electrodes for stimulation diverges from the other sensing materials used. This is a valid, minor observation.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 22
h2.append([
    "The reviewer points out that the electrical stimulation off voltage in Figure 5G is not 0.0 V and asks for clarification on the correlation. This is a specific, significant observation requiring clarification.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 23
h2.append([
    "The reviewer raises valid concerns about the risk of delamination when body temperature decreases and the potential adverse effects of continuous electrothermal stimulation. These are significant practical issues for the implant's design.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 24
h2.append([
    "The reviewer notes that T1 and T2 in Figure 6F are not explained in the caption. This is a specific, missing detail that is marginally significant.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 25
h2.append([
    "The reviewer observes that the ECG traces in Figure 6G do not show actual cardiac capture following the pacing spikes, contradicting the claim of effective pacing. This is a significant domain expert observation.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 26
h2.append([
    "The reviewer points out that Figures 6M and 6N only show changes in frequency, while strain sensors should ideally show resistance (amplitude) changes. This is a significant observation regarding the sensor data presentation.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 27
h2.append([
    "The reviewer notes that the paper lacks details on the system used to generate water flow and the properties of the rubber tube in the blood vessel simulation. This is a correct observation of missing methodological details.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 28
h2.append([
    "The reviewer points out that a single-day stimulation test is insufficient to determine the effects of continued stimulation over a longer period. This is a valid, significant concern for implant safety.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 29
h2.append([
    "The reviewer notes that the 2-week histological analysis does not clarify if the device remained operable and properly attached after withstanding repeated pulsations. This is a valid question regarding long-term device durability.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 30
h2.append([
    "The reviewer points out a typo in the caption of Figure S11a (AgNW/PI instead of AgNW/PDMS). This is a minor but correct observation.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 31
h2.append([
    "The reviewer spots a spelling error (PINPAM instead of PNIPAM). This is a valid, minor typographical correction.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 32
h2.append([
    "The reviewer spots a spelling error in Figure S51D (Basi instead of Basic). This is a valid, minor typographical correction.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
# 33
h2.append([
    "The reviewer notes that there should be a space before units in the text and figures. This is a valid, minor formatting correction.",
    "Correct", "Marginally Significant", "Sufficient", "correct_marginal_sufficient"
])
add_reviewer("Human_2", h2)

# claude-opus-4-5
c = []
# 1
c.append([
    "The reviewer correctly notes that the 14-day histological study presented in the supplementary materials is insufficient to support the paper's claim of 'long-term operation' for an implantable device, which typically requires much longer evaluation.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 2
c.append([
    "The reviewer points out that the required heating to 40°C exceeds the standard safety threshold of 2°C above body temperature for implants, posing a risk of thermal damage. This is a highly significant and well-supported safety concern.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 3
c.append([
    "The reviewer notes that the cardiac strain measurements are not quantitatively validated against gold-standard clinical metrics like ejection fraction, weakening the claim of quantifying cardiac contractility.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 4
c.append([
    "The reviewer accurately observes that many of the organ-level applications are validated on artificial phantom models rather than in vivo models, which limits the translational validity of the claims.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 5
c.append([
    "The reviewer points out that while the paper acknowledges decreases in wireless power transfer efficiency under deformation, it does not quantitatively address whether the remaining power is sufficient for reliable operation in dynamic in vivo environments.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
add_reviewer("claude-opus-4-5", c)

# gemini-3.0-pro-preview
g = []
# 1
g.append([
    "The reviewer correctly notes that the required elevated temperature of 40°C poses a severe risk of thermal damage, which violates safety margins for implants operating near the 37°C body temperature.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 2
g.append([
    "The reviewer observes that the reported 80% wireless power transfer efficiency is unrealistically high for an implantable scenario and likely represents an idealized measurement, which is misleading for practical application.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 3
g.append([
    "The reviewer correctly points out that bulk PNIPAM hydrogels inherently actuate slowly, contradicting the paper's claims of 'real-time' control and spatiotemporal precision for rapid processes like cardiac pacing.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 4
g.append([
    "The reviewer raises valid long-term safety concerns regarding the potential cytotoxicity and leakage of Silver Nanowires, which are not adequately addressed by the short-term biocompatibility tests.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 5
g.append([
    "The reviewer notes the significant mechanical mismatch between the stiff Polyimide (PI) matrix and soft biological tissues, which contradicts the claims of tissue-matching softness.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
add_reviewer("gemini-3.0-pro-preview", g)

# gpt-5.2
gpt = []
# 1
gpt.append([
    "The reviewer correctly points out that the core claims regarding organ-level implant utility are largely based on simplified benchtop models rather than the intended physiological environments, weakening the validity of the broad claims.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 2
gpt.append([
    "The reviewer accurately observes that the paper infers 'high biocompatibility' primarily from conformal fit, which is an over-interpretation since conformality alone does not guarantee long-term biological safety.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 3
gpt.append([
    "The reviewer notes that the paper claims safety for its wireless power transfer and electrothermal actuation but lacks tissue-relevant dosimetry and thermal safety evidence to substantiate this claim against heating risks.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
# 4
gpt.append([
    "The reviewer points out that the strain-sensor readouts are not calibrated to established physiological metrics of contractility, meaning the claim of 'quantifying cardiac contractility' is not adequately supported.",
    "Correct", "Significant", "Sufficient", "correct_significant_sufficient"
])
add_reviewer("gpt-5.2", gpt)

with open("/Users/seungonekim/Documents/vscode/cmu-paper-reviewer/peerreview_bench/metareview_bench/results/litellm_proxy__gemini__gemini_3_1_pro_preview/litellm_proxy__gemini__gemini_3_1_pro_preview_paper84_metareview.json", "w") as f:
    json.dump(data, f, indent=2)

