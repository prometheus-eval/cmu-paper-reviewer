# Selected Paper Statistics

## 1. Subfield Distribution

**Q1a:** The corpus of 82 papers spans **27 distinct mid-level categories** under 3 broad subject areas, following the Nature Communications Subject Taxonomy.

**Q1b:**

Broad subject areas (3 categories):
- Physical Sciences: 38 papers
- Biological Sciences: 30 papers
- Health Sciences: 14 papers

Mid-level categories (27 distinct, Nature Communications Subject Taxonomy):
- Physics: 7
- Neuroscience: 7
- Optics and photonics: 7
- Engineering: 6
- Astronomy and Planetary Science: 5
- Microbiology: 5
- Medical Research: 5
- Chemistry: 4
- Neurology: 4
- Nanoscience and Technology: 4
- Energy Science and Technology: 3
- Ecology: 3
- Risk factors: 3
- Computational Biology and Bioinformatics: 3
- Drug Discovery: 2
- Materials Science: 2
- Genetics: 2
- Systems Biology: 1
- Biomarkers: 1
- Biophysics: 1
- Biological techniques: 1
- Biotechnology: 1
- Molecular biology: 1
- Immunology: 1
- Evolution: 1
- Plant Sciences: 1
- Oncology: 1

Paper subjects (per-paper, 172 distinct tags from Nature article metadata, 76 papers with metadata):
- Electrical and electronic engineering: 6
- Epilepsy: 4
- Machine learning: 4
- Biomedical engineering: 4
- Computational chemistry: 3
- Computer science: 3
- Computational neuroscience: 3
- Translational research: 3
- Solar physics: 3
- Network models: 3
- Microbial ecology: 3
- Software: 3
- Batteries: 3
- Polymers: 3
- Nanophotonics and plasmonics: 3
- Chemical physics: 2
- Method development: 2
- Molecular dynamics: 2
- Computational methods: 2
- Atomistic models: 2
- Cognitive neuroscience: 2
- Diseases of the nervous system: 2
- Consciousness: 2
- Sleep: 2
- Astrophysical magnetic fields: 2
- Astrophysical plasmas: 2
- Magnetically confined plasmas: 2
- Magnetospheric physics: 2
- Microbial communities: 2
- Computational models: 2
- Risk factors: 2
- Biodiversity: 2
- Neurological disorders: 2
- Bacterial infection: 2
- Synthetic biology: 2
- SARS-CoV-2: 2
- Environmental impact: 2
- Epidemiology: 2
- Fibre optics and optical communications: 2
- Imaging and sensing: 2
- Evolutionary ecology: 2
- Evolutionary genetics: 2
- Genome informatics: 2
- Mutation: 2
- Drug development: 2
- Drug discovery and development: 2
- Sensors and biosensors: 2

The remaining 125 tags appear in 1 paper each. Papers typically carry 2-5 subject tags.

**Q1c:** We use the Nature Communications Subject Taxonomy, a two-level hierarchy: 3 broad subject areas (Physical Sciences, Biological Sciences, Health Sciences) with 27 mid-level categories. Each paper is assigned exactly one mid-level category. Additionally, each paper carries 2-5 fine-grained subject tags from Nature's article metadata (172 distinct tags across 76 papers with metadata).

---

## 2. Peer Review Composition

**Q2a:** Each paper was reviewed by a mean of **2.9 original peer reviewers** (from the Nature peer review files). Range: 1-5. Distribution: 1 reviewer (2 papers), 2 reviewers (24 papers), 3 reviewers (56 papers), 4 reviewers (15 papers), 5 reviewers (2 papers). Median: 3. We cap to max 3 reviewers in case there are same or more than 4 reviewers for the paper for our study.

**Q2b:** The peer review files downloaded from Nature include **all rounds** of review (initial review, author rebuttals, revision requests, and subsequent reviewer responses). They are the complete "Peer Review File" PDFs published by Nature Communications and other Nature Portfolio journals as part of their transparent peer review policy. For our study, we only use the peer review data for the initial round, since we do not have intermediate versions of papers on researchsquare.

**Q2c:** All reviewer comments from all rounds are included in the peer review files. However, our expert annotators evaluated review items extracted from the AI and human reviews generated for each paper (3 human reviewers + 3 AI reviewers per paper), not the original Nature peer review comments. The original peer review files serve as context for understanding the paper's review history but are not the direct subject of annotation.

---

## 3. Publication Outcome

**Q3a:** All 82 papers were **accepted and published** in Nature-family journals. Every paper has a DOI and a confirmed publication date. The journals are: Nature Communications (73), Nature (2), Nature Computational Science (2), Nature Ecology & Evolution (2), Nature Methods (1), Nature Physics (1), Nature Microbiology (1).

**Q3b:** N/A -- all papers were published at the time of data collection.

---

## 4. Paper Length Statistics

**Q4a:** Mean manuscript length: **8,820 words** (from the markdown-converted preprint content, which includes main text, figure captions, and methods but excludes references and supplementary).

**Q4b:** Range: **3,787 - 22,830 words**. Median: 8,500 words.

**Q4c:**

**Q4d:** 47 out of 78 papers in PeerReview Bench (60%) include supplementary materials in their file tree. 

---

## 5. Peer Review Length Statistics

These statistics are for the review items in our expert annotation dataset (the structured review items evaluated by domain scientists), not the original Nature peer review files.

**Q5a:** Average total word count per human review (sum of all extracted items): **632 words**. Median: 490. Range: 25-4,645.

**Q5b:** Average number of distinct review items per human review: **11.5 items**. Median: 8. Range: 1-73. For AI reviews: mean 5.6 items, median 5, range 2-10 (capped at 5 per the reviewer prompt, but some AI reviewers produced up to 10 in earlier runs).

**Q5c:** Shortest human review: 25 words (1 item). Longest: 4,645 words (73 items).

---

## 6. Temporal Distribution

**Q6a:** Publication year distribution:

| Year | Papers |
|------|--------|
| 2020 | 1      |
| 2021 | 6      |
| 2022 | 18     |
| 2023 | 18     |
| 2024 | 25     |
| 2025 | 14     |

Range: January 2020 - October 2025.

**Q6b:** 2024 has the most papers (25), followed by 2022 and 2023 (18 each). 2020 has the fewest (1 paper). The distribution skews toward more recent publications, reflecting both the availability of transparent peer review files and annotator selection of recent papers in their fields.

---

## 7. Code and Data Availability

**Q7a:** **66 out of 76 papers with metadata** (87%) provide publicly accessible source code links in their Nature metadata. For the 78-paper PeerReview Bench subset, 56 papers (72%) include code files in their submitted preprint directories.

**Q7b:** **70 out of 76 papers with metadata** (92%) include source data links. 47 out of 78 PeerReview Bench papers (60%) include supplementary data files in their preprint directories.

**Q7c:** All AI reviewer agents have filesystem access to the paper's complete preprint directory (main text, images, code, supplementary materials) via the OpenHands SDK. Papers with code allow agents to inspect and run analysis scripts; papers with only PDFs/images limit agents to reading and visual inspection. The reviewer prompt does not change based on code availability -- the agent autonomously decides which tools to use.

---

## 8. Figure and Table Counts

**Q8a:** Average number of main-text figures per paper: **6.0**. Median: 5. Range: 0-21.

**Q8b:** Average number of main-text tables per paper: **1.3**. Median: 1. Range: 0-7.

**Q8c:** All 82 papers include at least some visual content (figures, diagrams, or tables). AI reviewers receive the paper's images via multimodal input (all models used are vision-capable). The figure/table counts above are estimated from references in the paper text (e.g., "Fig. 3", "Table 2"), not from counting image files directly.

---

## 9. Author and Institutional Diversity

**Q9a:**

**Q9b:**

---

## 10. Selection Process Details

**Q10a:** Paper selection was **annotator-driven**. We first recruited domain expert annotators, then each annotator selected papers from their own subfield that met the eligibility criteria.

**Q10b:** We first recruited annotators and then asked them to find papers in their subfields. Each annotator identified 1-5 papers they had sufficient domain expertise to evaluate.

**Q10c:** The initial pool of eligible papers was identified through Nature Portfolio's transparent peer review initiative. Papers were required to meet three criteria: (1) published in a Nature-family journal with an open-access CC BY 4.0 license, (2) a transparent peer review file publicly available (Nature Communications and select other Nature journals publish these), and (3) a preprint version available on Research Square (Nature's preprint server) so that the AI reviewer could be given the pre-publication manuscript rather than the final published version. Annotators searched Research Square and Nature journal websites to find papers in their domain meeting these criteria.

---

## 11. Review File Processing

**Q11a:** Peer review files were downloaded as **PDFs** from Nature's website (the "Peer Review File" supplement published alongside each paper). These were converted to markdown for processing. The review items were then extracted and structured into the annotation schema (reviewer_id, item_number, review_content, review_claim, review_evidence, review_cited_references).

**Q11b:** We kept **only reviewer comments**. Editor decision letters, editorial notes, and author rebuttals were excluded from the review items presented to annotators. However, the full peer review file (including editor comments and author responses) was available to annotators as context if needed.

**Q11c:** Reviewers were **anonymized** in the original Nature peer review files (labeled as "Reviewer #1", "Reviewer #2", etc.). In our dataset, human reviewers are further anonymized to generic labels (Human_1, Human_2, Human_3) that are consistent within a paper but do not correspond to the same person across papers.
