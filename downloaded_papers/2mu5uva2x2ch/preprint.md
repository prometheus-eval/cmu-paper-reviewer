# UniversalRAG: Retrieval-Augmented Generation over Corpora of Diverse Modalities and Granularities

Anonymous ACL submission

# Abstract

Retrieval-Augmented Generation (RAG) has shown substantial promise in improving factual accuracy by grounding model responses with external knowledge relevant to queries. However, most existing approaches are limited to a text-only corpus, and while recent efforts have extended RAG to other modalities such as images and videos, they typically operate over a single modality-specific corpus. In contrast, real-world queries vary widely in the type of knowledge they require, which a single type of knowledge source cannot address. To address this, we introduce UniversalRAG, designed to retrieve and integrate knowledge from heterogeneous sources with diverse modalities and granularities. Specifically, motivated by the observation that forcing all modalities into a unified representation space derived from a single aggregated corpus causes a modality gap, where the retrieval tends to favor items from the same modality as the query, we propose modality-aware routing, which dynamically identifies the most appropriate modality-specific corpus and performs targeted retrieval within it, and further justify its effectiveness with a theoretical analysis. Moreover, beyond modality, we organize each modality into multiple granularity levels, enabling fine-tuned retrieval tailored to the complexity and scope of the query. We validate UniversalRAG on 10 benchmarks of multiple modalities, showing its superiority over various modality-specific and unified baselines.

# 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable performance across various tasks, and have been widely adopted to assist users in everyday life (Anil et al., 2023; OpenAI, 2025). However, LLMs often generate factually incorrect or misleading information, especially on topics they were less or not exposed to during training (Zhang et al., 2023; Huang et al., 2025). To address this, Retrieval-Augmented Generation (RAG)

![img-0.jpeg](img-0.jpeg)
Figure 1: Conceptual illustration comparing existing RAG strategies with our proposed UniversalRAG.

has emerged as a promising approach, which allows the model responses to be grounded in the query-relevant knowledge retrieved from external knowledge sources, enhancing factual accuracy (Lewis et al., 2020; Gao et al., 2024; Chen et al., 2024a).

Despite its effectiveness, existing approaches are typically designed for a single corpus and modality, limiting their ability to address queries that require diverse knowledge sources. In practice, as shown in Figure 1, user queries vary widely in the type of knowledge they require: some are best answered using text (e.g., surface-level facts), others demand visual understanding from images or videos (spatial or temporal cues), and still others require combinations of these modalities. Yet, the field of RAG primarily originates with a textual corpus (Lewis et al., 2020; Jiang et al., 2023; Yan et al., 2024), and although recent efforts have expanded it to modalities beyond text (such as images and videos) (Riedler and Langer, 2024; Abootorabi et al., 2025; Jeong et al., 2025), existing RAG methods (when considered individually) are typically modality- and corpus-specific; therefore, they may be suboptimal to serve as a universal, one-for-all framework that can flexibly handle the wide range of queries, whose knowledge requirements vary.

![img-1.jpeg](img-1.jpeg)
Figure 2: t-SNE plot of the unified embedding space.

In this work, we present UniversalRAG, a novel RAG framework that brings together knowledge distributed across multiple modality-specific corpora, and leverages them to generate grounded responses to queries in a universal workflow. To operationalize this, one straightforward approach might be to aggregate all entries from the collected, heterogeneous knowledge corpora, and embed them into a unified space using a multimodal encoder (which is typically trained to align inputs from different modalities if they are semantically similar). However, despite such alignment efforts, we find that this strategy suffers from modality gaps (Zhang et al., 2025b; Bolya et al., 2025; Meng et al., 2025), the tendency that inputs are clustered based on their modality rather than their semantic meaning (visualized in Figures 2 and 7). As a result, retrieval becomes biased toward knowledge sources that share the same modality as the query, overlooking relevant content from other modalities.

To address this challenge, rather than forcing all modalities into a single embedding space, we take a different direction and introduce modality-aware routing. UniversalRAG predicts its modality requirements and routes retrieval to the corresponding modality-specific corpora (potentially multiple, when the query calls for cross-modal evidence), after which the retrieved knowledge is jointly used for grounding. Notably, this strategy not only sidesteps modality gaps by avoiding every cross-modal comparison but also enables seamless integration of new modalities by extending the routing logic without modifying existing modality-specific retrievers.

Beyond modality, data granularity (i.e., the size or unit of each entry in the corpus) also affects retrieval precision and generation quality (Chen et al., 2024b; Zhong et al., 2025), since different queries benefit from different granularities even within the same modality: overly fine-grained entries can dilute context, while overly coarse ones may bundle unrelated information. For example, complex analytical questions may require full documents or videos, while simple factoid questions are better served with a single paragraph or short video clip.

To accommodate this, we further decompose each modality into multiple granularity levels, organizing them into distinct corpora: documents are additionally segmented into paragraphs and stored in a paragraph-level corpus, and similarly, full-length videos are divided into short clips and stored, while images are kept intact since they are inherently piecemeal. Overall, with these modality- and granularity-aware corpora (including paragraphs, documents, tables, images, clips, and videos) in place, as well as an additional no-retrieval option to efficiently handle straightforward queries (that require no external knowledge), our UniversalRAG dynamically routes each query to the most relevant knowledge sources, ultimately supporting the diverse information needs of real-world users.

We validate UniversalRAG on 10 datasets spanning diverse modalities and granularities, where it outperforms all baselines by large margins on average, confirming its effectiveness in handling diverse types of queries. Moreover, UniversalRAG improves efficiency via modality-aware retrieval and appropriate granularity selection, while maintaining robustness on out-of-distribution datasets.

# 2 Method

We begin by describing the preliminaries.

# 2.1 Preliminaries

Large Vision Language Models Let us first define LLMs, which take an input sequence of tokens  $\pmb{x} = [x_{1}, x_{2}, \dots, x_{n}]$  and generate an output sequence of tokens  $\pmb{y} = [y_{1}, y_{2}, \dots, y_{m}]$ , as follows:  $\pmb{y} = \mathsf{LLM}(\pmb{x})$ , where  $x_{i}$  and  $y_{i}$  are represented in text. Building on top of LLMs, Large Vision-Language Models (LVLMs) extend their capability to support multimodal understanding by incorporating visual encoders (Bai et al., 2023; Chen et al., 2024c; Liu et al., 2024; Li et al., 2025), to process both the textual and visual inputs. Formally, similar to LLMs, LVLMs can be functionalized as  $\pmb{y} = \mathsf{LVLM}(\pmb{x})$ , whose input token  $x_{i}$  is extended to either textual or visual. However, although they are extensively trained, LVLMs themselves are limited to their parametric knowledge, and often struggle with queries that require (fine-grained or up-to-date) information, less or not exposed for training.

Retrieval-Augmented Generation To address the aforementioned limitations of using only the parametric knowledge, RAG has been widely used, whose core idea is to retrieve query-relevant infor

mation from a large corpus and incorporate it into the generation process. Formally, in response to a query $\bm{q}$, a retrieval model $\mathcal{T}$ fetches the relevant context $\bm{c}$ from a corpus $\mathcal{C}$: $\bm{c}=\mathcal{T}(\bm{q};\mathcal{C})$. Then, in the subsequent generation step, LVLM generates a response $\bm{a}$ conditioned on the query and retrieved context: $\bm{a}=\texttt{LVLM}(\bm{q},\bm{c})$. However, most existing RAG approaches are restricted to retrieving from a single corpus consisting of entries from a single modality (such as only the textual documents), limiting their ability to handle diverse queries with knowledge requirements that vary across them.

### 2.2 UniversalRAG

We introduce UniversalRAG that dynamically identifies and routes queries to the most appropriate modality and granularity for targeted retrieval.

#### Challenges in Multi-Corpus Retrieval

To accommodate the diverse knowledge needs of real-world queries, which may involve heterogeneous sources spanning different modalities, we consider a set of modality-specific corpora, where each corpus $\mathcal{C}_{m}$ contains items of modality $m$. Notably, one straightforward approach to operationalize this is to aggregate all corpora into a unified corpus $\mathcal{C}_{\texttt{unified}}=\bigcup_{m\in M}\mathcal{C}_{m}$ and embed all items into a shared space using a multimodal encoder, as for retrieval over a single corpus: $\bm{c}=\mathcal{T}(\bm{q};\mathcal{C}_{\texttt{unified}})$. However, we find this approach suffers from modality gap (Figures 2 and 7), where queries, being textual, align more closely with elements in the text corpus regardless of the modality required. Therefore, instead of forcing all heterogeneous elements into a unified corpus, we propose selectively engaging the most relevant corpora needed for queries.

#### Modality-Aware Retrieval

To sidestep the issue of modality gap (introduced by handling all modalities over the unified space), we instead propose to break down the overall retrieval process into two subsequent stages: (1) identifying the most relevant set of modalities for the query; and (2) performing targeted retrieval within the selected modality-specific corpora. Specifically, instead of aggregating all modality-specific corpora, we preserve each corpus in its original form with an independent embedding space. After that, to direct queries to their best-aligned knowledge sources, we introduce a routing module $\mathcal{R}$ that dynamically predicts the modalities best suited for a query $\bm{q}$, yielding $\mathcal{R}(\bm{q})=M_{\bm{q}}$ where $M_{\bm{q}}$ is the set of modalities for $\bm{q}$. Retrieval is then restricted to the corresponding corpora $\{\mathcal{C}_{m}\mid m\in M_{\bm{q}}\}$, using any off-the-shelf retriever $\mathcal{T}_{m}$ tailored to each modality, thereby avoiding the modality gap issue present in a unified space. Proposition 1 formalizes the advantage of modality-aware routing over unified embeddings, and we provide its proof in Appendix C.

###### Proposition 1.

Let the similarity score in a unified embedding space $\mathcal{C}_{\texttt{unified}}$ be defined as

$s(\bm{q},\bm{c})=\alpha\cdot\mathbf{1}\{m(\bm{q})=m(\bm{c})\}+\beta\cdot r(\bm{q},\bm{c})+\varepsilon,$

where $\alpha>0$ induces modality bias and $r(\cdot,\cdot)$ measures relevance. If $\alpha$ dominates the variability of $r$, modality-aware routing retrieves items from the required modality $m^{*}(\bm{q})$ with higher probability than unified embedding retrieval.

However, while this routing principle mitigates the modality gap, organizing corpora solely by the modality might still be suboptimal since different queries require varying levels of granularity.

#### Granularity-Aware Retrieval

To accommodate the varying complexity and information scope of different queries, we extend UniversalRAG to operate not only across modalities but also across different levels of granularity within each modality. To be specific, rather than treating each modality-specific corpus as a flat collection of items, we organize it into representations at multiple resolutions, enabling retrieval to target either fine-grained details or broader context as required by the query. To reflect this richer organization of corpora, the routing module $\mathcal{R}$ expands its prediction space to include modality-granularity pairs best suited to a query, as well as a no-retrieval option for cases where external context is unnecessary: $\mathcal{R}:Q\rightarrow\{\varnothing\}\cup\mathcal{P}\left(\bigcup_{m\in M}\{m\}\times G_{m}\right),$ where $M$ is the set of modalities and $G_{m}$ is the set of granularities available for modality $m$. Once the router predicts the relevant pairs, retrieval is performed over the corresponding corpora, using retrievers specialized for each modality to obtain the relevant content $\bm{c}$. Finally, the LVLM generates the answer $\bm{a}$ with $\bm{c}$, customized to the modality and granularity for each individual query, thereby enabling the universal, one-for-all RAG framework.

### 2.3 Router Implementation Strategies

A key component of UniversalRAG is the router, which is responsible for determining the optimal modality and granularity of knowledge for a query.

#### Training-based Router

To perform the routing task, we first consider training the available models to predict the appropriate modality–granularity pair for each query. However, since ground-truth labels (for the modality and granularity the query should be routed to) are not available, we leverage inductive biases in existing benchmarks, mapping each dataset to routing targets that match its task characteristics, allowing us to automatically obtain a labeled corpus without manual annotation. We then train open-source LVLMs to serve as the router using a multi-hot label representation and cross-entropy loss. At inference time, the router produces a sigmoid distribution over modality-granularity pairs and returns all configurations whose scores exceed a predefined threshold, enabling cross-modal and multi-granularity retrieval when necessary.

#### Training-free Router

Alternatively, we also explore a training-free approach that leverages the broad knowledge and robust reasoning capabilities of modern frontier models, such as Gemini *(Anil et al., 2023)*. Instead of learning from labeled data, the model is directly prompted to act as a router. To achieve this, we first design the prompt template (used to elicit routing), which describes the objective and includes examples demonstrating how different types of queries correspond to specific retrieval targets (See Figure 8 for details). Then, at inference, the model is prompted with this template to predict the most suitable modality-granularity pairs from a predefined set. This eliminates the need for supervised labels or task-specific training, offering the flexibility to adapt to new domains.

## 3 Experiment

### 3.1 Experimental Setup

We now explain the experimental setup, including datasets, models, and implementation details.

#### Datasets

To evaluate UniversalRAG, we compile a comprehensive benchmark covering RAG tasks across seven modalities and granularities. For the no-retrieval setting, we use MMLU *(Hendrycks et al., 2021)*. For text-based RAG, we include Natural Questions (NQ) *(Kwiatkowski et al., 2019)* for single-hop, paragraph-level retrieval, and HotpotQA *(Yang et al., 2018)* for multi-hop, document-level retrieval. To consider diverse scenarios, we further incorporate HybridQA *(Chen et al., 2020)* for reasoning over text and tables, MRAG-Bench (MRAG) *(Hu et al., 2025)* for image RAG, and WebQA *(Chang et al., 2022)* and InfoSeek *(Chen et al., 2023a)* for cross-modal RAG over text and images. Lastly, for the RAG with videos, we use LVBench *(Wang et al., 2024)* for queries over short or localized video segments, as well as VideoRAG-Wiki and VideoRAG-Synth *(Jeong et al., 2025)* for queries grounded on long-form or complete videos. Please refer to Appendix A for more details.

#### Knowledge Corpora

To support the aforementioned, diverse RAG scenarios with various modalities and granularities, we consider their corresponding corpora. Recall that we define seven routing pathways: None, Paragraph, Document, Table, Image, Clip, and Video, with cross-modal routing allowing queries to span multiple modalities. For the paragraph and document corpora, we use Wikipedia at the levels of paragraphs *(Karpukhin et al., 2020)* and documents *(Jiang et al., 2024b)*. The table corpus is built by collecting tables from the HybridQA benchmark. For the image, we adopt corpora from MRAG-Bench, WebQA, and InfoSeek datasets. Lastly, we construct two video corpora at different scales: a video-level corpus consisting of full-length videos from LVBench and VideoRAG datasets, and a clip-level constructed by segmenting these videos into multiple short clips.

#### Methods

We compare our UniversalRAG to a diverse set of 12 baselines, grouped into four categories. The first is Naïve, which directly answers queries without retrieving external knowledge. In addition, the group of Unimodal RAGs includes ParagraphRAG, DocumentRAG, TableRAG, ImageRAG, ClipRAG, and VideoRAG methods, which retrieve information exclusively from their respective corpora and leverage it for response generation. The third group of Unified Embedding Multimodal RAGs uses multimodal encoders to align different modalities into a shared embedding space for retrieval, including UniRAG *(Sharifymoghaddam et al., 2025)*, GME *(Zhang et al., 2025b)*, PE_{core} *(Bolya et al., 2025)*, and VLM2Vec-V2 *(Meng et al., 2025)*. MultiRAG is included in the last group of Multi-corpus Multimodal RAGs, which performs retrieval over all the available corpora and incorporates the retrieved results for response generation. Notably, as UniversalRAG can be operationalized with different routing strategies, we implement training-based variants, which leverage Qwen3-VL-2B-Instruct *(Bai et al., 2025)*, InternVL3.5-1B *(Wang et al., 2025)*, and T5Gemma 2 270M *(Zhang et al., 2025a)* (fine

Table 1: Results of diverse RAG methods with Qwen3-VL-8B-Instruct across modalities. Bold denotes the best performance and underlined indicates the second-best among UniversalRAG variants, using either trained or training-free routers. R-L and BERT correspond to ROUGE-L and BERTScore, respectively.

[tbl-0.html](tbl-0.html)

![img-2.jpeg](img-2.jpeg)
Figure 3: Comparison of averaged evaluation results across different RAG methods and LVLMs.

tuned on the automatically constructed routing dataset), as well as training-free variants, which prompt GPT-5 (OpenAI, 2025) and Qwen3-VL8B-Instruct (Bai et al., 2025) to select appropriate modality-granularity pairs. Finally, we include an oracle setup (Oracle), which routes each query to its ideal corpora, non-comparable with others.

Implementation Details For response generation, we utilize multiple LVLMs, Qwen3-VL-8B-Instruct (Bai et al., 2025), InternVL3.5-8B (Wang et al., 2025), and Molmo2-4B (Clark et al., 2025). Also, to take advantage of UniversalRAG in routing the retrieval process to the modality-specific corpus, we use modality-specific encoders: Qwen3-Embedding-4B (Zhang et al., 2025c) for text, VLM2Vec-V2 (Meng et al., 2025) for vision, and dense row-level embedding (Ji et al., 2025) with the text encoder for tables. We provide further details (including router training) in Appendix B.

# 3.2 Experimental Results and Analyses

Now we present the overall results across diverse RAG scenarios, followed by a detailed analysis of the observed performance improvements.

Overall Results We present the modality- and granularity-specific results in Table 1, along with the averaged results with different LVLMs in Fig-

ure 3, from which we observe that UniversalRAG consistently achieves the best performance on average. Specifically, in Table 1, the results compared against the unimodal RAG baselines corroborate our hypothesis that retrieving from the modality (or granularity) that aligns best with the information needs of the queries achieves the highest accuracy; however, mismatches between the query and retrieval source results in significant degradation, which supports our claim that considering diverse modalities in the universal workflow is necessary for realistic RAG. Also, the level of granularity within each modality affects performance, suggesting that fine-grained retrieval and generation are necessary. In addition to them, UniversalRAG significantly outperforms unified embedding multimodal RAG baselines, confirming the issue of the modality gap inherent within them (See Figures 2 and 7). Lastly, when compared with the MultiRAG baseline (within the multi-corpus multimodal RAG category), which results in suboptimal performance due to the inclusion of noise from irrelevant modalities in generation, our UniversalRAG remains effective. Its strong performance is due to its core idea around modality-aware routing, enabling the dynamic retrieval from the most relevant modalities and granularities for each query, yielding performance gains despite using several corpora.

![img-3.jpeg](img-3.jpeg)
Figure 4: Distribution of the retrieved data modalities.

![img-4.jpeg](img-4.jpeg)
Figure 5: Retrieval latency per query across corpus sizes.

![img-5.jpeg](img-5.jpeg)
Figure 6: Router accuracy with varying the router model size.

Table 2: Performance comparison of uni-modal and cross-modal approaches across different router models. Among models, GPT-5 is the only training-free router.

[tbl-1.html](tbl-1.html)

Table 3: Modality accuracy (in corpus selection) and recall of retrieved items for retrieval methods. Among UniversalRAG variants, GPT-5 is only training-free router.

[tbl-2.html](tbl-2.html)

Effectiveness of Cross-Modal Retrieval While many queries can be addressed by using a single, most prominent modality, certain tasks benefit from integrating evidence across multiple modalities. For instance, HybridQA requires reasoning that spans both structured tables and accompanying textual sources, while WebQA involves visually grounded questions that pair text with images. Table 2 shows that, compared to uni-modal retrieval, for which each query is routed to a single relevant source, cross-modal retrieval achieves consistently stronger performance. By enabling queries to be routed across multiple modalities, the cross-modal retrieval can leverage complementary evidence that would otherwise be missed by the uni-modal approach. These highlight the effectiveness of UniversalRAG's flexible routing mechanism, which dynamically retrieves information from multiple sources rather than relying on a single modality.

Effectiveness of Modality Routing To investigate the effectiveness of our routing method, we compare the distribution of retrieved modalities for VLM2Vec-V2, GME, and UniversalRAG (with Qwen3-VL-2B) in Figure 4. Using 200 sampled queries per benchmark and normalizing distributions, we find that VLM2Vec-V2 retrieves exclusively text, while GME similarly exhibits a strong bias toward text regardless of the query's required modality, reflecting the modality gap inherent to unified embedding spaces. In contrast, UniversalRAG retrieves more evenly across modalities, indicating that the router effectively mitigates modality

bias and adaptively selects appropriate knowledge sources. This leads to higher modality retrieval accuracy, and consequently, higher retrieval recall, as shown in Table 3. While GME achieves comparable recall on text and image corpora, its inability to accurately retrieve from the correct modality leads to lower recall on multimodal corpora that include videos. Yet, UniversalRAG consistently retrieves from the correct modality, enabling it to achieve higher recall than baselines across all scenarios.

Effectiveness of Multigranularity Given the observed benefits of corpus selection in Table 1, we investigate its impact beyond modality by comparing UniversalRAG at varying levels of granularity $^{1}$ . Table 4 shows that incorporating granularity-aware corpus selection leads to consistent performance gains by avoiding the retrieval of context that is either insufficient (e.g., a short paragraph lacking key entities for multi-hop reasoning) or excessive (e.g., a full video when only a short clip is relevant), both of which can hinder accurate response generation. Also, as additional granularity levels are introduced, we observe further improvements in some cases, though gains are not strictly monotonic across tasks, reflecting the trade-off between context sufficiency and noise. Please see Appendix C.2 for a theoretical analysis supporting these findings.

Efficiency of Modality-Specific Retrieval Beyond accuracy, UniversalRAG also improves effi

Table 4: Performance across different numbers of granularity (#Gn) for training-free router models. The prompt used to route to finer granularities is shown in Figure 9.

[tbl-3.html](tbl-3.html)

ciency by reducing the search space: it leverages modality- and granularity-aware routing to restrict retrieval to only the most relevant sources, instead of querying a unified embedding index that aggregates all modalities into a single mega-corpus. Also, the overhead for routing is small as this cost is outweighed at scale by the size of the search space, leading to sub-linear latency growth as corpus size increases, as shown in Figure 5. Here, UniversalRAG eventually achieves lower latency than unified embedding methods at large corpus sizes, with the gap widening further at very large scales (beyond 10M entries). This scalability makes UniversalRAG a practical solution for real-world applications, where corpora are significantly larger than our experimental settings. We provide an in-depth theoretical analysis of efficiency in Appendix C.3.

Analysis on Router Size To examine whether the routing cost can be further reduced by using smaller models as routers without sacrificing accuracy, we train three models (Wang et al., 2025; Zhang et al., 2025a; Marafioti et al., 2025) ranging from 256M to 4B parameters and measure router accuracy. As shown in Figure 6, router accuracy consistently improves with increasing model size within each architecture, suggesting the scalability of our routing approach. While the largest models achieve near-perfect routing performance, a 1B-parameter model attains approximately  $90\%$  accuracy, indicating that compact models can serve as effective routers in UniversalRAG.

# Generalizability on Out-of-Domain Scenarios

As shown in Table 1, UniversalRAG with trained routers outperforms the training-free router (sometimes even approaching oracle performance), and a natural follow-up question is how these routers behave on unseen, out-of-domain (OOD) datasets. To investigate this, we evaluate on six OOD datasets (detailed in Appendix A.2), with results presented in Tables 5 and 10. In contrast to the in-domain set

Table 5: Router accuracy and generation performance across retrieval methods on two settings. Among UniversalRAG variants, GPT-5 is the only training-free router.

[tbl-4.html](tbl-4.html)

ting, trained routers exhibit noticeable performance degradation, whereas the training-free router generalizes robustly and even surpasses the trained variants. Nevertheless, UniversalRAG remains effective in OOD scenarios and consistently outperforms all baselines, including those using the unified embedding spaces or random modality and granularity assignment, highlighting the benefit of adaptive, modality- and granularity-aware retrieval.

Ensemble Strategy for Robust Routing Building on the trade-off between the high in-domain accuracy of trained routers and the strong OOD generalization of training-free routers, we propose ensemble strategies that leverage their complementary strengths. Specifically, we explore confidence-based ensembling, which uses the trained router's prediction when its confidence exceeds a threshold and otherwise falls back to the training-free router, as well as majority voting, which selects the majority prediction from three routers (training-based and free) with random tie-breaking. Table 5 shows that UniversalRAG with the ensemble routing achieves a robust balance between accuracy and generalization, making it well suited for real-world scenarios with unseen or shifting distributions.

Case Study We present a case study of UniversalRAG in Table 6. The query asks for the number of statues of people on the Michigan Soldiers and Sailors Monument. Both TextRAG and ImageRAG retrieve the relevant and correct evidence; however, each modality alone is insufficient to determine the full count. TextRAG lacks the information needed to aggregate all statues, while ImageRAG suffers from partial occlusion. VideoRAG fails to retrieve relevant evidence, as the video corpus does not contain information useful for this query. In contrast, UniversalRAG routes the query to both the "Paragraph" and "Image" corpora, allowing cross-modal reasoning and correctly identifying all nine statues. More case studies are provided in Appendix F.

Table 6: Case study comparing unimodal RAGs with fixed modality and granularity against UniversalRAG (Ours).

[tbl-5.html](tbl-5.html)

# 4 Related Work

Large Vision Language Models Building on the impressive performance of LLMs (Anil et al., 2023; Hurst et al., 2024), recent studies have extended them to visual domains. Liu et al. (2023) incorporates a CLIP-based (Radford et al., 2021) image encoder to align visual inputs with language representations, followed by models using diverse encoders (Bai et al., 2023; Chen et al., 2024c; Liu et al., 2024) and extensions to video (Li et al., 2025; Wang et al., 2025; Bai et al., 2025). However, despite improved performance on multimodal benchmarks (Mathew et al., 2021; Yue et al., 2024; Li et al., 2024a; Fu et al., 2024) from larger datasets and with improved architectures, LVLMs still often suffer from hallucinations (Huang et al., 2025) when relying solely on parametric knowledge.

Retrieval-Augmented Generation To address the aforementioned limitation of parametric-only models, RAG incorporates external knowledge during response generation. While conventional RAG focuses on the textual corpus (Lewis et al., 2020; Ram et al., 2023), recent work extends it to multimodal sources such as images and videos (Chen et al., 2022; Riedler and Langer, 2024; Jeong et al., 2025). However, these approaches assume a fixed single-modality retrieval, making them less adaptable to real-world queries that may require information from different modalities. Multimodal encoders (Radford et al., 2021; Zhang et al., 2025b; Bolya et al., 2025; Meng et al., 2025) enable unified embedding spaces across modalities, and Sharifymoghaddam et al. (2025) retrieves from such spaces, but often fails to retrieve visual content for text queries. RAG-Anything (Guo et al., 2025) sidesteps this by converting all the multimodal knowledge into textual form, at the cost of heavy preprocessing and loss of modality-specific information. Other approaches (Cui et al., 2024; Liu et al., 2025a) retrieve from all modalities, followed

by extra selection mechanisms, incurring notable computational cost. Lastly, adaptive retrieval strategies (Jeong et al., 2024; Islam et al., 2024; Ding et al., 2024; Yao et al., 2024; Tang et al., 2025) address query diversity but remain restricted to a single corpus (Zhang et al., 2024; Li et al., 2024b).

Retrieval Granularity While most of the existing RAG methods operate at fixed granularity (e.g., full documents, passages, or sentences), real-world queries often require information at varying levels of specificity depending on the knowledge needed, which in turn impacts performance and efficiency in both textual (Chen et al., 2024b; Liu et al., 2025b; Zhong et al., 2025) and video-based retrieval systems (Chen et al., 2023b). In contrast, UniversalRAG performs query-level routing across modality and granularity dimensions, enabling retrieval from the most relevant source at the appropriate level.

# 5 Conclusion

In this paper, we proposed UniversalRAG, a novel RAG framework designed to retrieve from corpora of diverse modalities and granularities. Through a modality- and granularity-aware routing mechanism, UniversalRAG dynamically selects the most suitable knowledge sources for each query, effectively addressing the limitations posed by modality gaps and fixed-granularity retrieval, which we further justify with theoretical results. Empirical evaluations across 10 benchmarks demonstrate that UniversalRAG outperforms both modality-specific and unified baselines, showcasing robust performance across diverse modalities. Also, our analyses highlight the importance of fine-grained retrieval and the complementary strengths of training-free and trained routers. We believe these findings demonstrate the potential of UniversalRAG as an adaptive solution for grounding LVLMs with heterogeneous external knowledge, paving the way for the one-for-all RAG that unifies the fragmented landscape of existing corpus-specific RAGs.

Limitations

The proposed UniversalRAG is designed for leveraging heterogeneous, multimodal corpora at RAG, enabling corpus-aware routing to flexibly utilize modality- and granularity-specific corpora. It is worth noting that the routing mechanism is its central part, and to improve its accuracy, high-quality samples for training may be required; however, existing datasets or benchmarks lack ground-truth labels indicating ideal modality or granularity for each query. Nonetheless, we address this by automatically annotating queries (based on inductive biases inherent in datasets or downstream performance measured with all the available corpora), as detailed in Appendix A. However, since they may contain some noise, constructing high-quality, human-annotated routing datasets would be a valuable direction for future work. Also, due to similar reasons: the absence of annotated data (specifically, the query-granularity pairs), we segment each (text and video) modality into two levels of granularity to obtain supervision signals for router training. Again, collecting more fine-grained annotations that cover a wider range of query-modality and query-granularity pairs would be an exciting direction to expand the applicability of UniversalRAG.

## Ethical Considerations

The proposed UniversalRAG can be seamlessly integrated with any LVLMs and compatible retrieval corpora, reducing hallucination with the corpus-specific routing. However, there can be potential private, harmful, or biased content present in the retrieved or generated outputs, depending on the nature of the underlying corpora or the internalized knowledge within LVLMs. To mitigate such risks, it is recommended to apply safeguard mechanisms and filtering techniques in retrieval and generation, to ensure the safe and responsible deployment.

## References

- [1] Mohammad Mahdi Abootorabi, Amirhosein Zobeiri, Mahdi Dehghani, Mohammadali Mohammadkhani, Bardia Mohammadi, Omid Ghahroodi, Mahdieh Soleymani Baghshah, and Ehsaneddin Asgari. 2025. Ask in any modality: A comprehensive survey on multimodal retrieval-augmented generation. Preprint, arXiv:2502.08826.
- [2] Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M. Dai, Anja Hauth, Katie Millican, David Silver, Slav Petrov, Melvin Johnson, Ioannis Antonoglou, Julian Schrittwieser, Amelia Glaese, Jilin Chen, Emily Pitler, Timothy P. Lillicrap, and 33 others. 2023. Gemini: A family of highly capable multimodal models. arXiv preprint arXiv:2312.11805.
- [3] Jinze Bai, Shuai Bai, Shusheng Yang, Shijie Wang, Sinan Tan, Peng Wang, Junyang Lin, Chang Zhou, and Jingren Zhou. 2023. Qwen-vl: A versatile vision-language model for understanding, localization, text reading, and beyond. Preprint, arXiv:2308.12966.
- [4] Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, and 1 others. 2025. Qwen3-vl technical report. arXiv preprint arXiv:2511.21631.
- [5] Valeriia Bolotova-Baranova, Vladislav Blinov, Sofya Filippova, Falk Scholer, and Mark Sanderson. 2023. WikiHowQA: A comprehensive benchmark for multi-document non-factoid question answering. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5291–5314, Toronto, Canada. Association for Computational Linguistics.
- [6] Daniel Bolya, Po-Yao Huang, Peize Sun, Jang Hyun Cho, Andrea Madotto, Chen Wei, Tengyu Ma, Jiale Zhi, Jathushan Rajasegaran, Hanoona Rasheed, Junke Wang, Marco Monteiro, Hu Xu, Shiyu Dong, Nikhila Ravi, Daniel Li, Piotr Dollár, and Christoph Feichtenhofer. 2025. Perception encoder: The best visual embeddings are not at the output of the network. Preprint, arXiv:2504.13181.
- [7] Brandon Castellano. 2014. PySceneDetect.
- [8] Yingshan Chang, Guihong Cao, Mridu Narang, Jianfeng Gao, Hisami Suzuki, and Yonatan Bisk. 2022. Webqa: Multihop and multimodal QA. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2022, New Orleans, LA, USA, June 18-24, 2022, pages 16474–16483. IEEE.
- [9] Jiawei Chen, Hongyu Lin, Xianpei Han, and Le Sun. 2024a. Benchmarking large language models in retrieval-augmented generation. In Thirty-Eighth AAAI Conference on Artificial Intelligence, AAAI 2024, Thirty-Sixth Conference on Innovative Applications of Artificial Intelligence, IAAI 2024, Fourteenth Symposium on Educational Advances in Artificial Intelligence, EAAI 2014, February 20-27, 2024, Vancouver, Canada, pages 17754–17762. AAAI Press.
- [10] Tong Chen, Hongwei Wang, Sihao Chen, Wenhao Yu, Kaixin Ma, Xinran Zhao, Hongming Zhang, and Dong Yu. 2024b. Dense X retrieval: What retrieval granularity should we use? In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024, Miami, FL, USA, November 12-16, 2024, pages 15159–15177. Association for Computational Linguistics.

Wenhu Chen, Hexiang Hu, Xi Chen, Pat Verga, and William W. Cohen. 2022. Murag: Multimodal retrieval-augmented generator for open question answering over images and text. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pages 5558–5570. Association for Computational Linguistics.
- Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. 2020. HybridQA: A dataset of multi-hop question answering over tabular and textual data. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1026–1036, Online. Association for Computational Linguistics.
- Chen, Yan Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang Wang. 2023a. Can pre-trained vision and language models answer visual information-seeking questions? In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14948–14968, Singapore. Association for Computational Linguistics.
- Chen, Xun Jiang, Xing Xu, Zuo Cao, Yijun Mo, and Heng Tao Shen. 2023b. Joint searching and grounding: Multi-granularity video content retrieval. In Proceedings of the 31st ACM International Conference on Multimedia, MM 2023, Ottawa, ON, Canada, 29 October 2023- 3 November 2023, pages 975–983. ACM.
- Clark, Jieyu Zhang, Zixian Ma, Jae Sung Park, Mohammadreza Salehi, Rohun Tripathi, Sangho Lee, Zhongzheng Ren, Chris Dongjoo Kim, Yinuo Yang, Vincent Shao, Yue Yang, Weikai Huang, Ziqi Gao, Taira Anderson, Jianrui Zhang, Jitesh Jain, George Stoica, Winson Han, and 2 others. 2025. Molmo2: Open weights and data for vision-language models with video understanding and grounding.
- Cui, Keping Bi, Jiafeng Guo, and Xueqi Cheng. 2024. MORE: Multi-mOdal REtrieval augmented generative commonsense reasoning. In Findings of the Association for Computational Linguistics: ACL 2024, pages 1178–1192, Bangkok, Thailand. Association for Computational Linguistics.
- Ding, Liang Pang, Zihao Wei, Huawei Shen, and Xueqi Cheng. 2024. Retrieve only when it needs: Adaptive retrieval augmentation for hallucination mitigation in large language models. Preprint, arXiv:2402.10612.
- Du, Ma, and Yihua Chen. 2025. Matthijs Douze, Alexandr Guzhva, Chengqi Deng, Jeff Johnson, Gergely Szilvasy, Pierre-Emmanuel Mazaré, Maria Lomeli, Lucas Hosseini, and Hervé Jégou. 2025. The faiss library. IEEE Transactions on Big Data, pages 1–17.
- Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Rongrong Ji, and Xing Sun. 2024. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. Preprint, arXiv:2405.21075.
- Gao, Yun Xiong, Xinyu Gao, Kangxiang Jia, Jinliu Pan, Yuxi Bi, Yi Dai, Jiawei Sun, Meng Wang, and Haofen Wang. 2024. Retrieval-augmented generation for large language models: A survey. Preprint, arXiv:2312.10997.
- Guo, Xubin Ren, Lingrui Xu, Jiahao Zhang, and Chao Huang. 2025. Rag-anything: All-in-one rag framework. arXiv preprint arXiv:2510.12323.
- Hembrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021. Measuring massive multitask language understanding. In 9th International Conference on Learning Representations, ICLR 2021, Virtual Event, Austria, May 3-7, 2021.
- Ho, Anh-Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In Proceedings of the 28th International Conference on Computational Linguistics, pages 6609–6625, Barcelona, Spain (Online). International Committee on Computational Linguistics.
- Hu, Jia-Chen Gu, Zi-Yi Dou, Mohsen Fayyaz, Pan Lu, Kai-Wei Chang, and Nanyun (Violet) Peng. 2025. Mrag-bench: Vision-centric evaluation for retrieval-augmented multimodal models. In International Conference on Representation Learning, volume 2025, pages 95558–95581.
- Lu, Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Inf. Syst., 43(2).
- Lin, Aaron Hurst, Adam Lerer, Adam P Goucher, Adam Perelman, Aditya Ramesh, Aidan Clark, AJ Ostrow, Akila Welihinda, Alan Hayes, Alec Radford, and 1 others. 2024. Gpt-4o system card. arXiv preprint arXiv:2410.21276.
- Lin, Shayekh Bin Islam, Md Asib Rahman, K. S. M. Tozammel Hossain, Enamul Hoque, Shafiq Joty, and Md. Rizwan Parvez. 2024. Open-rag: Enhanced retrieval augmented reasoning with open-source large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, Miami,

Florida, USA, November 12-16, 2024, pages 14231–14244. Association for Computational Linguistics.

Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024. Adaptive-rag: Learning to adapt retrieval-augmented large language models through question complexity. In *Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), NAACL 2024, Mexico City, Mexico, June 16-21, 2024*, pages 7036–7050. Association for Computational Linguistics.

Soyeong Jeong, Kangsan Kim, Jinheon Baek, and Sung Ju Hwang. 2025. VideoRAG: Retrieval-augmented generation over video corpus. In *Findings of the Association for Computational Linguistics: ACL 2025*, pages 21278–21298, Vienna, Austria. Association for Computational Linguistics.

Xingyu Ji, Parker Glenn, Aditya G. Parameswaran, and Madelon Hulsebos. 2025. TARGET: Benchmarking table retrieval for generative tasks. *Preprint*, arXiv:2505.11545.

Ting Jiang, Minghui Song, Zihan Zhang, Haizhen Huang, Weiwei Deng, Feng Sun, Qi Zhang, Deqing Wang, and Fuzhen Zhuang. 2024a. E5-v: Universal embeddings with multimodal large language models. *Preprint*, arXiv:2407.12580.

Zhengbao Jiang, Frank F. Xu, Luyu Gao, Zhiqing Sun, Qian Liu, Jane Dwivedi-Yu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Active retrieval augmented generation. In *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023*, pages 7969–7992. Association for Computational Linguistics.

Ziyan Jiang, Xueguang Ma, and Wenhu Chen. 2024b. Longrag: Enhancing retrieval-augmented generation with long-context llms. *Preprint*, arXiv:2406.15319.

Mandar Joshi, Eunsol Choi, Daniel Weld, and Luke Zettlemoyer. 2017. TriviaQA: A large scale distantly supervised challenge dataset for reading comprehension. In *Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 1601–1611, Vancouver, Canada. Association for Computational Linguistics.

Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In *Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pages 6769–6781, Online. Association for Computational Linguistics.

Tom Kwiatkowski, Jennimaria Palomaki, Olivia Redfield, Michael Collins, Ankur Parikh, Chris Alberti, Danielle Epstein, Illia Polosukhin, Jacob Devlin, Kenton Lee, Kristina Toutanova, Llion Jones, Matthew Kelcey, Ming-Wei Chang, Andrew M. Dai, Jakob Uszkoreit, Quoc Le, and Slav Petrov. 2019. Natural Questions: A benchmark for question answering research. *Transactions of the Association for Computational Linguistics*, 7:452–466.

Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-augmented generation for knowledge-intensive nlp tasks. In *Advances in Neural Information Processing Systems*, volume 33, pages 9459–9474.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2025. LLaVA-onevision: Easy visual task transfer. *Transactions on Machine Learning Research*.

Kunchang Li, Yali Wang, Yinan He, Yizhuo Li, Yi Wang, Yi Liu, Zun Wang, Jilan Xu, Guo Chen, Ping Lou, Limin Wang, and Yu Qiao. 2024a. Mvbench: A comprehensive multi-modal video understanding benchmark. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024*, pages 22195–22206. IEEE.

Yangning Li, Yinghui Li, Xinyu Wang, Yong Jiang, Zhen Zhang, Xinran Zheng, Hui Wang, Hai-Tao Zheng, Pengjun Xie, Philip S. Yu, Fei Huang, and Jingren Zhou. 2024b. Benchmarking multimodal retrieval augmented generation with dynamic vqa dataset and self-adaptive planning agent. *Preprint*, arXiv:2411.02937.

Chin-Yew Lin. 2004. ROUGE: A package for automatic evaluation of summaries. In *Text Summarization Branches Out*, pages 74–81, Barcelona, Spain. Association for Computational Linguistics.

Sheng-Chieh Lin, Chankyu Lee, Mohammad Shoeybi, Jimmy Lin, Bryan Catanzaro, and Wei Ping. 2025. MM-EMBED: Universal multimodal retrieval with multimodal LLMS. In *The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025*.

Stephanie Lin, Jacob Hilton, and Owain Evans. 2022. TruthfulQA: Measuring how models mimic human falsehoods. In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pages 3214–3252, Dublin, Ireland. Association for Computational Linguistics.

Haotian Liu, Chunyuan Li, Yuheng Li, and Yong Jae Lee. 2024. Improved baselines with visual instruction tuning. In *IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024*, pages 26286–26296. IEEE.

Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee. 2023. Visual instruction tuning. In *Advances in Neural Information Processing Systems 36: Annual

Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, and Jun Ma. 2025a. Hm-rag: Hierarchical multi-agent multimodal retrieval augmented generation. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, page 2781–2790, New York, NY, USA. Association for Computing Machinery.

Zuhong Liu, Charles-Elie Simon, and Fabien Caspani. 2025b. Passage segmentation of documents for extractive question answering. Preprint, arXiv:2501.09940.

Andrés Marafioti, Orr Zohar, Miquel Farré, Merve Noyan, Elie Bakouch, Pedro Cuenca, Cyril Zakka, Loubna Ben Allal, Anton Lozhkov, Nouamane Tazi, and 1 others. 2025. Smolvlm: Redefining small and efficient multimodal models. arXiv preprint arXiv:2504.05299.

Minesh Mathew, Dimosthenis Karatzas, and C. V. Jawahar. 2021. Docvqa: A dataset for VQA on document images. In IEEE Winter Conference on Applications of Computer Vision, WACV 2021, Waikoloa, HI, USA, January 3-8, 2021, pages 2199–2208. IEEE.

Rui Meng, Ziyan Jiang, Ye Liu, Mingyi Su, Xinyi Yang, Yuepeng Fu, Can Qin, Zeyuan Chen, Ran Xu, Caiming Xiong, and 1 others. 2025. Vlm2vec-v2: Advancing multimodal embedding for videos, images, and visual documents. arXiv preprint arXiv:2507.04590.

Antoine Miech, Dimitri Zhukov, Jean-Baptiste Alayrac, Makarand Tapaswi, Ivan Laptev, and Josef Sivic. 2019. Howto100m: Learning a text-video embedding by watching hundred million narrated video clips. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV).

OpenAI. 2025. Gpt-5 system card.

Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever. 2021. Learning transferable visual models from natural language supervision. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 8748–8763. PMLR.

Pranav Rajpurkar, Jian Zhang, Konstantin Lopyrev, and Percy Liang. 2016. SQuAD: 100,000+ questions for machine comprehension of text. In Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing, pages 2383–2392, Austin, Texas. Association for Computational Linguistics.

Ori Ram, Yoav Levine, Itay Dalmedigos, Dor Muhlgay, Amnon Shashua, Kevin Leyton-Brown, and Yoav Shoham. 2023. In-context retrieval-augmented language models. Transactions of the Association for Computational Linguistics, 11:1316–1331.

Ruchit Rawal, Khalid Saifullah, Miquel Farré, Ronen Basri, David Jacobs, Gowthami Somepalli, and Tom Goldstein. 2024. CinePile: A long video question answering dataset and benchmark. Preprint, arXiv:2405.08813.

Monica Riedler and Stefan Langer. 2024. Beyond text: Optimizing rag with multimodal inputs for industrial applications. Preprint, arXiv:2410.21943.

Sahel Sharifymoghaddam, Shivani Upadhyay, Wenhu Chen, and Jimmy Lin. 2025. UniRAG: Universal retrieval augmentation for large vision language models. In Findings of the Association for Computational Linguistics: NAACL 2025, pages 2026–2039, Albuquerque, New Mexico. Association for Computational Linguistics.

Xiaqiang Tang, Qiang Gao, Jian Li, Nan Du, Qi Li, and Sihong Xie. 2025. MBA-RAG: a bandit approach for adaptive retrieval-augmented generation through question complexity. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 3248–3254. Association for Computational Linguistics.

Weihan Wang, Zehai He, Wenyi Hong, Yean Cheng, Xiaohan Zhang, Ji Qi, Xiaotao Gu, Shiyu Huang, Bin Xu, Yuxiao Dong, Ming Ding, and Jie Tang. 2024. Lvbench: An extreme long video understanding benchmark. Preprint, arXiv:2406.08035.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, and 1 others. 2025. Internvl3. 5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. arXiv preprint arXiv:2508.18265.

Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, and Wenya Wang. 2025. Visual-rag: Benchmarking text-to-image retrieval augmented generation for visual knowledge intensive queries. Preprint, arXiv:2502.16636.

Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, and Zhen-Hua Ling. 2024. Corrective retrieval augmented generation. Preprint, arXiv:2401.15884.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. 2018. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 2369–2380, Brussels, Belgium. Association for Computational Linguistics.

Zijun Yao, Weijian Qi, Liangming Pan, Shulin Cao, Linmei Hu, Weichuan Liu, Lei Hou, and Juanzi Li. 2024. Seakr: Self-aware knowledge retrieval for

adaptive retrieval augmented generation. Preprint, arXiv:2406.19215.
- [2024] Xiang Yue, Yuansheng Ni, Tianyu Zheng, Kai Zhang, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, Yibo Liu, Wenhao Huang, and 3 others. 2024. MMMU: A massive multi-discipline multimodal understanding and reasoning benchmark for expert AGI. In IEEE/CVF Conference on Computer Vision and Pattern Recognition, CVPR 2024, Seattle, WA, USA, June 16-22, 2024, pages 9556–9567. IEEE.
- [2020] Biao Zhang, Paul Suganthan, Gaël Liu, Ilya Philippov, Sahil Dua, Ben Hora, Kat Black, Gus Martins, Omar Sanseviero, Shreya Pathak, and 1 others. 2025a. T5gemma 2: Seeing, reading, and understanding longer. arXiv preprint arXiv:2512.14856.
- [2020] Tianyi Zhang, Varsha Kishore, Felix Wu, Kilian Q. Weinberger, and Yoav Artzi. 2020. BERTScore: Evaluating text generation with BERT. In 8th International Conference on Learning Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30, 2020.
- [2020] Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, and Min Zhang. 2025b. Bridging modalities: Improving universal multimodal retrieval by multimodal large language models. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 9274–9285.
- [2023] Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, and 1 others. 2025c. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.
- [2023] Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, and Shuming Shi. 2023. Siren’s song in the ai ocean: A survey on hallucination in large language models. Preprint, arXiv:2309.01219.
- [2024] Zihan Zhang, Meng Fang, and Ling Chen. 2024. Retrievalqa: Assessing adaptive retrieval-augmented generation for short-form open-domain question answering. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11-16, 2024, pages 6963–6975. Association for Computational Linguistics.
- [2025] Zijie Zhong, Hanwen Liu, Xiaoya Cui, Xiaofan Zhang, and Zengchang Qin. 2025. Mix-of-granularity: Optimize the chunking granularity for retrieval-augmented generation. In Proceedings of the 31st International Conference on Computational Linguistics, COLING 2025, Abu Dhabi, UAE, January 19-24, 2025, pages 5756–5774. Association for Computational Linguistics.

A Additional Details on Dataset

Table 7 provides an overview of all datasets and their corresponding knowledge corpora used in our experiments, including the target modality type as well as the size of the queries and corpora. We divide each dataset into a 3:7 ratio for training and testing. We offer the detail of each dataset below.

### A.1 In-Domain Dataset

#### MMLU

As a dataset comprising queries that can be answered without the need for retrieval, we use MMLU *(Hendrycks et al., 2021)*, a benchmark that spans a wide range of tasks, including problem-solving abilities (e.g., elementary mathematics, computer science) and world knowledge (e.g., law, world religions). Specifically, we use questions from all tasks in the development split.

#### Natural Questions (NQ)

We also use Natural Questions *(Kwiatkowski et al., 2019)*, a question answering dataset consisting of real user queries issued to the Google search engine, with answers annotated based on supporting Wikipedia articles. We randomly sample 2,000 QA pairs from the dev split, and formulate the text corpus by segmenting the Wikipedia corpus into paragraphs of at most 100 words.

#### HotpotQA

HotpotQA *(Yang et al., 2018)* is a Wikipedia-based QA benchmark, but it contains complex queries that are annotated to reason over multiple articles. We utilize 2,000 randomly sampled QA pairs of the test split. As it requires multi-hop reasoning over multiple documents, we formulate the text corpus by grouping multiple related documents following LongRAG *(Jiang et al., 2024b)*, which can be longer than 4K tokens.

#### HybridQA

HybridQA *(Chen et al., 2020)* is a benchmark that requires reasoning over both tabular and textual information. Each question is grounded in a Wikipedia table, but often requires linking to associated text information to locate the correct answer. We randomly sample 2,000 QA pairs from the dev split. Unlike the original benchmark, which directly connects tables and textual evidence, we separate them into distinct table and text corpora to better validate our modality-specific routing-based retrieval framework.

#### MRAG-Bench (MRAG)

We utilize MRAG-Bench *(Hu et al., 2025)*, a vision-centric RAG benchmark that requires only relevant images and does not rely on other modalities, and evaluate on all 1,353 questions. Unlike conventional text-only queries, each query in MRAG-Bench is multi-modal, consisting of a textual question interleaved with a query image. We construct a single image corpus by collecting all images across questions.

#### WebQA

WebQA *(Chang et al., 2022)* is a benchmark designed to evaluate the ability of LVLMs to reason over multiple sources of information, including both text and images, in an open-domain setting. As the dataset is originally constructed with question-specific retrieval sources that combine text and images, we extract a subset of questions that require retrieval of image for answering. We then further filter these using GPT-4o *(Hurst et al., 2024)* with the prompt shown in Figure 10 to make sure questions are not grounded to a certain image, resulting in a final set of 2,000 QA pairs. Finally, we construct separate text and image corpora by extracting and aggregating evidence from each modality.

#### InfoSeek

InfoSeek *(Chen et al., 2023a)* is an open-domain benchmark comprising questions interleaved with images, which are best answered by retrieving relevant textual and visual information. For our experiments, we sample 2,000 QA pairs from dev split and collect the text and image evidence associated with each question to construct the corresponding text and image corpora.

#### LVBench

LVBench *(Wang et al., 2024)* is a benchmark developed for long video understanding, featuring questions generated by annotators based on YouTube videos with an average duration of over one hour. Since the benchmark was originally designed for non-RAG tasks, we rephrase the original text-video interleaved queries into a text-only format to align with our experimental setup using GPT-4o, with video metadata and a prompt (Figure 11). Each query is associated with a specific video and a corresponding time range. Notably, the majority of queries are annotated with timestamps spanning less than five minutes, thereby focusing on short segments within the longer videos. Since some videos are currently unavailable, we conduct our evaluation on the available videos and their corresponding questions. For training, we use these short-timestamp queries as a clip-level dataset.

#### VideoRAG

We also utilize VideoRAG-Wiki and VideoRAG-Synth benchmarks, introduced in Vide

Table 7: Dataset summary for in-domain and out-of-domain benchmarks. Average corpus length denotes the mean token count for text corpora and the mean duration for video corpora.

[tbl-6.html](tbl-6.html)

oRAG (Jeong et al., 2025), which are designed to evaluate RAG over a video corpus. These benchmarks are built on the HowTo100M (Miech et al., 2019) corpus (a large-scale collection of instructional YouTube videos) with queries sourced from WikiHowQA (Bolotova-Baranova et al., 2023) and synthetically generated QA pairs based on the videos. Since they lack timestamp annotations, we employ GPT-4o to identify video-level queries that are better answered through full video retrieval rather than short segments from the ground-truth video, which are then used as a video-level dataset for training the router.

# A.2 Out-of-Domain Dataset

Unlike the in-domain datasets, the out-of-domain datasets are used solely for evaluation to assess the generalizability of our routing approach and consist only of test splits.

TruthfulQA TruthfulQA (Lin et al., 2022) includes general knowledge questions designed to test whether LLMs can avoid common false beliefs or misconceptions, on diverse categories, including health, law, and politics. We use the multiple-choice version of the dataset, which includes only a single correct answer per question.

TriviaQA TriviaQA (Joshi et al., 2017) is a reading comprehension dataset consisting of trivia questions paired with evidence texts sourced from Wikipedia and the web. To distinguish between queries that require text retrieval and those that do

not, we categorize each query based on whether GPT-4o can produce an exact-match answer without access to external text. We randomly sample QA pairs from the dev split. Following the preprocessing strategies used in SQuAD and NQ, all supporting evidence documents are segmented into paragraphs of no more than 100 words.

SQuAD SQuAD v1.1 (Rajpurkar et al., 2016) is a benchmark dataset consisting of questions generated by crowdworkers based on a set of Wikipedia articles. Each question is answerable given the appropriate context paragraph. From the dataset's  $100,000+$  QA pairs, we randomly sample 2,000 pairs of dev split. For context retrieval, we utilize the full provided Wikipedia corpus, segmenting each article into paragraphs of at most 100 words.

2WikiMultiHopQA We also utilize 2WikiMultiHopQA (Ho et al., 2020), a benchmark designed to evaluate multi-hop reasoning across two Wikipedia articles. We randomly sample 2,000 QA pairs from the dev split and construct a document-level corpus by aggregating all annotated candidate paragraph-level contexts for each question.

Visual-RAG Visual-RAG (Wu et al., 2025) is a question-answering benchmark designed for visual knowledge-intensive questions, specifically tailored for text-to-image retrieval tasks. We utilize the full set of provided queries but sample five images per category to construct the image retrieval pool, ensuring efficient text-to-image retrieval.

#### CinePile

CinePile *(Rawal et al., 2024)* is a long-video question-answering benchmark that features questions based on movie clips from YouTube. Since the benchmark was originally designed for video understanding tasks rather than RAG, we reformulate each query using the same procedure as LVBench. For each of the 144 available videos, we randomly select 10 questions from the test split. Since CinePile does not provide granularity annotations, we classify the questions into two categories (such as clip-level and full-video-level granularity) using GPT-4o, following the same approach used in VideoRAG.

### A.3 Evaluation Metrics

We report results with standard metrics. For datasets with multiple-choice questions, we report Top-1 Accuracy (Acc), the proportion of questions answered correctly. For short-answer datasets, we use Exact Match (EM) and F1, which respectively measure exact agreement and word-level overlap between predictions and references; for InfoSeek, we use the custom accuracy metric defined in the original paper and official repository. For datasets with longer free-form answers, we use ROUGE-L, which captures the longest common subsequences between the prediction and reference *(Lin, 2004)*, and BERTScore, which assesses their semantic similarity *(Zhang et al., 2020)*. We report the average score by averaging first within each modality, then across modalities. Results are obtained from a single run under limited computational resources, while we validate the generality of our framework across multiple backbone models.

## Appendix B Additional Implementation Details

To effectively leverage both visual and textual information for visual element retrieval, we employ an ensemble approach that combines visual and textual similarity scores with a weighting ratio of 0.8 for visual information. The textual information consists of image captions for images and scripts for videos. To handle long videos, we utilize PySceneDetect *(Castellano, 2014)*, an open-source tool that detects scene boundaries by analyzing content changes (e.g., color histogram differences or threshold-based detection), to segment long videos into shorter clips with an average length of no more than 3 minutes. Moreover, for both the retrieval and generation stages, we uniformly sample 32 frames per video. For baseline models that do not natively support video input, specifically UniRAG (which utilizes CLIP) and GME, we average the embeddings of these sampled frames to obtain a single representative embedding vector.

Training-based routers employ a lightweight classifier head on top of the backbone model to produce logits over multi-label prediction. Multi-label targets are converted into multi-hot vectors, and training is performed via binary cross-entropy loss between these targets and the predicted logits. The router is trained for 5 epochs with a learning rate of 2e-5 and a LoRA rank of $r=32$. At inference time, routing decisions are made using a predefined threshold of 0.8, selecting all modality-granularity combinations whose sigmoid probabilities exceed the threshold. In contrast, for the training-free variant, we prompt the model using a curated prompt that specifies task objectives and few-shot examples, as shown in Figure 8. Most experiments are conducted on NVIDIA RTX Pro 6000 Max-Q GPUs with 96GB of VRAM.

## Appendix C Theoretical Analyses of UniversalRAG

In this section, we present formal analyses of each module in UniversalRAG, including the effectiveness of modality routing (Section C.1) and multi-granularity (Section C.2), as well as the efficiency of modality-aware routing (Section C.3).

### C.1 Effectiveness of Modality Routing

For a rigorous analysis of the effectiveness of modality routing, we restate Proposition 1 with additional detail and provide a complete proof.

###### Proposition 1 (Restated).

Let the similarity score in the unified embedding space of $\mathcal{C}_{\textup{unified}}$ be defined as

$s(\bm{q},\bm{c})=\alpha\cdot\mathbf{1}\{m(\bm{q})=m(\bm{c})\}+\beta\cdot r(\bm{q},\bm{c})+\varepsilon,$

where $\alpha>0$ is a modality bias, $m(\cdot)$ denotes the modality, and $r(\cdot,\cdot)$ measures semantic relevance. If $\alpha$ is sufficiently large relative to the variance of $r$, the probability of retrieving items from the required modality $m^{*}(\bm{q})$ is less than under modality-aware routing followed by within-modality retrieval.

###### Proof.

Without loss of generality, let us consider the top-1 retrieval, as the extension to the top-$k$ case follows directly. Let the unified retrieval corpus

$\mathcal{C}_{\text{unified}}$ be decomposed into three disjoint sets:

$S$ $=\{\bm{c}:\>m(\bm{c})=m(\bm{q})\}$ (1)
$R$ $=\{\bm{c}:\>m(\bm{c})=m^{*}(\bm{q})\}$
$O$ $=\mathcal{C}_{\text{unified}}\setminus(S\cup R).$

Let us consider the scenario where $m^{*}(\bm{q})\neq m(\bm{q})$ and $S,R\neq\emptyset$. Define $X_{c}\coloneqq\beta\cdot r(\bm{q},\bm{c})+\varepsilon_{\bm{c}}$ and suppose $\{X_{\bm{c}}\}_{\bm{c}\in\mathcal{C}_{\text{unified}}}$ are independent, mean-zero, sub-Gaussian with variance proxy $\sigma^{2}=\beta^{2}\cdot\text{Var}[r(\bm{q},\bm{c})]+\text{Var}[\varepsilon_{\bm{c}}]$. Then the similarity scores can be expressed as

\[ s(\bm{q},\bm{c})=\begin{cases}\alpha+X_{\bm{c}},&\bm{c}\in S\\
X_{\bm{c}},&\bm{c}\in R\cup O.\end{cases} \] (2)

Let $M_{S}=\max_{\bm{s}\in S}X_{\bm{s}}$, $M_{R}=\max_{\bm{r}\in R}X_{\bm{r}}$, and $M_{O}=\max_{\bm{o}\in O}X_{\bm{o}}$. Under the unified embedding retrieval, the top-1 item lies in $R$ if and only if

$M_{R}\geq\alpha+\max\{M_{S},M_{O}\}.$

Hence, we can obtain the upper bound of the probability where top-1 retrieval comes from $R$:

$\mathbb{P}(\mathcal{T}_{\text{unified}}(\bm{q};\mathcal{C}_{\text{unified}})\in R)$ (3)
$=\mathbb{P}(M_{R}\geq\alpha+\max\{M_{S},M_{O}\}$
$\leq\mathbb{P}(M_{R}-M_{S}\geq\alpha).$

As $\{M_{R}-M_{S}\geq\alpha\}\subseteq\cup_{(\bm{r},\bm{s})\in R\times S}\{X_{\bm{r}}-X_{\bm{s}}\geq\alpha\}$, by the union bound we have

$\mathbb{P}(M_{R}-M_{S}\geq\alpha)\leq\sum_{(\bm{r},\bm{s})\in R\times S}\mathbb{P}(X_{\bm{r}}-X_{\bm{s}}\geq\alpha).$

As $X_{\bm{r}}-X_{\bm{s}}$ is sub-Gaussian with variance proxy $2\sigma^{2}$, the Chernoff bound of the tail probability combined with Equation (3) leads to

$\mathbb{P}(\mathcal{T}_{\text{unified}}(\bm{q};\mathcal{C}_{\text{unified}})\in R)$ (4)
$\leq|R||S|\exp\left(-\frac{\alpha^{2}}{4\sigma^{2}}\right).$

By contrast, if the retrieval is done at the modality-specific corpus after modality-aware routing with accuracy $r$, the probability where the top-1 item is in $R$ is $r$. Combining this with Equation (4),

$\mathbb{P}(\mathcal{T}_{\text{unified}}(\bm{q};\mathcal{C}_{\text{unified}})\in R)$ (5)
$\leq|R||S|\exp\left(-\frac{\alpha^{2}}{4\sigma^{2}}\right)$
$<r=\mathbb{P}(\mathcal{T}_{\mathcal{R}(\bm{q})}(\bm{q};\mathcal{C}_{\mathcal{R}(\bm{q})})\in R)$

whenever $\alpha>2\sigma\sqrt{\frac{\log(|R||S|)}{r}}$. Meanwhile, the right-hand side of Equation (4) decays to 0 as $\alpha/\sigma\to\infty$. Hence, for $\alpha$ large enough relative to the variance of $r$, unified embedding retrieval is strictly worse than retrieving from modality-specific corpus after modality-aware routing. ∎

###### Remark.

Consider very large corpora with $|R|=|S|=10^{12}$. In this setting, if $p=0.8$ and $\sigma=0.01$, then $\alpha>2\sigma\sqrt{\frac{\log(|R||S|)}{p}}\simeq 0.17$ is sufficient to ensure that routing-based retrieval outperforms unified embedding retrieval. Given that most multimodal encoders exhibit inherent modality biases (as illustrated in Figures 2 and 7), this underscores the necessity of modality-aware routing.

### C.2 Effectiveness of Multigranularity

In Section 3.2 and Appendix D.2, we show that routing with multiple granularities within each modality improves performance (see Tables 4 and 9). We also provide a simple statement and proof that support these empirical findings.

###### Proposition 2.

Let $F(Q;m,g)$ be the expected response quality when retrieving from modality $m$ using granularity $g$. If there exist queries $\bm{q}_{1},\bm{q}_{2}$ and granularities $g_{f},g_{c}$ such that $F(\bm{q}_{1};m,g_{f})>F(\bm{q}_{1};m,g_{c})$ and $F(\bm{q}_{2};m,g_{c})>F(\bm{q}_{2};m,g_{f})$, then a routing policy that assigns $g_{f}$ to $\bm{q}_{1}$ and $g_{c}$ to $\bm{q}_{2}$ attains strictly higher expected quality than any fixed-granularity policy.

###### Proof.

Consider any fixed policy that always uses a single granularity $g\in\{g_{f},g_{c}\}$. If $g=g_{f}$, then we have

$F(\bm{q}_{1};m,g_{f})$ $+F(\bm{q}_{2};m,g_{f})$ (6)
$<F(\bm{q}_{1};m,g_{f})+F(\bm{q}_{2};m,g_{c}).$

Similarly, if $g=g_{c}$, then we have

$F(\bm{q}_{1};m,g_{c})$ $+F(\bm{q}_{2};m,g_{c})$ (7)
$<F(\bm{q}_{1};m,g_{f})+F(\bm{q}_{2};m,g_{c}).$

In both cases, the sum of response quality with the routing policy that applies $g_{f}$ to $\bm{q}_{1}$ and $g_{c}$ to $\bm{q}_{2}$ strictly exceeds that of any fixed granularity $g$. ∎

### C.3 Efficiency of Modality-Specific Retrieval

While the empirical results in Section 3.2 demonstrate the efficiency benefits of modality-aware routing (with latency trends shown in Figure 5),

we provide a more rigorous analysis on its computational advantages. Let $N$ denote the size of each modality- and granularity-specific corpus, assuming uniform corpus sizes for simplicity, and let $k$ be the number of available routing choices (i.e., the number of modality-granularity pairs). Under a unified embedding approach, retrieval is performed over a single aggregated corpus of size $kN$, incurring a search cost that scales with the total corpus size. In contrast, UniversalRAG first performs lightweight routing to select the most relevant modality-granularity subset, and then conducts retrieval over only a small selected subset.

###### Proposition 3.

Let $T(m)$ denote the expected retrieval latency of a single query over a corpus of size $m$ under a fixed retrieval backend, and let the routing cost to be a fixed constant $C$, independent of the number of available routing choices $k>1$. Then, UniversalRAG achieves lower latency than unified embedding space retrieval on large-scale corpora.

###### Proof.

Under unified embedding, all modality-granularity corpora are merged into a single index of size $kN$. Then, the expected per-query retrieval latency is $T_{\texttt{unified}}=T(kN)$. Under UniversalRAG, routing incurs a constant overhead $C$ and then retrieval is executed only on a small number of routed corpora. Assuming retrieval calls of selected corpus are executed in parallel, the end-to-end latency of whole retrieval process is $T_{\texttt{routing}}=C+T(N)$. Let us first consider the case of exact retrieval with embeddings, where the backend exhibits linear scaling $T(m)=\Theta(m)$, then we obtain

$\frac{T_{\texttt{unified}}}{T_{\texttt{routing}}}\gtrsim\frac{kN}{N+C}=\frac{k}{1+C/N}.$ (8)

Taking $N\rightarrow\infty$ yields

$\liminf_{N\rightarrow\infty}\frac{T_{\texttt{unified}}}{T_{\texttt{routing}}}=\Theta(k),$ (9)

resulting in a linear-in-$k$ speedup. Meanwhile, many modern retrieval systems adopt approximate nearest neighbor search *(Douze et al., 2025)*, which can achieve logarithmic query-time scaling $T(m)=\Theta(\log m)$ (in the best case). Then, for sufficiently large $N$,

$\frac{T_{\texttt{unified}}}{T_{\texttt{routing}}}\gtrsim\frac{\log(kN)}{\log N+C}=\frac{\log N+\log k}{\log N+C}.$ (10)

Letting $N\rightarrow\infty$, we have

$\liminf_{N\rightarrow\infty}\frac{T_{\texttt{unified}}}{T_{\texttt{routing}}}\geq 1.$ (11)

Thus, even with the approximate retrieval with logarithmic scaling, UniversalRAG achieves a constant-factor asymptotic speedup. Combining these results, UniversalRAG attains strictly lower asymptotic retrieval latency than unified embedding space retrieval for any retrieval methods. ∎

## Appendix D Additional Experimental Results

### D.1 Additional Results using Different LVLMs

Table 8 shows detailed generation results of baselines and UniversalRAG models on 10 benchmarks using InternVL3.5-8B and Molmo2-4B as generation models. In both settings, UniversalRAG outperforms all baselines and achieves average scores comparable to Oracle. These results demonstrate that UniversalRAG is robust and generalizable in various LVLM generators.

### D.2 Additional Results on Multigranularity

Table 4 demonstrates the correlation between the number of granularity levels and end-to-end performance for two training-free models, leveraging the flexibility of our approach in scenarios without labeled data. We further extend this analysis to training-based routers, comparing performance with and without granularity. Table 9 reports results across three training-based router models, consistently demonstrating a performance advantage when granularity is incorporated. These findings underscore the efficacy of including granularity in routing decisions for both training-free and training-based approaches.

### D.3 Detailed Results on Out-of-Domain Dataset

We provide the generation results of UniversalRAG variants and baseline methods on each out-of-domain dataset in Table 10. Overall, UniversalRAG consistently outperforms all baselines on average. Notably, the training-free router variants exhibit strong performance across all datasets, showing their outstanding generalization ability to unseen queries. In contrast, trained routers achieve relatively lower performance than on in-domain datasets; nevertheless, they remain robust and still surpass the baseline methods by a large margin.

Table 8: Results of diverse RAG methods with diverse LVLMs (InternVL3.5-8B and Molmo2-4B) across modalities. Bold denotes the best performance and underlined indicates the second-best among UniversalRAG variants, using either trained or training-free routers. R-L and BERT correspond to ROUGE-L and BERTScore, respectively.

[tbl-7.html](tbl-7.html)

Table 9: Effect of granularity on the performance for training-based router models. Gn denotes Granularity.

[tbl-8.html](tbl-8.html)

# E Modality Gap in Unified Embedding Space

Figure 7 visualizes the modality gap within the unified embedding space of five multimodal encoders (Jiang et al., 2024a; Bolya et al., 2025; Lin et al., 2025; Zhang et al., 2025b; Meng et al., 2025). The PCA plot reveals that embeddings cluster by modality, with text embeddings (shown in green) exhibiting larger distances from those of other modalities. Recent methods like E5-V and

GME focus on better aligning these modalities to narrow the gap. However, despite these efforts, a noticeable separation between modalities remains, indicating that current multimodal encoders still struggle to fully unify the embedding space across text, images, and videos. Therefore, the modality routing mechanism of UniversalRAG is required to dynamically direct each query to its corresponding modality-specific embedding space, thereby effectively bridging the modality gap and enhancing retrieval performance.

# F Qualitative Results

We present case studies to demonstrate the effectiveness of UniversalRAG. Table 11 compares the results of various RAG approaches, including traditional single-modality methods and UniversalRAG, on queries from the WebQA dataset. Traditional approaches such as TextRAG and VideoRAG fail to generate accurate answers: TextRAG retrieves passages lacking relevant visual details, while VideoRAG is better suited for temporal rea

Table 10: Results of diverse RAG methods on out-of-domain dataset with Qwen3-VL-8B-Instruct across modalities. Bold denotes the best performance and underlined indicates the second-best among UniversalRAG variants, using either trained or training-free routers. R-L and BERT correspond to ROUGE-L and BERTScore, respectively.

[tbl-9.html](tbl-9.html)

soning tasks. In contrast, UniversalRAG correctly routes the query to the image modality, recognizing that visual information about color is necessary, and successfully generates the correct response. This highlights the advantage of modality-aware routing in leveraging the appropriate data from the correct modality corpus, demonstrating UniversalRAG's ability to adaptively select the most informative modalities and granularities for accurate answer generation.

In addition to modality routing, we observe that UniversalRAG also benefits from retrieving information at the appropriate granularity. Table 12 shows results from HotpotQA, where the query requires complex reasoning over multiple text sources. While paragraph-level granularity fails to provide sufficient context for reasoning, UniversalRAG routes the query to the document-level corpus to retrieve all the textual information necessary for accurate reasoning. Similarly, for video queries, Table 13 shows results from LVBench on the query that requires only a short segment of the full long video to answer. While full-video-level retrieval includes irrelevant content and uniformly sampled frames fail to capture the necessary information, clip-level retrieval focuses on smaller, more relevant segments of the video to ensure that only the most pertinent visual details are considered, leading to a more accurate answer.

UniversalRAG performs cross-modal retrieval, allowing the router to select multiple modality

granularity combinations when required, rather than restricting routing to a single source. Table 14 presents an example from HybridQA, where queries primarily rely on tabular data but benefit substantially from complementary textual evidence. In such cases, factual information is best captured from paragraphs, whereas structured knowledge, such as numerical values, is more effectively represented in tables. By jointly retrieving from both modalities, UniversalRAG effectively aggregates complementary evidence and provides the information necessary to answer the query correctly. In contrast, a unimodal variant that restricts retrieval to a single modality retrieves incomplete evidence and fails to support correct reasoning.

However, there are some cases where the routing mechanism fails, particularly when the query exhibits ambiguity in modality requirement or when the required information spans across multiple modalities. Table 15 shows failure cases in which UniversalRAG, employing GPT-5 as a training-free router, incorrectly routes the modality. In the first example, the router's prediction deviates from the inductive ground-truth label as GPT-5, as a modern frontier model, has prior knowledge beyond the predefined routing taxonomy. Although this results in a nominal misclassification, it does not affect the final generation quality, as the model can answer the query without external retrieval. The router also struggles to distinguish between closely related modalities. As illustrated in the second case,

![img-6.jpeg](img-6.jpeg)

![img-7.jpeg](img-7.jpeg)

![img-8.jpeg](img-8.jpeg)

![img-9.jpeg](img-9.jpeg)

![img-10.jpeg](img-10.jpeg)

![img-11.jpeg](img-11.jpeg)
Figure 7: Visualization of the unified embedding space across various multimodal encoders.

![img-12.jpeg](img-12.jpeg)

![img-13.jpeg](img-13.jpeg)

![img-14.jpeg](img-14.jpeg)

![img-15.jpeg](img-15.jpeg)

a query requiring temporally localized visual evidence is incorrectly routed from clip-level retrieval to static image retrieval. Moreover, the router sometimes exhibits difficulty in determining the appropriate retrieval granularity. Queries that lie near the boundary between different granularity levels are sometimes misrouted, as shown in the third and fourth examples. Finally, the router occasionally fails to recognize cross-modal information needs, leading to incorrect routing decisions for queries that require joint reasoning across modalities, as illustrated in the last example.

# G Use of Large Language Models (LLMs)

We use large language models merely as a writing assistant. Their role is confined to improving grammar and paraphrasing sentences for clarity, and all the core ideas regarding problem definition, UniversalRAG framework, experimental design, and interpretation of results are entirely our own.

Table 11: Case study comparing RAG on single modality against UniversalRAG (Ours). Traditional RAG models are limited by modality-specific retrieval, often missing relevant knowledge from other modalities. UniversalRAG (Ours) dynamically routes each query to the most appropriate modality, enabling more accurate retrieval and response generation.

[tbl-10.html](tbl-10.html)

Table 12: Case study comparing different levels of text granularity. The user query requires complex retrieval involving multiple entities. ParagraphRAG retrieves limited context centered around a single entity, leading to an incorrect answer. UniversalRAG (Ours) routes the query to the document corpus and retrieves richer document-level information, allowing it to capture both relevant entities and generate the correct response.

[tbl-11.html](tbl-11.html)

Table 13: Case study comparing different levels of video granularity. The user query requires only a segment of the video to determine the answer. VideoRAG retrieves a broad range of frames across the video, which may include irrelevant content or miss key frames, leading to an incorrect response. UniversalRAG (Ours) routes the query to the clip-level granularity, retrieving more focused and relevant visual information, enabling it to generate the correct response.

[tbl-12.html](tbl-12.html)

(Timestamp Range: 00:00~1:01:05)
Response: A X

UniversalRAG (Ours) Routed to: Clip

Retrieved:

![img-16.jpeg](img-16.jpeg)

![img-17.jpeg](img-17.jpeg)

(Timestamp Range: 33:46~36:56)

Response: C

Table 14: Case study comparing UniversalRAG across uni-modal and cross-modal scenarios. In the uni-modal setup, where only a single prominent modality is used, information can sometimes be incomplete as they require evidence across modalities. UniversalRAG (Ours), with its cross-modal capability, gathers evidence from multiple modalities to generate a more comprehensive response.

[tbl-13.html](tbl-13.html)

Table 15: Failure cases in modality routing with UniversalRAG (Ours).

[tbl-14.html](tbl-14.html)

Classify the following query into one or more categories from: [No, Paragraph, Document, Table, Image, Clip, Video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:

- No: The query can be answered directly with common knowledge, reasoning, or computation without external data.
- Paragraph: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- Document: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- Table: The query requires information that is best represented in a tabular format, often involving comparisons or structured data.
- Image: The query focuses on visual aspects like appearances, structures, or spatial relationships.
- Clip: The query targets a short, specific moment or event within a video, without needing full context.
- Video: The query requires understanding dynamic events, motion, or sequences over time in a video.

# Examples:

- "What is the capital of France?"  $\rightarrow$  No
- "What is the birth date of Alan Turing?"  $\rightarrow$  Paragraph
- "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?"  $\rightarrow$  Document
- "Among the recipients of the Turing Award, who had the earliest birth year?"  $\rightarrow$  Table
- "Describe the appearance of a blue whale."  $\rightarrow$  Image
- "Describe the moment Messi scored his goal in the 2022 World Cup final."  $\rightarrow$  Clip
- "Explain how Messi scored his goal in the 2022 World Cup final."  $\rightarrow$  Video
- "Solve  $12 \times 8$ ."  $\rightarrow$  No
- "Who played a key role in the development of the iPhone?"  $\rightarrow$  Paragraph
- "Which Harvard University graduate played a key role in the development of the iPhone?"  $\rightarrow$  Document
- "What is the cheapest iPhone model available in 2023?"  $\rightarrow$  Table
- "Describe the structure of the Eiffel Tower."  $\rightarrow$  Image
- "Describe the moment Darth Vader reveals he is Luke's father in Star Wars."  $\rightarrow$  Clip
- "Analyze the sequence of events leading to the fall of the Empire in Star Wars."  $\rightarrow$  Video
- "Describe the visual appearance and habitat of the blue whale."  $\rightarrow$  Paragraph+Image
- "Compare the architectural features shown in Gothic and Renaissance cathedrals."  $\rightarrow$  Image+Table
- "Describe the moment of the moon landing and explain the mission details."  $\rightarrow$  Paragraph+Clip

Classify the following query: {query}

Provide only the category or categories combined with  $+$ .

Figure 8: Prompt for query routing in a training-free manner. The prompt defines each category with concise criteria and illustrative examples. Specifically, examples are designed to contrast closely related cases: for example, Paragraph vs. Document for simple fact retrieval vs. multi-hop reasoning; and Clip vs. Video for short specific moments vs. long-term sequential understanding, highlighting the key aspect that differentiates each category.

Classify the following query into one or more categories from: [No, Paragraph, Passage, Section, Document, ..., Clip, Sequence, Segment, Video], based on whether it requires retrieval-augmented generation (RAG) and the most appropriate modality. Consider:

- Paragraph: The query requires retrieving factual descriptions, straightforward explanations, or concise summaries from a single source.
- Passage: The query requires a detailed block of text (a few paragraphs) from a single source, with added context.
- Section: The query requires retrieving an extensive section of a document explaining a sub-topic, possibly with examples or elaboration.
- Document: The query requires multi-hop reasoning, combining information from multiple sources or documents to form a complete answer.
- Clip: The query targets a short, specific moment or event within a video, without needing full context.
- Sequence: The query targets a continuous stretch of related shots (about 10 minutes) that together form a self-contained mini-narrative or process, providing more context and flow than a standalone clip.
- Segment: The query targets a longer portion of a video (about 30 minutes) capturing a meaningful sub-scene or subplot-rich and cohesive enough to serve as its own chapter-like unit.
- Video: The query requires understanding dynamic events, motion, or sequences over time in a video.

# Examples:

- "What is the birth date of Alan Turing?"  $\rightarrow$  Paragraph
- "Summarize Alan Turing's concept of the Turing Machine."  $\rightarrow$  Passage
- "Explain Alan Turing's contributions to cryptography during WWII."  $\rightarrow$  Section
- "Which academic discipline do computer scientist Alan Turing and mathematician John von Neumann have in common?"  $\rightarrow$  Document
- "Describe the moment Messi scored his goal in the 2022 World Cup final."  $\rightarrow$  Clip
- "Detail the sequence of passes and movements leading to Messi's goal in the 2022 World Cup final."  $\rightarrow$  Sequence
- "Describe the build-up sequence during the mid-game period of the 2022 World Cup final."  $\rightarrow$  Segment
- "Analyze how Argentina won the 2022 World Cup."  $\rightarrow$  Video

Classify the following query: {query}

Provide only the category or categories combined with  $+$ .

Figure 9: Prompt for query routing in a training-free manner with additional granularity choices. Only the components that differ from Figure 8 are shown, including the task objective and few-shot examples.

Evaluate whether the query can be answered using general knowledge about the image's subject rather than relying solely on details unique to the provided image, and verify that the answer is obtainable from the image and the query.

- Respond "yes" if:

1. The query can be fully answered using general knowledge about the subject.
2. The answer can be derived solely from the image and the query, without needing image-specific details.

- Respond "no" if either condition is not met.

# Example 1:

- Image: A portrait of Donald Trump
- Query: What is the color of Trump's hair?
- Answer: White
- Response: "yes"

# Example 2:

- Image: A close-up photo of a light bulb
- Query: What is the color of the light bulb in this image?
- Answer: Yellow
- Response: "no"

Figure 10: Prompt to filter queries for WebQA.

You will receive a query from a video QA dataset and the title of the corresponding video on YouTube. I want you to paraphrase the query by replacing "in the video?", "of the video", or similar phrases with references to the video content naturally. The output should sound as if a human is asking ChatGPT, and should not explicitly mention the exact name of the video or even parts of the title. However, the rephrased query should contain enough implicit information about the video to allow the model to identify it. Try to reduce the chance of the model getting confused between multiple possible video candidates. If there could be multiple video matches for a given query, try to include more information in the rephrased query.

# Example 1:

- Query: What year appears in the opening caption of the video?
- Video Title: Blue Eye Samurai | Hammerscale | Full Episode | Netflix
- Upload Date: 2023-11-05
- Channel Name: Netflix
- Rephrased Output: What year appears in the opening caption of the Blue Eye Samurai episode on Netflix?

# Example 2:

- Query: After the vlogger sees a dog with an advertisement from the company named Smitten, camera changes to the scene with .
- Video Title: My ICELAND Experience | Ultimate Travel Vlog
- Upload Date: 2022-10-26
- Channel Name: Kallmekris
- Rephrased Output: After spotting a dog with a Smitten advertisement, what scene does the camera transition to in Kallmekris's Iceland travel vlog from 2022?

Figure 11: Prompt to rephrase queries using video metadata for LVBench and CinePile.