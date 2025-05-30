#  Investigating Thematic Patterns and User Preferences in Large Language Model Interactions Using BERTopic


![Architecture Diagram](https://i.ibb.co/Kz5KSdv2/Screenshot-2025-05-30-194345.png)

##  Abstract

This study applies **BERTopic**, a transformer-based topic modeling technique, to the **lmsys-chat-1m** dataset—a multilingual conversational corpus built from head-to-head evaluations of large language models (LLMs). Each user prompt is paired with two anonymized LLM responses and a human preference label, enabling analysis of how users evaluate competing model outputs.

The primary objective is to **uncover thematic patterns** in these conversations and examine their relationship to user preferences—particularly whether certain LLMs are consistently preferred within specific topics. To handle the complexities of multilingual variation, imbalanced dialogue turns, and redacted or noisy content, a robust preprocessing pipeline was developed.

Using BERTopic, the system extracted over **29 coherent topics**, covering areas such as artificial intelligence, programming, ethics, and cloud infrastructure. We then analyzed how user preferences mapped to these topics to identify patterns in **model-topic alignment**.

A range of visualizations supported the analysis, including:
- **Inter-topic distance maps**
- **Topic probability distributions**
- **Model-vs-topic preference matrices**

These insights have practical implications for **domain-specific LLM fine-tuning**, helping optimize performance and improve real-world user satisfaction.

> **Keywords:** Topic Modeling · BERTopic · Large Language Models · LMSYS-Chat-1M · Natural Language Processing
## Reproducibility & Resources

To ensure full reproducibility and facilitate further experimentation, we provide open access to both the implementation and all relevant data artifacts:

-    **Colab Notebook**  
	   - A complete, executable notebook containing the entire BERTopic-based topic modeling pipeline, from data preprocessing to evaluation.  
    ➤ [Open in Google Colab](https://colab.research.google.com/drive/1V_vJt-1qsvT-ZPgdl0_ll21ZrPrWR1oA?usp=sharing)
    
-    **Model and Dataset Repository**  
    Hosted on Google Drive, this includes:  
	    -  The original cleaned input dataset  
	    -  The annotated result dataset with topic assignments and model metadata  
	    - The trained BERTopic model (`.pkl`) for reuse and replication 
	   ➤ [Access Repository](https://drive.google.com/drive/folders/1hK-FzeTYoyusk-VGc3mb_Ad4W3feYaur?usp=drive_link)
    

These resources allow readers to replicate the findings, test variations, or extend the analysis with alternative LLMs or topic modeling strategies.
## Methodology

This section outlines the end-to-end methodology adopted to extract interpretable and semantically coherent topics from the LMSYS-Chat-1M dataset using the BERTopic framework. The overall architecture is summarized below.

###  Dataset Acquisition and Characterization

We used the **LMSYS-Chat-1M** dataset, a large-scale collection of head-to-head conversations between various large language models (LLMs) and users. Each entry contains:

-   A user prompt
    
-   Two anonymized LLM responses
    
-   A user preference label (`A`, `B`, or `Tie`)
    

### Preliminary Data Exploration

Exploratory analysis included:

-   **Model Appearance Frequency:** To examine dataset balance.
    
-   **Win/Loss/Tie Distribution:** To assess comparative performance of LLMs.
    
-   **Response Length Preference:** To determine user tendencies toward longer or shorter answers.
    

Visualizations supporting these insights are presented in Section 5.

###  Data Preprocessing

A robust preprocessing pipeline was implemented with the following steps:

-   **Language Filtration:** Only English prompts and responses were retained using the `fasttext` language detection library.
    
-   **Text Normalization:** Non-ASCII characters, emojis, URLs, and noisy tokens were removed using regular expressions.
    
-   **Stop Prompt Removal Consideration:** We evaluated removing templated or nonsensical prompts but found no measurable impact on topic quality, so this step was not adopted.
    

### Topic Modeling with BERTopic

We employed the BERTopic framework, which integrates transformer-based embeddings with density-based clustering and a topic representation layer.



#### Document Embedding

We used the `all-MiniLM-L6-v2` model to embed each user prompt into a 384-dimensional vector space:

$$
\mathbf{E} \in \mathbb{R}^{n \times 384}
$$

---

#### Dimensionality Reduction with UMAP

UMAP was used to reduce the embedding space to two dimensions while preserving local and global relationships. The objective is to minimize the weighted sum of squared distances between projected points:

$$
\underset{Y}{\mathrm{argmin}} \sum_{(i, j)} w_{ij} \| Y_i - Y_j \|^2
$$

where:

$$
w_{ij} \text{ are edge weights in the nearest-neighbor graph constructed in high-dimensional space.}
$$

---

#### Clustering with HDBSCAN

The reduced vectors were clustered using HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise), a density-based algorithm capable of identifying clusters of varying densities and separating noise.

HDBSCAN builds a hierarchy of clusters and extracts the most stable ones using the following:

- Let:

  $$
  \text{min\_cluster\_size} = k
  $$

- Define the **mutual reachability distance** between points \(x\) and \(y\) as:

  $$
  d_{\text{mreach-k}}(x, y) = \max \left\{ \text{core}_k(x), \text{core}_k(y), d(x, y) \right\}
  $$

- Construct a weighted graph using mutual reachability distances.
- Build a minimum spanning tree and extract clusters based on **density stability**.

Clusters are selected by maximizing the cluster's **persistence** in the dendrogram, resulting in:

$$
\text{Cluster stability} = \int_{\lambda_{\text{birth}}}^{\lambda_{\text{death}}} \left|C(\lambda)\right| \, d\lambda
$$

where:

$$
C(\lambda) \text{ is the cluster at threshold } \lambda, \quad \left|C(\lambda)\right| \text{ is its size.}
$$



> **Figure: HDBSCAN Architecture**  
> ![HDBSCAN ARCHITECTURE](https://i.ibb.co/5gdnVDSX/image.png)


#### Topic Representation with c-TF-IDF

For each cluster, the top terms were extracted using class-based TF–IDF (c-TF-IDF), which adapts traditional TF-IDF to group-level term importance:

$$
\text{c-TF-IDF}_{t,c} = \frac{f_{t,c}}{\sum_{t' \in c} f_{t',c}} \cdot \log \left( \frac{N}{n_t} \right)
$$

$$
\begin{aligned}
f_{t,c} \quad &\text{= frequency of term } t \text{ in class } c \\\\
n_t \quad &\text{= number of classes containing term } t \\\\
N \quad &\text{= total number of classes}
\end{aligned}
$$

This yields interpretable keywords per topic using `CountVectorizer`.


### Noise Handling

Topic `-1`, generated by HDBSCAN to represent noise or outliers, was excluded from the primary analysis to preserve thematic clarity and statistical integrity.

###  Hyperparameter Optimization

Key parameters were iteratively tuned:

-   **UMAP:** `n_neighbors`, `min_dist`
    
-   **HDBSCAN:** `min_cluster_size` and `min_samples`
    

Through qualitative validation and coherence assessment, the model converged on **29 meaningful topics**, excluding one outlier.

### Comparative Validation

For robustness, alternative modeling pipelines were considered:

-   **Latent Semantic Analysis (LSA)**
    
-   **Embedding variants (e.g., MPNet, BERT)**
    
-   **Clustering alternatives (e.g., KMeans, Agglomerative)**
    

BERTopic with MiniLM + UMAP + HDBSCAN consistently offered the best balance between interpretability and granularity.

### Conceptual Visualization of Clusters

BERTopic enables visualizing topic clusters in 2D space using UMAP, where prompts are colored by topic. This allows intuitive assessment of topic separability, outlier density, and thematic coherence.



##  Topic Analysis and Evaluation

The final BERTopic model identified **29 distinct and semantically rich topics** from the LMSYS-Chat-1M dataset. These topics span a wide range of domains including programming, health, ethics, logic, entertainment, and advanced technologies. A few illustrative examples include:

| Topic ID | Description                                |
|----------|--------------------------------------------|
| Topic 0  | Gaming and user-assistant interaction      |
| Topic 4  | Programming, SQL, RDBMS, Database          |
| Topic 6  | Machine Learning and Advanced AI Concepts  |
| Topic 9  | Health Advice and Medical Concerns         |
| Topic 16 | JavaScript, React, Web Development         |


For a complete list of topics, refer to **Table 1** in the accompanying paper.

----------

###  Hierarchical Clustering of Topics

![Hierarchical Clustering of Topics](https://i.ibb.co/XZvLWwK1/dendo.png)

This dendrogram visualizes the hierarchical structure among topics using cluster linkage. It highlights semantic relationships and overlaps between topic clusters—for example, programming topics tend to group closely, while medical and ethical concerns form a separate branch.

----------

###  Coverage of Top 10 Topics

![Top 10 Topic Coverage](https://i.ibb.co/99k8vGsk/top10.png)

This bar chart displays the proportion of dataset entries belonging to the ten most dominant topics. It illustrates thematic concentration and helps determine which subject areas were most prevalent in user prompts.

----------

###  Normalized Topic Performance Heatmap

![Heatmap of Topic Performance](https://i.ibb.co/BH7hvKjV/heatmap.png)

This heatmap evaluates how frequently each of the top-5 LLMs secured wins within the top 10 most populated topics (normalized by appearances). Darker shades indicate stronger model-topic alignment. This provides insight into domain-specific LLM strengths.

----------

###  Most Balanced Topics by Model Performance

![Balanced Topics](https://i.ibb.co/mr2yhM1w/balanced.png)

This plot identifies topics where model win rates were closest, indicating competitive balance. These balanced domains often represent either ambiguous or well-mastered subject areas, making them useful for benchmarking model generality.

----------

###  Topic-wise Winning Percentage

![Topic-wise Win Rate](https://i.ibb.co/N6Pm4h6X/g.png)

This chart shows which models dominated which topics in terms of win rate. Certain models exhibit clear domain-specific advantages.  
For a comprehensive breakdown of the **top-5 winning models per topic**, refer to **Table 2** in the accompanying paper.

## Conclusion

This project successfully leveraged the BERTopic framework to analyze the LMSYS-Chat-1M dataset, uncovering 29 semantically coherent topics across a diverse spectrum of conversational prompts. By aligning these topics with human preference labels from model comparisons, we demonstrated that certain Large Language Models (LLMs) consistently outperform others within specific thematic areas. However, no single model exhibited dominance across all topics, underscoring the importance of domain specialization alongside general versatility.

All findings are grounded in real-world user preferences rather than synthetic benchmarks, enhancing the practical relevance of the insights. Our methodology provides an interpretable, topic-centric framework to evaluate LLM performance that complements traditional aggregate metrics.

Future extensions of this work may explore the incorporation of multimodal data—such as visual inputs—and deeper investigations into topical consistency and adaptability in LLMs. These advancements could drive the development of more responsive, user-aligned conversational agents.

### Acknowledgments

-   **BERTopic**: Used for transformer-based topic modeling, dimensionality reduction, clustering, and topic representation.
    
-   **LMSYS-Chat-1M**: The dataset used in this study, comprising head-to-head LLM comparisons with human preference labels.
    
-   **fastText**: Employed for language identification during preprocessing.
    
-   **HDBSCAN**, **UMAP**, and **scikit-learn**: Key components for unsupervised clustering and dimensionality reduction.
    
-   **Matplotlib**, **Seaborn**, and **Plotly**: Used for generating all visualizations and statistical plots.