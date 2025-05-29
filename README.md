# Topic Modeling Insights on lmsys-chat-1m Dataset Using BERTopic

[Notebook Link](https://colab.research.google.com/drive/1V_vJt-1qsvT-ZPgdl0_ll21ZrPrWR1oA?usp=sharing)

## Model and Datasets link:
[Model and Datasets](https://drive.google.com/drive/folders/1hK-FzeTYoyusk-VGc3mb_Ad4W3feYaur?usp=drive_link)

## Task at hand and Why?

- This Notebook provides a comprehensive analysis using **BERTopic**, a state-of-the-art topic modeling technique, applied to the [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset on Hugging Face.
- The primary goal of this analysis is to delve into the dataset to **uncover and understand the prevalent topics** discussed in user interactions.
- This insight is **crucial for training new models and finetuning older models** on most sought after topics.

## Dataset Overview

- The [**lmsys-chat-1m**](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset comprises over one million user interaction entries, making it a rich source for understanding natural language processing in **conversational AI**.
- As the dataset is **request access** and not completely open, The notebook needs to have a basic authentication token, It can be done using **Google Colab Secrets** for safety reasons.
![Notebook Image](https://i.imgur.com/DW7Okuq.jpeg)
- Following are the columns of the dataset ->
**Columns/Dataset Features:** **`conversation_id, model, conversation, turn, language, openai_moderation, redacted`**
**Rows/Number of Datapoints:** **`1,000,000`**

![Dataset Image](https://i.imgur.com/qscD8vB.png)

- Each entry in the "conversation" represents an **array of dictionaries** of both user prompts and assistant responses.
- General structure of the conversation looks like ->

$$\begin{array}{l}
\text{[} \\
\quad \{ \text{"content": "User Prompt 1", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response 1", "role": "assistant"} \}, \\
\quad \{ \text{"content": "User Prompt 2", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response 2", "role": "assistant"} \}, \\
\quad \ldots, \\
\quad \{ \text{"content": "User Prompt n", "role": "user"} \}, \\
\quad \{ \text{"content": "Chatbot Response n", "role": "assistant"} \} \\
\text{]}
\end{array}$$

- Although due to resource constraints of Google Colab, I have used only ***10%*** subset of the original dataset.

**Loading the Data ->**

**1. Conversion of Hugging Face Dataset to Pandas DataFrame**

**Why?:**

[Pandas](https://pandas.pydata.org/) DataFrames offer a more intuitive interface for data manipulation and are generally faster due to the optimized nature of the library.

**Challenge:**

The dataset in question was substantial in size, leading to **excessive RAM consumption** when attempting to load the entire dataset into memory for conversion at once. This issue was causing the computational **notebook to crash**.

**Implementation Solution:**

To mitigate the memory overload issue, a **batching system was implemented**. The process involves:

1. Extracting segments of the dataset of smaller size.
1. Sequentially appending each batch to a Pandas DataFrame.
1. Iterating over 1 & 2 until n% of the data is loaded.

**Data Preprocessing**

***1. Handling Multilingual Data in the Dataset***

**Challenge: Multilingual Data**

The dataset contains **multilingual data**, which can lead to inconsistencies and inaccuracies in the analysis, as it complicates the linguistic processing and may skew the interpretation of topics.

**Implementation Solution:**

A **language based filtering** step was implemented to retain **only English** language texts.

This approach ensures that the dataset is **homogenized**, reducing complexity and improving the predictability of the model outcomes.

The language based filtered out data is stored in df\_english.

***2. Handling Multiple Interaction Turns in Conversations & Extraction of Text from the Complex Structure***

**First Challenge : Multiple Interaction Turns**

The dataset includes conversations that feature **multiple interaction turns** between a user and an assistant within a single conversation entry. For effective topic modeling, it is essential to **capture the essence of all user prompts** throughout the conversation, not just the initial ones.

**Second Challenge: Complex Dataset Structure and Targeted Content Extraction**

The dataset exhibits a **complex nested structure** and for effective implementation of [BERTopic](https://maartengr.github.io/BERTopic/index.html), which requires a **simplified array of strings format**, it is crucial to perform selective content extraction.

The challenge lies in efficiently **isolating and extracting only the user prompts** from this intricate conversation structure, as these prompts contain the primary content necessary for our topic modeling analysis.

**Data Format:**

Each conversation row is structured as follows:

["content": "User Prompt 1", "role": "user","content": "Chatbot Response 1", "role": "assistant","content": "User Prompt 2", "role": "user","content": "Chatbot Response 2", "role": "assistant",…,"content": "User Prompt n", "role": "user","content": "Chatbot Response n", "role": "assistant"]

**Implementation Solution:**

To address this challenge, the concatenate\_user\_messages() function was developed.

This function **extracts all user prompts** from a conversation by matching the "role" of each dictionary with "user" in the array and then **concatenates these messages into a single continuous string**.

This method ensures that every part of the user's input is considered, providing a comprehensive basis for subsequent topic modeling and analysis.

The function concatenate\_user\_messages is applied to each conversation within the df\_english['conversation'] column **iteratively** and stored in a **new column** named combined\_user\_prompts.

***3. General text cleaning using REGEX***

**Challenge : Unwanted Characters, Emoji, Icons, non ASCII**

The dataset includes many unwanted characters, emojis, icons, non ASCII

**Implementation Solution:**

To address this challenge, we used regex to further text cleanup

**Topic Modeling for Processed Data using BERTopic**

**What is BERTopic?**

BERTopic is a cutting-edge topic modeling technique that uses transformer-based embeddings, such as BERT, combined with clustering and dimensionality reduction techniques to uncover hidden topics in textual data. Unlike traditional models like Latent Dirichlet Allocation (LDA), BERTopic leverages contextual embeddings, enabling it to produce more semantically meaningful topics.

-----
**Key Components of BERTopic**

**1. BERT Embeddings**

- **What it does:** Utilizes transformer models (e.g., paraphrase-mpnet-base-v2) to generate contextual embeddings that represent text in a high-dimensional space.
- **Advantage:** Captures semantic nuances of text, outperforming traditional bag-of-words models.

**2. Dimensionality Reduction**

- **Technique:** UMAP (Uniform Manifold Approximation and Projection)
- **Purpose:** Reduces high-dimensional embeddings into a lower-dimensional space while preserving semantic relationships.
- **Parameters:**
  - n\_neighbors: Controls local vs. global structure preservation.
  - n\_components: Number of dimensions in the reduced space.
  - min\_dist: Controls clustering density.
  - metric: Defines distance calculations (e.g., cosine).

**3. Clustering**

- **Technique:** HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
- **Purpose:** Groups similar embeddings into clusters.
- **Parameters:**
  - min\_cluster\_size: Minimum documents per cluster.
  - min\_samples: Controls outlier sensitivity.
  - metric: Defines the clustering distance measure (e.g., euclidean).

**4. Topic Representation**

- **Technique:** c-TF-IDF (Class-based Term Frequency-Inverse Document Frequency)
- **Purpose:** Identifies the most representative terms for each topic.
- **Result:** Enhances interpretability of discovered topics.
-----
**Workflow Diagram**
![gmisharch2](https://github.com/user-attachments/assets/ff956df9-953e-4356-8c2f-9a869c68ccfa)


Below is a visual representation of BERTopic’s workflow:

-----
**Configuration of Our BERTopic Model**

Below is a detailed explanation of the configurations used in our BERTopic model:

|**Component**|**Parameter**|**Value**|**Effect**|
| :- | :- | :- | :- |
|**Embedding Model**|embedding\_model|paraphrase-mpnet-base-v2|Provides contextual embeddings with high semantic accuracy.|
|**UMAP**|n\_neighbors|20|Balances local and global structure preservation.|
||n\_components|8|Sets the dimensionality of the reduced space.|
||min\_dist|0\.1|Lower values create denser clusters.|
||metric|cosine|Measures similarity using cosine distance.|
|**HDBSCAN**|min\_cluster\_size|50|Ensures meaningful cluster sizes.|
||min\_samples|10|Balances sensitivity to outliers.|
||metric|euclidean|Defines clustering distance calculations.|
|**Topic Representation**|top\_n\_words|20|Highlights the top 20 words for each topic.|

-----
**Results: Prevalent Topics**

Below are some of the prominent topics discovered by our BERTopic model:

|**Topic ID**|**Name**|**Count**|**Representative Terms**|
| :- | :- | :- | :- |
|0|Roleplay|3,987|story, girl, character, game, roleplay|
|1|AI Assistant Queries|2,809|assistant, completion, repeat, system, instruction|
|2|Programming|1,079|import, int, self, const, return, class|
|3|Business and Industry|962|china, ltd, co, introduction, chemical|
|4|Coding Assistance|721|code, function, loop, debug, variable|
|5|Educational Resources|643|book, tutorial, explain, learn, teach|
|6|Physics & Mathematics|489|equation, solve, gravity, acceleration, theorem|

-----
**Visualizations**

**LLM Interaction Analysis using BERTopic**

**Overview**

Our approach uses BERTopic for unsupervised topic modeling on an LLM interaction dataset collected from human interactions with Large Language Models (LLMs). The dataset contains human preferences, allowing insights into topics effectively modeled by different LLM versions.

The dataset (January-April 2023 via the Vikuna Arena and Chatbot Arena) consists of 20K user preferences, involving over 31K unique IP addresses. Multilingual inputs include positive reinforcement (preferred LLM answers), negative reinforcement (less preferred answers), and neutral choices. Multilingual embedding, pre-trained via SentenceTransformers (paraphrase-multilingual-MiniLM-L12-v2), and UMAP dimensionality reduction, followed by HDBSCAN clustering, enhanced clustering performance.

**Methodology**

To evaluate BERTopic's effectiveness, we compared it against traditional clustering approaches:

- Traditional Methods: K-Means, LDA, and PCA.
- BERTopic: Enhanced results with hierarchical clustering, providing better interpretability.

BERTopic clustering identified clear, intuitively understandable groups, such as:

- **Technology & Science**
- **Finance**
- **Creativity and Arts**

Hierarchical clustering provided meaningful insight into topic clusters with a dendrogram clearly illustrating topic relationships.

**Identified Topics (Selection)**

- **Topic 0**: Games, Puzzles, Coding, and Logic
- **Topic 2**: Problem Solving, Logic
- **Topic 4**: Cooking, Recipes, Food
- **Topic 8**: Science, Health, and Environment
- **Topic 11**: Finance, Banking, Investments
- **Topic 15**: Creative Writing and Storytelling
- **Topic 18**: Lifestyle, Relationships, and Personal Well-being

(See the original image/document for complete topic details.)

**Visualization and Output Analysis**

**Topic Insights**

- **Top Topics Cumulative Coverage**:
  - Top 10 topics cover a significant proportion, validating their dominance in human preferences.

- **Topic Distribution Analysis**:
  - Indicates relative strengths of LLM versions across topics.
  - Heatmap provides intuitive visualization of topic strengths per model.

**Model Performance Insights**

Top-performing models include:

- gpt-4-1106-preview
- gpt-3.5-turbo-0613
- claude-2.1
- gpt-4-0125
- claude-3-haiku

Notable findings:

- gpt-4-1106-preview excels significantly in Gaming, Logic, Cooking.
- claude-2.1 performs notably in Problem-solving.
- gpt-3.5-turbo-0613 demonstrates strong balanced performance across topics.

**Balanced Model Performance**

One surprising insight: gpt-3.5-turbo-0314 showed the highest win rate (68.53%) relative to appearances, indicating exceptional balanced performance.

**Visualizations Provided**

- **Dendrogram**: Clearly illustrates topic relationships. 
![topic_hierarchy](https://github.com/user-attachments/assets/e0315ae8-84f9-42ed-b1ee-8d005ec4e2bc)


- **Bar Graph**: Cumulative topic coverage. 
![barchart](https://github.com/user-attachments/assets/e08889af-761b-45d2-9404-df6b3c58f846)


- **Heatmap**: Topic distribution and model strengths. 
![heatmap](https://github.com/user-attachments/assets/c9d6057b-610d-47ce-9394-e13880051534)

- **Bar Chart**: Balanced model win rate per appearance.
![balancedmodel](https://github.com/user-attachments/assets/6321533a-baa0-4f70-a803-81ac9d0472e9)


**Conclusions and Practical Implications**

- The BERTopic-based analysis delivers clear, interpretable insights into human preferences for LLM interactions.
- Specialized models show strengths in domain-specific tasks, while balanced models excel broadly.
- Insights guide targeted fine-tuning and highlight optimization opportunities for specialized use-cases.
-----
*This analysis informs practical applications, emphasizing effective model deployment strategies tailored to specific use-cases, optimizing both user satisfaction and model performance.*

**Additional Insights**

- **Topic Evolution:** The model enables dynamic topic modeling, making it possible to track changes in topics over time or across specific intervals in the dataset.
- **Application Scope:**
  - **Customer Feedback Analysis:** Extract recurring themes from user reviews or customer service logs.
  - **Content Categorization:** Automatically classify documents, articles, or blogs into relevant categories.
  - **Academic Research:** Reveal trends, methodologies, and key topics across scientific literature.
  - **AI Training:** Provides a foundation for training and fine-tuning conversational AI models based on prevalent topics.
-----
**Conclusion**

This Notebook provides a comprehensive analysis using **BERTopic**, a state-of-the-art topic modeling technique, applied to the [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) dataset on Hugging Face. The **primary goal** was to delve into the dataset to **uncover and understand the prevalent topics discussed in user interactions**. These insights are crucial for training new models and fine-tuning older models on the most sought-after topics.

The task involved uncovering dominant themes and their distributions in a dataset with multilingual content and varying conversational structures. The robust preprocessing pipeline solved subproblems such as:

- **Handling multilingual data** to ensure consistent insights.
- **Resolving conversational imbalances** due to varying turn lengths.
- **Managing redacted or noisy information** and addressing numerical anomalies in the data.
- **Streamlining text** using lemmatization and stop word elimination through spaCy.

The sequential processing workflow enabled a clean, consistent dataset that could be effectively modeled.

**Key Achievements**

1. **Topic Insights:** Dominant themes such as **AI Assistant Queries, Programming Concepts, Educational Resources**, and others were identified and visualized using inter-topic distance maps and probability distributions.
1. **Advanced Preprocessing:** Successfully addressed multilingual challenges, redacted data issues, and conversational imbalances.
1. **Versatile Applications:** The model's insights are applicable across domains like content categorization, conversational AI training, and customer feedback analysis.
1. **Visualization Mastery:** Interactive visualizations provided deep insights into topic relationships, term significance, and dataset structure.

**Future Scope**

- **Real-time Evolution Tracking:** Enhance the pipeline to dynamically track topic changes across time.
- **Multilingual Integration:** Extend support for diverse datasets with varied linguistic and structural patterns.
- **Customized Applications:** Tailor the workflow to meet specific industry needs, such as domain-specific categorization or customer experience optimization.
- **Robust Training for Robust Performance** : Training the model more robustly covering 100% of the dataset and handle the outlier cases better for better insights.

The methodologies demonstrated here highlight the power of modern NLP techniques in managing and interpreting complex datasets, paving the way for innovative, actionable applications in both research and industry.

**Acknowledgments**

- [lmsys-chat-1m](https://huggingface.co/datasets/lmsys/lmsys-chat-1m) - A dataset of user interactions with an AI chatbots.
- [BERTopic](https://github.com/MaartenGr/BERTopic) - A state-of-the-art topic modeling technique.


