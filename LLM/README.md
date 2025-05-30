# 🚀 Enhancing Financial Sentiment Analysis with Synthetic Data: Fine-Tuning Qwen3-0.6B on FiQA-SA

# Why is sentiment analysis important in finance?
It is well known that unstructured data sources such as news articles and social media sentiment affect capital markets. Accurately classifying the sentiment of such content can be incorporated into investment theses and financial models to improve returns.

Computational models trained to process natural language and text present in these unstructured data sources are known as natural language processing (NLP) models. In recent years, large language models (LLMs) have proved exceptional in understanding and generating textual data, including NLP-related tasks such as sentiment analysis.

# What is fine-tuning and why do we need it?
Fine-tuning an LLM is used to enhance the performance of an LLM on specific tasks requiring specialized knowledge and understanding. For example, LLMs that have been 'fine-tuned' on finance-specific datasets typically perform better at a variety of tasks on financial datasets compared to the foundational, 'out-of-the-box' models such as ChatGPT.

We see an example of this from the table below, where LLMs fine-tuned on financial data outperform base, foundation models despite having significantly fewer parameters.

|Model          |Year|Backbone         |Open-source (Y/N)|Fine-Tuned on Financial Data?|HuggingFace Link                                                                                                                 |Financial Phrase Bank (micro F1) - 5-shot prompts|FiQA-SA (weighted F1) - 5-shot prompts|FOMC (micro-F1) - Zero-shot prompts|
|---------------|----|-----------------|-----------------|-----------------------------|---------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------|--------------------------------------|-----------------------------------|
|FinGPT         |2023|LLaMA 2-7B (base)|Y                |Yes                          |[FinGPT on HuggingFace](https://huggingface.co/FinGPT/fingpt-mt_llama2-7b_lora)                                                  |0.861                                            |0.825                                 |-                                  |
|InvestLM 65B   |2023|LLaMA 2-65B      |Y                |Yes                          |[InvestLM GitHub](https://github.com/AbaciNLP/InvestLM), [InvestLM on HuggingFace](https://huggingface.co/yixuantt/InvestLM2-AWQ)|0.71                                             |0.9                                   |0.61                               |
|FinLLaMA       |2024|LLaMA 3 8B       |-                |Yes                          |[FinLLaMA on HuggingFace](https://huggingface.co/TheFinAI/FinLLaMA-instruct)                                                     |0.7025                                           |0.7534                                |0.5                                |
|FinMA 7B Full  |2023|LLaMA 7B/30B     |Y                |Yes                          |[FinMA on HuggingFace](https://huggingface.co/TheFinAI/finma-7b-full)                                                            |0.87/0.88                                        |0.79                                  |0.52/0.49                          |
|BloombergGPT   |2023|BLOOM 50B        |N                |Yes                          |N/A                                                                                                                              |0.5107                                           |0.7505                                |-                                  |
|Mistral 7B     |2023|-                |Y                |No                           |[Mistral 7B on HuggingFace](https://huggingface.co/mistralai/Mistral-7B-v0.1)                                                    |0.29                                             |0.16                                  |0.37                               |
|LLaMA 2 7B Chat|2023|-                |Y                |No                           |[LLaMA 2 7B Chat](https://huggingface.co/meta-llama/Llama-2-7b-chat)                                                             |0.39                                             |0.76                                  |0.35                               |
|LLaMA 2 70B    |2023|-                |Y                |No                           |[LLaMA 2 70B](https://huggingface.co/meta-llama/Llama-2-70b)                                                                     |0.73                                             |0.83                                  |0.49                               |
|LLaMA 3 8B     |2024|-                |-                |No                           |[LLaMA 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B)                                                                   |0.6965                                           |0.5229                                |0.41                               |
|ChatGPT        |2023|-                |N                |No                           |-                                                                                                                                |0.78                                             |0.6                                   |0.64                               |
|GPT-4          |2023|-                |N                |No                           |-                                                                                                                                |0.78                                             |0.8                                   |0.71                               |
|Gemini         |2023|-                |N                |No                           |-                                                                                                                                |0.77                                             |0.81                                  |0.4                                |

# 🔍 Project Overview

In this project, we explore how **synthetic data** generated using ChatGPT-4o can be used to improve the performance of a small LLM to correctly classify the sentiment of financial news headlines and social media tweets. Specifically, we fine-tune the open-source **Qwen3-0.6B** model on the **FiQA-SA** dataset and demonstrate how synthetic data boosts performance on the validation dataset.

---

## 📈 Key Findings

- **Baseline performance**: Qwen3-0.6B fine-tuned on real FiQA-SA data achieves an **F1 score of 75.29%**.
- **With synthetic augmentation**: Qwen3-0.6B fine-tuned on a hybrid of real and synthetic data improves F1 score to **85.17%**.
- **Ablation study**: We show how different proportions of real + synthetic data affect performance.

---

## 🧪 Repository Structure

| Notebook | Purpose |
|----------|---------|
| `01-Preprocessing.ipynb` | Downloads, pre-processes and performs exploratory data analysis on the FiQA-SA dataset. |
| `02-Qwen3_sentiment_analysis.ipynb` | Fine-tunes Qwen3-0.6B on real FiQA-SA data and evaluates the model on the validation dataset |
| `03-Synthetic-Data-Generation.ipynb` | Uses ChatGPT-4o to generate synthetic financial text with sentiment labels. |
| `04-Qwen3_sentiment_analysis_with_synthetic_and_real_data.ipynb` | Fine-tunes Qwen3-0.6B on real + synthetic data and evaluates the model on the validation dataset. |
| `05-Synthetic_Proportion_Impact_Analysis.ipynb` | Compares the performance of Qwen3 models fine-tuned on different proportions of real + synthetic data.|


All notebooks are designed for **Google Colab** with their free T4 GPU.

---
