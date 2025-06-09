# Synthetic-Data-For-Finance
This repository complements the CFA Institute's Research and Policy Center [Synthetic Data in Investment Management](some_url) report. It aims to serve as a centralized hub for generative AI (genAI) approaches to synthetic data generation and their applications within finance. The repository provides a curated list of libraries,  papers and case studies that can be used for synthetic data generation to aid practitioners and is regularly updated. 

## 📘 Contents
- [🧠 Overview](#-overview)
- [🛠️ Libraries](#-libraries)
- [📁 Case Studies](#-case-studies)
- [📚 Papers](#papers)
---

## 🧠 Overview

Synthetic data is artificially generated data designed to resemble real data. It can be used to address data-related challenges such as:
- Lack of historical data 
- Privacy and compliance concerns around data-sharing
- Overfitting in backtesting and model training
- Imbalanced datasets
- Lack of scenario diversity for stress testing

This repository focuses on genAI approaches to synthetic data generation, focusing on the following techniques:
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Diffusion models
- Large Language Models (LLMs)

These methods are more flexible than traditional statistical methodologies, allowing for each data type to be modelled - from textual datasets to time-series and tabular data, all common to financial professionals. As a result, synthetic data has a wide range of use cases from risk modelling to forecasting, sentiment analysis and stress-testing. 

---

## 🛠️ Libraries 

- [Synthetic Data Vault](https://github.com/sdv-dev/SDV): General-purpose synthetic data generation with statistical and genAI approaches.
- [Synthetic Data SDK, MOSTLY.AI](https://github.com/mostly-ai/mostlyai): Python Library for high-quality synthetic data generation|
- [HuggingFace Synthetic Data Generator](https://huggingface.co/blog/synthetic-data-generator): No-code natural language synthetic dataset builder.
- [nbsynthetic](https://github.com/NextBrain-ai/nbsynthetic): GAN-based synthetic tabular dataset creation.
- [synthcity](https://github.com/vanderschaarlab/synthcity)
- [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger): GAN-based time-series generation.
- [CTGAN](https://github.com/sdv-dev/CTGAN): GAN-based synthetic tabular dataset creation.
---


## 📁 Case Studies (LLM Example)

See [`/LLM`](./LLM) for an example using synthetic data to fine-tune a small LLM (Qwen3-0.6B) for financial sentiment classification. 

## 📚 Papers

### Variational Autoencoders
| Paper     | Release Date | Type of Data Modeled | Codebase|
|-----------|--------------|-----------------------|----------------|
|[An Overview of Variational Autoencoders for Source Separation, Finance, and Bio-Signal Applications](https://www.mdpi.com/1099-4300/24/1/55) | 2021 | *N/A* | *No official repo* |


### Generative Adversarial Networks

| Paper     | Release Date | Type of Data Modeled | Codebase|
|-----------|--------------|-----------------------|----------------|
| [SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning](https://arxiv.org/abs/2410.21203)| 2024         | Time Series           | [GitHub](https://github.com/samresume/SeriesGAN) |
| [Time-series Generative Adversarial Networks](https://papers.nips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) | 2019 | Time Series| [GitHub](https://github.com/jsyoon0823/TimeGAN/tree/master) | 
| [Simulating Asset Prices using Conditional Time-Series GAN](https://dl.acm.org/doi/10.1145/3677052.3698638) |2024         | Time Series          | [GitHub](https://github.com/riasataliistiaque/cts-gan) |
| [CorrGAN: Sampling Realistic Financial Correlation Matrices Using Generative Adversarial Networks](https://arxiv.org/abs/1910.09504)| 2019         | Financial Correlation Matrices | *No official repo* |
| [cCorrGAN: Conditional Correlation GAN for Learning Empirical Conditional Distributions in the Elliptope](https://arxiv.org/abs/2107.10606) |2021         | Financial Correlation Matrices | *No official repo* |
| [Conditional Sig-Wasserstein GANs for Time Series Generation](https://arxiv.org/abs/2006.05421) |2020 | Time Series | [GitHub](https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs) | 
| [Deep Hedging: Learning to Simulate Equity Option Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756)| 2019 | Equity Options | *No official repo* |
| [GANs and synthetic financial data: calculating VaR](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2365456) |2024 | Time-Series | *No official repo*|
| [A Modified CTGAN-Plus-Features Based Method for Optimal Asset Allocation](https://arxiv.org/abs/2302.02269) | 2023 | Time-Series | *No official repo* |






### Diffusion models

### Large Language Models
