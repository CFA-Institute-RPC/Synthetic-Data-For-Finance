# Synthetic-Data-For-Finance
This repository complements the CFA Institute's Research and Policy Center [Synthetic Data in Investment Management](https://rpc.cfainstitute.org/research/reports/2025/synthetic-data-in-investment-management) report. It aims to serve as a centralized hub for generative AI (genAI) approaches to synthetic data generation and their applications within finance. The repository provides a curated list of libraries,  papers and case studies that can be used for synthetic data generation to aid practitioners and is regularly updated. 

## 📘 Contents
- [🧠 Overview](#-overview)
- [🛠️ Libraries](#-libraries)
- [📁 Case Studies](#-case-studies)
- [📚 Papers](#-papers)
---

## 🧠 Overview

Synthetic data is artificially generated data designed to resemble real data. It can be used to address data-related challenges such as:
- Lack of historical data 
- Privacy and compliance concerns around data-sharing
- Overfitting in backtesting and model training
- Imbalanced datasets

This repository focuses on genAI approaches to synthetic data generation, focusing on the following:
- Variational Autoencoders (VAEs)
- Generative Adversarial Networks (GANs)
- Diffusion models
- Large Language Models (LLMs)

These methods are more flexible than traditional statistical methodologies, allowing for each data type to be modelled - from textual datasets to time-series and tabular data. As a result, synthetic data has a wide range of use cases within the industry, from enhanced risk modelling and portfolio optimization approaches to forecasting and sentiment analysis. 

---

## 🛠️ Libraries 

- [Synthetic Data Vault](https://github.com/sdv-dev/SDV): General-purpose synthetic data generation with statistical and genAI approaches.
- [Synthetic Data SDK, MOSTLY.AI](https://github.com/mostly-ai/mostlyai): Python Library for high-quality synthetic data generation
- [HuggingFace Synthetic Data Generator](https://huggingface.co/blog/synthetic-data-generator): No-code natural language synthetic dataset builder.
- [nbsynthetic](https://github.com/NextBrain-ai/nbsynthetic): GAN-based synthetic tabular dataset creation.
- [synthcity](https://github.com/vanderschaarlab/synthcity): GenAI based synthetic data library covering various data types.
- [DoppelGANger](https://github.com/fjxmlzn/DoppelGANger): GAN-based time-series generation.
- [CTGAN](https://github.com/sdv-dev/CTGAN): GAN-based model for synthetic tabular datasets.
---


## 📁 Case Studies

See [`/LLM`](./LLM) for an example using synthetic data to improve the performance of a fine-tuned small LLM (Qwen3-0.6B) for financial sentiment classification. 

## 📚 Papers

### Variational Autoencoders
| Paper     | Release Date | Type of Data Modeled | Codebase|
|-----------|--------------|-----------------------|----------------|
|[An Overview of Variational Autoencoders for Source Separation, Finance, and Bio-Signal Applications](https://www.mdpi.com/1099-4300/24/1/55) | 2021 | *N/A* | *No official repo* |
|[TimeVAE: A Variational Auto-Encoder for Multivariate Time Series Generation](https://arxiv.org/abs/2111.08095) | 2021 | Time Series | [GitHub](https://github.com/abudesai/timeVAE) |
|[Variational Autoencoders:  A Hands-Off Approach to Volatility](https://arxiv.org/abs/2102.03945) | 2021 | N/A | Implied Volatility | *No official repo* |




### Generative Adversarial Networks

| Paper     | Release Date | Type of Data Modeled | Codebase|
|-----------|--------------|-----------------------|----------------|
|[SeriesGAN: Time Series Generation via Adversarial and Autoregressive Learning](https://arxiv.org/abs/2410.21203)| 2024 | Time Series | [GitHub](https://github.com/samresume/SeriesGAN) |
|[Time-series Generative Adversarial Networks](https://papers.nips.cc/paper_files/paper/2019/hash/c9efe5f26cd17ba6216bbe2a7d26d490-Abstract.html) | 2019 | Time Series| [GitHub](https://github.com/jsyoon0823/TimeGAN/tree/master) | 
|[Simulating Asset Prices using Conditional Time-Series GAN](https://dl.acm.org/doi/10.1145/3677052.3698638) |2024 | Time Series | [GitHub](https://github.com/riasataliistiaque/cts-gan) |
|[CorrGAN: Sampling Realistic Financial Correlation Matrices Using Generative Adversarial Networks](https://arxiv.org/abs/1910.09504)| 2019 | Financial Correlation Matrices | *No official repo* |
|[cCorrGAN: Conditional Correlation GAN for Learning Empirical Conditional Distributions in the Elliptope](https://arxiv.org/abs/2107.10606) | 2021 | Financial Correlation Matrices | *No official repo* |
|[Conditional Sig-Wasserstein GANs for Time Series Generation](https://arxiv.org/abs/2006.05421) | 2020 | Time Series | [GitHub](https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs) | 
|[Deep Hedging: Learning to Simulate Equity Option Markets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3470756)| 2019 | Equity Options | *No official repo* |
|[GANs and synthetic financial data: calculating VaR](https://www.tandfonline.com/doi/full/10.1080/00036846.2024.2365456) | 2024 | Time-Series | *No official repo* |
|[A Modified CTGAN-Plus-Features Based Method for Optimal Asset Allocation](https://arxiv.org/abs/2302.02269) | 2023 | Time-Series | *No official repo* |
|[Autoencoding Conditional GAN for Portfolio Allocation Diversification](https://arxiv.org/abs/2207.05701) | 2022 | Time-Series | *No official repo* |
|[Data Synthesis based on Generative Adversarial Networks](https://arxiv.org/abs/1806.03384) | 2018 | Tabular | [GitHub](https://github.com/mahmoodm2/tableGAN) |
|[Financial Thought Experiment: A GAN-based Approach to Vast Robust Portfolio Selection](https://dl.acm.org/doi/10.5555/3491440.3492077) | 2021 | Time Series | *No official repo* |
|[Improved Data Generation for Enhanced Asset Allocation: A Synthetic Dataset Approach for the Fixed Income Universe](https://arxiv.org/abs/2311.16004) | 2023 | Financial Correlation Matrices | *No official repo* |
|[MTSS-GAN: Multivariate Time Series Simulation Generative Adversarial Networks](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3616557) | 2020 | Time Series | [GitHub](https://github.com/firmai/mtss-gan) |
|[PAGAN: Portfolio Analysis with Generative Adversarial Networks](https://arxiv.org/abs/1909.10578) | 2019 | Time Series | *No official repo* | 
|[Quant GANs: Deep Generation of Financial Time Series](https://arxiv.org/abs/1907.06673) | 2019 | Time Series | *No official repo*|
|[Tail-GAN:  Learning to Simulate Tail Risk Scenarios](https://arxiv.org/abs/2203.01664) | 2022| Time Series | [GitHub](https://github.com/chaozhang-ox/Tail-GAN) |
|[Time Series Simulation by Conditional Generative Adversarial Net](https://arxiv.org/abs/1904.11419) | 2019 | Time Series | *No official repo* |


### Diffusion models
| Paper     | Release Date | Type of Data Modeled | Codebase|
|-----------|--------------|-----------------------|----------------|
|[Denoising Diffusion Probabilistic Model for Realistic Financial Correlation Matrices](https://dl.acm.org/doi/10.1145/3677052.3698640) | 2024 | Financial Correlation Matrices | [GitHub](https://github.com/szymkubiak/DDPM-for-Correlation-Matrices) | 
|[FinDiff: Diffusion Models for Financial Tabular Data Generation](https://arxiv.org/abs/2309.01472) | 2023 | Tabular | [GitHub](https://github.com/sattarov/FinDiff) |
|[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) | 2021 | Image | [GitHub](https://github.com/CompVis/latent-diffusion) | 

### Large Language Models
| Paper     | Release Date | Type of Data Modeled | Codebase|
|-----------|--------------|-----------------------|----------------|
|[AugGPT: Leveraging ChatGPT for Text Data Augmentation](https://arxiv.org/abs/2302.13007) | 2023 | Text | [GitHub](https://github.com/yhydhx/AugGPT) |
|[Data Augmentation using LLMs: Data Perspectives, Learning Paradigms and Challenges](https://aclanthology.org/2024.findings-acl.97/) | 2024 | N/A | *No official repo* |
|[FinLLMs: A Framework for Financial Reasoning Dataset Generation with Large Language Models](https://arxiv.org/abs/2401.10744) | 2024 | Text | *No official repo* |
|[Simulating Financial Market via Large Language Model based Agents](https://arxiv.org/abs/2406.19966) | 2024 | Time Series | *No official repo* |

## 📣 Contribute
Feel free to contribute if you’d like to add a new paper, case study or tool.


