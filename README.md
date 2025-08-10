# 🧠 Bias-Aware Content Analysis Platform

A Flask-based web application that retrieves, clusters, and analyzes textual content for potential biases, diversity, and fairness. The system uses advanced NLP and machine learning techniques to provide **Bias Inclusivity Quality (BIQ)**, **Content Fairness Score (CFS)**, and **Content Diversity Score (CDS)** for the retrieved documents.

---

## 🚀 Features

- **Bias-Aware Retrieval** – Retrieves relevant documents from domain-specific datasets while accounting for potential biases.
- **Content Clustering** – Groups similar documents using Gaussian Mixture Models (GMM) for better diversity analysis.
- **Bias Metrics Calculation** – Calculates Inclusivity, Diversity, and Fairness scores to evaluate content objectively.
- **Multi-Domain Support** – Works with `healthcare`, `finance`, `education`, and `general` domains.
- **Interactive API** – Exposes an `/analyze` endpoint to receive queries and return results in JSON format.

---

## 🛠 Tech Stack

- **Backend Framework:** Flask
- **Machine Learning:** scikit-learn, SentenceTransformers
- **Vectorization:** TF-IDF, Sentence Embeddings (`all-MiniLM-L6-v2`)
- **Clustering:** Gaussian Mixture Models (GMM)
- **Bias Detection Metrics:** Inclusivity, Diversity (NMI), Fairness (Disparate Impact Ratio)
- **Languages:** Python 3



